"""Native decoder-only Transformer with randomly initialized weights."""
from dataclasses import asdict, dataclass
import torch
from torch import nn
from torch.nn import functional as F

@dataclass
class KatAIConfig:
    vocab_size: int = 8000
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    max_seq_len: int = 256
    dropout: float = 0.1
    def validate(self):
        dims=(self.vocab_size,self.d_model,self.n_heads,self.n_layers,self.max_seq_len)
        if any(v<=0 for v in dims): raise ValueError("All model dimensions must be positive")
        if self.d_model % self.n_heads: raise ValueError("d_model must be divisible by n_heads")
        if not 0 <= self.dropout < 1: raise ValueError("dropout must be in [0, 1)")
    def to_dict(self): return asdict(self)

class CausalSelfAttention(nn.Module):
    def __init__(self,cfg):
        super().__init__(); self.n_heads=cfg.n_heads; self.head_dim=cfg.d_model//cfg.n_heads
        self.qkv=nn.Linear(cfg.d_model,3*cfg.d_model); self.proj=nn.Linear(cfg.d_model,cfg.d_model)
        self.attn_dropout=nn.Dropout(cfg.dropout); self.resid_dropout=nn.Dropout(cfg.dropout)
        self.register_buffer("mask",torch.tril(torch.ones(cfg.max_seq_len,cfg.max_seq_len)).view(1,1,cfg.max_seq_len,cfg.max_seq_len),persistent=False)
    def forward(self,x):
        b,t,c=x.shape; q,k,v=self.qkv(x).chunk(3,dim=-1)
        split=lambda z:z.view(b,t,self.n_heads,self.head_dim).transpose(1,2)
        q,k,v=split(q),split(k),split(v)
        scores=(q@k.transpose(-2,-1))*(self.head_dim**-0.5)
        scores=scores.masked_fill(self.mask[:,:,:t,:t]==0,torch.finfo(scores.dtype).min)
        out=self.attn_dropout(F.softmax(scores,dim=-1))@v
        return self.resid_dropout(self.proj(out.transpose(1,2).contiguous().view(b,t,c)))

class Block(nn.Module):
    def __init__(self,cfg):
        super().__init__(); self.norm1=nn.LayerNorm(cfg.d_model); self.attn=CausalSelfAttention(cfg)
        self.norm2=nn.LayerNorm(cfg.d_model)
        self.mlp=nn.Sequential(nn.Linear(cfg.d_model,4*cfg.d_model),nn.GELU(),nn.Linear(4*cfg.d_model,cfg.d_model),nn.Dropout(cfg.dropout))
    def forward(self,x): return x+self.mlp(self.norm2(x+self.attn(self.norm1(x))))

class KatAITransformer(nn.Module):
    def __init__(self,cfg):
        super().__init__(); cfg.validate(); self.config=cfg
        self.token_embedding=nn.Embedding(cfg.vocab_size,cfg.d_model)
        self.position_embedding=nn.Embedding(cfg.max_seq_len,cfg.d_model)
        self.dropout=nn.Dropout(cfg.dropout); self.blocks=nn.Sequential(*(Block(cfg) for _ in range(cfg.n_layers)))
        self.final_norm=nn.LayerNorm(cfg.d_model); self.lm_head=nn.Linear(cfg.d_model,cfg.vocab_size,bias=False)
        self.lm_head.weight=self.token_embedding.weight; self.apply(self._init_weights)
    @staticmethod
    def _init_weights(m):
        if isinstance(m,(nn.Linear,nn.Embedding)):
            nn.init.normal_(m.weight,mean=0.0,std=0.02)
            if isinstance(m,nn.Linear) and m.bias is not None: nn.init.zeros_(m.bias)
    def forward(self,input_ids,targets=None):
        _,length=input_ids.shape
        if length>self.config.max_seq_len: raise ValueError("Input exceeds configured context length")
        pos=torch.arange(length,device=input_ids.device).unsqueeze(0)
        x=self.dropout(self.token_embedding(input_ids)+self.position_embedding(pos))
        logits=self.lm_head(self.final_norm(self.blocks(x))); loss=None
        if targets is not None: loss=F.cross_entropy(logits.reshape(-1,logits.size(-1)),targets.reshape(-1))
        return logits,loss
    @torch.inference_mode()
    def generate(self,input_ids,max_new_tokens=100,temperature=.8,top_k=40,top_p=.9,repetition_penalty=1.05,eos_id=None):
        if temperature<=0 or not 0<top_p<=1: raise ValueError("Invalid sampling parameters")
        self.eval()
        for _ in range(max_new_tokens):
            logits,_=self(input_ids[:,-self.config.max_seq_len:]); logits=logits[:,-1,:]/temperature
            if repetition_penalty>1:
                for token in set(input_ids[0].tolist()):
                    logits[0,token]=logits[0,token]/repetition_penalty if logits[0,token]>0 else logits[0,token]*repetition_penalty
            if top_k and top_k>0:
                k=min(top_k,logits.size(-1)); threshold=torch.topk(logits,k).values[:,-1].unsqueeze(-1)
                logits=logits.masked_fill(logits<threshold,float("-inf"))
            if top_p<1:
                vals,idx=torch.sort(logits,descending=True); probs=F.softmax(vals,dim=-1)
                remove=torch.cumsum(probs,dim=-1)-probs>top_p; vals[remove]=float("-inf")
                logits=torch.full_like(logits,float("-inf")).scatter(1,idx,vals)
            token=torch.multinomial(F.softmax(logits,dim=-1),1); input_ids=torch.cat([input_ids,token],dim=1)
            if eos_id is not None and int(token[0,0])==eos_id: break
        return input_ids
