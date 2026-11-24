from pathlib import Path
import math, random
import torch
from torch.utils.data import Dataset,DataLoader
from kat_ai.model.transformer import KatAIConfig,KatAITransformer
from kat_ai.data.jsonl import read_jsonl_texts

class TokenBlockDataset(Dataset):
    def __init__(self,texts,processor,block_size):
        ids=[]
        for text in texts: ids.extend(processor.encode(text,out_type=int)); ids.append(processor.eos_id())
        self.data=torch.tensor(ids,dtype=torch.long); self.block_size=block_size
        if self.data.numel()<block_size+1: raise ValueError(f"Need at least {block_size+1} tokens; got {self.data.numel()}")
    def __len__(self): return max(1,(self.data.numel()-1)//self.block_size)
    def __getitem__(self,index):
        start=index*self.block_size; chunk=self.data[start:start+self.block_size+1]
        if chunk.numel()<self.block_size+1: chunk=torch.cat([chunk,self.data[:self.block_size+1-chunk.numel()]])
        return chunk[:-1],chunk[1:]

def _save(payload,path):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+".tmp"); torch.save(payload,temp); temp.replace(path)

def train_model(data_path,tokenizer_path,output_path,steps=2000,batch_size=4,learning_rate=3e-4,
                seed=42,d_model=128,n_heads=4,n_layers=4,max_seq_len=256,resume=False,
                validation_interval=100,warmup_steps=100,validation_batches=20):
    import sentencepiece as spm
    random.seed(seed); torch.manual_seed(seed)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor=spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    cfg=KatAIConfig(vocab_size=processor.get_piece_size(),d_model=d_model,n_heads=n_heads,n_layers=n_layers,max_seq_len=max_seq_len)
    texts=read_jsonl_texts(data_path); random.shuffle(texts)
    split=max(1,int(len(texts)*.95)) if len(texts)>=10 else len(texts)
    train_texts,val_texts=texts[:split],texts[split:] or texts
    train_ds=TokenBlockDataset(train_texts,processor,max_seq_len); val_ds=TokenBlockDataset(val_texts,processor,max_seq_len)
    train_dl=DataLoader(train_ds,batch_size=batch_size,shuffle=True); val_dl=DataLoader(val_ds,batch_size=batch_size)
    model=KatAITransformer(cfg).to(device); opt=torch.optim.AdamW(model.parameters(),lr=learning_rate,weight_decay=.01); step=0
    if resume:
        ckpt=torch.load(output_path,map_location=device,weights_only=False)
        if ckpt["config"]!=cfg.to_dict(): raise ValueError("Checkpoint config mismatch")
        model.load_state_dict(ckpt["model_state"]); opt.load_state_dict(ckpt["optimizer_state"]); step=int(ckpt["step"])
    def lr_at(s):
        if warmup_steps and s<warmup_steps: return learning_rate*(s+1)/warmup_steps
        p=min(max((s-warmup_steps)/max(1,steps-warmup_steps),0),1)
        return learning_rate*.5*(1+math.cos(math.pi*p))
    while step<steps:
        for x,y in train_dl:
            if step>=steps: break
            for group in opt.param_groups: group["lr"]=lr_at(step)
            x,y=x.to(device),y.to(device); opt.zero_grad(set_to_none=True); _,loss=model(x,y)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); step+=1
            if step==1 or step%25==0: print(f"step={step}/{steps} train_loss={loss.item():.4f}")
            if validation_interval and step%validation_interval==0:
                model.eval(); losses=[]
                with torch.inference_mode():
                    for i,(vx,vy) in enumerate(val_dl):
                        if i>=validation_batches: break
                        _,vl=model(vx.to(device),vy.to(device)); losses.append(float(vl))
                model.train(); print(f"step={step} validation_loss={sum(losses)/max(1,len(losses)):.4f}")
            _save({"model_state":model.state_dict(),"optimizer_state":opt.state_dict(),"step":step,
                   "config":cfg.to_dict(),"tokenizer_path":str(tokenizer_path),"training_seed":seed},output_path)
    return model
