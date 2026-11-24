"""KatAI K.AI 1 decoder-only Transformer architecture.

Architecture: pre-norm RMSNorm, RoPE, SwiGLU MLP, and PyTorch SDPA.
This is a newly initialized model; it does not load pretrained weights.
"""
from dataclasses import asdict, dataclass
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class KatAI1Config:
    vocab_size: int = 8000
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    max_seq_len: int = 256
    dropout: float = 0.05
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5

    def validate(self):
        dims = (self.vocab_size, self.d_model, self.n_heads, self.n_layers, self.max_seq_len)
        if any(v <= 0 for v in dims):
            raise ValueError("All model dimensions must be positive")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if (self.d_model // self.n_heads) % 2:
            raise ValueError("head dimension must be even for rotary embeddings")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.rope_theta <= 0 or self.norm_eps <= 0:
            raise ValueError("rope_theta and norm_eps must be positive")

    def to_dict(self):
        return asdict(self)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        scale = torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x.float() * scale).to(dtype=x.dtype) * self.weight


def _rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary(q, k, theta: float):
    """Apply RoPE to [batch, heads, sequence, head_dim] query/key tensors."""
    seq_len = q.size(-2)
    dim = q.size(-1)
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, device=q.device, dtype=torch.float32) / dim))
    positions = torch.arange(seq_len, device=q.device, dtype=torch.float32)
    angles = torch.outer(positions, inv_freq)
    emb = torch.cat((angles, angles), dim=-1)
    cos = emb.cos().to(dtype=q.dtype)[None, None, :, :]
    sin = emb.sin().to(dtype=q.dtype)[None, None, :, :]
    return (q * cos + _rotate_half(q) * sin,
            k * cos + _rotate_half(k) * sin)


class CausalSelfAttentionV1(nn.Module):
    def __init__(self, cfg: KatAI1Config):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.rope_theta = cfg.rope_theta
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.attn_dropout = cfg.dropout
        self.resid_dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        batch, seq_len, width = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        def split_heads(t):
            return t.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)
        q, k = apply_rotary(q, k, self.rope_theta)
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=True,
        )
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, width)
        return self.resid_dropout(self.out_proj(out))


class SwiGLU(nn.Module):
    def __init__(self, cfg: KatAI1Config):
        super().__init__()
        hidden = (8 * cfg.d_model) // 3
        hidden = ((hidden + 63) // 64) * 64
        self.gate_proj = nn.Linear(cfg.d_model, hidden, bias=False)
        self.up_proj = nn.Linear(cfg.d_model, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, cfg.d_model, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class TransformerBlockV1(nn.Module):
    def __init__(self, cfg: KatAI1Config):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = CausalSelfAttentionV1(cfg)
        self.mlp_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.mlp = SwiGLU(cfg)

    def forward(self, x):
        x = x + self.attn(self.attn_norm(x))
        return x + self.mlp(self.mlp_norm(x))


class KatAI1Transformer(nn.Module):
    """K.AI 1 model; random initialization only."""
    architecture_version = "1.0.0"

    def __init__(self, cfg: KatAI1Config):
        super().__init__()
        cfg.validate()
        self.config = cfg
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)
        self.blocks = nn.Sequential(*(TransformerBlockV1(cfg) for _ in range(cfg.n_layers)))
        self.final_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, targets=None):
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        if input_ids.size(1) > self.config.max_seq_len:
            raise ValueError("Input exceeds configured context length")
        x = self.dropout(self.token_embedding(input_ids))
        x = self.final_norm(self.blocks(x))
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            if targets.shape != input_ids.shape:
                raise ValueError("targets must have the same shape as input_ids")
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.inference_mode()
    def generate(self, input_ids, max_new_tokens=100, temperature=0.8, top_k=40,
                 top_p=0.9, repetition_penalty=1.05, eos_id=None):
        if temperature <= 0 or not 0 < top_p <= 1 or top_k < 0:
            raise ValueError("Invalid sampling parameters")
        if input_ids.ndim != 2 or input_ids.size(0) != 1:
            raise ValueError("generate currently expects input shape [1, sequence]")
        self.eval()
        for _ in range(max_new_tokens):
            context = input_ids[:, -self.config.max_seq_len:]
            logits, _ = self(context)
            logits = logits[:, -1, :] / temperature
            if repetition_penalty > 1:
                seen = input_ids[0].unique()
                values = logits[0, seen]
                logits[0, seen] = torch.where(values > 0, values / repetition_penalty,
                                               values * repetition_penalty)
            if top_k:
                k = min(top_k, logits.size(-1))
                threshold = torch.topk(logits, k).values[:, -1, None]
                logits = logits.masked_fill(logits < threshold, float("-inf"))
            if top_p < 1:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                sorted_probs = F.softmax(sorted_logits, dim=-1)
                remove = sorted_probs.cumsum(dim=-1) - sorted_probs > top_p
                sorted_logits[remove] = float("-inf")
                filtered = torch.full_like(logits, float("-inf"))
                logits = filtered.scatter(1, sorted_indices, sorted_logits)
            probs = F.softmax(logits, dim=-1)
            if not torch.isfinite(probs).all():
                raise RuntimeError("Generation produced invalid token probabilities")
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat((input_ids, next_token), dim=1)
            if eos_id is not None and int(next_token.item()) == eos_id:
                break
        return input_ids
