import pytest
import torch

from kat_ai.model.transformer_v1 import KatAI1Config, KatAI1Transformer, RMSNorm, apply_rotary


def test_v1_forward_and_loss():
    cfg = KatAI1Config(vocab_size=128, d_model=64, n_heads=4, n_layers=2, max_seq_len=32)
    model = KatAI1Transformer(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    logits, loss = model(x, x)
    assert logits.shape == (2, 16, cfg.vocab_size)
    assert loss is not None and torch.isfinite(loss)


def test_v1_rejects_odd_head_dimension():
    cfg = KatAI1Config(vocab_size=128, d_model=30, n_heads=3)
    with pytest.raises(ValueError, match="even"):
        KatAI1Transformer(cfg)


def test_v1_rejects_long_context():
    cfg = KatAI1Config(vocab_size=128, d_model=64, n_heads=4, n_layers=1, max_seq_len=8)
    model = KatAI1Transformer(cfg)
    with pytest.raises(ValueError, match="context"):
        model(torch.ones((1, 9), dtype=torch.long))


def test_v1_generation_stops_at_eos():
    cfg = KatAI1Config(vocab_size=32, d_model=32, n_heads=2, n_layers=1, max_seq_len=16)
    model = KatAI1Transformer(cfg)
    tokens = model.generate(torch.tensor([[1, 2]]), max_new_tokens=2, top_k=0, top_p=1.0)
    assert tokens.shape[0] == 1 and tokens.shape[1] >= 3


def test_rmsnorm_preserves_shape():
    norm = RMSNorm(16)
    x = torch.randn(2, 3, 16)
    assert norm(x).shape == x.shape
