import json
import pytest
import torch

from kat_ai.training.trainer_v1 import TokenBlockDataset


class FakeTokenizer:
    def encode(self, text, out_type=int):
        return [int(x) for x in text.split()]
    def eos_id(self):
        return 9


def test_token_block_dataset_builds_next_token_pairs():
    ds = TokenBlockDataset(["1 2 3 4 5 6"], FakeTokenizer(), 3)
    x, y = ds[0]
    assert x.tolist() == [1, 2, 3]
    assert y.tolist() == [2, 3, 4]


def test_token_block_dataset_rejects_too_small_corpus():
    with pytest.raises(ValueError, match="Need at least"):
        TokenBlockDataset(["1"], FakeTokenizer(), 8)
