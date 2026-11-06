import math
from types import SimpleNamespace

import pytest
import torch

from toksw.metrics import bits_per_byte, roundtrip_ok


class UniformModel(torch.nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.vocab = vocab

    def forward(self, input_ids):
        return SimpleNamespace(logits=torch.zeros(*input_ids.shape, self.vocab))


@pytest.mark.parametrize("block,batch", [(1, 1), (7, 3), (64, 2)])
def test_bpb_analytic_uniform_including_tail_and_utf8(old_tok, block, batch):
    texts = ["def f():\n    return 42\n", "中文 café", "x"]
    model = UniformModel(old_tok.get_vocab_size()).eval()
    result = bits_per_byte(model, old_tok, texts, None, block, batch, "cpu")
    n = sum(len(old_tok.encode(t, add_special_tokens=False).ids) + 1 for t in texts)
    assert result["n_tokens"] == n
    assert result["bpb"] == pytest.approx(n * math.log2(model.vocab) / sum(len(t.encode()) for t in texts), rel=1e-6)
    assert not model.training


def test_roundtrip_including_special_token_and_unicode(old_tok):
    assert roundtrip_ok(old_tok, ["a<|endoftext|>b", "中文🙂\r\n\t café"]) == 1


def test_bpb_rejects_empty(old_tok):
    with pytest.raises(ValueError, match="nonempty"):
        bits_per_byte(UniformModel(old_tok.get_vocab_size()), old_tok, [""], None, device="cpu")
