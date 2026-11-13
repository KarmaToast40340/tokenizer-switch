from types import SimpleNamespace

import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from toksw.metrics import bits_per_byte


class ModeProbe(torch.nn.Module):
    def __init__(self, vocab, fail=False):
        super().__init__()
        self.dropout = torch.nn.Dropout(.1)
        self.vocab, self.fail = vocab, fail
        self.seen = []

    def forward(self, input_ids):
        self.seen.append((self.training,self.dropout.training,torch.is_grad_enabled()))
        if self.fail:
            raise RuntimeError('deliberate evaluation failure')
        return SimpleNamespace(logits=torch.zeros(*input_ids.shape,self.vocab))


@pytest.mark.parametrize('training',[True,False])
def test_evaluation_disables_dropout_and_restores_prior_mode(old_tok,training):
    model=ModeProbe(old_tok.get_vocab_size()).train(training)
    bits_per_byte(model,old_tok,['中文 abc'],None,32,1,'cpu')
    assert model.seen and all(e==(False,False,False) for e in model.seen)
    assert model.training==training and model.dropout.training==training
    assert torch.is_grad_enabled()


@pytest.mark.parametrize('training',[True,False])
def test_evaluation_restores_mode_after_forward_error(old_tok,training):
    model=ModeProbe(old_tok.get_vocab_size(),fail=True).train(training)
    with pytest.raises(RuntimeError,match='deliberate'):
        bits_per_byte(model,old_tok,['中文 abc'],None,32,1,'cpu')
    assert model.training==training and model.dropout.training==training
    assert torch.is_grad_enabled()


def test_same_checkpoint_evaluation_is_independent_of_prior_mode(old_tok):
    torch.manual_seed(2026)
    model=GPT2LMHeadModel(GPT2Config(vocab_size=old_tok.get_vocab_size(),n_positions=32,
                                   n_embd=16,n_layer=1,n_head=2,attn_pdrop=.1,embd_pdrop=.1,resid_pdrop=.1))
    texts=['中文 tokenizer comparison','same checkpoint, same inference']
    model.train()
    a=bits_per_byte(model,old_tok,texts,None,32,1,'cpu')
    assert model.training
    model.eval()
    b=bits_per_byte(model,old_tok,texts,None,32,1,'cpu')
    assert not model.training
    assert a==b
    assert all(p.grad is None for p in model.parameters())
