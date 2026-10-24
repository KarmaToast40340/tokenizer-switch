"""Build GPT-2 small for each experimental condition.

  A  keep         GPT-2 tokenizer, GPT-2 embeddings
  B  swap/fvt     new tokenizer, new rows initialized with FVT
  C  swap/random  new tokenizer, new rows initialized at random
  E  progressive  union vocabulary (old + new-only rows, FVT), pruned at the end

GPT-2 ties input and output embeddings, so resizing ``wte`` is enough.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from tokenizers import Tokenizer
from transformers import GPT2LMHeadModel

from . import EOT
from .data import token_byte_lengths
from .vocab import UnionVocab, _id_to_token, build_union, init_embeddings


@dataclass
class Setup:
    model: GPT2LMHeadModel
    old_map: np.ndarray | None     # raw old id -> model id
    new_map: np.ndarray | None     # raw new id -> model id
    token_bytes: np.ndarray        # model id -> bytes
    union: UnionVocab | None = None


def _set_embeddings(model: GPT2LMHeadModel, emb: torch.Tensor) -> None:
    model.resize_token_embeddings(emb.shape[0], mean_resizing=False)
    model.transformer.wte.weight.data.copy_(emb)
    model.tie_weights()
    assert model.lm_head.weight.data_ptr() == model.transformer.wte.weight.data_ptr()


def build(
    condition: str,
    old_tok: Tokenizer,
    new_tok: Tokenizer | None,
    base: str = "gpt2",
    init: str = "fvt",
    seed: int = 0,
) -> Setup:
    model = GPT2LMHeadModel.from_pretrained(base)
    old_emb = model.transformer.wte.weight.data.float().cpu()
    old_tokens = _id_to_token(old_tok)
    assert len(old_tokens) == old_emb.shape[0], "base model and old tokenizer disagree"

    if condition == "keep":
        return Setup(model, np.arange(len(old_tokens)), None, token_byte_lengths(old_tokens, {EOT}))

    assert new_tok is not None
    if condition == "swap":
        new_tokens = _id_to_token(new_tok)
        _set_embeddings(model, init_embeddings(old_emb, old_tok, new_tokens, init, seed))
        return Setup(model, None, np.arange(len(new_tokens)), token_byte_lengths(new_tokens, {EOT}))

    if condition == "progressive":
        union = build_union(old_tok, new_tok)
        print(f"[union] {union.n_old} old + {union.n_new_only} new-only = {union.size} rows")
        _set_embeddings(model, init_embeddings(old_emb, old_tok, union.tokens, init, seed))
        return Setup(model, union.old2union, union.new2union, token_byte_lengths(union.tokens, {EOT}), union)

    raise ValueError(condition)
