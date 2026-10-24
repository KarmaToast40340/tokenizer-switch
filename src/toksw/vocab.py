"""Union vocabulary, Fast Vocabulary Transfer (FVT) and pruning.

Both GPT-2 and our tokenizers are byte-level BPE with the same byte-to-unicode
alphabet, so two tokens are "the same" exactly when their vocabulary strings
are equal.

Index spaces:
  * old ids   : ids of the GPT-2 tokenizer (0..50256)
  * new ids   : ids of the new tokenizer
  * model ids : rows of the model's embedding matrix. Depends on the condition:
      keep        -> model ids = old ids
      swap        -> model ids = new ids
      progressive -> model ids = union ids (old ids first, then new-only tokens)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from tokenizers import Tokenizer


@dataclass
class UnionVocab:
    tokens: list[str]          # union id -> token string
    old2union: np.ndarray      # old id -> union id (identity)
    new2union: np.ndarray      # new id -> union id
    n_old: int

    @property
    def size(self) -> int:
        return len(self.tokens)

    @property
    def n_new_only(self) -> int:
        return self.size - self.n_old


def _id_to_token(tok: Tokenizer) -> list[str]:
    vocab = tok.get_vocab(with_added_tokens=True)
    out = [None] * len(vocab)
    for s, i in vocab.items():
        out[i] = s
    assert all(s is not None for s in out), "vocabulary ids are not contiguous"
    return out


def build_union(old_tok: Tokenizer, new_tok: Tokenizer) -> UnionVocab:
    old_tokens = _id_to_token(old_tok)
    new_tokens = _id_to_token(new_tok)
    index = {s: i for i, s in enumerate(old_tokens)}
    tokens = list(old_tokens)
    new2union = np.empty(len(new_tokens), dtype=np.int64)
    for j, s in enumerate(new_tokens):
        if s not in index:
            index[s] = len(tokens)
            tokens.append(s)
        new2union[j] = index[s]
    return UnionVocab(
        tokens=tokens,
        old2union=np.arange(len(old_tokens), dtype=np.int64),
        new2union=new2union,
        n_old=len(old_tokens),
    )


def fvt_pieces(old_tok: Tokenizer, token: str) -> list[int]:
    """Old-tokenizer ids that FVT averages for one new token.

    ``token`` is already in byte-level form, so we run the old BPE model on it
    directly (no pre-tokenization), which is exactly the segmentation the old
    tokenizer would give this byte string inside a single pre-token.
    """
    special = old_tok.token_to_id(token)
    if special is not None:
        return [special]
    return [t.id for t in old_tok.model.tokenize(token)]


@torch.no_grad()
def init_embeddings(
    old_emb: torch.Tensor,
    old_tok: Tokenizer,
    target_tokens: list[str],
    method: str = "fvt",
    seed: int = 0,
) -> torch.Tensor:
    """Build an embedding matrix for ``target_tokens`` from GPT-2's embeddings.

    Tokens that exist in the old vocabulary keep their row. Other tokens get:
      * ``fvt``   : the mean of the old rows of their old-tokenizer pieces
      * ``random``: a Gaussian draw matching the mean/std of the old matrix
    """
    old_vocab = old_tok.get_vocab(with_added_tokens=True)
    d = old_emb.shape[1]
    out = torch.empty(len(target_tokens), d, dtype=old_emb.dtype)
    g = torch.Generator().manual_seed(seed)
    mu, sigma = old_emb.mean(0), old_emb.std(0)
    n_copied = n_init = 0
    for i, s in enumerate(target_tokens):
        j = old_vocab.get(s)
        if j is not None:
            out[i] = old_emb[j]
            n_copied += 1
        elif method == "fvt":
            out[i] = old_emb[fvt_pieces(old_tok, s)].mean(0)
            n_init += 1
        elif method == "random":
            out[i] = mu + sigma * torch.randn(d, generator=g, dtype=old_emb.dtype)
            n_init += 1
        else:
            raise ValueError(method)
    print(f"[init:{method}] copied {n_copied} rows, initialized {n_init} new rows")
    return out


@torch.no_grad()
def prune_to_new(union_emb: torch.Tensor, union: UnionVocab) -> torch.Tensor:
    """Keep only the rows of the new vocabulary, in new-id order."""
    return union_emb[torch.as_tensor(union.new2union)].clone()
