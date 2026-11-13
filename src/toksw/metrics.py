"""Compression metrics (NSL, bytes per token) and bits per byte."""
from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from .tok import eot_id


def token_counts(tok: Tokenizer, texts: Sequence[str]) -> np.ndarray:
    return np.array([len(e.ids) for e in tok.encode_batch(list(texts), add_special_tokens=False)])


def nsl(tok: Tokenizer, ref: Tokenizer, texts: Sequence[str]) -> float:
    """Normalized Sequence Length: sum |T(D_i)| / sum |T_ref(D_i)|."""
    return float(token_counts(tok, texts).sum() / token_counts(ref, texts).sum())


def bytes_per_token(tok: Tokenizer, texts: Sequence[str]) -> float:
    n_bytes = sum(len(t.encode("utf-8")) for t in texts)
    return n_bytes / float(token_counts(tok, texts).sum())


def roundtrip_ok(tok: Tokenizer, texts: Sequence[str]) -> float:
    """Fraction of documents with decode(encode(x)) == x."""
    encs = tok.encode_batch(list(texts), add_special_tokens=False)
    return float(np.mean([tok.decode(e.ids, skip_special_tokens=False) == t for e, t in zip(encs, texts)]))


def mask_logits(logits: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
    """Give zero probability to the vocabulary rows where ``allowed`` is False.

    ``allowed`` broadcasts against ``logits`` over the last (vocabulary) axis.
    Use float32 logits: the fill value does not fit in BF16.
    """
    return logits.masked_fill(~allowed, torch.finfo(logits.dtype).min)


@torch.no_grad()
def bits_per_byte(
    model: torch.nn.Module,
    tok: Tokenizer,
    texts: Sequence[str],
    id_map: np.ndarray | None,
    block_size: int = 1024,
    batch_size: int = 8,
    device: str = "cuda",
    allowed: torch.Tensor | None = None,
) -> dict:
    """bpb = -sum log2 p(x_t | x_<t) / #bytes over ``texts``.

    Documents are concatenated as ``EOT doc_1 EOT doc_2 ...`` and cut into
    non-overlapping windows; every document token is predicted once. The EOT
    targets are included in the loss (one per document, for every tokenizer).
    ``id_map`` maps raw tokenizer ids to model ids (``None`` = identity).
    ``allowed`` (bool, one per model id) restricts the softmax to those rows;
    ``None`` uses every row.
    """
    if not texts or not sum(len(t.encode("utf-8")) for t in texts):
        raise ValueError("bits per byte requires nonempty text")
    if block_size <= 0 or batch_size <= 0:
        raise ValueError("block_size and batch_size must be positive")
    was_training = model.training
    try:
        model.eval()
        eot = eot_id(tok)
        stream = [eot]
        for e in tok.encode_batch(list(texts), add_special_tokens=False):
            stream.extend(e.ids)
            stream.append(eot)
        ids = np.asarray(stream, dtype=np.int64)
        if id_map is not None:
            ids = id_map[ids]
        ids_t = torch.from_numpy(ids)
        n_bytes = sum(len(t.encode("utf-8")) for t in texts)

        def nll_sum(xs: list[torch.Tensor], ys: list[torch.Tensor]) -> float:
            x, y = torch.stack(xs).to(device), torch.stack(ys).to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                logits = model(input_ids=x).logits
            logits = logits.float()
            if allowed is not None:
                logits = mask_logits(logits, allowed.to(logits.device))
            return F.cross_entropy(logits.flatten(0, 1), y.flatten(), reduction="sum").item()

        n_pred = len(ids_t) - 1
        n_full = n_pred // block_size
        total_nll = 0.0
        for b in range(0, n_full, batch_size):
            starts = range(b * block_size, min(n_full, b + batch_size) * block_size, block_size)
            total_nll += nll_sum(
                [ids_t[s : s + block_size] for s in starts],
                [ids_t[s + 1 : s + 1 + block_size] for s in starts],
            )
        s = n_full * block_size
        if s < n_pred:  # last, shorter window
            total_nll += nll_sum([ids_t[s:n_pred]], [ids_t[s + 1 :]])
        n_tok = n_pred
        return {
            "bpb": total_nll / math.log(2) / n_bytes,
            "nll_per_token": total_nll / n_tok,
            "n_tokens": n_tok,
            "n_bytes": n_bytes,
        }

    finally:
        model.train(was_training)
