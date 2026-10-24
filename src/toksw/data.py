"""Token storage (uint32 memmaps) and the mixed-tokenization batch sampler.

On disk, one encoded corpus is two files:
  ``<name>.bin``  token ids (raw ids of that tokenizer), each document
                  followed by the end-of-text token; uint16 when the vocabulary
                  fits (GPT-2 and our tokenizers do), else uint32
  ``<name>.idx``  int64 offsets: start of document i, plus a final end offset

The same ``train.jsonl`` is encoded once with every tokenizer we need, so
document i is the same text in every ``.bin``. This is what lets the sampler
switch tokenization per sequence while keeping the data order fixed.
"""
from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer

from .tok import eot_id


def iter_jsonl(path: str | Path, limit: int | None = None) -> Iterator[dict]:
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            yield json.loads(line)


def chunk_text(text: str, max_chars: int) -> list[str]:
    """Cut ``text`` into consecutive pieces of at most ``max_chars`` characters,
    at line breaks when possible. Joining the pieces gives back ``text``."""
    pieces, start = [], 0
    while len(text) - start > max_chars:
        cut = text.rfind("\n", start, start + max_chars)
        cut = start + max_chars if cut < start else cut + 1
        pieces.append(text[start:cut])
        start = cut
    pieces.append(text[start:])
    return pieces


def encode_corpus(
    tok: Tokenizer,
    jsonl: str | Path,
    out_prefix: str | Path,
    batch_docs: int = 2048,
    max_bytes: int | None = None,
) -> tuple[int, int]:
    """Encode the documents of ``jsonl``; returns (n_docs, n_tokens).

    ``max_bytes`` stops after the document that reaches this many UTF-8 bytes of
    text. The cut depends only on the text, so every tokenizer encodes the same
    documents (the sampler requires it)."""
    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    eot = eot_id(tok)
    dtype = np.uint16 if tok.get_vocab_size(with_added_tokens=True) <= 2**16 else np.uint32
    offsets = [0]
    n_tokens = 0
    with open(out_prefix.with_suffix(".bin"), "wb") as fbin:
        batch: list[str] = []

        def flush():
            nonlocal n_tokens
            for enc in tok.encode_batch(batch, add_special_tokens=False):
                ids = np.asarray(enc.ids + [eot], dtype=dtype)
                ids.tofile(fbin)
                n_tokens += len(ids)
                offsets.append(n_tokens)
            batch.clear()

        n_bytes = 0
        for doc in iter_jsonl(jsonl):
            batch.append(doc["text"])
            n_bytes += len(doc["text"].encode("utf-8"))
            if len(batch) == batch_docs:
                flush()
            if max_bytes is not None and n_bytes >= max_bytes:
                break
        if batch:
            flush()
    np.asarray(offsets, dtype=np.int64).tofile(out_prefix.with_suffix(".idx"))
    return len(offsets) - 1, n_tokens


class TokenFile:
    def __init__(self, prefix: str | Path):
        prefix = Path(prefix)
        self.offsets = np.fromfile(prefix.with_suffix(".idx"), dtype=np.int64)
        # the token width (uint16 or uint32) follows from the file size
        n_tokens = int(self.offsets[-1])
        width = prefix.with_suffix(".bin").stat().st_size // max(1, n_tokens)
        dtype = {2: np.uint16, 4: np.uint32}[width]
        self.tokens = np.memmap(prefix.with_suffix(".bin"), dtype=dtype, mode="r")
        self.n_docs = len(self.offsets) - 1

    def window(self, doc: int, length: int) -> np.ndarray:
        """``length`` tokens starting at the start of ``doc`` (crossing into the
        following documents). Wraps around at the end of the corpus."""
        start = int(self.offsets[doc])
        end = start + length
        if end <= len(self.tokens):
            return np.asarray(self.tokens[start:end], dtype=np.int64)
        head = np.asarray(self.tokens[start:], dtype=np.int64)
        tail = np.asarray(self.tokens[: end - len(self.tokens)], dtype=np.int64)
        return np.concatenate([head, tail])


class MixedBatchSampler:
    """Draws training sequences, each encoded with the old or the new tokenizer.

    For a given (seed, step, micro-step), the documents drawn and the uniform
    numbers used for the tokenizer choice are the same in every condition, so
    conditions A, B, C and E see the same data in the same order. Each sequence
    starts at a document boundary, so its two possible encodings start on the
    same text.

    ``old_map`` / ``new_map`` turn raw tokenizer ids into model ids (see
    ``vocab.py``); pass ``None`` for a tokenizer the condition never uses.
    """

    def __init__(
        self,
        old: TokenFile | None,
        new: TokenFile | None,
        old_map: np.ndarray | None,
        new_map: np.ndarray | None,
        token_bytes: np.ndarray,
        p_new: Callable[[int], float],
        block_size: int,
        batch_size: int,
        seed: int = 0,
    ):
        ref = old if old is not None else new
        assert ref is not None
        if old is not None and new is not None:
            assert old.n_docs == new.n_docs, "old and new corpora must hold the same documents"
        self.old, self.new = old, new
        self.old_map, self.new_map = old_map, new_map
        self.token_bytes = token_bytes  # model id -> UTF-8 byte length of the token
        self.p_new = p_new
        self.block_size = block_size
        self.batch_size = batch_size
        self.seed = seed
        self.n_docs = ref.n_docs

    def get_batch(self, step: int, micro: int = 0):
        rng = np.random.default_rng([self.seed, step, micro])
        docs = rng.integers(0, self.n_docs, size=self.batch_size)
        u = rng.random(self.batch_size)
        p = self.p_new(step)
        rows, n_new = [], 0
        for d, ui in zip(docs, u):
            use_new = ui < p
            src, mp = (self.new, self.new_map) if use_new else (self.old, self.old_map)
            if src is None:
                raise RuntimeError(f"step {step}: p_new={p} needs a tokenizer that is not loaded")
            rows.append(mp[src.window(int(d), self.block_size + 1)])
            n_new += int(use_new)
        arr = torch.from_numpy(np.stack(rows))
        x, y = arr[:, :-1].contiguous(), arr[:, 1:].contiguous()
        n_bytes = int(self.token_bytes[y.numpy()].sum())
        return x, y, {"n_bytes": n_bytes, "frac_new": n_new / self.batch_size, "p_new": p}


def token_byte_lengths(tokens: list[str], special: set[str]) -> np.ndarray:
    """UTF-8 byte length of each token. In byte-level BPE every character of a
    token string stands for exactly one byte; special tokens count 0 bytes."""
    return np.array([0 if t in special else len(t) for t in tokens], dtype=np.int64)
