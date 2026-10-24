import numpy as np
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from toksw import EOT
from toksw.data import MixedBatchSampler, TokenFile, encode_corpus, iter_jsonl, token_byte_lengths
from toksw.metrics import bits_per_byte, nsl
from toksw.vocab import build_union


def _files(tmp_path, corpus, old_tok, new_tok):
    encode_corpus(old_tok, corpus, tmp_path / "tok" / "old")
    encode_corpus(new_tok, corpus, tmp_path / "tok" / "new")
    return TokenFile(tmp_path / "tok" / "old"), TokenFile(tmp_path / "tok" / "new")


def test_encode_roundtrip(tmp_path, corpus, old_tok, new_tok):
    old, _ = _files(tmp_path, corpus, old_tok, new_tok)
    docs = [d["text"] for d in iter_jsonl(corpus)]
    assert old.n_docs == len(docs)
    ids = old.tokens[old.offsets[3] : old.offsets[4] - 1].tolist()  # drop EOT
    assert old_tok.decode(ids) == docs[3]


def _sampler(old, new, u, tokens, p):
    return MixedBatchSampler(old, new, u.old2union, u.new2union, token_byte_lengths(tokens, {EOT}),
                             lambda s: p, block_size=32, batch_size=6, seed=7)


def test_sampler_same_documents_across_conditions(tmp_path, corpus, old_tok, new_tok):
    old, new = _files(tmp_path, corpus, old_tok, new_tok)
    u = build_union(old_tok, new_tok)
    xa, _, ia = _sampler(old, new, u, u.tokens, 0.0).get_batch(5, 1)
    xb, _, ib = _sampler(old, new, u, u.tokens, 1.0).get_batch(5, 1)
    assert ia["frac_new"] == 0 and ib["frac_new"] == 1
    # same starting text, different encodings
    da = old_tok.decode([int(i) for i in xa[0]])
    inv = {int(v): k for k, v in enumerate(u.new2union)}
    db = new_tok.decode([inv[int(i)] for i in xb[0]])
    n = min(len(da), len(db)) // 2
    assert da[:n] == db[:n]
    # determinism
    xa2, _, _ = _sampler(old, new, u, u.tokens, 0.0).get_batch(5, 1)
    assert torch.equal(xa, xa2)


def test_bits_per_byte_runs(old_tok, texts):
    cfg = GPT2Config(vocab_size=old_tok.get_vocab_size(), n_positions=64, n_embd=32, n_layer=2, n_head=2)
    model = GPT2LMHeadModel(cfg)
    res = bits_per_byte(model, old_tok, texts[:20], None, block_size=64, batch_size=4, device="cpu")
    total = sum(len(t.encode()) for t in texts[:20])
    assert res["n_bytes"] == total and res["bpb"] > 0
    n_tok = sum(len(old_tok.encode(t).ids) + 1 for t in texts[:20])
    assert res["n_tokens"] == n_tok


def test_nsl_identity(old_tok, texts):
    assert nsl(old_tok, old_tok, texts[:10]) == 1.0


def test_chunk_text_roundtrip():
    from toksw.data import chunk_text
    text = "\n".join(f"line {i} " + "x" * (i % 17) for i in range(300))
    pieces = chunk_text(text, 120)
    assert "".join(pieces) == text
    assert all(len(p) <= 120 for p in pieces)
    assert all(p.endswith("\n") for p in pieces[:-1])  # cut at line breaks
    assert chunk_text("a" * 50, 20) == ["a" * 20, "a" * 20, "a" * 10]  # no line break: hard cut
    assert chunk_text("short", 100) == ["short"]


def test_encode_uint16_and_max_bytes(tmp_path, corpus, old_tok, new_tok):
    n_old, _ = encode_corpus(old_tok, corpus, tmp_path / "o", max_bytes=2_000)
    n_new, _ = encode_corpus(new_tok, corpus, tmp_path / "n", max_bytes=2_000)
    assert n_old == n_new < len(list(iter_jsonl(corpus)))  # same documents for every tokenizer
    f = TokenFile(tmp_path / "o")
    assert f.tokens.dtype == np.uint16 and f.n_docs == n_old
    docs = [d["text"] for d in iter_jsonl(corpus)]
    assert old_tok.decode(f.tokens[f.offsets[2] : f.offsets[3] - 1].tolist()) == docs[2]
