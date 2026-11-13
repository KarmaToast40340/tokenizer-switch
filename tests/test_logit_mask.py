"""Opt-in E fix: restrict each sequence's softmax to its own tokenizer's rows.

Tiny random models are test fixtures, never project results.
"""
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from test_metrics import UniformModel
from toksw import EOT
from toksw import model as M
from toksw.data import MixedBatchSampler, TokenFile, encode_corpus, token_byte_lengths
from toksw.metrics import bits_per_byte, mask_logits
from toksw.vocab import allowed_rows, build_union

ROOT = Path(__file__).resolve().parents[1]
TEXTS = ["def f(x):\n    return x + 1\n", "import numpy as np\nprint(np.arange(3))", "class A:\n    pass"]


def test_allowed_rows_cover_each_vocabulary(old_tok, new_tok):
    u = build_union(old_tok, new_tok)
    old_allowed, new_allowed = allowed_rows(u)
    assert old_allowed.sum() == u.n_old and old_allowed[: u.n_old].all()
    assert new_allowed.sum() == new_tok.get_vocab_size() and new_allowed[torch.as_tensor(u.new2union)].all()
    assert not new_allowed[u.n_old:].logical_not().any()  # every new-only row belongs to the new vocabulary


def test_masked_rows_get_zero_probability():
    logits = torch.randn(2, 3, 6)
    allowed = torch.tensor([True, False, True, True, False, True])
    p = torch.softmax(mask_logits(logits, allowed), -1)
    assert torch.all(p[..., ~allowed] == 0) and torch.allclose(p.sum(-1), torch.ones(2, 3))
    loss = torch.nn.functional.cross_entropy(mask_logits(logits, allowed).flatten(0, 1), torch.tensor([0, 2, 3, 5, 0, 2]))
    assert torch.isfinite(loss)


def test_bpb_uniform_with_mask_counts_only_allowed_rows(old_tok):
    vocab = old_tok.get_vocab_size()
    used = {i for e in old_tok.encode_batch(TEXTS, add_special_tokens=False) for i in e.ids}
    allowed = torch.zeros(vocab, dtype=torch.bool)
    allowed[sorted(used | {old_tok.token_to_id(EOT)})] = True  # only never-targeted rows are masked
    assert 0 < allowed.sum() < vocab
    model = UniformModel(vocab).eval()
    full = bits_per_byte(model, old_tok, TEXTS, None, 16, 2, "cpu")
    masked = bits_per_byte(model, old_tok, TEXTS, None, 16, 2, "cpu", allowed)
    assert masked["bpb"] == pytest.approx(full["bpb"] * math.log(allowed.sum().item()) / math.log(vocab), rel=1e-6)
    assert bits_per_byte(model, old_tok, TEXTS, None, 16, 2, "cpu", None) == full


def test_sampler_reports_row_choice_without_changing_batches(tmp_path, corpus, old_tok, new_tok):
    u = build_union(old_tok, new_tok)
    for name, tok in [("old", old_tok), ("new", new_tok)]:
        encode_corpus(tok, corpus, tmp_path / name)
    old, new = TokenFile(tmp_path / "old"), TokenFile(tmp_path / "new")
    s = MixedBatchSampler(old, new, u.old2union, u.new2union, token_byte_lengths(u.tokens, {EOT}),
                          lambda step: 0.5, 16, 8, seed=3)
    _, new_allowed = allowed_rows(u)
    old_only = torch.arange(u.size) >= u.n_old
    for step in range(5):
        x, y, info = s.get_batch(step, 1)
        flags = info["use_new"]
        assert flags.dtype == torch.bool and flags.shape == (8,)
        assert info["frac_new"] == flags.float().mean().item()
        for row, use_new in zip(torch.cat([x, y[:, -1:]], 1), flags):
            assert new_allowed[row].all() if use_new else not old_only[row].any()


def test_masked_union_step0_equals_keep_and_swap(tmp_path, old_tok, new_tok):
    base = tmp_path / "base"
    torch.manual_seed(0)
    GPT2LMHeadModel(GPT2Config(vocab_size=old_tok.get_vocab_size(), n_positions=64, n_embd=16, n_layer=1,
                               n_head=2)).save_pretrained(base)
    A = M.build("keep", old_tok, None, str(base))
    B = M.build("swap", old_tok, new_tok, str(base), "fvt", 0)
    E = M.build("progressive", old_tok, new_tok, str(base), "fvt", 0)
    old_allowed, new_allowed = allowed_rows(E.union)
    bpb = lambda s, tok, mp, allowed=None: bits_per_byte(s.model, tok, TEXTS, mp, 32, 2, "cpu", allowed)["bpb"]
    assert bpb(E, old_tok, E.old_map, old_allowed) == pytest.approx(bpb(A, old_tok, A.old_map), abs=1e-5)
    assert bpb(E, new_tok, E.new_map, new_allowed) == pytest.approx(bpb(B, new_tok, B.new_map), abs=1e-5)
    # unmasked union evaluation is not equal: foreign rows take probability mass
    assert bpb(E, old_tok, E.old_map) > bpb(A, old_tok, A.old_map)


def _train(tmp_path, out, *extra, script="scripts/train.py"):
    command = [sys.executable, str(script), "configs/train/E_progressive.yaml",
               f"base_model={tmp_path / 'base'}", f"old_tokenizer={tmp_path / 'old.json'}",
               f"new_tokenizer={tmp_path / 'new.json'}", "old_name=old", "new_name=new",
               f"raw_dir={tmp_path / 'raw'}", f"tokens_dir={tmp_path / 'tokens'}", f"out={out}",
               "run_name=E", "token_budget=128", "block_size=32", "micro_batch=2", "grad_accum=1",
               "eval_every=2", "eval_docs=1", "eval_batch=1", "eval_split=validation", "log_every=1", *extra]
    env = os.environ | dict(PYTHONPATH=str(ROOT / "src"), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                            OMP_NUM_THREADS="1", TOKENIZERS_PARALLELISM="false")
    subprocess.run(command, cwd=ROOT, env=env, check=True, capture_output=True, text=True, timeout=180)
    run = Path(out) / "E_0M"
    return run, [json.loads(line) for line in (run / "log.jsonl").read_text().splitlines()]


def test_cli_flag_off_is_bit_identical_and_flag_on_trains(tmp_path, corpus, old_tok, new_tok):
    from safetensors.torch import load_file
    GPT2LMHeadModel(GPT2Config(vocab_size=old_tok.get_vocab_size(), n_positions=64, n_embd=16, n_layer=1,
                               n_head=2)).save_pretrained(tmp_path / "base")
    old_tok.save(str(tmp_path / "old.json")); new_tok.save(str(tmp_path / "new.json"))
    raw = tmp_path / "raw/code"; raw.mkdir(parents=True)
    for split, text in [("validation", "validation only 中文"), ("test", "held-out test only")]:
        (raw / f"{split}.jsonl").write_text(json.dumps(dict(text=text)) + "\n")
    for name, tok in [("old", old_tok), ("new", new_tok)]:
        encode_corpus(tok, corpus, tmp_path / "tokens/code" / name)

    reference = tmp_path / "train_reference.py"
    reference.write_bytes((ROOT / "tests/fixtures/train_before_tensor_cleanup.py").read_bytes())
    ref_run, ref_rows = _train(tmp_path, tmp_path / "ref", script=reference)
    off_run, off_rows = _train(tmp_path, tmp_path / "off", "mask_foreign_logits=false")
    on_run, on_rows = _train(tmp_path, tmp_path / "on", "mask_foreign_logits=true")

    losses = lambda rows: [r["train_loss"] for r in rows if "train_loss" in r]
    assert losses(off_rows) == losses(ref_rows)
    before, after = load_file(str(ref_run / "model/model.safetensors")), load_file(str(off_run / "model/model.safetensors"))
    assert before.keys() == after.keys() and all(torch.equal(before[k], after[k]) for k in before)

    assert json.loads((on_run / "config.json").read_text())["mask_foreign_logits"] is True
    assert json.loads((off_run / "config.json").read_text())["mask_foreign_logits"] is False
    assert all(np.isfinite(v) for v in losses(on_rows)) and losses(on_rows) != losses(off_rows)
    final = next(r["final_eval"] for r in on_rows if "final_eval" in r)
    pruned = next(r["final_eval_pruned"] for r in on_rows if "final_eval_pruned" in r)
    # with the mask the union "new" evaluation already is the pruned softmax
    assert pruned["new"] == pytest.approx(final["new"], abs=1e-5)
