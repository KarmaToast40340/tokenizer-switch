"""Exercise saved-checkpoint evaluation offline using a tiny random test model."""
import json
import os
from pathlib import Path
import subprocess
import sys

from transformers import GPT2Config, GPT2LMHeadModel


def test_saved_checkpoint_bpb_and_speed_cli(tmp_path, old_tok):
    root = Path(__file__).resolve().parents[1]
    ckpt = tmp_path / "toy_test_only" / "model"
    ckpt.mkdir(parents=True)
    model = GPT2LMHeadModel(GPT2Config(vocab_size=old_tok.get_vocab_size(), n_positions=64,
                                     n_embd=16, n_layer=1, n_head=2))
    model.save_pretrained(ckpt)
    old_tok.save(str(ckpt / "tokenizer.json"))
    raw = tmp_path / "raw" / "code"
    raw.mkdir(parents=True)
    (raw / "test.jsonl").write_text(json.dumps(dict(text="def f():\n    return 42\n")) + "\n")
    out = tmp_path / "evaluation"
    env = os.environ | dict(PYTHONPATH=str(root / "src"), HF_HUB_OFFLINE="1", OMP_NUM_THREADS="1",
                            TOKENIZERS_PARALLELISM="false")
    subprocess.run([sys.executable, "scripts/evaluate_checkpoints.py", str(ckpt),
                    "--raw", str(raw.parent), "--out", str(out), "--block-size", "32",
                    "--speed", "--prompt-chars", "8", "--new-tokens", "4"],
                   cwd=root, env=env, check=True, capture_output=True, text=True, timeout=180)
    result = json.loads((out / "bpb.json").read_text())[0]
    assert result["n_bytes"] == len("def f():\n    return 42\n".encode())
    assert result["bpb"] > 0
    assert "chars_per_second" in (out / "speed.csv").read_text()
