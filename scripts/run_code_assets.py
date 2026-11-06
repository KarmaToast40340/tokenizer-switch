"""Build full-budget code corpora using the frozen validation-selected tokenizer.

No tokenizer refitting or model training. Pilot data/results stay separate.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from toksw.data import iter_jsonl
from toksw.splits import audit_splits, digest, file_sha256


def run(script, *args):
    command = [sys.executable, f"scripts/{script}.py", *map(str, args)]
    print("RUN", " ".join(command), flush=True)
    subprocess.run(command, check=True, timeout=1800)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-mb", type=float, default=250)
    args = ap.parse_args()
    raw, out = Path("data/code50/raw"), Path("results/code50")
    out.mkdir(parents=True, exist_ok=True)
    selected = json.loads(Path("results/milestone/selection.json").read_text())["selected"]["tokenizer"]
    if not (raw / "code").exists():
        start = time.monotonic()
        run("prepare_milestone", "--source", "code", "--out", raw, "--train-mb", args.train_mb,
            "--n-eval", 1000, "--seed", 2026, "--chunk-chars", 3000, "--gzip-train",
            "--revision", "35a59fb025bc0a102f7d96eac09d145b896d487b")
        (out / "preparation_timing.json").write_text(json.dumps(dict(seconds=time.monotonic()-start),indent=2)+"\n")
    stats = json.loads((raw / "code/stats.json").read_text())
    if not stats["complete"] or stats["requested_train_bytes"] != int(args.train_mb*1e6) or stats["audit"] != audit_splits(raw/"code"):
        raise ValueError("Code data audit failed; existing files preserved")
    # Enlarging held-out sets must not introduce an exact copy of a chunk the
    # already-fitted selected tokenizer saw in the pilot training corpus.
    train_hashes = {digest(d["text"]) for d in iter_jsonl("data/milestone/raw/code/train.jsonl")}
    for split in ("validation", "test"):
        if any(digest(d["text"]) in train_hashes for d in iter_jsonl(raw/"code"/f"{split}.jsonl")):
            raise ValueError(f"pilot-tokenizer text leakage into full {split}; inspect before proceeding")
    (out / "code_stats.json").write_text(json.dumps(stats,indent=2)+"\n")
    for name in ("gpt2",selected):
        prefix = Path(f"data/code50/tokens/code/{name}")
        if not any(prefix.with_suffix(s).exists() for s in (".bin",".idx",".json")):
            run("encode_corpus", "--source", "code", "--tokenizer", f"artifacts/milestone/tokenizers/{name}",
                "--name", name, "--raw", raw, "--out", "data/code50/tokens")
        elif not all(prefix.with_suffix(s).exists() for s in (".bin",".idx",".json")):
            raise ValueError(f"partial encoding preserved: {prefix}")
    run("verify_alignment", "--raw", raw/"code/train.jsonl", "--prefixes", "data/code50/tokens/code/gpt2",
        f"data/code50/tokens/code/{selected}", "--tokenizers", "artifacts/milestone/tokenizers/gpt2",
        f"artifacts/milestone/tokenizers/{selected}", "--out", out/"alignment.json")
    alignment = json.loads((out/"alignment.json").read_text())
    capacity = {r["prefix"].split("/")[-1]:r["n_tokens"] for r in alignment["encodings"]}
    result = dict(status="complete",selected=selected,token_capacity=capacity,
                  tokenizer_status="frozen from 20 MB pilot validation selection",
                  sufficient_without_corpus_reuse={str(b):all(n>=b for n in capacity.values()) for b in (50_000_000,200_000_000)},
                  raw_sha256=alignment["raw_sha256"], tokenizer_sha256=file_sha256(f"artifacts/milestone/tokenizers/{selected}/tokenizer.json"),
                  training_launched=False)
    (out/"manifest.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
