"""Evaluate final saved A/B/C/E model directories; E/model is already pruned.

Pre-pruning E metrics must come from its real log.jsonl (no union checkpoint is
saved by training). This script does not train or submit cluster jobs.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoints", nargs="+")
    ap.add_argument("--raw", default="data/milestone/raw")
    ap.add_argument("--domain", default="code")
    ap.add_argument("--out", default="results/checkpoints")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--block-size", type=int, default=1024)
    ap.add_argument("--speed", action="store_true")
    ap.add_argument("--prompt-chars", type=int, default=200)
    ap.add_argument("--new-tokens", type=int, default=128)
    args = ap.parse_args()
    for ck in args.checkpoints:
        if not (Path(ck) / "config.json").is_file() or not (Path(ck) / "tokenizer.json").is_file():
            ap.error(f"missing local saved checkpoint/tokenizer: {ck}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for i, ck in enumerate(args.checkpoints):
        dest = out / f"{i}_{Path(ck).parent.name}_bpb.json"
        subprocess.run([sys.executable, "scripts/eval_bpb.py", ck, "--raw", args.raw, "--domain", args.domain,
                        "--batch-size", str(args.batch_size), "--block-size", str(args.block_size),
                        "--out", str(dest)], check=True)
        results.append(json.loads(dest.read_text()))
    (out / "bpb.json").write_text(json.dumps(results, indent=2) + "\n")
    if args.speed:
        subprocess.run([sys.executable, "scripts/gen_speed.py", *args.checkpoints, "--raw", args.raw,
                        "--domain", args.domain, "--prompt-chars", str(args.prompt_chars),
                        "--new-tokens", str(args.new_tokens), "--out", str(out / "speed.csv")], check=True)


if __name__ == "__main__":
    main()
