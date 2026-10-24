"""Compression of tokenizers on the held-out test sets: NSL vs GPT-2 and
bytes per token, per subset. Also checks that decoding is lossless.

    python scripts/eval_tokenizers.py artifacts/tokenizers/* --subsets code wiki_en \
        --out results/tokenizers_code.csv
"""
import argparse
import csv
from pathlib import Path

from toksw.data import iter_jsonl
from toksw.metrics import bytes_per_token, nsl, roundtrip_ok
from toksw.tok import load_tokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tokenizers", nargs="+", help="tokenizer dirs or tokenizer.json files")
    ap.add_argument("--subsets", nargs="+", default=["code", "wiki_en"])
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--reference", default="gpt2")
    ap.add_argument("--out", default="results/tokenizers.csv")
    args = ap.parse_args()

    ref = load_tokenizer(args.reference)
    tests = {s: [d["text"] for d in iter_jsonl(Path(args.raw) / s / "test.jsonl")] for s in args.subsets}
    rows = []
    for path in [args.reference, *args.tokenizers]:
        tok = load_tokenizer(path)
        row = {"tokenizer": Path(path).name, "vocab_size": tok.get_vocab_size()}
        for s, texts in tests.items():
            row[f"nsl_{s}"] = round(nsl(tok, ref, texts), 4)
            row[f"bytes_per_token_{s}"] = round(bytes_per_token(tok, texts), 3)
            row[f"roundtrip_{s}"] = roundtrip_ok(tok, texts)
        rows.append(row)
        print(row)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
