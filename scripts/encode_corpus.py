"""Encode a source's train.jsonl with one tokenizer into .bin/.idx (uint16 when the vocabulary fits).
Run it once per (source, tokenizer) pair; the progressive condition needs
both the GPT-2 and the new encoding of the same file.

    python scripts/encode_corpus.py --source code --tokenizer gpt2 --name gpt2
    python scripts/encode_corpus.py --source code --tokenizer artifacts/tokenizers/code70_gpt4_32k --name code70_gpt4_32k
"""
import argparse
import json
from pathlib import Path

from toksw.data import encode_corpus, iter_jsonl
from toksw.tok import load_tokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/tokens")
    ap.add_argument("--max-mb", type=float, default=None,
                    help="encode only the first documents up to this many MB of text (same cut for every tokenizer)")
    args = ap.parse_args()

    tok = load_tokenizer(args.tokenizer)
    prefix = Path(args.out) / args.source / args.name
    max_bytes = int(args.max_mb * 1e6) if args.max_mb else None
    n_docs, n_tokens = encode_corpus(tok, Path(args.raw) / args.source / "train.jsonl", prefix,
                                     max_bytes=max_bytes)
    raw_bytes = json.loads((Path(args.raw) / args.source / "stats.json").read_text())["train_bytes"]
    if max_bytes is not None:
        raw_bytes = sum(len(d["text"].encode("utf-8")) for _, d in zip(range(n_docs), iter_jsonl(
            Path(args.raw) / args.source / "train.jsonl")))
    info = {"n_docs": n_docs, "n_tokens": n_tokens, "bytes_per_token": raw_bytes / n_tokens}
    prefix.with_suffix(".json").write_text(json.dumps(info, indent=2))
    print(info)


if __name__ == "__main__":
    main()
