"""Audit every encoded document against its raw text, not just document counts."""
import argparse
import json
from pathlib import Path

from toksw.data import TokenFile, iter_jsonl
from toksw.splits import file_sha256
from toksw.tok import eot_id, load_tokenizer


def verify(raw, prefixes, tokenizers):
    if len(prefixes) != len(tokenizers):
        raise ValueError("one tokenizer per prefix required")
    report = {"raw_sha256": file_sha256(raw), "encodings": []}
    for prefix, tokenizer in zip(prefixes, tokenizers):
        tok, tokens = load_tokenizer(tokenizer), TokenFile(prefix)
        count = 0
        for i, doc in enumerate(iter_jsonl(raw)):
            if i >= tokens.n_docs:
                raise ValueError("encoded corpus is shorter than raw corpus")
            ids = tokens.tokens[tokens.offsets[i]:tokens.offsets[i + 1]].tolist()
            if ids[-1] != eot_id(tok) or tok.decode(ids[:-1], skip_special_tokens=False) != doc["text"]:
                raise ValueError(f"alignment or roundtrip failure at document {i}: {prefix}")
            count += 1
        if count != tokens.n_docs:
            raise ValueError("encoded corpus has extra documents")
        report["encodings"].append(dict(prefix=prefix, n_docs=count, n_tokens=len(tokens.tokens),
                                         dtype=str(tokens.tokens.dtype),
                                         bin_sha256=file_sha256(Path(prefix).with_suffix(".bin")),
                                         idx_sha256=file_sha256(Path(prefix).with_suffix(".idx"))))
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--prefixes", nargs="+", required=True)
    ap.add_argument("--tokenizers", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    report = verify(args.raw, args.prefixes, args.tokenizers)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
