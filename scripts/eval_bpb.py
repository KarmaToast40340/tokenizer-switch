"""Bits per byte of a saved checkpoint on a held-out test set.

    python scripts/eval_bpb.py runs/B_swap_fvt_code_50M/model --domain code
"""
import argparse
import json
from pathlib import Path

import torch
from transformers import GPT2LMHeadModel

from toksw.data import iter_jsonl
from toksw.metrics import bits_per_byte
from toksw.tok import load_tokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--domain", required=True)
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--block-size", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GPT2LMHeadModel.from_pretrained(args.checkpoint).to(device)
    tok = load_tokenizer(args.checkpoint)
    texts = [d["text"] for d in iter_jsonl(Path(args.raw) / args.domain / "test.jsonl")]
    res = bits_per_byte(model, tok, texts, None, args.block_size, args.batch_size, device)
    print(json.dumps({"checkpoint": args.checkpoint, "domain": args.domain, **res}, indent=2))


if __name__ == "__main__":
    main()
