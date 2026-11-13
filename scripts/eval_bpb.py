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
from toksw.splits import file_sha256


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--domain", required=True)
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--block-size", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--split", choices=["validation", "test"], default="test")
    ap.add_argument("--out")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GPT2LMHeadModel.from_pretrained(args.checkpoint).to(device).eval()
    tok = load_tokenizer(args.checkpoint)
    texts = [d["text"] for d in iter_jsonl(Path(args.raw) / args.domain / f"{args.split}.jsonl")]
    res = bits_per_byte(model, tok, texts, None, args.block_size, args.batch_size, device)
    result = {"checkpoint": args.checkpoint, "domain": args.domain, "split": args.split,
              "raw_sha256": file_sha256(Path(args.raw) / args.domain / f"{args.split}.jsonl"),
              "block_size": args.block_size, "batch_size": args.batch_size, "device": device,
              "tokenizer_sha256": file_sha256(Path(args.checkpoint) / "tokenizer.json"),
              "model_training_after_eval": model.training, "torch": str(torch.__version__),
              "gpu": torch.cuda.get_device_name() if device == "cuda" else None, **res}
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
