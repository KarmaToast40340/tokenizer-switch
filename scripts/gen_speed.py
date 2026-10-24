"""Generation speed in characters per second (greedy, fixed number of new tokens).

    python scripts/gen_speed.py runs/A_keep_code_50M/model runs/B_swap_fvt_code_50M/model --domain code
"""
import argparse
import time
from pathlib import Path

import torch
from transformers import GPT2LMHeadModel

from toksw.data import iter_jsonl
from toksw.tok import load_tokenizer


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoints", nargs="+")
    ap.add_argument("--domain", required=True)
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--n-prompts", type=int, default=20)
    ap.add_argument("--prompt-chars", type=int, default=500)
    ap.add_argument("--new-tokens", type=int, default=256)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    prompts = [d["text"][: args.prompt_chars]
               for d in iter_jsonl(Path(args.raw) / args.domain / "test.jsonl", args.n_prompts)]
    for ck in args.checkpoints:
        model = GPT2LMHeadModel.from_pretrained(ck, torch_dtype=torch.bfloat16).to(device).eval()
        tok = load_tokenizer(ck)
        chars, secs = 0, 0.0
        for p in prompts:
            ids = torch.tensor([tok.encode(p).ids], device=device)
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            out = model.generate(ids, max_new_tokens=args.new_tokens, min_new_tokens=args.new_tokens,
                                 do_sample=False, pad_token_id=0)
            if device == "cuda":
                torch.cuda.synchronize()
            secs += time.time() - t0
            chars += len(tok.decode(out[0, ids.shape[1]:].tolist()))
        print(f"{ck}: {chars / secs:.0f} chars/s ({chars / (len(prompts) * args.new_tokens):.2f} chars/token)")


if __name__ == "__main__":
    main()
