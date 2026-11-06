"""Generation speed in characters per second (greedy, fixed number of new tokens).

    python scripts/gen_speed.py runs/A_keep_code_50M/model runs/B_swap_fvt_code_50M/model --domain code
"""
import argparse
import csv
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
    ap.add_argument("--split", choices=["validation", "test"], default="test")
    ap.add_argument("--out")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    prompts = [d["text"][: args.prompt_chars]
               for d in iter_jsonl(Path(args.raw) / args.domain / f"{args.split}.jsonl", args.n_prompts)]
    if not prompts or args.new_tokens <= 0:
        raise ValueError("nonempty prompts and positive new-tokens required")
    results = []
    for ck in args.checkpoints:
        model = GPT2LMHeadModel.from_pretrained(ck, torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32).to(device).eval()
        tok = load_tokenizer(ck)
        chars, secs = 0, 0.0
        max_prompt = model.config.n_positions - args.new_tokens
        encoded = [tok.encode(p, add_special_tokens=False).ids for p in prompts]
        if any(not ids or len(ids) > max_prompt for ids in encoded):
            raise ValueError("prompt exceeds context or is empty; reduce --prompt-chars or --new-tokens consistently for all checkpoints")
        # Warm up outside timing. Fixed token counts measure throughput, not task quality.
        model.generate(torch.tensor([encoded[0]], device=device), max_new_tokens=args.new_tokens,
                       min_new_tokens=args.new_tokens, do_sample=False, pad_token_id=0)
        for prompt_ids in encoded:
            ids = torch.tensor([prompt_ids], device=device)
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            out = model.generate(ids, max_new_tokens=args.new_tokens, min_new_tokens=args.new_tokens,
                                 do_sample=False, pad_token_id=0)
            if device == "cuda":
                torch.cuda.synchronize()
            secs += time.time() - t0
            chars += len(tok.decode(out[0, ids.shape[1]:].tolist(), skip_special_tokens=True))
        print(f"{ck}: {chars / secs:.0f} chars/s ({chars / (len(prompts) * args.new_tokens):.2f} chars/token)")
        results.append(dict(checkpoint=ck, domain=args.domain, split=args.split, device=device,
                            n_prompts=len(prompts), prompt_chars=args.prompt_chars,
                            new_tokens=args.new_tokens, chars=chars, seconds=secs,
                            chars_per_second=chars/secs, tokens_per_second=len(prompts)*args.new_tokens/secs))
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0]))
            writer.writeheader()
            writer.writerows(results)


if __name__ == "__main__":
    main()
