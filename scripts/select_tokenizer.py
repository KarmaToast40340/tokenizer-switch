"""Select on validation only. Lossless; English NSL <= cap; minimize code NSL.

If no candidate meets the cap, stop rather than silently change the rule.
"""
import argparse
import csv
import json
from pathlib import Path


def select(rows, english_cap, domain="code"):
    eligible = [r for r in rows if r["tokenizer"] != "gpt2"
                and float(r[f"roundtrip_{domain}"]) == float(r["roundtrip_wiki_en"]) == 1
                and float(r["nsl_wiki_en"]) <= english_cap]
    if not eligible:
        raise ValueError("No lossless candidate meets the predeclared English cap")
    return min(eligible, key=lambda r: (float(r[f"nsl_{domain}"]), int(r["vocab_size"]), r["tokenizer"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("validation_csv")
    ap.add_argument("--english-cap", type=float, default=1.25)
    ap.add_argument("--domain", default="code")
    ap.add_argument("--out", default="results/milestone/selection.json")
    args = ap.parse_args()
    rows = list(csv.DictReader(open(args.validation_csv)))
    chosen = select(rows, args.english_cap, args.domain)
    result = dict(validation_csv=args.validation_csv, english_nsl_cap=args.english_cap,
                  domain=args.domain,
                  rule=f"lossless on both validation domains; English NSL <= cap; minimum {args.domain} NSL, then smaller vocab",
                  selected=chosen)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
