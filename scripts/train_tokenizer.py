"""Train one byte-level BPE tokenizer on a mix of sources.

    python scripts/train_tokenizer.py --mix code=0.7 wiki_en=0.3 \
        --pretok gpt4 --vocab-size 32000 --total-mb 1000 --out artifacts/tokenizers/code70_gpt4_32k
"""
import argparse
import json
import time
from pathlib import Path

from toksw.data import iter_jsonl
from toksw.pretokenizers import PRETOKENIZERS
from toksw.tok import mixed_text_iterator, train_bpe


def parse_mix(items: list[str]) -> dict[str, float]:
    mix = {k: float(v) for k, v in (i.split("=") for i in items)}
    assert abs(sum(mix.values()) - 1) < 1e-6, f"mix must sum to 1: {mix}"
    return mix


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mix", nargs="+", required=True, help="source=fraction, e.g. code=0.7 wiki_en=0.3")
    ap.add_argument("--pretok", choices=PRETOKENIZERS, default="gpt4")
    ap.add_argument("--vocab-size", type=int, default=32000)
    ap.add_argument("--total-mb", type=float, default=1000)
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--min-frequency", type=int, default=2)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    mix = parse_mix(args.mix)
    sources = {k: (d["text"] for d in iter_jsonl(Path(args.raw) / k / "train.jsonl")) for k in mix}
    texts = mixed_text_iterator(sources, mix, int(args.total_mb * 1e6))
    t0 = time.time()
    tok = train_bpe(texts, args.vocab_size, args.pretok, args.min_frequency)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tok.save(str(out / "tokenizer.json"))
    meta = vars(args) | {"mix": mix, "train_seconds": round(time.time() - t0, 1),
                         "final_vocab_size": tok.get_vocab_size()}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
