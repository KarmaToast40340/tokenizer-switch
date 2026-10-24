"""Stream one source from the HF hub and write held-out test + train JSONL.

The first ``--n-test`` documents (after a seeded shuffle) are the test set, so
they are held out before any tokenizer or model sees the data. For code, every
other file of a test file's repository is also kept out of the training set. Tokenizers are
later trained on a prefix of ``train.jsonl``.

    python scripts/prepare_data.py --source code    --train-mb 1500
    python scripts/prepare_data.py --source wiki_en --train-mb 1200
    python scripts/prepare_data.py --source wiki_zh --train-mb 1500 --min-chars 200
"""
import argparse
import hashlib
import json
from pathlib import Path

from tqdm import tqdm

from toksw.data import chunk_text
from toksw.sources import SOURCES, stream_source


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=SOURCES, required=True)
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--n-test", type=int, default=1000)
    ap.add_argument("--train-mb", type=float, required=True, help="UTF-8 megabytes of training text")
    ap.add_argument("--min-chars", type=int, default=0, help="drop shorter documents (stubs)")
    ap.add_argument("--max-chars", type=int, default=0, help="drop longer documents (0 = keep all)")
    ap.add_argument("--chunk-chars", type=int, default=0,
                    help="split training documents into pieces of at most this many characters, at line "
                         "breaks (0 = off). Training windows start at document starts, so without it only "
                         "the first ~1024 tokens of a long file are ever trained on. Test documents stay whole.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shuffle-buffer", type=int, default=10_000)
    args = ap.parse_args()

    out = Path(args.out) / args.source
    out.mkdir(parents=True, exist_ok=True)
    seen: set[bytes] = set()
    test_groups: set[str] = set()
    stats = {"source": args.source, "hf": list(SOURCES[args.source][:2]), "seed": args.seed,
             "dropped_dup": 0, "dropped_long": 0, "dropped_test_group": 0}
    budget = int(args.train_mb * 1e6)

    ftest = open(out / "test.jsonl", "w", encoding="utf-8")
    ftrain = open(out / "train.jsonl", "w", encoding="utf-8")
    n_test = n_train = train_bytes = test_bytes = 0
    bar = tqdm(total=budget, unit="B", unit_scale=True, desc=args.source)
    for doc in stream_source(args.source, args.seed, args.shuffle_buffer, args.min_chars):
        if args.max_chars and len(doc["text"]) > args.max_chars:
            stats["dropped_long"] += 1
            continue
        h = hashlib.sha1(doc["text"].encode("utf-8")).digest()
        if h in seen:  # exact duplicates (also guarantees test/train are disjoint)
            stats["dropped_dup"] += 1
            continue
        seen.add(h)
        n = len(doc["text"].encode("utf-8"))
        line = json.dumps(doc, ensure_ascii=False) + "\n"
        if n_test < args.n_test:
            if doc["group"] is not None:
                test_groups.add(doc["group"])
            ftest.write(line)
            n_test += 1
            test_bytes += n
            continue
        if doc["group"] in test_groups:  # no file from a test repository in train
            stats["dropped_test_group"] += 1
            continue
        pieces = chunk_text(doc["text"], args.chunk_chars) if args.chunk_chars else [doc["text"]]
        for k, piece in enumerate(pieces):
            sub = doc if len(pieces) == 1 else doc | {"id": f"{doc['id']}#{k}", "text": piece}
            ftrain.write(json.dumps(sub, ensure_ascii=False) + "\n")
            n_train += 1
        train_bytes += n
        bar.update(n)
        if train_bytes >= budget:
            break
    bar.close()
    ftest.close()
    ftrain.close()
    stats.update(n_test=n_test, test_bytes=test_bytes, n_train=n_train, train_bytes=train_bytes,
                 mean_train_doc_bytes=train_bytes / max(1, n_train))
    if train_bytes < budget:
        print(f"[warn] stream exhausted: only {train_bytes / 1e6:.0f} MB of {args.train_mb:.0f} MB")
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
