"""Prepare bounded, revision-pinned code/English data with three disjoint splits.

Never overwrites a directory. Whole documents are assigned before train chunking.
Exact duplicate documents and chunks are removed; near-duplicates are not removed.
"""
import argparse
from contextlib import ExitStack
import json
import gc
import gzip
from pathlib import Path

from huggingface_hub import HfApi
from tqdm import tqdm

from toksw.data import chunk_text
from toksw.sources import SOURCES, stream_source
from toksw.splits import audit_splits, digest, split_for


def prepare(docs, out, source, train_bytes, n_eval, seed, chunk_chars, provenance, gzip_train=False):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    counts = dict.fromkeys(("train", "validation", "test"), 0)
    sizes = counts.copy()
    seen, emitted = set(), set()
    dropped = dict(duplicate_document=0, duplicate_chunk=0, missing_group=0)
    with ExitStack() as stack:
        files = {}
        for s in counts:
            if s == "train" and gzip_train:
                files[s] = stack.enter_context(gzip.open(out / "train.jsonl.gz", "wt", encoding="utf-8", compresslevel=5))
            else:
                files[s] = stack.enter_context((out / f"{s}.jsonl").open("w", encoding="utf-8"))
        bar = stack.enter_context(tqdm(total=train_bytes, unit="B", unit_scale=True, desc=source))
        for doc in docs:
            if source == "code" and not doc.get("group"):
                dropped["missing_group"] += 1
                continue
            h = digest(doc["text"])
            if h in seen:
                dropped["duplicate_document"] += 1
                continue
            seen.add(h)
            split = split_for(doc, seed)
            if (split == "train" and sizes[split] >= train_bytes) or (split != "train" and counts[split] >= n_eval):
                continue
            pieces = chunk_text(doc["text"], chunk_chars) if split == "train" and chunk_chars else [doc["text"]]
            for i, piece in enumerate(pieces):
                ph = digest(piece)
                if ph in emitted:
                    dropped["duplicate_chunk"] += 1
                    continue
                emitted.add(ph)
                row = doc | {"text": piece, "parent_id": doc["id"], "sha256": ph}
                if len(pieces) > 1:
                    row["id"] = f"{doc['id']}#{i}"
                files[split].write(json.dumps(row, ensure_ascii=False) + "\n")
                counts[split] += 1
                n = len(piece.encode("utf-8"))
                sizes[split] += n
                if split == "train":
                    bar.update(n)
            if sizes["train"] >= train_bytes and all(counts[s] >= n_eval for s in ("validation", "test")):
                break
    audit = audit_splits(out)
    complete = sizes["train"] >= train_bytes and all(counts[s] >= n_eval for s in ("validation", "test"))
    stats = dict(source=source, seed=seed, split_method="sha256(seed:repository_or_text_hash) modulo 100: 80/10/10",
                 requested_train_bytes=train_bytes, requested_eval_docs=n_eval, chunk_chars=chunk_chars,
                 n_train=counts["train"], train_bytes=sizes["train"], dropped=dropped,
                 complete=complete, provenance=provenance, audit=audit)
    stats["train_storage"] = "train.jsonl.gz" if gzip_train else "train.jsonl"
    (out / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    if not complete:
        raise RuntimeError(f"stream exhausted before quotas; see {out}/stats.json")
    print(json.dumps(stats, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["code", "wiki_en", "wiki_zh"], required=True)
    ap.add_argument("--out", default="data/milestone/raw")
    ap.add_argument("--train-mb", type=float, required=True)
    ap.add_argument("--n-eval", type=int, default=100)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--shuffle-buffer", type=int, default=1000)
    ap.add_argument("--chunk-chars", type=int, default=0)
    ap.add_argument("--revision")
    ap.add_argument("--gzip-train", action="store_true")
    ap.add_argument("--reader", choices=["auto", "hf", "sequential"], default="auto")
    args = ap.parse_args()
    if args.train_mb <= 0 or args.n_eval <= 0 or args.chunk_chars < 0:
        ap.error("budgets must be positive; chunk size must be nonnegative")
    out = Path(args.out) / args.source
    if out.exists():
        raise FileExistsError(f"preserving existing output: {out}")
    revision = args.revision or HfApi().dataset_info(SOURCES[args.source][0]).sha
    min_chars = 200 if args.source.startswith("wiki_") else 1
    docs = stream_source(args.source, args.seed, args.shuffle_buffer, min_chars, revision=revision, reader=args.reader)
    try:
        prepare(docs, out, args.source, int(args.train_mb * 1e6), args.n_eval, args.seed, args.chunk_chars,
                dict(hf_repo=SOURCES[args.source][0], config=SOURCES[args.source][1], revision=revision,
                     shuffle_buffer=args.shuffle_buffer, min_chars=min_chars,
                     reader="sequential-parquet-hf-shuffle-v1" if args.source.startswith("wiki_") and args.reader != "hf" else "hf-stream-v1",
                     conversion="OpenCC t2s (1.1.9)" if args.source == "wiki_zh" else None), args.gzip_train)
    finally:
        # Release partially consumed Parquet iterators while Python is still
        # running, before Arrow's native thread-pool finalizers execute.
        docs.close()
        del docs
        gc.collect()


if __name__ == "__main__":
    main()
