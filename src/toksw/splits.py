"""Stable group splits and audits, independent of dataset download machinery."""
import hashlib
import json
from pathlib import Path
from .jsonio import open_binary, iter_rows


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def split_for(doc, seed):
    # A repository always belongs to one split, even when the stream is truncated.
    key = doc.get("group") or digest(doc["text"])
    bucket = int(digest(f"{seed}:{key}")[:16], 16) % 100
    return "train" if bucket < 80 else "validation" if bucket < 90 else "test"


def file_sha256(path):
    h = hashlib.sha256()
    # For JSONL.gz this is the logical, decompressed JSONL checksum. It stays
    # stable if an existing raw corpus is compressed to save disk.
    with open_binary(path) as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def audit_splits(root):
    """Reject exact text duplicates and repository overlap within a source."""
    root = Path(root)
    hashes, groups, report = {}, {}, {}
    for split in ("train", "validation", "test"):
        n, size, lengths = 0, 0, []
        for doc in iter_rows(root / f"{split}.jsonl"):
            h = digest(doc["text"])
            if h in hashes:
                raise ValueError(f"duplicate text in {hashes[h]} and {split}")
            hashes[h] = split
            group = doc.get("group")
            if group is not None:
                if group in groups and groups[group] != split:
                    raise ValueError(f"repository leakage: {group}")
                groups[group] = split
            n += 1
            b = len(doc["text"].encode("utf-8"))
            size += b
            lengths.append(b)
        lengths.sort()
        report[split] = dict(n_docs=n, utf8_bytes=size,
                             median_doc_bytes=lengths[len(lengths)//2] if lengths else 0,
                             max_doc_bytes=max(lengths, default=0),
                             sha256=file_sha256(root / f"{split}.jsonl"))
    return report
