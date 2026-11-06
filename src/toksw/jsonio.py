"""JSONL storage helpers; gzip is transparent to existing logical .jsonl paths."""
import gzip
import json
from pathlib import Path


def stored_path(path):
    path = Path(path)
    if not path.exists() and path.suffix == ".jsonl" and Path(str(path) + ".gz").exists():
        return Path(str(path) + ".gz")
    return path


def open_binary(path):
    path = stored_path(path)
    return gzip.open(path, "rb") if path.name.endswith(".jsonl.gz") else path.open("rb")


def open_text(path):
    path = stored_path(path)
    return gzip.open(path, "rt", encoding="utf-8") if path.name.endswith(".jsonl.gz") else path.open(encoding="utf-8")


def iter_rows(path):
    with open_text(path) as f:
        for line in f:
            yield json.loads(line)
