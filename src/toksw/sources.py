"""Streaming loaders for the three text sources used in the project.

Every loader yields plain dicts ``{"id": str, "text": str, "group": str | None}``
(``group`` is the GitHub repository for code, used to keep test repositories
out of the training set) so the rest of the
pipeline never depends on the Hugging Face schema of a dataset.
"""
from __future__ import annotations

from collections.abc import Iterator

WIKI_DUMP = "20231101"

SOURCES = {
    # name: (hf repo, config, text field, id field, group field)
    "wiki_en": ("wikimedia/wikipedia", f"{WIKI_DUMP}.en", "text", "id", None),
    "wiki_zh": ("wikimedia/wikipedia", f"{WIKI_DUMP}.zh", "text", "id", None),
    "code": ("codeparrot/codeparrot-clean", None, "content", "hash", "repo_name"),
}


def _wiki_rows(repo, revision, files):
    """Read remote Parquet sequentially without Arrow dataset scan workers.

    Disabling threaded reads avoids the observed macOS scan-pool shutdown hang
    after a bounded streaming iterator is stopped early. Range requests keep
    disk usage bounded; no full Wikipedia download is materialized.
    """
    from huggingface_hub import HfFileSystem
    import pyarrow.parquet as pq

    fs = HfFileSystem()
    for file in files:
        with fs.open(f"datasets/{repo}@{revision}/{file}", "rb", block_size=4 * 1024**2) as handle:
            with pq.ParquetFile(handle) as parquet:
                for batch in parquet.iter_batches(batch_size=128, use_threads=False):
                    yield from batch.to_pylist()


def _t2s():
    """Return a traditional -> simplified Chinese converter (OpenCC)."""
    try:
        import opencc  # C++ binding, fast

        conv = opencc.OpenCC("t2s.json")
    except Exception:  # pure-python fallback: opencc-python-reimplemented
        from opencc import OpenCC

        conv = OpenCC("t2s")
    return conv.convert


def stream_source(
    name: str,
    seed: int = 0,
    shuffle_buffer: int = 10_000,
    min_chars: int = 0,
    revision: str | None = None,
    reader: str = "auto",
) -> Iterator[dict]:
    """Stream documents of one source in a reproducible, shuffled order.

    ``shuffle_buffer`` mixes documents across repositories (code) and
    alphabetical article ranges (Wikipedia); without it the first documents of
    the stream are strongly correlated.
    """
    from datasets import IterableDataset, load_dataset

    repo, config, text_field, id_field, group_field = SOURCES[name]
    if name.startswith("wiki_") and reader != "hf":
        from huggingface_hub import HfApi
        files = sorted(p for p in HfApi().list_repo_files(repo, repo_type="dataset", revision=revision)
                       if p.startswith(f"{config}/train-") and p.endswith(".parquet"))
        if not files:
            raise ValueError(f"no Parquet files for {repo}/{config}@{revision}")
        ds = IterableDataset.from_generator(_wiki_rows, gen_kwargs=dict(repo=repo, revision=revision, files=files))
    else:
        ds = load_dataset(repo, config, split="train", streaming=True, revision=revision)
    if shuffle_buffer > 0:
        ds = ds.shuffle(seed=seed, buffer_size=shuffle_buffer)

    convert = _t2s() if name == "wiki_zh" else None
    for ex in ds:
        text = ex[text_field]
        if convert is not None:
            text = convert(text)
        if len(text) < min_chars:
            continue
        group = ex[group_field] if group_field else None
        yield {"id": str(ex[id_field]), "text": text, "group": group}
