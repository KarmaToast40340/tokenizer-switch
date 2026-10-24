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
) -> Iterator[dict]:
    """Stream documents of one source in a reproducible, shuffled order.

    ``shuffle_buffer`` mixes documents across repositories (code) and
    alphabetical article ranges (Wikipedia); without it the first documents of
    the stream are strongly correlated.
    """
    from datasets import load_dataset

    repo, config, text_field, id_field, group_field = SOURCES[name]
    ds = load_dataset(repo, config, split="train", streaming=True)
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
