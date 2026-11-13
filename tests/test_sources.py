import datasets
from huggingface_hub import HfApi

from toksw import sources


class FakeStream:
    def shuffle(self, seed, buffer_size):
        assert (seed, buffer_size) == (2026, 1000)
        return self

    def __iter__(self):
        yield dict(id="1", text="English article")


def test_legacy_hf_reader_remains_available(monkeypatch):
    calls = []
    def load(repo, config, **kwargs):
        calls.append((repo, config, kwargs))
        return FakeStream()
    monkeypatch.setattr(datasets, "load_dataset", load)
    rows = list(sources.stream_source("wiki_en", 2026, 1000, revision="pinned", reader="hf"))
    assert rows == [dict(id="1", text="English article", group=None)]
    assert calls[0][2] == dict(split="train", streaming=True, revision="pinned")


def test_new_wiki_reader_selects_sequential_parquet(monkeypatch):
    calls = []
    monkeypatch.setattr(HfApi, "list_repo_files", lambda *a, **k: ["20231101.en/train-00000.parquet", "README.md"])
    def make(generator, gen_kwargs):
        calls.append((generator, gen_kwargs))
        return FakeStream()
    monkeypatch.setattr(datasets.IterableDataset, "from_generator", make)
    assert len(list(sources.stream_source("wiki_en", 2026, 1000, revision="pinned"))) == 1
    assert calls[0][0] is sources._wiki_rows
    assert calls[0][1]["files"] == ["20231101.en/train-00000.parquet"]
