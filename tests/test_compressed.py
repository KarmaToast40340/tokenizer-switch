import gzip
import json

from toksw.data import encode_corpus, iter_jsonl
from toksw.splits import file_sha256, audit_splits


def test_compressed_input_retains_logical_hash_and_alignment(tmp_path, old_tok):
    text = "".join(json.dumps(dict(text=t, group=None)) + "\n" for t in ["中文 café", "def f(): return 1"])
    raw = tmp_path / "train.jsonl"
    raw.write_text(text)
    checksum = file_sha256(raw)
    with gzip.open(str(raw) + ".gz", "wt", encoding="utf-8") as f:
        f.write(text)
    raw.unlink()
    assert file_sha256(raw) == checksum
    assert len(list(iter_jsonl(raw))) == 2
    assert encode_corpus(old_tok, raw, tmp_path / "tokens")[0] == 2
    for split, t in [("validation", "held out validation"), ("test", "held out test")]:
        (tmp_path / f"{split}.jsonl").write_text(json.dumps(dict(text=t, group=None)) + "\n")
    assert audit_splits(tmp_path)["train"]["sha256"] == checksum
