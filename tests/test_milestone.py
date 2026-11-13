import importlib.util
import json
from pathlib import Path

import pytest

from toksw.data import encode_corpus
from toksw.splits import audit_splits, split_for


def script(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_prepare_splits_dedup_chunk_and_preserve(tmp_path):
    groups = {}
    for i in range(1000):
        doc = dict(id=str(i), group=f"repo{i}", text=f"file {i}\n" + "x" * 40)
        groups.setdefault(split_for(doc, 2026), doc)
    docs = [groups["train"], groups["train"], groups["validation"], groups["test"]]
    out = tmp_path / "code"
    script("prepare_milestone").prepare(docs, out, "code", 1, 1, 2026, 20, {})
    stats = json.loads((out / "stats.json").read_text())
    assert stats["complete"] and stats["dropped"]["duplicate_document"] == 1
    assert audit_splits(out)["train"]["max_doc_bytes"] <= 20
    with pytest.raises(FileExistsError):
        script("prepare_milestone").prepare([], out, "code", 1, 1, 2026, 20, {})


def test_alignment_detects_reordered_documents(tmp_path, corpus, old_tok):
    tok = tmp_path / "tokenizer.json"
    old_tok.save(str(tok))
    prefix = tmp_path / "encoded"
    encode_corpus(old_tok, corpus, prefix)
    verify = script("verify_alignment").verify
    assert verify(corpus, [str(prefix)], [str(tok)])["encodings"][0]["n_docs"] == 400
    lines = corpus.read_text().splitlines()
    corpus.write_text("\n".join(reversed(lines)) + "\n")
    with pytest.raises(ValueError, match="alignment"):
        verify(corpus, [str(prefix)], [str(tok)])


def test_selection_respects_english_and_losslessness():
    def row(name, code, english, lossless=1):
        return dict(tokenizer=name, nsl_code=code, nsl_wiki_en=english, roundtrip_code=lossless,
                    roundtrip_wiki_en=1, vocab_size=32000)
    rows = [row("gpt2", 1, 1), row("best_code", .5, 1.3), row("valid", .7, 1.2), row("lossy", .4, 1, .9)]
    select = script("select_tokenizer").select
    assert select(rows, 1.25)["tokenizer"] == "valid"
    with pytest.raises(ValueError):
        select(rows, 1.1)
