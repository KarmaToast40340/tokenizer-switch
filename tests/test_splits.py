import json

import pytest

from toksw.splits import audit_splits, split_for


def test_repository_assignment_is_stable():
    a = {"group": "owner/repo", "text": "first file"}
    b = {"group": "owner/repo", "text": "different file"}
    assert split_for(a, 2026) == split_for(b, 2026)
    assert split_for(a, 2026) == split_for(a, 2026)


@pytest.mark.parametrize("duplicate", [True, False])
def test_audit_rejects_cross_split_leakage(tmp_path, duplicate):
    for split, text, group in [("train", "hello", "a"),
                                ("validation", "hello" if duplicate else "world", "a"),
                                ("test", "test", "b")]:
        (tmp_path / f"{split}.jsonl").write_text(json.dumps(dict(text=text, group=group)) + "\n")
    with pytest.raises(ValueError, match="duplicate|leakage"):
        audit_splits(tmp_path)
