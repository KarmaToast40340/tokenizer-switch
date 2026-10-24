from toksw.pretokenizers import pretokenize
from toksw.tok import train_bpe


def test_punct_separates_attribute_access():
    assert "." in pretokenize("punct", "x.append(1)")
    assert ".append" in pretokenize("gpt4", "x.append(1)")


def test_digits_grouped_by_three():
    assert pretokenize("gpt4", "1234567")[:2] == ["123", "456"]
    assert pretokenize("gpt2", "1234567") == ["1234567"]


def test_punct_drops_contractions():
    assert "'s" in pretokenize("gpt4", "it's")
    assert "'s" not in pretokenize("punct", "it's")


def test_all_pretokenizers_are_reversible(texts):
    sample = texts[:50] + ["  tabs\tand\r\nCRLF  ", "中文 混合 English 123", "émoji 🚀 ok"]
    for name in ["gpt2", "gpt4", "punct", "identity"]:
        tok = train_bpe(texts, 350, name, min_frequency=1)
        for t in sample:
            assert tok.decode(tok.encode(t).ids) == t, (name, t)
