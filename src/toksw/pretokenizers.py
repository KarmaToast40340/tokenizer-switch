"""Pre-tokenization schemes compared in Dagan et al. (2024), Figure 3."""
from __future__ import annotations

from tokenizers import Regex, pre_tokenizers

# GPT-2 (Radford et al., 2019). Applied by ByteLevel(use_regex=True).
GPT2_PATTERN = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

# GPT-4 / cl100k_base (OpenAI, 2023).
GPT4_PATTERN = (
    r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}"""
    r"""| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
)

# Punct (Dagan et al., 2024, Appendix E): no English contractions, and a
# letter-only chunk may only be preceded by a space (so "." or "\t" never
# merge with the following word, e.g. ".append" -> "." "append").
PUNCT_PATTERN = (
    r""" ?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
)

PRETOKENIZERS = ("gpt2", "gpt4", "punct", "identity")


def build_pretokenizer(name: str):
    """Return a HF ``tokenizers`` pre-tokenizer. All are byte-level and reversible."""
    if name == "gpt2":
        return pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    byte_level = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    if name == "identity":
        return byte_level
    pattern = {"gpt4": GPT4_PATTERN, "punct": PUNCT_PATTERN}[name]
    return pre_tokenizers.Sequence(
        [pre_tokenizers.Split(Regex(pattern), behavior="isolated", invert=False), byte_level]
    )


def pretokenize(name: str, text: str) -> list[str]:
    """Split ``text`` into pre-tokens (as readable strings, not byte-level symbols)."""
    if name == "identity":
        return [text]
    pattern = {"gpt2": GPT2_PATTERN, "gpt4": GPT4_PATTERN, "punct": PUNCT_PATTERN}[name]
    pre = pre_tokenizers.Split(Regex(pattern), behavior="isolated", invert=False)
    return [piece for piece, _ in pre.pre_tokenize_str(text)]
