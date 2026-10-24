"""Training, loading and measuring byte-level BPE tokenizers."""
from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, trainers
from tokenizers.pre_tokenizers import ByteLevel

from . import EOT
from .pretokenizers import build_pretokenizer


def train_bpe(
    texts: Iterable[str],
    vocab_size: int,
    pretok: str = "gpt4",
    min_frequency: int = 2,
    length: int | None = None,
) -> Tokenizer:
    """Train a byte-level BPE tokenizer, without normalization (fully reversible).

    The end-of-text token gets id 0. It is the only special token.
    """
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = build_pretokenizer(pretok)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=[EOT],
        initial_alphabet=ByteLevel.alphabet(),
        show_progress=True,
    )
    tok.train_from_iterator(texts, trainer=trainer, length=length)
    return tok


def load_tokenizer(name_or_path: str) -> Tokenizer:
    """Load ``gpt2`` (from the HF hub) or a local ``tokenizer.json``."""
    p = Path(name_or_path)
    if p.is_dir():
        p = p / "tokenizer.json"
    if p.exists():
        return Tokenizer.from_file(str(p))
    return Tokenizer.from_pretrained(name_or_path)


def eot_id(tok: Tokenizer) -> int:
    i = tok.token_to_id(EOT)
    if i is None:
        raise ValueError(f"tokenizer has no {EOT} token")
    return i


def mixed_text_iterator(
    sources: dict[str, Iterator[str]], fractions: dict[str, float], total_bytes: int
) -> Iterator[str]:
    """Yield documents from several sources until each has contributed its share
    of ``total_bytes`` (UTF-8). Documents are interleaved so that the trainer
    never sees one source in a single block."""
    budget = {k: int(total_bytes * f) for k, f in fractions.items() if f > 0}
    used = {k: 0 for k in budget}
    active = [k for k in budget]
    while active:
        for k in list(active):
            try:
                text = next(sources[k])
            except StopIteration:
                print(f"[warn] source {k} exhausted at {used[k] / 1e6:.0f} MB")
                active.remove(k)
                continue
            used[k] += len(text.encode("utf-8"))
            yield text
            if used[k] >= budget[k]:
                active.remove(k)
