import json
import random

import pytest

from toksw.tok import train_bpe

WORDS = ["def", "return", "self", "import", "numpy", "value", "the", "of", "language", "model",
         "token", "函数", "模型", "语言", "数据", "print", "append", "range", "class", "lambda"]


def make_texts(n: int, seed: int = 0) -> list[str]:
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        lines = []
        for _ in range(rng.randint(2, 6)):
            ws = rng.choices(WORDS, k=rng.randint(3, 10))
            lines.append(("    " if rng.random() < 0.5 else "") + " ".join(ws) + rng.choice([".", "()", ":", ""]))
        out.append("\n".join(lines))
    return out


@pytest.fixture(scope="session")
def texts():
    return make_texts(400)


@pytest.fixture(scope="session")
def old_tok(texts):
    # stand-in for GPT-2: small GPT-2-regex tokenizer
    return train_bpe(texts, 300, "gpt2", min_frequency=1)


@pytest.fixture(scope="session")
def new_tok(texts):
    return train_bpe(texts, 400, "gpt4", min_frequency=1)


@pytest.fixture()
def corpus(tmp_path, texts):
    path = tmp_path / "train.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for i, t in enumerate(texts):
            f.write(json.dumps({"id": str(i), "text": t}, ensure_ascii=False) + "\n")
    return path
