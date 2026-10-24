from __future__ import annotations

import json
from pathlib import Path

import yaml


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    """Load a YAML config; ``base: other.yaml`` is merged first, then
    ``key=value`` overrides from the command line (values parsed as YAML)."""
    path = Path(path)
    cfg = yaml.safe_load(path.read_text()) or {}
    if "base" in cfg:
        parent = load_config(path.parent / cfg.pop("base"))
        parent.update(cfg)
        cfg = parent
    for item in overrides or []:
        k, v = item.split("=", 1)
        cfg[k] = yaml.safe_load(v)
    return cfg


class JsonlLogger:
    def __init__(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.f = open(path, "a", encoding="utf-8")

    def log(self, **row) -> None:
        self.f.write(json.dumps(row) + "\n")
        self.f.flush()
