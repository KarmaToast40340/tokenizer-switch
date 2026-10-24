"""Run the Phase 1 (or Chinese) tokenizer grid: vary one lever at a time
around a center point. Prints the commands with --dry-run.

    python scripts/tokenizer_grid.py configs/tokenizer_grid_code.yaml
"""
import argparse
import subprocess
import sys
from pathlib import Path

import yaml


def grid_points(cfg: dict):
    c = cfg["center"]
    yield dict(c)
    for frac in cfg["levers"].get("domain_frac", []):
        if frac != c["domain_frac"]:
            yield c | {"domain_frac": frac}
    for p in cfg["levers"].get("pretok", []):
        if p != c["pretok"]:
            yield c | {"pretok": p}
    for v in cfg["levers"].get("vocab_size", []):
        if v != c["vocab_size"]:
            yield c | {"vocab_size": v}


def name(cfg: dict, pt: dict) -> str:
    v = pt["vocab_size"]
    size = f"{v // 1000}k" if v % 1000 == 0 else str(v)
    return f"{cfg['domain']}{round(pt['domain_frac'] * 100)}_{pt['pretok']}_{size}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    out_root = Path(cfg.get("out", "artifacts/tokenizers"))
    for pt in grid_points(cfg):
        out = out_root / name(cfg, pt)
        if (out / "tokenizer.json").exists():
            print(f"[skip] {out}")
            continue
        mix = [f"{cfg['domain']}={pt['domain_frac']}"]
        if pt["domain_frac"] < 1:
            mix.append(f"{cfg['other']}={1 - pt['domain_frac']:.4f}")
        mix = [m for m in mix if float(m.split("=")[1]) > 0]
        total_mb = cfg["total_mb"]
        if pt["pretok"] == "identity":  # see README: whole documents become BPE "words"
            total_mb = cfg.get("identity_total_mb", total_mb)
        cmd = [sys.executable, str(Path(__file__).with_name("train_tokenizer.py")), "--mix", *mix,
               "--pretok", pt["pretok"], "--vocab-size", str(pt["vocab_size"]),
               "--total-mb", str(total_mb), "--out", str(out)]
        print(" ".join(cmd))
        if not args.dry_run:
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
