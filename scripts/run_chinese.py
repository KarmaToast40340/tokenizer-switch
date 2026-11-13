"""Prepare and verify Chinese tokenizer/data assets on CPU; never train models."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

from toksw.splits import audit_splits, file_sha256


def run(script, *args, timeout=None):
    cmd = [sys.executable, f"scripts/{script}.py", *map(str, args)]
    print("RUN", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, timeout=timeout)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--prepare-only", action="store_true")
    ap.add_argument("--train-mb", type=float, default=400)
    args = ap.parse_args()
    tag = "chinese-smoke" if args.smoke else "chinese"
    raw = Path(f"data/{tag}/raw")
    out = Path(f"results/{tag}")
    artifacts = Path(f"artifacts/{tag}/tokenizers")
    out.mkdir(parents=True, exist_ok=True)
    mb, n_eval = (.2, 10) if args.smoke else (args.train_mb, 1000)
    if shutil.disk_usage(".").free < 1024**3:
        raise RuntimeError("Need at least 1 GiB free before preparing Chinese assets")
    if not (raw / "wiki_zh").exists():
        start = time.monotonic()
        run("prepare_milestone", "--source", "wiki_zh", "--out", raw, "--train-mb", mb,
            "--n-eval", n_eval, "--seed", 2026, "--chunk-chars", 1000, "--gzip-train",
            "--revision", "b04c8d1ceb2f5cd4588862100d08de323dccfbaa", timeout=300 if args.smoke else 1800)
        (out / "preparation_timing.json").write_text(json.dumps(dict(seconds=time.monotonic()-start, train_mb=mb), indent=2)+"\n")
    stats = json.loads((raw / "wiki_zh/stats.json").read_text())
    if (not stats["complete"] or stats["seed"] != 2026 or stats["requested_train_bytes"] != int(mb*1e6)
            or stats["audit"] != audit_splits(raw / "wiki_zh")):
        raise ValueError("Chinese data audit/provenance failed; preserving existing files")
    (out / "wiki_zh_stats.json").write_text(json.dumps(stats, indent=2)+"\n")
    if args.prepare_only:
        return
    english = raw / "wiki_en"
    if not english.exists():
        english.symlink_to(Path("data/milestone/raw/wiki_en").resolve(), target_is_directory=True)
    ref = artifacts / "gpt2"
    if not ref.exists():
        ref.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree("artifacts/milestone/tokenizers/gpt2", ref)
    if args.smoke:
        candidate = artifacts / "wiki_zh70_gpt4_32k"
        if not candidate.exists():
            run("train_tokenizer", "--mix", "wiki_zh=0.7", "wiki_en=0.3", "--total-mb", .1,
                "--raw", raw, "--out", candidate)
    else:
        run("tokenizer_grid", "configs/tokenizer_mini_zh.yaml", "--raw", raw)
    candidates = sorted(artifacts.glob("wiki_zh*"))
    for candidate in candidates:
        meta = json.loads((candidate / "meta.json").read_text())
        for source, sha in meta["train_sha256"].items():
            if file_sha256(raw / source / "train.jsonl") != sha:
                raise ValueError(f"stale tokenizer: {candidate}")
    run("eval_tokenizers", *candidates, "--reference", ref, "--raw", raw, "--split", "validation",
        "--subsets", "wiki_zh", "wiki_en", "--out", out / "validation.csv")
    run("select_tokenizer", out / "validation.csv", "--domain", "wiki_zh", "--english-cap", 2.5 if args.smoke else 1.25,
        "--out", out / "selection.json")
    selected = json.loads((out / "selection.json").read_text())["selected"]["tokenizer"]
    run("eval_tokenizers", artifacts / selected, "--reference", ref, "--raw", raw,
        "--subsets", "wiki_zh", "wiki_en", "--out", out / "test.csv")
    for name in ("gpt2", selected):
        prefix = Path(f"data/{tag}/tokens/wiki_zh/{name}")
        if not any(prefix.with_suffix(s).exists() for s in (".bin", ".idx", ".json")):
            run("encode_corpus", "--source", "wiki_zh", "--tokenizer", artifacts / name, "--name", name,
                "--raw", raw, "--out", f"data/{tag}/tokens")
        elif not all(prefix.with_suffix(s).exists() for s in (".bin", ".idx", ".json")):
            raise ValueError(f"partial encoding preserved: {prefix}")
    run("verify_alignment", "--raw", raw / "wiki_zh/train.jsonl", "--prefixes",
        f"data/{tag}/tokens/wiki_zh/gpt2", f"data/{tag}/tokens/wiki_zh/{selected}",
        "--tokenizers", ref, artifacts / selected, "--out", out / "alignment.json")
    alignment = json.loads((out / "alignment.json").read_text())
    for split in ("validation", "test"):
        run("plot_results", "--compression", out / f"{split}.csv", "--out", out / f"compression_{split}")
    capacity = {r["prefix"].split("/")[-1]:r["n_tokens"] for r in alignment["encodings"]}
    result = dict(status="smoke" if args.smoke else "complete", selected=selected, token_capacity=capacity,
                  sufficient_without_corpus_reuse={str(b): all(n>=b for n in capacity.values()) for b in (50_000_000,200_000_000)},
                  tokenizers={p.name:dict(sha256=file_sha256(p/"tokenizer.json"),metadata=json.loads((p/"meta.json").read_text())) for p in [ref,*candidates]},
                  packages={p:importlib.metadata.version(p) for p in ["opencc","datasets","pyarrow","tokenizers"]},
                  training_launched=False)
    (out / "manifest.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k not in ("tokenizers","packages")},indent=2))


if __name__ == "__main__":
    main()
