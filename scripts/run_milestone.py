"""Bounded CPU data/tokenizer milestone; no training, cluster access or git writes.

Run with PYTHONPATH=src, HF_HOME=data/hf-cache and TOKENIZERS_PARALLELISM=false.
Existing completed splits are checksum-verified; partial directories are preserved
and rejected. A separate smoke root avoids contaminating the measured experiment.
"""
import argparse
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys

from huggingface_hub import HfApi, hf_hub_download

from toksw.splits import audit_splits, file_sha256


def run(script, *args):
    cmd = [sys.executable, f"scripts/{script}.py", *map(str, args)]
    print("RUN", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    tag = "smoke" if args.smoke else "milestone"
    raw = Path(f"data/{tag}/raw")
    artifacts = Path(f"artifacts/{tag}/tokenizers")
    results = Path(f"results/{tag}")
    results.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(".").free < 1024 ** 3:
        raise RuntimeError("At least 1 GiB free disk required for this bounded run")
    budgets = {"code": .2 if args.smoke else 50, "wiki_en": .2 if args.smoke else 20}
    for source, mb in budgets.items():
        dest = raw / source
        if not dest.exists():
            saved = results / f"{source}_stats.json"
            revisions = {"code": "35a59fb025bc0a102f7d96eac09d145b896d487b",
                         "wiki_en": "b04c8d1ceb2f5cd4588862100d08de323dccfbaa"}
            revision = ["--revision", json.loads(saved.read_text())["provenance"]["revision"]
                        if saved.exists() else revisions[source]]
            run("prepare_milestone", "--source", source, "--out", raw, "--train-mb", mb,
                "--n-eval", 10 if args.smoke else 100, "--chunk-chars", 3000 if source == "code" else 0, *revision)
        stats = json.loads((dest / "stats.json").read_text())
        if (not stats["complete"] or stats["audit"] != audit_splits(dest)
                or stats["requested_train_bytes"] != int(mb * 1e6)
                or stats["seed"] != 2026 or stats["requested_eval_docs"] != (10 if args.smoke else 100)):
            raise ValueError(f"incomplete or modified splits: {dest}")
        (results / f"{source}_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    ref = artifacts / "gpt2"
    if not ref.exists():
        saved = results / "manifest.json"
        revision = (json.loads(saved.read_text())["tokenizers"]["gpt2"]["metadata"]["revision"]
                    if saved.exists() else HfApi().model_info("gpt2").sha)
        path = hf_hub_download("gpt2", "tokenizer.json", revision=revision)
        ref.mkdir(parents=True)
        shutil.copyfile(path, ref / "tokenizer.json")
        (ref / "meta.json").write_text(json.dumps(dict(hf_repo="gpt2", revision=revision), indent=2) + "\n")
    if args.smoke:
        candidate = artifacts / "code70_gpt4_32k"
        if not candidate.exists():
            run("train_tokenizer", "--mix", "code=0.7", "wiki_en=0.3", "--total-mb", .1,
                "--raw", raw, "--out", candidate)
    else:
        run("tokenizer_grid", "configs/tokenizer_mini_code.yaml", "--raw", raw)
    candidates = sorted(artifacts.glob("code*"))
    for p in candidates:
        meta = json.loads((p / "meta.json").read_text())
        for source, sha in meta["train_sha256"].items():
            if file_sha256(raw / source / "train.jsonl") != sha:
                raise ValueError(f"stale tokenizer trained on different data: {p}")
    # Ref and candidate names are stable across machines, and only train data
    # enters tokenizer fitting. Test is evaluated only after selection is saved.
    run("eval_tokenizers", *candidates, "--reference", ref, "--raw", raw, "--split", "validation",
        "--out", results / "validation.csv")
    run("select_tokenizer", results / "validation.csv", "--english-cap", 2 if args.smoke else 1.25,
        "--out", results / "selection.json")
    selected = json.loads((results / "selection.json").read_text())["selected"]["tokenizer"]
    run("eval_tokenizers", artifacts / selected, "--reference", ref, "--raw", raw,
        "--out", results / "test.csv")
    for name in ("gpt2", selected):
        prefix = Path(f"data/{tag}/tokens/code") / name
        if prefix.with_suffix(".bin").exists() or prefix.with_suffix(".idx").exists():
            if not prefix.with_suffix(".json").exists():
                raise ValueError(f"partial encoded output preserved: {prefix}")
        else:
            run("encode_corpus", "--source", "code", "--tokenizer", artifacts / name, "--name", name,
                "--raw", raw, "--out", f"data/{tag}/tokens")
    run("verify_alignment", "--raw", raw / "code/train.jsonl", "--prefixes",
        f"data/{tag}/tokens/code/gpt2", f"data/{tag}/tokens/code/{selected}", "--tokenizers",
        ref, artifacts / selected, "--out", results / "alignment.json")
    for split in ("validation", "test"):
        run("plot_results", "--compression", results / f"{split}.csv", "--out", results / f"compression_{split}")
    manifest = dict(python=sys.version, selected=selected,
                    packages={p: importlib.metadata.version(p) for p in
                              ("torch", "transformers", "tokenizers", "datasets", "numpy", "matplotlib", "huggingface-hub")},
                    tokenizers={p.name: dict(sha256=file_sha256(p / "tokenizer.json"),
                                            metadata=json.loads((p / "meta.json").read_text())) for p in [ref, *candidates]})
    (results / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Completed {tag}; selected {selected}; outputs in {results}")


if __name__ == "__main__":
    main()
