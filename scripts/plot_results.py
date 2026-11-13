"""Plot measured compression CSVs or real training JSONL; never synthesize data."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def compression(csv_path, out, title=None):
    rows = list(csv.DictReader(open(csv_path)))
    labels = [r["tokenizer"] for r in rows]
    domain = "wiki_zh" if "nsl_wiki_zh" in rows[0] else "code"
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for ax, metric, ylabel in zip(axes, ("nsl", "bytes_per_token"), ("NSL vs GPT-2 (lower is better)", "UTF-8 bytes per token (higher is better)")):
        for offset, subset, color in [(-.19, domain, "#386CB0"), (.19, "wiki_en", "#E68A2E")]:
            ax.bar([i + offset for i in range(len(rows))], [float(r[f"{metric}_{subset}"]) for r in rows],
                   width=.38, label={"code":"Python", "wiki_en":"English", "wiki_zh":"Chinese"}[subset], color=color)
        ax.set_xticks(range(len(rows)), labels, rotation=35, ha="right")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(axis="y", alpha=.2)
    axes[0].axhline(1, color="black", linewidth=.8, linestyle="--")
    fig.suptitle(title or f"Measured tokenizer compression — {Path(csv_path).stem}")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=180)
    plt.close(fig)


def training(logs, out, title=None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    if title:
        fig.suptitle(title)
    measured = False
    eval_splits = set()
    for log in logs:
        path = Path(log)
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        config_path = path.parent / "config.json"
        cfg = json.loads(config_path.read_text()) if config_path.exists() else None
        train = [r for r in rows if "train_bpb" in r and "tokens" in r]
        if train:
            measured = True
            axes[0].plot([r["tokens"] / 1e6 for r in train], [r["train_bpb"] for r in train], label=path.parent.name)
        ev = [r for r in rows if "eval" in r]
        if ev and cfg:
            eval_splits.add(cfg.get("eval_split", "test"))
            tps = cfg["block_size"] * cfg["micro_batch"] * cfg["grad_accum"]
            for encoding in ("old", "new"):
                pts = [r for r in ev if encoding in r["eval"]]
                if pts:
                    measured = True
                    axes[1].plot([r["step"] * tps / 1e6 for r in pts], [r["eval"][encoding] for r in pts],
                                 label=f"{path.parent.name}/{encoding}")
    if not measured:
        plt.close(fig)
        raise ValueError("No measured training/eval points found")
    eval_label = "Validation bpb" if eval_splits == {"validation"} else "Evaluation bpb"
    axes[1].set_title("Periodic evaluation")
    for ax, label in zip(axes, ("Training bpb", eval_label)):
        ax.set_xlabel("Training tokens (millions)")
        ax.set_ylabel(label)
        if ax.lines:
            ax.legend(fontsize=7)
        ax.grid(alpha=.2)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=180)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--compression")
    group.add_argument("--logs", nargs="+")
    ap.add_argument("--out", required=True, help="output prefix without extension")
    ap.add_argument("--title", help="explicit scope label for figures")
    args = ap.parse_args()
    if args.compression:
        compression(args.compression, args.out, args.title)
    else:
        training(args.logs, args.out, args.title)


if __name__ == "__main__":
    main()
