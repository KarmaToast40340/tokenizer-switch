"""Gather every training run into one results table.

    python scripts/collect_results.py collected --out results/runs.csv

Looks for run folders (a ``config.json`` next to a ``log.jsonl``, as written by ``train.py``) anywhere
under the given folders, typically ``collected/<account>/<run>/`` copied back from the cluster.
Standard library only, so it runs on a laptop without torch. Unfinished runs (crash, time limit) are
kept with ``status=incomplete`` so that they are not silently missing from the figures.

Every ``bpb*`` column except ``bpb_step0_new`` is measured on the 1000 held-out *test* documents
(train.py's final evaluation). ``eval_split`` only says which split the periodic evaluations and the
step-0 evaluation used. Folders renamed after a failure (``<run>.failed1``) are skipped.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

LETTER = {("keep", None): "A", ("swap", "fvt"): "B", ("swap", "random"): "C", ("progressive", "fvt"): "E"}
COLUMNS = [
    "run", "letter", "condition", "init", "rho", "domain", "budget_M", "status",
    "bpb", "bpb_old", "bpb_new", "bpb_pruned", "bpb_step0_new", "peak_train_bpb",
    "tokens", "bytes_seen", "wall_h", "tok_per_s",
    "eval_split", "micro_batch", "grad_accum", "dropout", "seed", "account", "commit", "path",
]
SHORT_HASH = 7  # `git rev-parse --short`; some launchers record the full 40-character hash


def headline_bpb(condition: str, final: dict, pruned: dict) -> float | None:
    """The one bits-per-byte number that represents a run in tables and figures.

    ``final`` is train.py's ``final_eval`` on the 1000 test documents: ``{"old": ..., "new": ...}``,
    with only the keys the condition can evaluate (A: old; B, C: new; E: both, on the union vocabulary).
    ``pruned`` is ``final_eval_pruned`` (E only, ``{"new": ...}`` after dropping the old-only rows), else {}.
    Return None when the needed value is missing.
    """
    if condition == "keep":
        return final.get("old")
    elif condition == "swap":
        return final.get("new")
    elif condition == "progressive":
        # the pruned model has exactly B's vocabulary: same object as B, fair comparison
        return pruned.get("new")
    return None


def run_status(condition: str, final: dict | None, pruned: dict) -> str:
    """"done" when the run wrote everything its headline number needs, else "incomplete".

    train.py writes ``final_eval`` first; for E it then prunes the old-only rows and writes
    ``final_eval_pruned``. A job killed in between leaves ``final`` set and ``pruned`` empty.
    """
    if final is None:
        return "incomplete"
    return "done" if headline_bpb(condition, final, pruned) is not None else "incomplete"


def read_log(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass  # last line cut by a killed job
    return rows


def summarize(run_dir: Path, root: Path) -> dict:
    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    rows = read_log(run_dir / "log.jsonl")
    final = next((r["final_eval"] for r in rows if "final_eval" in r), None)
    pruned = next((r["final_eval_pruned"] for r in rows if "final_eval_pruned" in r), {})
    wall_s = next((r["wall_s"] for r in rows if "wall_s" in r), None)
    train = [r for r in rows if "train_bpb" in r]
    step0 = next((r["eval"] for r in rows if r.get("step") == 0 and "eval" in r), {})

    condition, init = cfg.get("condition"), cfg.get("init")
    key_init = None if condition == "keep" else init or "fvt"
    rho = {"keep": None, "swap": 0.0}.get(condition, cfg.get("rho", 0.5))
    rel = run_dir.relative_to(root)
    account = cfg.get("account") or (rel.parts[0] if len(rel.parts) > 1 else "")

    return {
        "run": run_dir.name,
        "letter": LETTER.get((condition, key_init), "?"),
        "condition": condition,
        "init": key_init or "",
        "rho": "" if rho is None else rho,
        "domain": cfg.get("domain"),
        "budget_M": cfg.get("token_budget", 0) // 1_000_000,
        "status": run_status(condition, final, pruned),
        "bpb": headline_bpb(condition, final, pruned) if final is not None else None,
        "bpb_old": (final or {}).get("old"),
        "bpb_new": (final or {}).get("new"),
        "bpb_pruned": pruned.get("new"),
        "bpb_step0_new": step0.get("new"),
        "peak_train_bpb": max((r["train_bpb"] for r in train), default=None),
        "tokens": train[-1]["tokens"] if train else None,
        "bytes_seen": train[-1]["bytes_seen"] if train else None,
        "wall_h": round(wall_s / 3600, 2) if wall_s is not None else None,
        "tok_per_s": round(train[-1]["tok_per_s"]) if train else None,
        "eval_split": cfg.get("eval_split", "test"),  # train.py's default
        "micro_batch": cfg.get("micro_batch"),
        "grad_accum": cfg.get("grad_accum"),
        "dropout": cfg.get("expected_dropout"),
        "seed": cfg.get("seed"),
        "account": account,
        "commit": str(cfg.get("commit", "")),
        "path": rel.as_posix(),
    }


def find_runs(roots: list[Path]) -> list[tuple[Path, Path]]:
    found = []
    for root in roots:
        for cfg in sorted(root.rglob("config.json")):
            if ".failed" in cfg.parent.name:
                continue  # an earlier attempt, renamed before resubmitting
            if (cfg.parent / "log.jsonl").exists():
                found.append((cfg.parent, root))
    return found


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("roots", nargs="+", type=Path, help="folders to search, e.g. collected")
    ap.add_argument("--out", type=Path, default=Path("results/runs.csv"))
    args = ap.parse_args(argv)

    runs = find_runs(args.roots)
    if not runs:
        print("no run folder (config.json + log.jsonl) found", file=sys.stderr)
        return 1
    table = [summarize(d, root) for d, root in runs]
    table.sort(key=lambda r: (r["domain"] or "", r["budget_M"], r["letter"], str(r["rho"]), r["run"]))

    names = [r["run"] for r in table]
    for name in sorted({n for n in names if names.count(n) > 1}):
        print(f"warning: {name} appears {names.count(name)} times (copied from two accounts?)", file=sys.stderr)
    commits = {r["commit"][:SHORT_HASH] for r in table if r["commit"]}
    if len(commits) > 1:
        print(f"warning: runs come from {len(commits)} commits: {', '.join(sorted(commits))}", file=sys.stderr)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(table)
    done = sum(r["status"] == "done" for r in table)
    print(f"{len(table)} runs ({done} done, {len(table) - done} incomplete) -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
