"""Table and figure of the rho ablation, from the real logs only.

    python results/code50_djam_rho_ablation_dropout01/plot_rho.py

Reads the three runs of this folder and, as references, A, B and E (rho 0.5) from
results/code50_djam_corrected_dropout01. Writes rho_ablation.{csv,png,pdf} here.
"""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
REF = HERE.parent / "code50_djam_corrected_dropout01"
# label, folder, Slurm job, sha256 of the saved model.safetensors (None: listed in REF/comparison.csv)
RUNS = [
    ("A keep GPT-2", REF / "A", 774132, None),
    ("B FVT swap", REF / "B", 774666, None),
    ("E rho 0", HERE / "rho0", 776145, "1cfb23f9f0c342d124418f00efc9f6faf961380bc1334f60b230af256dbfd123"),
    ("E rho 0.25", HERE / "rho025", 776550, "12dbf6052b7bf5ba196a4203b178111824c5279bf75c38a056d12976e4311161"),
    ("E rho 0.5", REF / "E", 775737, None),
    ("E rho 0.75", HERE / "rho075", 777515, "a4b5e666bb8050070020fcd30b3721ffb088e3aecca121877d31348f9767cd0a"),
]


def load(folder):
    cfg = json.loads((folder / "config.json").read_text())
    events = [json.loads(line) for line in (folder / "log.jsonl").read_text().splitlines() if line.strip()]
    train = {e["step"]: e for e in events if "train_bpb" in e}
    final = next(e for e in events if "final_eval" in e)
    pruned = next((e["final_eval_pruned"]["new"] for e in events if "final_eval_pruned" in e), None)
    key = "old" if cfg["condition"] == "keep" else "new"
    # validation curve: bytes read are taken from the training event logged at the same step
    curve = [(train[e["step"]]["bytes_seen"] if e["step"] else 0, e["eval"][key], e["eval"].get("old"))
             for e in events if "eval" in e and (e["step"] == 0 or e["step"] in train)]
    last = train[max(train)]
    return dict(cfg=cfg, curve=curve, old=final["final_eval"].get("old"), new=final["final_eval"].get("new"),
                pruned=pruned, bpb=pruned if pruned is not None else final["final_eval"][key],
                bytes_seen=last["bytes_seen"], wall_h=final["wall_s"] / 3600)


def main():
    data = [(label, job, sha, load(folder)) for label, folder, job, sha in RUNS]
    b = next(d["bpb"] for label, _, _, d in data if label.startswith("B"))
    a = next(d["bpb"] for label, _, _, d in data if label.startswith("A"))
    with open(HERE / "rho_ablation.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "rho", "slurm_job", "test_bpb", "test_bpb_minus_B", "test_bpb_new_before_pruning",
                    "test_bpb_old_before_pruning", "training_bytes_read", "trainer_hours", "commit", "weights_sha256"])
        for label, job, sha, d in data:
            rho = d["cfg"]["rho"] if d["cfg"]["condition"] == "progressive" else ""
            w.writerow([label, rho, job, f"{d['bpb']:.7f}", f"{d['bpb'] - b:+.7f}",
                        "" if d["pruned"] is None else f"{d['new']:.7f}",
                        "" if d["pruned"] is None else f"{d['old']:.7f}",
                        d["bytes_seen"], f"{d['wall_h']:.2f}", d["cfg"]["commit"], sha or "see ../code50_djam_corrected_dropout01/comparison.csv"])

    e_runs = [(d["cfg"]["rho"], d) for label, _, _, d in data if label.startswith("E")]
    colors = dict(zip([0, 0.25, 0.5, 0.75], ["#1b9e77", "#7570b3", "#d95f02", "#e7298a"]))
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), constrained_layout=True)
    ax = axes[0]
    ax.plot([r for r, _ in e_runs], [d["bpb"] for _, d in e_runs], "o-", color="black", label="E, pruned")
    ax.axhline(b, color="#386CB0", linestyle="--", label=f"B FVT swap ({b:.4f})")
    ax.axhline(a, color="#888888", linestyle=":", label=f"A keep GPT-2 ({a:.4f})")
    for r, d in e_runs:
        ax.annotate(f"{d['bpb']:.4f}", (r, d["bpb"]), textcoords="offset points", xytext=(6, -12))
    ax.set_xticks([0, 0.25, 0.5, 0.75])
    ax.set_xlabel("rho (fraction of training spent ramping to the new tokenizer)")
    ax.set_ylabel("test bpb, 1,000 documents (lower is better)")
    ax.set_title("Final score against rho")
    for ax, title, xlabel, x_of in [
        (axes[1], "Validation bpb (new tokenizer) against text read", "training bytes read (MB)", None),
        (axes[2], "Validation bpb (old tokenizer) during training", "training bytes read (MB)", "old"),
    ]:
        for r, d in e_runs:
            pts = [(x / 1e6, old if x_of else new) for x, new, old in d["curve"]]
            ax.plot(*zip(*pts), "o-", ms=3, color=colors[r], label=f"E rho {r:g}")
        ref_label, ref = ("A keep GPT-2", "A") if x_of else ("B FVT swap", "B")
        d = next(d for label, _, _, d in data if label.startswith(ref))
        ax.plot(*zip(*[(x / 1e6, y) for x, y, _ in d["curve"]]), "s--", ms=3, color="#386CB0", label=ref_label)
        ax.set_ylim(0.68, 1.4)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("validation bpb, 200 documents")
        ax.set_title(title)
    for ax in axes:
        ax.grid(alpha=.2)
        ax.legend()
    fig.suptitle("Code 50M, rho ablation of the progressive transition (one seed per point)")
    for ext in ("png", "pdf"):
        fig.savefig(HERE / f"rho_ablation.{ext}", dpi=180)


if __name__ == "__main__":
    main()
