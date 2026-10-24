"""Continue pre-training GPT-2 small under one condition (A, B, C or E).

    python scripts/train.py configs/train/B_swap_fvt.yaml token_budget=50_000_000
"""
import argparse
import json
import math
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from toksw import model as M
from toksw.data import MixedBatchSampler, TokenFile, iter_jsonl
from toksw.metrics import bits_per_byte
from toksw.schedule import lr_at, p_new
from toksw.tok import load_tokenizer
from toksw.utils import JsonlLogger, load_config
from toksw.vocab import prune_to_new


def evaluate(setup, old_tok, new_tok, texts, cfg, device):
    out = {}
    if setup.old_map is not None:
        out["old"] = bits_per_byte(setup.model, old_tok, texts, setup.old_map,
                                   cfg["block_size"], cfg["eval_batch"], device)["bpb"]
    if setup.new_map is not None:
        out["new"] = bits_per_byte(setup.model, new_tok, texts, setup.new_map,
                                   cfg["block_size"], cfg["eval_batch"], device)["bpb"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("overrides", nargs="*", help="key=value")
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)
    torch.manual_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    run = Path(cfg["out"]) / f"{cfg['run_name']}_{cfg['token_budget'] // 1_000_000}M"
    run.mkdir(parents=True, exist_ok=True)
    (run / "config.json").write_text(json.dumps(cfg, indent=2))
    log = JsonlLogger(run / "log.jsonl")

    old_tok = load_tokenizer(cfg["old_tokenizer"])
    new_tok = load_tokenizer(cfg["new_tokenizer"]) if cfg["condition"] != "keep" else None
    setup = M.build(cfg["condition"], old_tok, new_tok, cfg["base_model"], cfg.get("init", "fvt"), cfg["seed"])
    model = setup.model.to(device)

    tokens = Path(cfg["tokens_dir"]) / cfg["domain"]
    old_file = TokenFile(tokens / cfg["old_name"]) if setup.old_map is not None else None
    new_file = TokenFile(tokens / cfg["new_name"]) if setup.new_map is not None else None

    tokens_per_step = cfg["block_size"] * cfg["micro_batch"] * cfg["grad_accum"]
    total_steps = cfg["token_budget"] // tokens_per_step
    rho = {"keep": None, "swap": 0.0, "progressive": cfg.get("rho", 0.5)}[cfg["condition"]]
    sched = (lambda s: 0.0) if rho is None else (lambda s: p_new(s, total_steps, rho))
    sampler = MixedBatchSampler(old_file, new_file, setup.old_map, setup.new_map, setup.token_bytes,
                                sched, cfg["block_size"], cfg["micro_batch"], cfg["data_seed"])

    test_path = Path(cfg["raw_dir"]) / cfg["domain"] / "test.jsonl"
    test_texts = [d["text"] for d in iter_jsonl(test_path)]
    eval_texts = test_texts[: cfg["eval_docs"]]

    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": cfg["weight_decay"]}, {"params": no_decay, "weight_decay": 0.0}],
        lr=cfg["lr"], betas=tuple(cfg["betas"]), fused=device == "cuda",
    )
    fwd = torch.compile(model) if cfg.get("compile") else model
    warmup = max(1, int(cfg["warmup_frac"] * total_steps))
    print(f"[train] {run.name}: {total_steps} steps x {tokens_per_step} tokens, device={device}")

    log.log(step=0, eval=evaluate(setup, old_tok, new_tok, eval_texts, cfg, device))
    t0 = time.time()
    bytes_seen = 0
    for step in range(total_steps):
        lr = lr_at(step, total_steps, cfg["lr"], warmup, cfg["min_lr_ratio"])
        for g in opt.param_groups:
            g["lr"] = lr
        nll_sum, n_bytes, frac_new = 0.0, 0, 0.0
        for micro in range(cfg["grad_accum"]):
            x, y, info = sampler.get_batch(step, micro)
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                logits = fwd(input_ids=x).logits
            loss = F.cross_entropy(logits.float().flatten(0, 1), y.flatten())
            (loss / cfg["grad_accum"]).backward()
            nll_sum += loss.item() * y.numel()
            n_bytes += info["n_bytes"]
            frac_new += info["frac_new"] / cfg["grad_accum"]
        bytes_seen += n_bytes
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        opt.step()
        opt.zero_grad(set_to_none=True)

        if step % cfg["log_every"] == 0:
            elapsed = time.time() - t0
            log.log(step=step, lr=lr, train_bpb=nll_sum / math.log(2) / n_bytes,
                    train_loss=nll_sum / (tokens_per_step), p_new=info["p_new"], frac_new=frac_new,
                    grad_norm=float(gnorm), tokens=(step + 1) * tokens_per_step, bytes_seen=bytes_seen,
                    tok_per_s=(step + 1) * tokens_per_step / max(elapsed, 1e-9))
        if (step + 1) % cfg["eval_every"] == 0 and step + 1 < total_steps:
            log.log(step=step + 1, eval=evaluate(setup, old_tok, new_tok, eval_texts, cfg, device))

    final = evaluate(setup, old_tok, new_tok, test_texts, cfg, device)
    log.log(step=total_steps, final_eval=final, wall_s=time.time() - t0)
    print(f"[final] {final}")

    ckpt = run / "model"
    if cfg["condition"] == "progressive":
        # Drop old-only rows: the final model has exactly the new vocabulary.
        model.cpu()
        pruned = prune_to_new(model.transformer.wte.weight.data, setup.union)
        M._set_embeddings(model, pruned)
        setup = M.Setup(model, None, np.arange(pruned.shape[0]), setup.token_bytes[setup.union.new2union])
        model.to(device)
        pruned_eval = evaluate(setup, old_tok, new_tok, test_texts, cfg, device)
        log.log(step=total_steps, final_eval_pruned=pruned_eval)
        print(f"[final pruned] {pruned_eval}")
    model.save_pretrained(ckpt)
    tok_src = cfg["new_tokenizer"] if cfg["condition"] != "keep" else None
    if tok_src:
        src = Path(tok_src)
        shutil.copy(src / "tokenizer.json" if src.is_dir() else src, ckpt / "tokenizer.json")
    else:
        old_tok.save(str(ckpt / "tokenizer.json"))


if __name__ == "__main__":
    main()
