"""Allocated-GPU compatibility/memory check using the frozen Chinese A/B/E assets.

No checkpoint is saved and no test text is read. Outputs are diagnostics only.
Run through cluster/job.sbatch after storage, environment and checksum checks.
"""
import argparse
import gc
import json
import math
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
import torch.nn.functional as F

from toksw import model as M
from toksw.data import MixedBatchSampler, TokenFile, iter_jsonl
from toksw.metrics import bits_per_byte
from toksw.tok import load_tokenizer
from toksw.utils import load_config
from toksw.vocab import prune_to_new


def check_condition(cfg, old_tok, new_tok, texts, device='cuda'):
    """Match train.py's precision, loss, optimizer and batch memory footprint."""
    torch.manual_seed(cfg['seed'])
    setup = M.build(cfg['condition'], old_tok, new_tok, cfg['base_model'], 'fvt', cfg['seed'])
    model = setup.model.to(device)
    model.train()  # Match the explicit policy in the corrected trainer.
    training_forward_mode = model.training
    vocab_size = model.config.vocab_size
    tokens = Path(cfg['tokens_dir']) / cfg['domain']
    old_file = TokenFile(tokens / cfg['old_name']) if setup.old_map is not None else None
    new_file = TokenFile(tokens / cfg['new_name']) if setup.new_map is not None else None
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{'params': decay, 'weight_decay': cfg['weight_decay']},
                             {'params': no_decay, 'weight_decay': 0.}],
                            lr=cfg['lr'], betas=tuple(cfg['betas']), fused=device == 'cuda')
    # In E exercise both old and new paths, including optimizer state allocation.
    p = (lambda s: float(s)) if cfg['condition'] == 'progressive' else (lambda s: float(cfg['condition'] == 'swap'))
    sampler = MixedBatchSampler(old_file, new_file, setup.old_map, setup.new_map, setup.token_bytes,
                                p, cfg['block_size'], cfg['micro_batch'], cfg['data_seed'])
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    start, losses = time.monotonic(), []
    for step in range(2):
        for micro in range(cfg['grad_accum']):
            x, y, _ = sampler.get_batch(step, micro)
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=device == 'cuda'):
                logits = model(input_ids=x).logits
            loss = F.cross_entropy(logits.float().flatten(0, 1), y.flatten())
            if not torch.isfinite(loss):
                raise RuntimeError('non-finite smoke loss')
            (loss / cfg['grad_accum']).backward()
            losses.append(loss.item())
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'])
        opt.step()
        opt.zero_grad(set_to_none=True)

    def evaluate(tok, mapping):
        result = bits_per_byte(model, tok, texts, mapping, cfg['block_size'], 1, device)
        if not math.isfinite(result['bpb']):
            raise RuntimeError('non-finite validation diagnostic')

    if setup.old_map is not None:
        evaluate(old_tok, setup.old_map)
    if setup.new_map is not None:
        evaluate(new_tok, setup.new_map)
    if setup.union is not None:
        # Keep optimizer alive as train.py does during its final pruning/eval.
        model.cpu()
        emb = prune_to_new(model.transformer.wte.weight.data, setup.union)
        M._set_embeddings(model, emb)
        model.to(device)
        evaluate(new_tok, np.arange(len(emb)))
    if device == 'cuda':
        torch.cuda.synchronize()
    return dict(status='passed', condition=cfg['condition'], vocab_size=vocab_size,
                training_forward_mode=training_forward_mode,
                final_vocab_size=model.config.vocab_size, optimizer_steps=2,
                block_size=cfg['block_size'], micro_batch=cfg['micro_batch'], grad_accum=cfg['grad_accum'],
                elapsed_s=time.monotonic()-start, finite_losses=len(losses),
                peak_allocated_bytes=torch.cuda.max_memory_allocated() if device == 'cuda' else None,
                peak_reserved_bytes=torch.cuda.max_memory_reserved() if device == 'cuda' else None)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--old-tokenizer', default='artifacts/tokenizers/gpt2')
    ap.add_argument('--new-tokenizer', default='artifacts/tokenizers/wiki_zh70_gpt4_50k')
    ap.add_argument('--tokens-dir', default='data/tokens')
    ap.add_argument('--raw-dir', default='data/raw')
    ap.add_argument('--base-model', default='gpt2')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        ap.error(f'preserving existing diagnostic: {args.out}')
    if not torch.cuda.is_available():
        ap.error('CUDA unavailable; run inside a Slurm GPU allocation')
    gpu = torch.cuda.get_device_name()
    if 'RTX 5050' not in gpu or not torch.cuda.is_bf16_supported():
        ap.error(f'planned RTX 5050 with bf16 required, got {gpu}')
    # Execute a real kernel; architecture-list inspection alone is insufficient.
    (torch.ones((32, 32), device='cuda') @ torch.ones((32, 32), device='cuda')).sum().item()
    old_tok, new_tok = load_tokenizer(args.old_tokenizer), load_tokenizer(args.new_tokenizer)
    texts = [d['text'] for d in iter_jsonl(Path(args.raw_dir) / 'wiki_zh/validation.jsonl', 1)]
    if not texts:
        ap.error('validation text is missing/empty')
    report = dict(status='running', purpose='GPU smoke only; excluded from model comparisons',
                  gpu=gpu, capability=torch.cuda.get_device_capability(), torch=torch.__version__,
                  cuda=torch.version.cuda, commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                  slurm_job_id=__import__('os').environ.get('SLURM_JOB_ID'), conditions=[])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as f:
        f.write(json.dumps(report, indent=2) + '\n')
    try:
        for config in ['A_keep', 'B_swap_fvt', 'E_progressive']:
            cfg = load_config(f'configs/train/{config}.yaml')
            cfg.update(domain='wiki_zh', tokens_dir=args.tokens_dir, raw_dir=args.raw_dir,
                       old_name='gpt2', new_name='wiki_zh70_gpt4_50k', base_model=args.base_model)
            result = check_condition(cfg, old_tok, new_tok, texts)
            report['conditions'].append(result)
            print(json.dumps(result), flush=True)
            gc.collect()
            torch.cuda.empty_cache()
            args.out.write_text(json.dumps(report, indent=2) + '\n')
        report['status'] = 'passed'
    except Exception as e:
        report.update(status='failed', error=f'{type(e).__name__}: {e}')
        raise
    finally:
        args.out.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
