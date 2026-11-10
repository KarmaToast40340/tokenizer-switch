"""GTX 1060 feasibility diagnostic, not a model-quality experiment.

Each trial gets a fresh process so failed CUDA allocations cannot contaminate
later memory measurements. Uses validation only and retains no checkpoint.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def probe():
    import torch
    report = dict(cuda_available=torch.cuda.is_available(), torch=torch.__version__, cuda=torch.version.cuda)
    if not report['cuda_available']:
        return report
    report.update(gpu=torch.cuda.get_device_name(), capability=torch.cuda.get_device_capability(),
                  memory_bytes=torch.cuda.get_device_properties(0).total_memory,
                  compiled_architectures=torch.cuda.get_arch_list())
    try:
        report['native_bf16'] = torch.cuda.is_bf16_supported(including_emulation=False)
    except TypeError:
        report['native_bf16'] = report['capability'][0] >= 8
    for name, dtype in [('fp32', torch.float32), ('bf16', torch.bfloat16)]:
        try:
            a = torch.randn(64, 64, device='cuda', dtype=dtype, requires_grad=True)
            (a @ a).float().square().mean().backward()
            torch.cuda.synchronize()
            report[name + '_forward_backward'] = 'passed'
        except Exception as e:
            report[name + '_forward_backward'] = f'{type(e).__name__}: {e}'
    return report


def trial(args):
    import gc
    import tempfile
    import time
    import torch
    import torch.nn.functional as F
    # transformers 4.49 imports optional OpenCV/image modules while loading
    # GPT-2. Compute nodes lack libGL; this text-only process needs neither.
    # These flags affect only this worker, never the shared environment files.
    import transformers.utils.import_utils as imports
    imports._cv2_available = False
    imports._torchvision_available = False
    from toksw import model as M
    from toksw.data import MixedBatchSampler, TokenFile, iter_jsonl
    from toksw.tok import load_tokenizer
    from toksw.utils import load_config
    from toksw.vocab import prune_to_new

    cfg = load_config(f'configs/train/{args.condition}.yaml')
    old_tok = load_tokenizer('artifacts/tokenizers/gpt2')
    new_tok = load_tokenizer('artifacts/tokenizers/wiki_zh70_gpt4_50k')
    torch.manual_seed(cfg['seed'])
    setup = M.build(cfg['condition'], old_tok, new_tok, args.base_model, 'fvt', cfg['seed'])
    model = setup.model.to('cuda')
    model.train()  # Match the explicit policy in the corrected trainer.
    training_forward_mode = model.training
    old = TokenFile('data/tokens/wiki_zh/gpt2') if setup.old_map is not None else None
    new = TokenFile('data/tokens/wiki_zh/wiki_zh70_gpt4_50k') if setup.new_map is not None else None
    tokens_per_step = cfg['block_size'] * cfg['micro_batch'] * cfg['grad_accum']
    accum = tokens_per_step // (cfg['block_size'] * args.micro_batch)
    assert accum * args.micro_batch * cfg['block_size'] == tokens_per_step
    schedule = (lambda s: float(s > 0)) if setup.union is not None else (lambda s: float(new is not None))
    sampler = MixedBatchSampler(old, new, setup.old_map, setup.new_map, setup.token_bytes,
                                schedule, cfg['block_size'], args.micro_batch, cfg['data_seed'])
    opt = torch.optim.AdamW([
        dict(params=[p for p in model.parameters() if p.dim() >= 2], weight_decay=cfg['weight_decay']),
        dict(params=[p for p in model.parameters() if p.dim() < 2], weight_decay=0.)],
        lr=cfg['lr'], betas=tuple(cfg['betas']), fused=True)
    torch.cuda.reset_peak_memory_stats()

    retained = {}
    def micro(step, index):
        x, y, _ = sampler.get_batch(step, index)
        x, y = x.cuda(), y.cuda()
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=args.precision == 'bf16'):
            retained['logits'] = model(input_ids=x).logits
        retained['loss'] = F.cross_entropy(retained['logits'].float().flatten(0, 1), y.flatten())
        if not torch.isfinite(retained['loss']):
            raise RuntimeError('nonfinite loss')
        (retained['loss'] / accum).backward()
        return retained['loss'].item()

    def update():
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'])
        opt.step()
        opt.zero_grad(set_to_none=True)

    micro(0, 0)
    update()  # allocate optimizer states before timing
    torch.cuda.synchronize()
    start = time.perf_counter()
    for i in range(args.measure_micro):
        micro(1, i)  # E exercises old warmup, then new steady-state paths
    torch.cuda.synchronize()
    micro_s = (time.perf_counter() - start) / args.measure_micro
    start = time.perf_counter()
    update()
    torch.cuda.synchronize()
    update_s = time.perf_counter() - start
    projected_step_s = accum * micro_s + update_s

    # Same validation text for every condition, no held-out test access.
    text = next(iter_jsonl('data/raw/wiki_zh/validation.jsonl'))['text']
    model.eval()
    eval_start = time.perf_counter()
    eval_tokens = 0
    def validate(tok, mapping):
        nonlocal eval_tokens
        ids = tok.encode(text, add_special_tokens=False).ids[:cfg['block_size'] + 1]
        if mapping is not None:
            ids = mapping[ids].tolist()
        x = torch.tensor([ids[:-1]], device='cuda')
        y = torch.tensor([ids[1:]], device='cuda')
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=args.precision == 'bf16'):
            logits = model(input_ids=x).logits
        loss = F.cross_entropy(logits.float().flatten(0, 1), y.flatten())
        if not torch.isfinite(loss):
            raise RuntimeError('nonfinite validation diagnostic')
        eval_tokens += y.numel()
    if setup.old_map is not None:
        validate(old_tok, setup.old_map)
    if setup.new_map is not None:
        validate(new_tok, setup.new_map)
    torch.cuda.synchronize()
    eval_s = time.perf_counter() - eval_start
    vocab_size = model.config.vocab_size
    if setup.union is not None:
        model.cpu()
        emb = prune_to_new(model.transformer.wte.weight.data, setup.union)
        M._set_embeddings(model, emb)
        model.cuda()
        validate(new_tok, None)
    torch.cuda.synchronize()
    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    model.cpu()
    del opt
    gc.collect()
    torch.cuda.empty_cache()
    # Measure actual save size in persistent project storage; remove only this
    # newly created diagnostic directory. Never touch existing runs/coursework.
    scratch = Path('runs/diagnostics')
    scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='gtx1060-save-', dir=scratch) as directory:
        start = time.perf_counter()
        model.save_pretrained(directory)
        (new_tok if new is not None else old_tok).save(str(Path(directory) / 'tokenizer.json'))
        checkpoint_bytes = sum(p.stat().st_size for p in Path(directory).rglob('*') if p.is_file())
        save_s = time.perf_counter() - start
    return dict(status='passed', condition=args.condition, precision=args.precision,
                training_forward_mode=training_forward_mode,
                vocab_size=vocab_size, final_vocab_size=model.config.vocab_size,
                block_size=cfg['block_size'], micro_batch=args.micro_batch, grad_accum=accum,
                tokens_per_optimizer_step=tokens_per_step, measured_micro_batches=args.measure_micro,
                micro_batch_seconds=micro_s, optimizer_update_seconds=update_s,
                estimated_training_tokens_per_second=tokens_per_step/projected_step_s,
                projected_50M_train_hours=(50_000_000//tokens_per_step)*projected_step_s/3600,
                validation_forward_seconds=eval_s, validation_forward_tokens=eval_tokens,
                peak_allocated_bytes=peak_allocated, peak_reserved_bytes=peak_reserved,
                checkpoint_bytes=checkpoint_bytes, checkpoint_save_seconds=save_s,
                caveat='Few-microbatch projection excludes periodic/final full evaluation, setup, scheduling and save time; diagnostic weights removed; not a training result.')


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--worker', choices=['probe', 'trial'])
    ap.add_argument('--condition', choices=['A_keep', 'B_swap_fvt', 'E_progressive'])
    ap.add_argument('--precision', choices=['fp32', 'bf16'], default='fp32')
    ap.add_argument('--micro-batch', type=int, default=2)
    ap.add_argument('--measure-micro', type=int, default=4)
    ap.add_argument('--base-model', default='data/hf-cache/hub/models--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e')
    ap.add_argument('--out', type=Path)
    args = ap.parse_args()
    if args.worker:
        try:
            result = probe() if args.worker == 'probe' else trial(args)
        except Exception as e:
            result = dict(status='failed', error=f'{type(e).__name__}: {e}')
        print('RESULT_JSON=' + json.dumps(result), flush=True)
        return
    if args.out is None or args.out.exists():
        ap.error('--out must be a new diagnostic output')
    report = dict(status='running', purpose='GTX 1060 feasibility only; excluded from model comparisons',
                  job_id=os.environ.get('SLURM_JOB_ID'),
                  commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(), trials=[])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as f:
        f.write(json.dumps(report, indent=2)+'\n')
    def run(extra):
        cmd = [sys.executable, '-u', __file__, '--worker', *extra, '--base-model', args.base_model]
        try:
            process = subprocess.run(cmd, capture_output=True, text=True, timeout=240)
        except subprocess.TimeoutExpired:
            return dict(status='failed', error='trial timeout after 240 seconds')
        print(process.stdout, flush=True)
        print(process.stderr, file=sys.stderr, flush=True)
        lines = [s for s in process.stdout.splitlines() if s.startswith('RESULT_JSON=')]
        if not lines:
            return dict(status='failed', exit_code=process.returncode, error=process.stderr[-2000:])
        return json.loads(lines[-1][len('RESULT_JSON='):])
    report['probe'] = run(['probe'])
    p = report['probe']
    if not p.get('cuda_available') or 'GTX 1060' not in p.get('gpu', '') or p.get('fp32_forward_backward') != 'passed':
        report['status'] = 'incompatible'
    else:
        for precision in ['bf16', 'fp32']:
            for condition in ['A_keep', 'B_swap_fvt', 'E_progressive']:
                for batch in [2, 1]:
                    result = run(['trial', '--condition', condition, '--precision', precision,
                                  '--micro-batch', str(batch)])
                    result.update(condition=condition, precision=precision, micro_batch=batch)
                    report['trials'].append(result)
                    args.out.write_text(json.dumps(report, indent=2)+'\n')
                    if result.get('status') == 'passed':
                        break
                    if 'out of memory' not in result.get('error', '').lower():
                        break
        report['status'] = 'assessed'
        report['text_only_import_override'] = 'Disabled optional cv2/torchvision discovery in worker processes only'
        report['full_training_launched'] = False
    args.out.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
