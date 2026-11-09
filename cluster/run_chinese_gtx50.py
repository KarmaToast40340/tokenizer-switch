"""Sequential assigned Chinese 50M A/B/E, gated by actual trainer smoke.

No installation, resume, overwrite or coursework cleanup. One Slurm allocation.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

CASES = [('A_keep', 'A_keep_zh'), ('B_swap_fvt', 'B_swap_fvt_zh'), ('E_progressive', 'E_progressive_zh')]
BASE = 'data/hf-cache/hub/models--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e'


def quota(remaining):
    try:
        result = subprocess.run(['quota','-v','-w','-f','/net/course'], capture_output=True, text=True, timeout=30, check=True)
        rows = [line.split() for line in result.stdout.splitlines() if re.match(r'^\S+\s+\d+\*?\s+\d+\s+\d+', line)]
        fields = rows[-1]
        used, limit = int(fields[1].rstrip('*')) * 1024, int(fields[3]) * 1024
        if not limit:
            raise ValueError('no positive hard quota')
        method = 'quota /net/course'
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        # Conservative fallback when quota tools are absent on a compute node.
        used = int(subprocess.check_output(['du','-sk',str(Path.home())], text=True).split()[0]) * 1024
        limit, method = 10 * (1 << 30), 'du against confirmed 10 GiB hard limit'
    needed = int((remaining * .6 + .6 + .5) * (1 << 30))
    if limit - used < needed:
        raise RuntimeError(f'insufficient quota: {limit-used} free bytes; {needed} reserved bytes required')
    return dict(used_bytes=used, hard_limit_bytes=limit, required_free_bytes=needed, method=method)


def command(script, *args):
    subprocess.run([sys.executable, '-u', 'cluster/text_only.py', script, *map(str,args)], check=True)


def train_args(config, name, budget, raw, out, commit):
    return [f'configs/train/{config}.yaml', f'run_name={name}', 'domain=wiki_zh', f'base_model={BASE}',
            'old_tokenizer=artifacts/tokenizers/gpt2', 'new_tokenizer=artifacts/tokenizers/wiki_zh70_gpt4_50k',
            'new_name=wiki_zh70_gpt4_50k', f'token_budget={budget}', 'micro_batch=1', 'grad_accum=64',
            'eval_split=validation', 'eval_batch=1', f'raw_dir={raw}', f'out={out}',
            f'account={os.environ.get("USER", "chloe")}', f"commit='{commit}'", 'gpu_profile=gtx1060_6gb']


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--preflight-only', action='store_true')
    args = ap.parse_args()
    release = subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    subprocess.run(['git','diff','--quiet'],check=True)
    subprocess.run(['git','diff','--cached','--quiet'],check=True)
    for _, name in CASES:
        if (Path('runs')/f'{name}_50M').exists():
            raise FileExistsError(f'preserving existing run: {name}_50M')
    if not Path(BASE,'model.safetensors').is_file():
        raise FileNotFoundError('pinned GPT-2 weights missing')
    storage = quota(3)
    if args.preflight_only:
        print(json.dumps(dict(commit=release, quota=storage, cases=CASES, actual_tokens_per_run=49_938_432),indent=2))
        return
    job = os.environ['SLURM_JOB_ID']
    out = Path(f'results/chinese_gtx50_{job}.json')
    report = dict(status='running', stage='preflight', job_id=job, commit=release,
                  started_at=datetime.now(timezone.utc).isoformat(), quota=storage, smoke=[], runs=[])
    def write(stage):
        report['stage'] = stage
        report['updated_at'] = datetime.now(timezone.utc).isoformat()
        out.write_text(json.dumps(report,indent=2)+'\n')
    def unchanged():
        if subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip() != release:
            raise RuntimeError('checkout changed during allocation; stop before running mixed revisions')
    if out.exists():
        raise FileExistsError(out)
    write('preflight')
    try:
        subprocess.run([sys.executable,'scripts/verify_assets.py','--layout','destination','--profiles','chinese'],check=True)
        probe = subprocess.check_output([sys.executable,'cluster/assess_gtx1060.py','--worker','probe'],text=True)
        gpu = json.loads(next(line.split('=',1)[1] for line in probe.splitlines() if line.startswith('RESULT_JSON=')))
        if not gpu.get('cuda_available') or 'GTX 1060' not in gpu.get('gpu','') or gpu.get('bf16_forward_backward')!='passed':
            raise RuntimeError(f'GPU gate failed: {gpu}')
        report['gpu'] = gpu
        # Exact unmodified trainer, two full accumulation steps, validation-only
        # surrogate test text. Temporary diagnostic weights are removed per case.
        parent = Path('runs/diagnostics'); parent.mkdir(parents=True,exist_ok=True)
        first = next(iter(Path('data/raw/wiki_zh/validation.jsonl').open()))
        for config, name in CASES:
            write('trainer_smoke_' + config)
            with tempfile.TemporaryDirectory(prefix=f'cli_gate_{job}_',dir=parent) as d:
                root = Path(d).resolve()
                raw = root/'raw/wiki_zh'; raw.mkdir(parents=True)
                for split in ['validation','test']:
                    (raw/f'{split}.jsonl').write_text(first)
                command('scripts/train.py',*train_args(config,name+'_smoke',131072,raw.parent,root/'runs',release),
                        'eval_docs=1','eval_every=100','log_every=1')
                run = root/'runs'/f'{name}_smoke_0M'
                rows=[json.loads(line) for line in (run/'log.jsonl').read_text().splitlines()]
                final=next(r['final_eval'] for r in rows if 'final_eval' in r)
                assert all(math.isfinite(value) for value in final.values())
                if config=='E_progressive':
                    assert any('final_eval_pruned' in r for r in rows)
                report['smoke'].append(dict(condition=config,status='passed',optimizer_steps=2,scope='validation-only diagnostic; checkpoint removed'))
        for index,(config,name) in enumerate(CASES):
            unchanged()
            report['quota_before_run'] = quota(3-index)
            write('training_' + config)
            command('scripts/train.py',*train_args(config,name,50_000_000,'data/raw','runs',release))
            run = Path('runs')/f'{name}_50M'
            weights = run/'model/model.safetensors'
            if not weights.is_file():
                raise FileNotFoundError(weights)
            report['runs'].append(dict(condition=config,run=str(run),status='saved',actual_tokens=49_938_432,
                                       weights_bytes=weights.stat().st_size,weights_sha256=sha256(weights)))
            write('saved_' + config)
        result_dir = Path(f'results/chinese_models_gtx50_{job}'); result_dir.mkdir()
        models=[Path('runs')/f'{name}_50M/model' for _,name in CASES]
        for model in models:
            write('evaluate_' + model.parent.name)
            command('scripts/eval_bpb.py',model,'--domain','wiki_zh','--raw','data/raw','--batch-size','1',
                    '--out',result_dir/(model.parent.name+'_bpb.json'))
        write('generation_speed')
        command('scripts/gen_speed.py',*models,'--domain','wiki_zh','--raw','data/raw','--prompt-chars','200',
                '--new-tokens','128','--out',result_dir/'speed.csv')
        subprocess.run([sys.executable,'scripts/collect_results.py','runs','--out',str(result_dir/'runs.csv')],check=True)
        logs=[str(model.parent/'log.jsonl') for model in models]
        subprocess.run([sys.executable,'scripts/plot_results.py','--logs',*logs,'--out',str(result_dir/'training')],check=True)
        report.update(status='completed',result_dir=str(result_dir))
        write('completed')
    except Exception as e:
        report.update(status='failed',error=f'{type(e).__name__}: {e}')
        write(report['stage'])
        raise


if __name__=='__main__':
    main()
