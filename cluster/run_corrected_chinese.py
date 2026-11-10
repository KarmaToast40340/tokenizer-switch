"""Fresh corrected Chinese 50M A/B/E; immutable pilot is read only.

The gate uses exact training CLIs on validation-only surrogates and deletes only
its own temporary checkpoints. Full runs require that gate and measured quotas.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from corrected_chinese import CASES,ACTUAL_TOKENS,checkpoint_sizes,runtime_plan,storage_plan,train_args,usage,write_json


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def command(script,*args):
    subprocess.run([sys.executable,'-u','cluster/text_only.py',script,*map(str,args)],check=True)


def gate(assets,out,commit):
    sizes=checkpoint_sizes(assets)
    used,limit,method=usage()
    # Temporary gate saves are one at a time, and removed before the next case.
    needed=max(sizes)+(384<<20)
    if limit-used<needed:raise RuntimeError('insufficient space for diagnostic checkpoint/save reserve')
    report=dict(status='running',scope='Corrected train-mode GPU/memory/runtime gate; not model quality',
        job_id=os.environ['SLURM_JOB_ID'],commit=commit,quota=dict(used_bytes=used,hard_limit_bytes=limit,
        required_free_bytes=needed,method=method),checkpoint_directory_bytes=sizes,trials=[])
    write_json(out,report)
    try:
        subprocess.run([sys.executable,'scripts/verify_assets.py','--root',str(assets),'--layout','destination','--profiles','chinese'],check=True)
        scratch=out.parent/'diagnostics';scratch.mkdir(exist_ok=True)
        first=next((assets/'data/raw/wiki_zh/validation.jsonl').open())
        for case,name in CASES:
            with tempfile.TemporaryDirectory(prefix='corrected-gate-',dir=scratch) as d:
                tmp=Path(d).resolve();raw=tmp/'raw/wiki_zh';raw.mkdir(parents=True)
                for split in ['validation','test']:(raw/f'{split}.jsonl').write_text(first)
                trace=tmp/'trace.json'
                command('scripts/trace_training_mode.py','--script','scripts/train.py','--out',trace,
                    '--scope','Corrected Chinese GPU gate: validation-only; not a research checkpoint','--',
                    *train_args(case,name,131072,raw.parent,tmp/'runs',assets,commit),
                    'eval_docs=1','eval_every=1','log_every=1')
                evidence=json.loads(trace.read_text())
                assert 'GTX 1060' in evidence['gpu']
                tr=evidence['training_forwards'];ev=evidence['evaluation_forwards']
                assert len(tr)==128 and all(e['model_training'] and e['active_dropout_modules']==e['configured_dropout_modules']>0 for e in tr)
                assert ev and all(not e['model_training'] and not e['active_dropout_modules'] for e in ev)
                assert evidence['gpu_peak_allocated_bytes']>0
                assert any(g>0 and math.isfinite(g) for g in evidence['backward_gradient_norms'])
                run=tmp/'runs'/f'{name}_0M'
                rows=[json.loads(s) for s in (run/'log.jsonl').read_text().splitlines()]
                evaluations=[r for r in rows if any(k in r for k in ['eval','final_eval','final_eval_pruned'])]
                assert all(r['model_training_after_eval'] for r in evaluations)
                final=next(r for r in rows if 'final_eval' in r)
                assert final['training_mode_checks']==2
                assert all(math.isfinite(v) for v in final['final_eval'].values())
                trial=dict(condition=case,status='passed',optimizer_steps=2,
                    all_training_forwards_train_mode=True,all_evaluations_restored_train_mode=True,
                    training_forwards=len(tr),nonzero_backward_calls=sum(g>0 for g in evidence['backward_gradient_norms']),
                    configured_dropout=.1,trainer_wall_s=final['wall_s'],cli_wall_s=evidence['cli_wall_seconds'],
                    peak_allocated_bytes=evidence['gpu_peak_allocated_bytes'],peak_reserved_bytes=evidence['gpu_peak_reserved_bytes'],
                    checkpoint_bytes=sum(p.stat().st_size for p in (run/'model').rglob('*') if p.is_file()),
                    gpu=evidence['gpu'],torch=evidence['torch'])
                report['trials'].append(trial);write_json(out,report)
        report.update(status='passed',runtime=runtime_plan(report['trials']))
        write_json(out,report)
    except Exception as e:
        report.update(status='failed',error=f'{type(e).__name__}: {e}');write_json(out,report);raise


def train(assets,out,commit,gate_file):
    check=json.loads(gate_file.read_text())
    if check['status']!='passed' or check['commit']!=commit:raise RuntimeError('matching corrected GPU gate required')
    plan=runtime_plan(check['trials'])
    sizes=[max(a,b['checkpoint_bytes']) for a,b in zip(checkpoint_sizes(assets),check['trials'])]
    used,limit,method=usage();reserve=storage_plan(used,limit,sizes)
    # All new output lives here; never under a pilot run directory.
    runs=assets/'runs/chinese_corrected_dropout01_20261110'
    runs.mkdir(exist_ok=False)
    report=dict(status='running',stage='preflight',job_id=os.environ['SLURM_JOB_ID'],commit=commit,
        training_mode='train',configured_dropout=.1,effective_dropout='enabled',protocol='corrected',
        runs_root=str(runs),gate_job_id=check['job_id'],runtime_plan=plan,
        quota=reserve|dict(method=method),runs=[])
    write_json(out,report)
    try:
        for index,(case,name) in enumerate(CASES):
            used,limit,method=usage()
            report['quota']=storage_plan(used,limit,sizes,3-index)|dict(method=method)
            report['stage']='training_'+case;write_json(out,report)
            subprocess.run(['git','diff','--quiet'],check=True)
            command('scripts/train.py',*train_args(case,name,50_000_000,assets/'data/raw',runs,assets,commit))
            run=runs/f'{name}_50M';weights=run/'model/model.safetensors'
            rows=[json.loads(s) for s in (run/'log.jsonl').read_text().splitlines()]
            final=next(r for r in rows if 'final_eval' in r)
            assert final['training_mode_checks']==762 and final['model_training_after_eval']
            assert all(r['model_training'] for r in rows if 'train_bpb' in r)
            assert all(r['model_training_after_eval'] for r in rows if 'eval' in r)
            report['runs'].append(dict(condition=case,run=str(run),status='saved',actual_tokens=ACTUAL_TOKENS,
                weights_sha256=sha(weights),weights_bytes=weights.stat().st_size))
            write_json(out,report)
        report['stage']='verification';write_json(out,report)
        command('cluster/verify_dropout_off_run.py','--protocol','corrected','--training-commit',commit,
            '--job-id',os.environ['SLURM_JOB_ID'],'--job-state',out,'--runs-root',runs,
            '--raw',assets/'data/raw','--tokenizers',assets/'artifacts/tokenizers',
            '--out',out.parent/f'corrected_verified_{os.environ["SLURM_JOB_ID"]}')
        report.update(status='completed',stage='completed');write_json(out,report)
    except Exception as e:
        report.update(status='failed',error=f'{type(e).__name__}: {e}');write_json(out,report);raise


def main():
    ap=argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--phase',choices=['gate','train'],required=True)
    ap.add_argument('--assets',type=Path,default=Path.home()/'tokenizer-switch')
    ap.add_argument('--gate-file',type=Path)
    args=ap.parse_args()
    commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    subprocess.run(['git','diff','--quiet'],check=True);subprocess.run(['git','diff','--cached','--quiet'],check=True)
    out=Path(f'results/corrected_{args.phase}_{os.environ["SLURM_JOB_ID"]}.json')
    if out.exists():raise FileExistsError(out)
    if args.phase=='gate':gate(args.assets,out,commit)
    else:
        if args.gate_file is None:ap.error('--gate-file required for training')
        train(args.assets,out,commit,args.gate_file)


if __name__=='__main__':main()
