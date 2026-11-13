"""Shared, measured resource policy for the corrected Chinese 50M sequence."""
import json
import math
import os
from pathlib import Path
import re
import subprocess

CASES=[('A_keep','A_keep_zh_dropout01'),('B_swap_fvt','B_swap_fvt_zh_dropout01'),
       ('E_progressive','E_progressive_zh_dropout01')]
ACTUAL_TOKENS=49_938_432
PILOT_COMMIT='31dbce7f645018eb221d55f5b7756872396de9b5'


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def usage():
    try:
        result=subprocess.run(['quota','-v','-w','-f','/net/course'],capture_output=True,text=True,check=True,timeout=30)
        rows=[s.split() for s in result.stdout.splitlines() if re.match(r'^\S+\s+\d+\*?\s+\d+\s+\d+',s)]
        r=rows[-1];used,limit=int(r[1].rstrip('*'))*1024,int(r[3])*1024
        if limit<=0:raise ValueError('missing hard quota')
        return used,limit,'quota /net/course'
    except (OSError,subprocess.SubprocessError,ValueError,IndexError):
        used=int(subprocess.check_output(['du','-sk',str(Path.home())],text=True).split()[0])*1024
        return used,10*(1<<30),'du against confirmed 10 GiB course-home quota'


def storage_plan(used,limit,checkpoint_sizes,remaining=3):
    # Measured final pilot directory sizes, plus 8 MiB/condition for metadata.
    sizes=[int(s)+(8<<20) for s in checkpoint_sizes]
    if len(sizes)!=3 or remaining not in range(4) or min(sizes)<=8<<20:
        raise ValueError('three measured checkpoint sizes required')
    new=sum(sizes[3-remaining:])
    save_workspace=max(sizes) if remaining else 0
    logs_headroom=384<<20
    required=new+save_workspace+logs_headroom
    report=dict(used_bytes=used,hard_limit_bytes=limit,free_bytes=limit-used,
        remaining_final_checkpoint_bytes=new,save_workspace_bytes=save_workspace,
        logs_and_headroom_bytes=logs_headroom,required_free_bytes=required,
        projected_reserved_peak_bytes=used+required,fits=used+required<=limit)
    if not report['fits']:
        raise RuntimeError('corrected checkpoint/save reserve exceeds quota: '+json.dumps(report))
    return report


def checkpoint_sizes(assets):
    state=json.loads((assets.parent/'tokenizer-switch-releases/zh50-31dbce7f6450/results/chinese_gtx50_757238.json').read_text())
    if state['commit']!=PILOT_COMMIT or str(state['job_id'])!='757238':
        raise ValueError('unexpected pilot state')
    records={r['condition']:r for r in state['runs'] if r['status']=='saved'}
    if set(records)!={c for c,_ in CASES}:raise ValueError('pilot must finish and save all three checkpoints')
    sizes=[]
    for case,_ in CASES:
        model=assets/records[case]['run']/'model'
        sizes.append(sum(p.stat().st_size for p in model.rglob('*') if p.is_file()))
    return sizes


def train_args(case,name,budget,raw,out,assets,commit):
    return [f'configs/train/{case}.yaml',f'run_name={name}','domain=wiki_zh',
        f'base_model={assets}/data/hf-cache/hub/models--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e',
        f'old_tokenizer={assets}/artifacts/tokenizers/gpt2',f'new_tokenizer={assets}/artifacts/tokenizers/wiki_zh70_gpt4_50k',
        'old_name=gpt2','new_name=wiki_zh70_gpt4_50k',f'tokens_dir={assets}/data/tokens',
        f'raw_dir={raw}',f'out={out}',f'token_budget={budget}',
        'block_size=1024','micro_batch=1','grad_accum=64','seed=0','data_seed=1234',
        'lr=0.0003','warmup_frac=0.02','min_lr_ratio=0.1','weight_decay=0.1',
        'betas=[0.9,0.95]','grad_clip=1.0','compile=false','rho=0.5',
        'eval_split=validation','eval_docs=200','eval_batch=1','eval_every=100','log_every=10',
        'expected_dropout=0.1','training_protocol=corrected_dropout01',
        f'account={os.environ.get("USER","chloe")}',f"commit='{commit}'",'gpu_profile=gtx1060_6gb']


def runtime_plan(trials,max_hours=96):
    if {r['condition'] for r in trials}!={c for c,_ in CASES} or len(trials)!=3:
        raise ValueError('three corrected GPU measurements required')
    cases=[]
    for r in trials:
        if (r['status']!='passed' or not r['all_training_forwards_train_mode'] or
            not r['all_evaluations_restored_train_mode'] or r['optimizer_steps']!=2):
            raise ValueError('failed/incomplete corrected training-mode gate')
        seconds=r['trainer_wall_s']/2
        if not math.isfinite(seconds) or seconds<=0:raise ValueError('invalid measured runtime')
        cases.append(dict(condition=r['condition'],measured_seconds_per_step=seconds,
            projected_training_hours=seconds*762/3600,
            reserved_hours=(seconds*762*1.3+3600)/3600))
    hours=math.ceil(sum(r['reserved_hours'] for r in cases))+1
    if hours>max_hours:raise RuntimeError(f'corrected runtime reserve {hours}h exceeds {max_hours}h QoS limit')
    return dict(cases=cases,requested_hours=hours,
        caveat='Two exact optimizer steps per condition; projection, not completed runtime. Includes 30% margin, 1h/case evaluation allowance and 1h allocation margin.')
