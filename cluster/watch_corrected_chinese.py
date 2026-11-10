"""Cluster-only follow-up: verified pilot -> measured GPU gate -> fresh A/B/E.

No laptop, SSH session, GitHub credentials, cancellation or deletion is needed.
An exclusive persistent lock prevents duplicate submission. Failures stop safely.
"""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import time

from corrected_chinese import checkpoint_sizes,runtime_plan,storage_plan,usage,write_json


def main():
    ap=argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--pilot-watcher',type=Path,required=True)
    ap.add_argument('--assets',type=Path,default=Path.home()/'tokenizer-switch')
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--interval',type=int,default=60)
    ap.add_argument('--hours',type=int,default=120)
    args=ap.parse_args()
    root=Path(__file__).resolve().parents[1]
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.with_suffix('.lock').open('x') as f:f.write(str(os.getpid())+'\n')
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    state=dict(status='waiting_pilot_verification',pid=os.getpid(),commit=commit,pilot_job_id='757238',
        corrected_gate_job_id=None,corrected_training_job_id=None,
        survives_laptop_disconnect=True,github_publication='Local/manual; no credentials on cluster')
    def write():
        state['updated_at']=datetime.now(timezone.utc).isoformat();write_json(args.out,state)
    def submit(name,phase,hours=None,gate=None):
        subprocess.run(['git','diff','--quiet'],cwd=root,check=True)
        subprocess.run(['git','diff','--cached','--quiet'],cwd=root,check=True)
        cmd=['sbatch','--parsable',f'--chdir={root}',f'--output={root}/logs/%x_%j.out',f'--job-name={name}']
        if hours is not None:cmd.append(f'--time={hours}:00:00')
        cmd += [str(root/'cluster/corrected_chinese.sbatch'),'--phase',phase,'--assets',str(args.assets)]
        if gate is not None:cmd += ['--gate-file',str(gate)]
        state['status']='submitting_'+phase;write()
        answer=subprocess.check_output(cmd,cwd=root,text=True).strip();job=answer.split(';')[0]
        if not job.isdecimal():raise ValueError(f'unexpected sbatch output: {answer}')
        return job
    write()
    deadline=time.monotonic()+args.hours*3600
    try:
        while time.monotonic()<deadline:
            queue=subprocess.check_output(['squeue','-h','-u',os.environ['USER'],'-o','%i'],text=True)
            if queue.strip():
                write();time.sleep(args.interval);continue
            if state['status']=='waiting_pilot_verification':
                pilot=json.loads(args.pilot_watcher.read_text())
                if pilot['status']=='failed':raise RuntimeError('pilot watcher failed; inspect before any retry')
                if pilot['status']!='submitted':
                    time.sleep(args.interval);continue
                state['pilot_verification_job_id']=pilot['verification_job_id']
                manifest=Path(pilot['result_dir'])/'manifest.json'
                report=Path(pilot['result_dir'])/'verification.json'
                if not manifest.is_file() or not report.is_file():
                    raise RuntimeError('pilot verification job ended without complete results')
                verification=json.loads(report.read_text())
                if verification['status']!='verified' or verification['provisional'] is not True:
                    raise RuntimeError('pilot checkpoints not verified/labelled')
                sizes=checkpoint_sizes(args.assets);used,limit,method=usage()
                state['storage']=storage_plan(used,limit,sizes)|dict(method=method)
                target=args.assets/'runs/chinese_corrected_dropout01_20261110'
                if target.exists():raise FileExistsError('preserving existing corrected output root')
                state['corrected_gate_job_id']=submit('zh50_corrected_gate','gate')
                state['status']='waiting_corrected_gate';write()
            elif state['status']=='waiting_corrected_gate':
                gate=root/f'results/corrected_gate_{state["corrected_gate_job_id"]}.json'
                check=json.loads(gate.read_text())
                if check['status']!='passed' or check['commit']!=commit:
                    raise RuntimeError('corrected GPU gate failed or revision mismatch; no training submitted')
                plan=runtime_plan(check['trials']);state['runtime']=plan
                # Use the larger of pilot sizes and the gate's measured saves.
                sizes=[max(a,b['checkpoint_bytes']) for a,b in zip(checkpoint_sizes(args.assets),check['trials'])]
                used,limit,method=usage();state['storage']=storage_plan(used,limit,sizes)|dict(method=method)
                state['corrected_training_job_id']=submit('zh50_corrected_ABE','train',plan['requested_hours'],gate)
                state['status']='waiting_corrected_training';write()
            elif state['status']=='waiting_corrected_training':
                result=root/f'results/corrected_train_{state["corrected_training_job_id"]}.json'
                check=json.loads(result.read_text())
                if check['status']!='completed':raise RuntimeError('corrected allocation ended before verified completion')
                state.update(status='completed',corrected_controller=str(result));write();return
            else:
                raise RuntimeError('ambiguous prior submission state; inspect queue before retry')
            time.sleep(args.interval)
        raise TimeoutError('cluster follow-up deadline elapsed')
    except Exception as e:
        state.update(status='blocked',error=f'{type(e).__name__}: {e}');write();raise


if __name__=='__main__':main()
