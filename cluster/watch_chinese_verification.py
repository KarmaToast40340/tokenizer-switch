"""Submit one read-only checkpoint verification after 757238 leaves the queue.

No training, cancellation, environment changes or writes into the active release.
The exclusive persistent lock prevents duplicate launchers, including after errors.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time

TRAIN_COMMIT = '31dbce7f645018eb221d55f5b7756872396de9b5'
CONDITIONS = {'A_keep', 'B_swap_fvt', 'E_progressive'}


def ready(state, queue):
    """Require finalized weights and an empty account queue (MaxSubmitPU=1)."""
    if queue.strip():
        return False
    if state.get('commit') != TRAIN_COMMIT or str(state.get('job_id')) != '757238':
        raise ValueError('unexpected training snapshot/job')
    saved = {r['condition'] for r in state.get('runs', []) if r.get('status') == 'saved'}
    if saved != CONDITIONS:
        raise ValueError('training ended without all three saved checkpoints; no submission')
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--job-state', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--interval', type=int, default=60)
    ap.add_argument('--hours', type=int, default=24)
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    lock = args.out.with_suffix('.lock')
    with lock.open('x') as f:
        f.write(str(os.getpid())+'\n')
    report = dict(status='waiting', training_job_id='757238', training_commit=TRAIN_COMMIT,
                  verification_commit=subprocess.check_output(['git','rev-parse','HEAD'], cwd=root, text=True).strip(),
                  pid=os.getpid())
    def write():
        report['updated_at'] = datetime.now(timezone.utc).isoformat()
        tmp=args.out.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2)+'\n');tmp.replace(args.out)
    write()
    try:
        deadline=time.monotonic()+args.hours*3600
        while time.monotonic()<deadline:
            queue=subprocess.check_output(['squeue','-h','-u',os.environ['USER'],'-o','%i'],text=True)
            if queue.strip():
                time.sleep(args.interval)
                continue
            state=json.loads(args.job_state.read_text())
            if ready(state,queue):
                accounting=subprocess.check_output(['sacct','-n','-X','-j','757238','--parsable2',
                    '--format=JobIDRaw,State,ExitCode'],text=True).strip()
                report['training_accounting']=accounting
                subprocess.run(['git','diff','--quiet'],cwd=root,check=True)
                subprocess.run(['git','diff','--cached','--quiet'],cwd=root,check=True)
                (root/'logs').mkdir(exist_ok=True)
                # Submit exactly once. Any ambiguous/error outcome retains the lock.
                report['status']='submitting';write()
                job=subprocess.check_output(['sbatch','--parsable',f'--chdir={root}',
                    f'--output={root}/logs/%x_%j.out',str(root/'cluster/verify_chinese757238.sbatch')],text=True).strip()
                number=job.split(';')[0]
                if not number.isdecimal():
                    raise ValueError(f'unexpected sbatch response: {job!r}')
                report.update(status='submitted',verification_job_id=number,
                    result_dir=str(root/f'results/chinese757238_verified_{number}'))
                write();return
        raise TimeoutError('watch period elapsed; no verification submitted')
    except Exception as e:
        report.update(status='failed',error=f'{type(e).__name__}: {e}');write();raise


if __name__=='__main__':
    main()
