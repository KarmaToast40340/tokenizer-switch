"""Collect verified small results and optionally publish to chloe/data-eval.

Runs locally, using the existing SSH master. Stops at the first authentication
failure or changed Git head; never retries passwords, copies credentials, pushes
main, merges a PR or downloads model weights. Durable output survives a stop.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import shlex
import shutil
import subprocess
import tarfile
import time

SSH=['ssh','-o','BatchMode=yes','-o','ControlPath=/tmp/tok-%C','-o','ConnectTimeout=10',
     'c115550802@hpclogin02.cs.nycu.edu.tw']
BRANCH='chloe/data-eval'


def checked_manifest(manifest, expected_provisional=True):
    if manifest.get('provisional') is not expected_provisional:
        raise ValueError('unexpected provisional flag for requested protocol')
    records=manifest['files']
    if not 1<=len(records)<=50 or sum(r['bytes'] for r in records)>30*(1<<20):
        raise ValueError('unexpected collection size')
    seen=set()
    for r in records:
        path=PurePosixPath(r['path'])
        if (path.is_absolute() or '..' in path.parts or str(path)!=r['path'] or
            path.suffix not in {'.json','.jsonl','.csv','.png','.pdf','.md'} or
            r['path'] in seen or not 0<=r['bytes']<10*(1<<20)):
            raise ValueError('unsafe/unexpected result path or size')
        if len(r['sha256'])!=64 or any(c not in '0123456789abcdef' for c in r['sha256']):
            raise ValueError('invalid SHA-256')
        seen.add(r['path'])
    if 'verification.json' not in seen:
        raise ValueError('missing verification report')
    return records


def output(cmd,cwd=None):
    return subprocess.check_output(cmd,cwd=cwd,text=True,stderr=subprocess.PIPE).strip()


def publish(repo,expected,download,watch,verification):
    # Refuse concurrent work or teammate updates. A normal push also rejects races.
    if output(['git','branch','--show-current'],repo)!=BRANCH:
        raise RuntimeError('local branch changed; collected only')
    if output(['git','rev-parse','HEAD'],repo)!=expected or output(['git','status','--porcelain'],repo):
        raise RuntimeError('local HEAD/worktree changed; collected only')
    remote=output(['git','ls-remote','origin',f'refs/heads/{BRANCH}'],repo).split()[0]
    if remote!=expected:
        raise RuntimeError('remote HEAD changed; collected only')
    pr=json.loads(output(['gh','pr','view','4','--json','body,state,headRefName'],repo))
    if pr['state']!='OPEN' or pr['headRefName']!=BRANCH:
        raise RuntimeError('PR state changed; collected only')
    dest=repo/'results/chinese757238_dropout_disabled'
    if dest.exists():
        raise FileExistsError('preserving existing published result directory')
    shutil.copytree(download,dest)
    rows=verification['cases']
    table='\n'.join(f"| {r['condition']} | {r['test_bpb']:.6f} | {r['characters_per_second']:.2f} |" for r in rows)
    section=('\n\n## Verified Chinese pilot results — job 757238\n\n'
        '**PROVISIONAL for the intended dropout-0.1 study:** all three checkpoints\n'
        'were trained in eval mode, with dropout disabled. These are valid inference\n'
        'measurements of those weights; reevaluation does not repair training.\n\n'
        f"Independent verification job **{watch['verification_job_id']}** checked all\n"
        'weight/tokenizer/raw hashes, 1,000-document test coverage, saved/logged-score\n'
        'agreement, unchanged weights and shared-prompt speed protocol.\n\n'
        '| Condition | Test bpb | Generation characters/s |\n|---|---:|---:|\n'+table+'\n\n'
        'Speed: GTX 1060, 20 shared 200-character prefixes, 128 greedy new tokens.\n'
        'Tokenizers are 20 MB fits, not the proposed 1 GB study. Corrected dropout\n'
        'training still requires fresh runs after a train-mode memory check.\n\n'
        '[Verified figures and provenance](../results/chinese757238_dropout_disabled/README.md)\n')
    progress=repo/'docs/chloe-progress.md'
    progress.write_text(progress.read_text()+section)
    readme=repo/'results/README.md'
    readme.write_text(readme.read_text()+'\nVerified inference measurements and explicitly provisional figures: '
        '[Chinese dropout-disabled pilot](chinese757238_dropout_disabled/README.md).\n')
    subprocess.run(['git','diff','--check'],cwd=repo,check=True)
    subprocess.run(['git','add','--',str(dest.relative_to(repo)),'docs/chloe-progress.md','results/README.md'],cwd=repo,check=True)
    subprocess.run(['git','commit','-m','Publish verified dropout-disabled Chinese 50M pilot measurements'],cwd=repo,check=True)
    subprocess.run(['git','push','origin',f'HEAD:refs/heads/{BRANCH}'],cwd=repo,check=True)
    head=output(['git','rev-parse','HEAD'],repo)
    if output(['git','ls-remote','origin',f'refs/heads/{BRANCH}'],repo).split()[0]!=head:
        raise RuntimeError('published HEAD verification failed')
    body=download.parent/'pr4-verified-body.md'
    # GitHub uses a repository-relative link, not a docs-relative link.
    body.write_text(pr['body']+section.replace('../results/','results/'))
    subprocess.run(['gh','pr','edit','4','--body-file',str(body)],cwd=repo,check=True)
    return head


def main():
    ap=argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--remote-root',required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--repo',type=Path,required=True)
    ap.add_argument('--expected-head',required=True)
    ap.add_argument('--publish',action='store_true')
    ap.add_argument('--interval',type=int,default=180)
    ap.add_argument('--hours',type=int,default=30)
    args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    status=dict(status='waiting',scope='PROVISIONAL: dropout-disabled Chinese 50M pilot',
                expected_head=args.expected_head)
    def write():
        status['updated_at']=datetime.now(timezone.utc).isoformat()
        (args.out/'collector.json').write_text(json.dumps(status,indent=2)+'\n')
    def read_remote(path):
        cmd='if test -f '+shlex.quote(path)+'; then cat '+shlex.quote(path)+'; else exit 3; fi'
        p=subprocess.run([*SSH,cmd],capture_output=True,text=True)
        if p.returncode==3:
            return None
        if p.returncode:
            raise RuntimeError(f'SSH stopped ({p.returncode}): {p.stderr.strip()}')
        return json.loads(p.stdout)
    write()
    try:
        deadline=time.monotonic()+args.hours*3600
        while time.monotonic()<deadline:
            watch=read_remote(args.remote_root+'/results/postwatch757238.json')
            if watch is None:
                raise RuntimeError('remote watcher state missing')
            status['watcher']=watch
            if watch['status']=='failed':
                raise RuntimeError('remote watcher failed: '+watch.get('error','unknown'))
            if watch['status']=='submitted':
                remote_dir=watch['result_dir']
                if not remote_dir.startswith(args.remote_root+'/results/chinese757238_verified_'):
                    raise ValueError('unexpected remote output directory')
                verification=read_remote(remote_dir+'/verification.json')
                if verification and verification['status']=='failed':
                    raise RuntimeError('verification failed: '+verification.get('error','unknown'))
                manifest=read_remote(remote_dir+'/manifest.json')
                if manifest:
                    records=checked_manifest(manifest)
                    archive=subprocess.check_output([*SSH,'tar -C '+shlex.quote(remote_dir)+' -cf - -- '+
                        ' '.join(shlex.quote(r['path']) for r in records)],stderr=subprocess.PIPE)
                    if len(archive)>35*(1<<20):
                        raise ValueError('oversized archive')
                    download=args.out/'verified';download.mkdir()
                    with tarfile.open(fileobj=io.BytesIO(archive),mode='r:') as tar:
                        members={m.name:m for m in tar.getmembers()}
                        for r in records:
                            m=members[r['path']]
                            if not m.isfile() or m.size!=r['bytes']:
                                raise ValueError('unexpected archive entry')
                            payload=tar.extractfile(m).read()
                            if hashlib.sha256(payload).hexdigest()!=r['sha256']:
                                raise ValueError('download hash mismatch')
                            dest=download/r['path'];dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(payload)
                    (download/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
                    verification=json.loads((download/'verification.json').read_text())
                    if verification['status']!='verified' or verification['provisional'] is not True:
                        raise ValueError('not a verified provisional comparison')
                    status.update(status='collected',verification_job_id=watch['verification_job_id'])
                    write()
                    if args.publish:
                        status['pushed_head']=publish(args.repo,args.expected_head,download,watch,verification)
                        status['status']='published'
                    write();return
            write();time.sleep(args.interval)
        raise TimeoutError('collector deadline elapsed')
    except Exception as e:
        status.update(status='stopped',error=f'{type(e).__name__}: {e}');write();raise


if __name__=='__main__':
    main()
