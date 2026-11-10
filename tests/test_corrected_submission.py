"""Exercise the cluster-only submission chain without contacting Slurm."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize('failure',[None,'gate','quota'])
def test_submits_only_after_verified_pilot_and_measured_safe_gate(tmp_path,monkeypatch,failure):
    source=Path(__file__).resolve().parents[1]/'cluster'
    monkeypatch.syspath_prepend(str(source))
    spec=importlib.util.spec_from_file_location('corrected_watch_test',source/'watch_corrected_chinese.py')
    watch=importlib.util.module_from_spec(spec);spec.loader.exec_module(watch)
    monkeypatch.setattr(watch,'__file__',str(tmp_path/'cluster/watch_corrected_chinese.py'))
    result=tmp_path/'results';result.mkdir()
    pilot=result/'pilot';pilot.mkdir()
    (pilot/'manifest.json').write_text('{}')
    (pilot/'verification.json').write_text(json.dumps(dict(status='verified',provisional=True)))
    old=result/'pilot_watcher.json';old.write_text(json.dumps(dict(status='submitted',verification_job_id='700000',result_dir=str(pilot))))
    out=result/'follow.json'
    trials=[dict(condition=c,status='passed',all_training_forwards_train_mode=True,
        all_evaluations_restored_train_mode=True,optimizer_steps=2,trainer_wall_s=90.,checkpoint_bytes=500_000_000)
        for c in ['A_keep','B_swap_fvt','E_progressive']]
    (result/'corrected_gate_800001.json').write_text(json.dumps(dict(status='failed' if failure=='gate' else 'passed',commit='abc',trials=trials)))
    (result/'corrected_train_800002.json').write_text(json.dumps(dict(status='completed')))
    monkeypatch.setattr(watch,'checkpoint_sizes',lambda _: [500_000_000]*3)
    monkeypatch.setattr(watch,'usage',lambda: (int((9.0 if failure=='quota' else 7.52)*(1<<30)),10*(1<<30),'test'))
    monkeypatch.setattr(watch.time,'sleep',lambda _:None)
    submitted=[]
    def output(cmd,**kwargs):
        if cmd[0]=='git':return 'abc\n'
        if cmd[0]=='squeue':return ''
        assert cmd[0]=='sbatch'
        submitted.append(cmd)
        return f'{800000+len(submitted)}\n'
    monkeypatch.setattr(watch.subprocess,'check_output',output)
    monkeypatch.setattr(watch.subprocess,'run',lambda *a,**k:SimpleNamespace(returncode=0))
    monkeypatch.setenv('USER','test-user')
    monkeypatch.setattr(sys,'argv',['watch','--pilot-watcher',str(old),'--out',str(out),'--assets',str(tmp_path/'assets')])
    if failure:
        with pytest.raises(RuntimeError):watch.main()
    else:watch.main()
    state=json.loads(out.read_text())
    if failure=='quota':assert not submitted
    elif failure=='gate':assert len(submitted)==1
    else:
        assert len(submitted)==2 and state['status']=='completed'
        assert state['corrected_gate_job_id']=='800001' and state['corrected_training_job_id']=='800002'
        assert '--phase' in submitted[0] and 'gate' in submitted[0]
        assert 'train' in submitted[1] and any(s.startswith('--time=') for s in submitted[1])
    if failure:assert state['status']=='blocked' and state['corrected_training_job_id'] is None
