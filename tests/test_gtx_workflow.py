import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


def workflow():
    path = Path(__file__).resolve().parents[1]/'cluster/run_chinese_gtx50.py'
    spec = importlib.util.spec_from_file_location('gtx_workflow',path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_quota_reserves_checkpoint_save_and_logs(monkeypatch):
    mod=workflow()
    monkeypatch.setattr(mod.subprocess,'run',lambda *a,**k:SimpleNamespace(stdout='netapp3:/vol/home/course 6399600 10485760 10485760 0 0 0\n'))
    q=mod.quota(3)
    assert q['hard_limit_bytes']==10*(1<<30)
    assert q['required_free_bytes']==int(2.9*(1<<30))


def test_quota_refuses_before_writing_when_reserve_would_not_fit(monkeypatch):
    mod=workflow()
    monkeypatch.setattr(mod.subprocess,'run',lambda *a,**k:SimpleNamespace(stdout='netapp3:/vol/home/course 10000000 10485760 10485760 0 0 0\n'))
    with pytest.raises(RuntimeError,match='insufficient quota'):
        mod.quota(1)


def test_all_conditions_keep_same_batch_schedule_and_validation():
    mod=workflow()
    for config,name in mod.CASES:
        args=mod.train_args(config,name,50_000_000,'data/raw','runs','abcdef0')
        overrides=dict(s.split('=',1) for s in args[1:])
        assert int(overrides['micro_batch'])*int(overrides['grad_accum'])*1024==65536
        assert overrides['eval_split']=='validation' and overrides['eval_batch']=='1'
        assert overrides['new_name']=='wiki_zh70_gpt4_50k'
