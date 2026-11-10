"""Corrected runs must fit measured resources, match settings, and preserve pilots."""
import importlib.util
from pathlib import Path
import sys

import pytest
import torch
from toksw.modes import require_mode
from toksw.utils import load_config

root=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('corrected_policy',root/'cluster/corrected_chinese.py')
policy=importlib.util.module_from_spec(spec);spec.loader.exec_module(policy)


def test_guards_catch_root_child_and_dropout_configuration_errors():
    model=torch.nn.Sequential(torch.nn.Dropout(.1))
    require_mode(model,True,.1)
    model.eval()
    with pytest.raises(RuntimeError,match='mode mismatch'):require_mode(model,True)
    model.train();model[0].eval()
    with pytest.raises(RuntimeError,match='mode mismatch'):require_mode(model,True)
    model.train()
    with pytest.raises(RuntimeError,match='configured dropout'):require_mode(model,True,.2)


def test_trainer_rejects_broken_evaluator_restoration(monkeypatch):
    spec=importlib.util.spec_from_file_location('corrected_trainer',root/'scripts/train.py')
    train=importlib.util.module_from_spec(spec);spec.loader.exec_module(train)
    model=torch.nn.Sequential(torch.nn.Dropout(.1)).train()
    from types import SimpleNamespace
    setup=SimpleNamespace(model=model,old_map=[0],new_map=None)
    def broken(*args):
        model.eval();return dict(bpb=1.)
    monkeypatch.setattr(train,'bits_per_byte',broken)
    with pytest.raises(RuntimeError,match='mode mismatch'):
        train.evaluate(setup,None,None,['text'],dict(block_size=32,eval_batch=1),'cpu')


def test_matched_fresh_settings_and_separate_output_names(tmp_path):
    settings=[]
    for case,name in policy.CASES:
        args=policy.train_args(case,name,50_000_000,tmp_path/'raw',tmp_path/'corrected',tmp_path/'assets','abc')
        cfg=load_config(root/args[0],args[1:]);settings.append(cfg)
        assert 'dropout01' in cfg['run_name']
        assert cfg['out']==str(tmp_path/'corrected')
        assert cfg['expected_dropout']==.1 and cfg['eval_split']=='validation'
        assert cfg['micro_batch']*cfg['grad_accum']*cfg['block_size']==65536
        assert cfg['token_budget']//65536*65536==policy.ACTUAL_TOKENS
    common=['base_model','seed','data_seed','lr','betas','weight_decay','warmup_frac','min_lr_ratio',
            'grad_clip','block_size','micro_batch','grad_accum','token_budget','expected_dropout','eval_every','eval_docs','eval_batch']
    for key in common:assert settings[0][key]==settings[1][key]==settings[2][key]


def test_space_reserve_includes_all_new_checkpoints_largest_save_and_headroom():
    sizes=[500_000_000]*3
    plan=policy.storage_plan(int(7.52*(1<<30)),10*(1<<30),sizes)
    assert plan['fits'] and plan['remaining_final_checkpoint_bytes']>=sum(sizes)
    assert plan['save_workspace_bytes']>=max(sizes)
    assert plan['logs_and_headroom_bytes']==384<<20
    with pytest.raises(RuntimeError,match='exceeds quota'):
        policy.storage_plan(9*(1<<30),10*(1<<30),sizes)
    assert policy.storage_plan(9*(1<<30),10*(1<<30),sizes,0)['fits']


def trials():
    return [dict(condition=c,status='passed',all_training_forwards_train_mode=True,
        all_evaluations_restored_train_mode=True,optimizer_steps=2,trainer_wall_s=90.) for c,_ in policy.CASES]


def test_runtime_requires_all_corrected_cases_and_includes_margin():
    result=policy.runtime_plan(trials())
    assert result['requested_hours']>sum(c['projected_training_hours'] for c in result['cases'])
    with pytest.raises(ValueError):policy.runtime_plan(trials()[:2])
    bad=trials();bad[0]['all_evaluations_restored_train_mode']=False
    with pytest.raises(ValueError):policy.runtime_plan(bad)
    with pytest.raises(RuntimeError,match='QoS'):policy.runtime_plan(trials(),max_hours=1)
