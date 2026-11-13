"""Protect active training and avoid scoring incomplete checkpoints."""
import importlib.util
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('verify_watch',Path(__file__).resolve().parents[1]/'cluster/watch_chinese_verification.py')
watch=importlib.util.module_from_spec(spec);spec.loader.exec_module(watch)


def test_waits_for_every_account_job_and_all_saved_weights():
    state=dict(commit=watch.TRAIN_COMMIT,job_id='757238',runs=[])
    assert not watch.ready(state,'757238\n')
    assert not watch.ready(state,'999999\n')  # Do not compete with a lab job.
    with pytest.raises(ValueError,match='all three'):
        watch.ready(state,'')
    state['runs']=[dict(condition=c,status='saved') for c in watch.CONDITIONS]
    assert watch.ready(state,'')
    state['runs'][-1]['status']='saving'
    with pytest.raises(ValueError,match='all three'):
        watch.ready(state,'')


def test_refuses_another_training_snapshot():
    with pytest.raises(ValueError,match='snapshot/job'):
        watch.ready(dict(commit='other',job_id='757238'),'')
