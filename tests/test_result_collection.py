"""Only small, explicitly scoped, hash-addressed result files may publish."""
import importlib.util
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('collect_verify',Path(__file__).resolve().parents[1]/'scripts/follow_chinese_verification.py')
collector=importlib.util.module_from_spec(spec);spec.loader.exec_module(collector)


def manifest(path='verification.json',size=10):
    return dict(provisional=True,files=[dict(path=path,bytes=size,sha256='a'*64)])


def test_accepts_small_provisional_verification():
    assert collector.checked_manifest(manifest())[0]['path']=='verification.json'


@pytest.mark.parametrize('bad',[
    manifest('../weights.json'),manifest('/absolute.json'),manifest('model.safetensors'),
    manifest(size=11*(1<<20)),dict(provisional=False,files=manifest()['files'])])
def test_rejects_unsafe_or_unlabelled_artifacts(bad):
    with pytest.raises(ValueError):
        collector.checked_manifest(bad)


def test_corrected_results_require_explicit_protocol():
    corrected=manifest();corrected['provisional']=False
    assert collector.checked_manifest(corrected,expected_provisional=False)
    with pytest.raises(ValueError):collector.checked_manifest(corrected)


def test_corrected_collection_rejects_pilot_scope():
    with pytest.raises(ValueError):
        collector.checked_manifest(manifest(),expected_provisional=False)
