"""Use tiny synthetic assets to verify the manifest-only transfer boundary."""
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import tarfile

import pytest


def setup(tmp_path,monkeypatch):
    root=Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root/'scripts'))
    spec=importlib.util.spec_from_file_location('packager',root/'scripts/build_code50_package.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    repo=tmp_path/'repo';repo.mkdir();(repo/'scripts').mkdir()
    (repo/'scripts/verify_assets.py').write_bytes((root/'scripts/verify_assets.py').read_bytes())
    files=[]
    def add(source,dest,blob,profile='code50'):
        path=repo/source;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(blob)
        files.append(dict(profile=profile,source=source,destination=dest,bytes=len(blob),sha256=hashlib.sha256(blob).hexdigest()))
    for tok in ['gpt2','code70_gpt4_32k']:
        for ext,blob in [('bin',struct.pack('<4H',1,2,3,4)),('idx',struct.pack('<3q',0,2,4)),
                         ('json',json.dumps(dict(n_docs=2,n_tokens=4)).encode())]:
            add(f'data/code50/tokens/code/{tok}.{ext}',f'data/tokens/code/{tok}.{ext}',blob)
        for filename in ['meta.json','tokenizer.json']:
            add(f'artifacts/milestone/tokenizers/{tok}/{filename}',f'artifacts/tokenizers/{tok}/{filename}',b'{}')
    for filename,blob in [('validation.jsonl',b'{"text":"validation"}\n'),('test.jsonl',b'{"text":"test"}\n'),
                          ('stats.json',b'{"train_bytes":20,"n_train":2}')]:
        add('data/code50/raw/code/'+filename,'data/raw/code/'+filename,blob)
    # A Chinese row must be ignored, not read or shipped.
    files.append(dict(profile='chinese',source='not-present',destination='data/raw/wiki_zh/test.jsonl',bytes=1,sha256='0'*64))
    manifest=repo/'manifest.json';manifest.write_text(json.dumps(dict(files=files)))
    return mod,repo,manifest,files


def test_reproducible_code_only_archive_and_checksums(tmp_path,monkeypatch):
    mod,repo,manifest,files=setup(tmp_path,monkeypatch)
    a=mod.build(repo,manifest,tmp_path/'out1','code50-djam','Verified {{VERIFIED_CODE_COMMIT}}','a'*40,budget=4)
    b=mod.build(repo,manifest,tmp_path/'out2','code50-djam','Verified {{VERIFIED_CODE_COMMIT}}','a'*40,budget=4)
    assert a['sha256']==b['sha256'] and a['asset_count']==13
    with tarfile.open(a['archive'],'r:gz') as tar:
        names=tar.getnames()
        assert len(names)==18 and all(m.isfile() for m in tar.getmembers())
        assert not any('wiki_zh' in n or 'train.jsonl' in n or '.venv' in n for n in names)
        assert tar.extractfile('code50-djam/README.md').read()==b'Verified '+b'a'*40
        sums=tar.extractfile('code50-djam/SHA256SUMS').read().decode().splitlines()
        for line in sums:
            digest,path=line.split('  ',1)
            assert hashlib.sha256(tar.extractfile('code50-djam/'+path).read()).hexdigest()==digest
    with pytest.raises(FileExistsError):
        mod.build(repo,manifest,tmp_path/'out1','code50-djam','README','a'*40,budget=4)


def test_corrupt_source_refused_before_packaging(tmp_path,monkeypatch):
    mod,repo,manifest,files=setup(tmp_path,monkeypatch)
    (repo/files[0]['source']).write_bytes(b'corrupt!')
    with pytest.raises(ValueError,match='verification failed'):
        mod.build(repo,manifest,tmp_path/'out','code50','README','a'*40,budget=4)


def test_incomplete_or_cross_domain_manifest_refused(tmp_path,monkeypatch):
    mod,repo,manifest,files=setup(tmp_path,monkeypatch)
    with pytest.raises(ValueError,match='incomplete'):
        mod.selected_manifest(dict(files=files[1:]))
    files[0]['destination']='data/raw/wiki_zh/test.jsonl'
    with pytest.raises(ValueError,match='unexpected'):
        mod.selected_manifest(dict(files=files))


def test_output_inside_git_checkout_refused(tmp_path,monkeypatch):
    mod,repo,manifest,files=setup(tmp_path,monkeypatch)
    with pytest.raises(ValueError,match='outside'):
        mod.build(repo,manifest,repo/'deliverables','code50','README','a'*40,budget=4)
