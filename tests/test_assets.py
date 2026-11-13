import importlib.util
import json
from pathlib import Path
import struct


def verifier():
    path = Path(__file__).resolve().parents[1] / 'scripts/verify_assets.py'
    spec = importlib.util.spec_from_file_location('verify_assets', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fixture(tmp_path, different_docs=False, bad_offsets=False):
    mod, files = verifier(), []
    for name in ['gpt2', 'new']:
        docs = 1 if different_docs and name == 'new' else 2
        offsets = [0, 4] if docs == 1 else [0, 2, 4]
        if bad_offsets:
            offsets[-1] = 3
        blobs = {'bin': struct.pack('<4H', 0, 1, 2, 3),
                 'idx': struct.pack('<' + 'q'*len(offsets), *offsets),
                 'json': json.dumps(dict(n_docs=docs, n_tokens=4)).encode()}
        for ext, blob in blobs.items():
            rel = f'data/chinese/tokens/wiki_zh/{name}.{ext}'
            dest = f'data/tokens/wiki_zh/{name}.{ext}'
            path = tmp_path / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(blob)
            files.append(dict(profile='chinese', source=rel, destination=dest,
                              bytes=len(blob), sha256=mod.sha256(path)))
    return mod, dict(files=files)


def test_verified_source_and_destination(tmp_path):
    mod, manifest = fixture(tmp_path)
    assert mod.verify(manifest, tmp_path, budget=4)['status'] == 'verified'
    for item in manifest['files']:
        dest = tmp_path / item['destination']
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((tmp_path / item['source']).read_bytes())
    assert mod.verify(manifest, tmp_path, layout='destination', budget=4)['status'] == 'verified'


def test_detect_same_size_corruption_and_short_budget(tmp_path):
    mod, manifest = fixture(tmp_path)
    (tmp_path / manifest['files'][0]['source']).write_bytes(b'xxxxxxxx')
    errors = mod.verify(manifest, tmp_path, budget=5)['errors']
    assert any('checksum' in e for e in errors)
    assert sum('tokens < 5' in e for e in errors) == 2


def test_offsets_must_match_metadata_even_with_valid_hashes(tmp_path):
    mod, manifest = fixture(tmp_path, bad_offsets=True)
    assert any('offsets' in e for e in mod.verify(manifest, tmp_path, budget=4)['errors'])


def test_encoding_document_counts_must_agree(tmp_path):
    mod, manifest = fixture(tmp_path, different_docs=True)
    assert any('different document counts' in e for e in mod.verify(manifest, tmp_path, budget=4)['errors'])


def test_unknown_profile_fails(tmp_path):
    mod, manifest = fixture(tmp_path)
    assert mod.verify(manifest, tmp_path, profiles=['missing'], budget=4)['status'] == 'failed'
