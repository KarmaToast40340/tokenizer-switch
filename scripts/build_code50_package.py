"""Build a code-only, manifest-whitelisted transfer archive outside Git.

No preparation, fitting, environment, checkpoint or Chinese data is included.
Payload names use the manifest's destination layout; source paths are resolved.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile

from verify_assets import sha256, verify


def selected_manifest(manifest):
    files=[f for f in manifest['files'] if f['profile']=='code50']
    if not files:raise ValueError('manifest has no code50 assets')
    seen=set()
    for f in files:
        for key in ['source','destination']:
            p=PurePosixPath(f[key])
            if p.is_absolute() or '..' in p.parts:raise ValueError('unsafe manifest path')
        dest=f['destination']
        allowed=('data/tokens/code/','data/raw/code/','artifacts/tokenizers/gpt2/','artifacts/tokenizers/code70_gpt4_32k/')
        if not dest.startswith(allowed) or dest in seen:raise ValueError('unexpected/duplicate code asset')
        if dest.startswith('data/raw/code/') and PurePosixPath(dest).name not in ['validation.jsonl','test.jsonl','stats.json']:
            raise ValueError('raw training data must not be shipped')
        seen.add(dest)
    expected={f'data/tokens/code/{tok}.{ext}' for tok in ['gpt2','code70_gpt4_32k'] for ext in ['bin','idx','json']}
    expected.update('data/raw/code/'+n for n in ['validation.jsonl','test.jsonl','stats.json'])
    expected.update(f'artifacts/tokenizers/{tok}/{n}' for tok in ['gpt2','code70_gpt4_32k'] for n in ['meta.json','tokenizer.json'])
    if seen!=expected:raise ValueError('incomplete code50 transfer manifest')
    return dict(status='verified',notes='Code50-only subset; exact source/destination rows retained.',files=files)


def build(repo,manifest_path,out_dir,name,readme,code_ref,budget=50_000_000):
    repo=Path(repo).resolve();out_dir=Path(out_dir).resolve()
    if out_dir==repo or repo in out_dir.parents:raise ValueError('large package must be outside the repository')
    if PurePosixPath(name).name!=name or name in ['', '.', '..']:raise ValueError('simple package name required')
    manifest_path=Path(manifest_path)
    original=json.loads(manifest_path.read_text());manifest=selected_manifest(original)
    verified=verify(manifest,repo,layout='source',profiles=['code50'],budget=budget)
    if verified['status']!='verified':raise ValueError('source verification failed: '+json.dumps(verified['errors']))
    out_dir.mkdir(parents=True,exist_ok=True)
    payload=out_dir/name;archive=out_dir/(name+'.tar.gz')
    if payload.exists() or archive.exists():raise FileExistsError('preserving existing package')
    payload.mkdir()
    for entry in manifest['files']:
        source=(repo/entry['source']).resolve(strict=True)
        dest=payload/entry['destination'];dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,dest)
        if dest.stat().st_size!=entry['bytes'] or sha256(dest)!=entry['sha256']:
            raise RuntimeError('copy checksum mismatch: '+entry['destination'])
    (payload/'assets-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (payload/'README.md').write_text(readme.replace('{{VERIFIED_CODE_COMMIT}}',code_ref))
    (payload/'tools').mkdir();shutil.copyfile(repo/'scripts/verify_assets.py',payload/'tools/verify_assets.py')
    stats=json.loads((payload/'data/raw/code/stats.json').read_text())
    meta=dict(profile='code50',verified_code_commit=code_ref,original_manifest_sha256=sha256(manifest_path),
        asset_count=len(manifest['files']),asset_bytes=sum(f['bytes'] for f in manifest['files']),
        requested_training_tokens=budget,tokenizer_fit_budget_bytes_per_variant=20_000_000,
        corpus_train_bytes=stats['train_bytes'],train_documents=stats['n_train'],
        encodings=verified['encodings'],scope='20 MB tokenizer pilots; separate model corpus; no environments, raw train, Chinese data or weights')
    (payload/'PACKAGE.json').write_text(json.dumps(meta,indent=2)+'\n')
    files=sorted(p for p in payload.rglob('*') if p.is_file())
    sums=''.join(f'{sha256(p)}  {p.relative_to(payload).as_posix()}\n' for p in files)
    (payload/'SHA256SUMS').write_text(sums)
    files.append(payload/'SHA256SUMS');files.sort()
    # Fixed metadata produces a portable deterministic archive with regular files only.
    with archive.open('xb') as raw:
        with gzip.GzipFile(fileobj=raw,mode='wb',filename='',mtime=0,compresslevel=6) as zipped:
            with tarfile.open(fileobj=zipped,mode='w|',format=tarfile.USTAR_FORMAT) as tar:
                for p in files:
                    info=tarfile.TarInfo(name+'/'+p.relative_to(payload).as_posix())
                    info.size=p.stat().st_size;info.mode=0o644;info.mtime=0;info.uid=info.gid=0
                    with p.open('rb') as f:tar.addfile(info,f)
    # Verify bytes inside the archive as well as the source and staging tree.
    expected={name+'/'+p.relative_to(payload).as_posix():sha256(p) for p in files}
    with tarfile.open(archive,'r:gz') as tar:
        members=tar.getmembers()
        if len(members)!=len(expected):raise RuntimeError('archive file count mismatch')
        for member in members:
            if not member.isfile() or member.name not in expected:raise RuntimeError('unexpected archive member')
            h=hashlib.sha256()
            with tar.extractfile(member) as stream:
                for block in iter(lambda:stream.read(1<<20),b''):h.update(block)
            if h.hexdigest()!=expected[member.name]:raise RuntimeError('archive hash mismatch')
    dest_verify=verify(manifest,payload,layout='destination',profiles=['code50'],budget=budget)
    if dest_verify['status']!='verified':raise RuntimeError('payload layout verification failed')
    result=dict(status='verified',archive=str(archive),bytes=archive.stat().st_size,sha256=sha256(archive),
        payload=str(payload),asset_count=len(manifest['files']),asset_bytes=meta['asset_bytes'],archive_file_count=len(files),
        verified_code_commit=code_ref,manifest_sha256=meta['original_manifest_sha256'],encodings=verified['encodings'])
    archive.with_suffix(archive.suffix+'.sha256').write_text(f"{result['sha256']}  {archive.name}\n")
    (out_dir/(name+'.package.json')).write_text(json.dumps(result,indent=2)+'\n')
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--root',type=Path,default=Path('.'))
    ap.add_argument('--manifest',type=Path,default=Path('results/assets-manifest.json'))
    ap.add_argument('--output-dir',type=Path,required=True)
    ap.add_argument('--name',default='code50-djam-2026-11-13')
    ap.add_argument('--readme',type=Path,default=Path('docs/djam-code50-package-readme.md'))
    ap.add_argument('--code-ref',required=True)
    args=ap.parse_args()
    print(json.dumps(build(args.root,args.manifest,args.output_dir,args.name,args.readme.read_text(),args.code_ref),indent=2))


if __name__=='__main__':main()
