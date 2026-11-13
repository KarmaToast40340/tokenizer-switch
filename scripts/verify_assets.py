"""Verify frozen transfer files and token capacity locally or after transfer.

Standard library only; never regenerates assets. Exit nonzero on any mismatch
or insufficient capacity. A passing report does not imply unique sampling.
"""
import argparse
from array import array
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def verify(manifest, root, layout='source', profiles=None, budget=50_000_000):
    if budget <= 0:
        raise ValueError('budget must be positive')
    root = Path(root)
    profiles = profiles or sorted({f['profile'] for f in manifest['files']})
    errors, checked, capacities = [], [], []
    for profile in profiles:
        entries = [f for f in manifest['files'] if f['profile'] == profile]
        if not entries:
            errors.append(f'unknown or empty profile: {profile}')
            continue
        by_source = {f['source']: f for f in entries}
        for item in entries:
            rel = Path(item[layout])
            if rel.is_absolute() or '..' in rel.parts:
                errors.append(f'unsafe manifest path: {rel}')
                continue
            path = root / rel
            try:
                size, digest = path.stat().st_size, sha256(path)
                ok = size == item['bytes'] and digest == item['sha256']
                checked.append(dict(profile=profile, path=str(rel), bytes=size, sha256=digest, ok=ok))
                if not ok:
                    errors.append(f'size/checksum mismatch: {rel}')
            except OSError as e:
                errors.append(f'{rel}: {e.strerror}')
        encoded = [f for f in entries if '/tokens/' in f['source'] and f['source'].endswith('.json')]
        if not encoded:
            errors.append(f'no encoding metadata: {profile}')
        doc_counts = set()
        for item in encoded:
            try:
                meta = json.loads((root / item[layout]).read_text())
                bin_item = by_source[str(Path(item['source']).with_suffix('.bin'))]
                idx_item = by_source[str(Path(item['source']).with_suffix('.idx'))]
                idx = array('q')
                idx.frombytes((root / idx_item[layout]).read_bytes())
                if sys.byteorder != 'little':
                    idx.byteswap()
                n_docs, n_tokens = meta['n_docs'], meta['n_tokens']
                if (len(idx) != n_docs + 1 or not idx or idx[0] != 0 or idx[-1] != n_tokens
                        or any(a >= b for a, b in zip(idx, idx[1:]))):
                    raise ValueError('invalid document offsets/counts')
                if n_tokens <= 0 or (root / bin_item[layout]).stat().st_size not in (2*n_tokens, 4*n_tokens):
                    raise ValueError('invalid token binary size')
                doc_counts.add(n_docs)
                capacities.append(dict(profile=profile, tokenizer=Path(item['source']).stem,
                                       n_docs=n_docs, n_tokens=n_tokens, sufficient=n_tokens >= budget))
                if n_tokens < budget:
                    errors.append(f'{profile}/{Path(item[layout]).stem}: {n_tokens} tokens < {budget}')
            except (OSError, ValueError, KeyError) as e:
                errors.append(f'invalid encoding {item[layout]}: {e}')
        if len(doc_counts) > 1:
            errors.append(f'encodings have different document counts: {profile}')
    return dict(status='verified' if not errors else 'failed',
                checked_at=datetime.now(timezone.utc).isoformat(), layout=layout,
                requested_token_budget=budget, files=checked, encodings=capacities, errors=errors,
                scope='Physical hashes and encoding structure; prior full alignment remains valid only for unchanged files.',
                sampling='Training samples document starts with replacement; capacity is not unique coverage.')


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--manifest', type=Path, default=Path('results/assets-manifest.json'))
    ap.add_argument('--root', type=Path, default=Path('.'))
    ap.add_argument('--layout', choices=['source', 'destination'], default='source')
    ap.add_argument('--profiles', nargs='+', choices=['code50', 'chinese'])
    ap.add_argument('--token-budget', type=int, default=50_000_000)
    ap.add_argument('--out', type=Path)
    args = ap.parse_args()
    report = verify(json.loads(args.manifest.read_text()), args.root, args.layout, args.profiles, args.token_budget)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open('x') as f:
            f.write(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'files'}, indent=2))
    return int(bool(report['errors']))


if __name__ == '__main__':
    raise SystemExit(main())
