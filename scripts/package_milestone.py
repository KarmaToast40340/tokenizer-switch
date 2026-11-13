"""Make a private, checksummed archive of the measured CPU milestone.

Includes source text: keep private and outside Git. No credentials, caches,
environments, model weights or cluster account files are included.
"""
import argparse
import json
from pathlib import Path
import tarfile

from toksw.splits import file_sha256


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="deliverables/chloe-code-milestone.tar.gz")
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(f"preserving existing archive: {out}")
    inputs = [Path(p) for p in ("data/milestone/raw", "data/milestone/tokens",
                                "artifacts/milestone/tokenizers", "results/milestone",
                                "requirements-milestone.txt", "configs/tokenizer_mini_code.yaml")]
    for p in inputs:
        if not p.exists():
            raise FileNotFoundError(p)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Add individual files so parent symlinks to non-synced working storage are
    # followed, while archive names remain portable relative repository paths.
    with tarfile.open(out, "w:gz") as tar:
        for p in inputs:
            files = sorted(f for f in p.rglob("*") if f.is_file()) if p.is_dir() else [p]
            for f in files:
                tar.add(f.resolve(), arcname=str(f), recursive=False)
    info = dict(archive=str(out), bytes=out.stat().st_size, sha256=file_sha256(out))
    Path("results/milestone/archive.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
