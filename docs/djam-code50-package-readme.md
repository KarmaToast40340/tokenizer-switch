# Verified code50 transfer for Djam — 13 November 2026

This archive contains exactly the 13 code50 assets selected by the repository's
`results/assets-manifest.json`, plus this README, a filtered manifest, PACKAGE.json,
SHA256SUMS and a standalone verifier. All are ordinary files, not machine links.
No environment, credentials, model weights/cache, raw training JSONL or Chinese
corpus is included. The large archive is not in Git.

Verified corrected code commit: **{{VERIFIED_CODE_COMMIT}}**. Use this merged main
commit, not an older training checkout. Full launch instructions are in
`docs/djam-code50-handoff.md` at that commit. Djam's working RTX environment is
reused without installing or changing packages.

## What these assets mean

- Frozen selected tokenizer `code70_gpt4_32k`: **20 MB pilot fitting mix**, not
  the proposed 1 GB study and not a fit on the entire model corpus.
- Model corpus: **250,012,440 UTF-8 train bytes**, **95,761 encoded chunks**.
- GPT-2 encoding **113,886,429 tokens**; selected encoding **63,154,284 tokens**.
- Validation/test: **1,000 documents each**. Both encodings cover 50M, not 200M.
- Planned per-condition budget: 50,000,000 requested tokens, rounded to
  **49,938,432 actual tokens** (762 × 65,536). Sampling is with replacement.

Source paths are recorded in assets-manifest.json. The payload already uses its
destination layout: `data/tokens/code/`, `data/raw/code/`,
`artifacts/tokenizers/{gpt2,code70_gpt4_32k}/`.

## Verify after transfer

Before extraction, compare the archive's external `.tar.gz.sha256` sidecar:

```bash
sha256sum -c code50-djam-2026-11-13.tar.gz.sha256
tar -xzf code50-djam-2026-11-13.tar.gz
cd code50-djam-2026-11-13
sha256sum -c SHA256SUMS
python3 tools/verify_assets.py --manifest assets-manifest.json --root . \
  --layout destination --profiles code50
```

Extract only into a fresh directory; don't overwrite another transfer. On macOS,
`shasum -a 256 -c` can replace `sha256sum -c`. Every checksum and capacity check
must pass. Keep this payload in persistent storage outside iCloud and `/tmp`.

Use a separate worktree of the verified merged main. To avoid another data copy,
create **local ignored** `data`/`artifacts` links to this extracted payload (see
the handoff document); do not commit those links. The archive itself has none.

## Corrected launch requirements

Activate the already working RTX 5050 environment with Djam's own activation
command. Capture `DJAM_PYTHON="$(command -v python)"` and resolve the existing
local GPT-2 snapshot pinned to `607a30d783dfa663caf39e06633721c8d4cfcd7e`.
Use `cluster/djam_code50.sbatch` one condition/job at a time: A_keep,
B_swap_fvt, C_swap_random, then E_progressive. It does not activate/install an
environment. It passes local tokenizers, validation evaluation, dropout 0.1,
micro 1 × accumulation 64 and exact commit/account metadata to the corrected
trainer. Training mode is checked before every step and restored after eval.

Never resubmit an active/completed run or overwrite a failed run. There is no
optimizer/RNG resume state: after inspecting a failure, preserve its old folder
under a unique `.failedN` name before a justified fresh retry. No 200M or extra
innovation run is requested by this package.
