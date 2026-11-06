# Chloé progress — 6 November 2026

Objective: compare GPT-2 small tokenizer switching on code and Chinese (A keep, B FVT swap, C random swap, E progressive/pruned). Compression gains are measured; quality gains from E remain hypotheses.

Milestone: **17 November 2026**, English, 5 minutes + 3 minutes Q&A. Deliver Chloé’s assets and presentation inputs by **11 November**; travel 12–21 November. Basic 50M comparisons precede rho ablations, 200M extensions or unspecified innovations.

## Responsibility and allocation

| Person | Implementation responsibility | Proposed GPU allocation |
|---|---|---|
| Chloé | Shared data, code/Chinese tokenizers, evaluation, figures | Chinese 50M A/B/E, later Chinese 200M B/E (~13 h estimate) |
| Mathieu | Training implementation, E/pruning, code review | Code 200M A/B, Chinese 200M A (~12.5 h estimate) |
| Djam | Cluster setup, coordination, collecting results | Code 50M A/B/C/E; later rho studies, code 200M E (~14 h estimate) |

GPU allocations do not change implementation ownership. Extra innovations are unspecified and have not been implemented. Account time estimates are unverified for our selected tokenizers; equal tokens do not imply equal runtime.

## Verified checklist

- [x] Verified all 56 preserved files (85,955,238 bytes) against yesterday’s SHA-256 inventory.
- [x] Preserved all 28 original edited/untracked files in `backups/2026-11-06-before-integration/` before integration.
- [x] Active repository: `$HOME/tokenizer-switch-local/repo`, branch `chloe/data-eval`, based on teammate main `4dfef36`. Original Desktop checkout is preserved; use the persistent checkout for further work.
- [x] Integrated upstream account plan/setup/collector; kept both sets of `.gitignore` rules. The unmerged Djam setup fix was inspected, not silently merged.
- [x] Code pilot: five equal 20 MB BPE fits, frozen validation selection, held-out metrics, PNG/PDF compression figures, complete alignment verification.
- [x] Code pilot selected `code70_gpt4_32k` (70% code, GPT-4 regex, 32,000 vocabulary). Validation NSL: Python 0.5589, English 1.0978. Code-only NSL 0.5463 failed the English <=1.25 cap (1.3872).
- [x] Selected code tokenizer test: Python NSL 0.5916, 2.973 bytes/token; English NSL 1.0813, 3.911 bytes/token. Round trips 100% on both.
- [x] Pilot encodings: same 19,067 chunks; GPT-2 22,915,659 tokens, selected 12,489,572. These are **not 50M/200M assets**.
- [x] Chinese sequential Parquet + OpenCC + compressed-storage smoke pipeline passed and exited cleanly. No repeat of the earlier Arrow scanner shutdown hang in this path.
- [x] 30 offline integration tests plus two source-reader tests passed, including analytic bpb, split leakage, gzip logical checksums, checkpoint/speed CLIs, and tiny A/B/C/E training/pruning integration.
- [x] Separate 250 MB code model-training assets: verified; see capacity table.
- [x] Chinese 400 MB model-training assets + five-point tokenizer grid: verified; see capacity table.
- [ ] Real model bpb, generation speed and training plots: require actual checkpoints/logs; none invented.
- [ ] Chinese Slurm runs: blocked by project storage/environment readiness; recheck the account queue before any submission.

## Capacity and storage

Data and environment live in persistent directories `data/`, `artifacts/`, `venv/` under `$HOME/tokenizer-switch-local`; repo symlinks resolve there, never into iCloud or `/private/tmp`. Results are actual files under the persistent repository. Temporary storage is used only for edit staging/tests, never required data or environments.

| Asset | GPT-2 tokens | New tokens | Both >=50M? | Both >=200M? |
|---|---:|---:|---|---|
| Code pilot (50 MB) | 22,915,659 | 12,489,572 | No | No |
| Code model corpus (250 MB) | 113,886,429 | 63,154,284 | True | False |
| Chinese model corpus (400 MB) | 283,162,335 | 91,679,979 | True | False |

Capacity describes encoded corpus size, not guaranteed unique-text sampling: the existing sampler draws document starts with replacement. A 200M run is a separate experiment with its own LR/ramp schedule; training saves final weights, not optimizer/RNG resume state. Do not extend 50M unchanged.

Compressed raw training JSONL is read transparently using logical `.jsonl` paths. SHA-256 in audit/encoding manifests refers to decompressed JSONL bytes. Existing uncompressed pilot checksums remain valid. `.bin`/`.idx` formats and document ordering are unchanged.

## Reproducibility and results

Seed 2026; shuffle buffer 1000; stable SHA-256 assignment 80/10/10 by code repository or non-code document hash. Bounded quotas are not an exact 80/10/10 size ratio. Exact documents/chunks are deduplicated; code repositories are disjoint; no near-dedup claim. Splits precede tokenizer fitting. Train chunk sizes: code 3000 characters, Chinese 1000; held-out documents stay whole.

Code source revision `35a59fb025bc0a102f7d96eac09d145b896d487b`; Wikipedia en/zh `20231101` configs, revision `b04c8d1ceb2f5cd4588862100d08de323dccfbaa`; Chinese OpenCC 1.1.9 t2s. Original GPT-2 tokenizer revision `607a30d783dfa663caf39e06633721c8d4cfcd7e`. Wikipedia reader backend is recorded; old English pilot data is reused rather than regenerated.

Pilot English train: 20,004,203 UTF-8 bytes; English validation/test each 100 documents. Code pilot train 50,002,307 bytes; code validation/test each 100. Separate model corpora use 1,000 validation and 1,000 test documents. Tokenizer fitting stays at **20 MB** per variant: this is a CPU milestone mini-grid, smaller than the proposal’s 1 GB design study.

Code pilot results: `results/milestone/{validation.csv,test.csv,selection.json,alignment.json,manifest.json,compression_*.png,compression_*.pdf}`. Model code corpus: `results/code50/`. Chinese results: `results/chinese/`. Smoke results are explicitly excluded from project comparisons.

## Exact local commands

```bash
cd "$HOME/tokenizer-switch-local/repo"
export PYTHONPATH=src TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2
export HF_HOME="$HOME/tokenizer-switch-local/data/hf-cache"
export MPLCONFIGDIR="$HOME/tokenizer-switch-local/data/mpl-cache"
.venv/bin/python -m pytest -q -p no:cacheprovider
# Existing completed steps are checked/reused; do not start a second concurrent copy.
.venv/bin/python scripts/run_milestone.py
.venv/bin/python scripts/run_code_assets.py --train-mb 250
.venv/bin/python scripts/run_chinese.py --train-mb 400
```

## Cluster and next steps

Read-only queue checks were performed without changing existing course jobs. Latest project blocker is home quota and lack of a verified CUDA 12.8 project environment. Shared torch is 2.6.0+cu124; do not modify the shared environment. A failed separate installation was removed. Djam must resolve storage/environment availability before project data transfer or GPU verification.

Handoff: consult `docs/data-handoff.md` for actual files and overrides. Djam/Mathieu need code assets; Chloé needs both Chinese encodings; Mathieu’s Chinese A needs only GPT-2 encoding **plus validation/test text and GPT-2 tokenizer/weights**. Copy no raw training text to the cluster.

Before a model job: verify a released code revision, asset checksums, selected names, validation override, eval_batch=1, CUDA build, checkpoint capacity, disk quota and queue. Basic A/B/E first. No duplicate job submission. No GPU training or benchmark has been launched by these CPU pipelines.

## Latest cluster checkpoint

Chinese tokenizer selected: `wiki_zh70_gpt4_50k`. Test NSL Chinese 0.3035 / English 1.0761; both lossless. Chinese assets are complete and verified for 50M A/B/E; Chinese GPT-2 alone covers 200M A. Other 200M assets remain incomplete.

The school environment attempt hit quota and was removed. A compatible project environment and storage remain blocked. No project GPU jobs have been launched. The latest course-job status and diagnostic details are kept privately outside Git; see `docs/data-handoff.md` for conditional project commands.
