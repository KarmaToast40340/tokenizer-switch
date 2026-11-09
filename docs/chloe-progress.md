# Chloé progress — 9 November 2026

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

- [x] Verified all 56 preserved files (85,955,238 bytes) against the 5 November SHA-256 inventory.
- [x] Preserved all 28 original edited/untracked files in `backups/2026-11-06-before-integration/` before integration.
- [x] Active repository: `$HOME/tokenizer-switch-local/repo`, branch `chloe/data-eval`, includes teammate main `4dfef36` (re-fetched and merged on 9 November; already an ancestor). Original Desktop checkout is preserved; use the persistent checkout for further work.
- [x] Integrated upstream account plan/setup/collector; kept both sets of `.gitignore` rules. Djam’s later GPT-2 download fix `60a4e3e` was reviewed and cherry-picked as `7689a74` on 9 November.
- [x] Code pilot: five equal 20 MB BPE fits, frozen validation selection, held-out metrics, PNG/PDF compression figures, complete alignment verification.
- [x] Code pilot selected `code70_gpt4_32k` (70% code, GPT-4 regex, 32,000 vocabulary). Validation NSL: Python 0.5589, English 1.0978. Code-only NSL 0.5463 failed the English <=1.25 cap (1.3872).
- [x] Selected code tokenizer test: Python NSL 0.5916, 2.973 bytes/token; English NSL 1.0813, 3.911 bytes/token. Round trips 100% on both.
- [x] Pilot encodings: same 19,067 chunks; GPT-2 22,915,659 tokens, selected 12,489,572. These are **not 50M/200M assets**.
- [x] Chinese sequential Parquet + OpenCC + compressed-storage smoke pipeline passed and exited cleanly. No repeat of the earlier Arrow scanner shutdown hang in this path.
- [x] 41 offline tests passed (9 November), including analytic bpb, split leakage, gzip logical checksums, checkpoint/speed CLIs, and tiny A/B/C/E training/pruning integration.
- [x] Separate 250 MB code model-training assets: verified; see capacity table.
- [x] Chinese 400 MB model-training assets + five-point tokenizer grid: verified; see capacity table.
- [ ] Real model bpb, generation speed and training plots: require actual checkpoints/logs; none invented.
- [x] GTX 1060 fallback validated in Slurm job 757153 using unchanged DL_HW. All Chinese A/B/E pass at BF16 micro-batch 1 / accumulation 64; checkpoint reserve fits. Separate RTX installation remains blocked, but no administrator help is needed for this fallback.
- [ ] Full Chinese 50M A/B/E: sequential workflow prepared with an actual trainer smoke gate, quota checks and post-training evaluation. See `docs/gtx1060-handoff.md` and the later job-status update.

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

## 9 November verification and publication

- [x] Persistent checkout was clean at `1b879e2`; GitHub authentication works and every existing commit was already pushed. Desktop preservation checkout was left untouched.
- [x] Fetched origin, reviewed PR/branch updates and requested integration of `origin/main`: already up to date at `4dfef36`. No new Mathieu training changes or `runs-v1` tag were found. No main push or force push.
- [x] Cherry-picked Djam’s `60a4e3e` download fix as `7689a74`: use huggingface_hub without importing torch or hf_xet on the constrained login node.
- [x] Published `bd44fc8`: reusable `verify_assets.py`, exact A/B/E GPU memory diagnostic, tiny CPU tests, full 26-file verification report and measured milestone summary PNG/PDF.
- [x] Rechecked all physical transfer hashes, document offsets, token counts and 1,000-document held-out splits. Rechecked both compressed training corpora’s logical SHA-256; unchanged full alignment reports remain applicable. No fitting/preparation experiments repeated.
- [x] Cloned the published branch to `~/tokenizer-switch` on Chloé’s cluster account. Uploaded only the 743.1 MiB Chinese transfer set; all 13 destination hashes passed. Local code assets remain ready for the other accounts; no duplicate uploads or raw training text.
- [x] Cached all seven GPT-2 files at revision `607a30d783dfa663caf39e06633721c8d4cfcd7e`. Offline resolution passed; weights SHA-256 matches its LFS content address, and tokenizer SHA-256 matches the encoded corpus’s GPT-2 tokenizer.
- [x] Read-only queue and storage checks: queue empty, no project run files, home 6.11 GiB after staging (6,405,356 KiB). No coursework, course jobs or environments changed.
- [x] Djam confirmed PyTorch must be updated; user authorized a separate project environment. Selected official torch 2.7.1+cu128 / Python 3.12; resolved 40 wheels with exact URLs and SHA-256.
- [x] Queried the actual course-home quota with `quota -vs -f /net/course` / `quota -vs -A`: hard limit 10 GiB, used 6.10 GiB, free 3.90 GiB. Wheel ZIP-directory sizing gives 6.28 GiB unpacked; environment plus existing data/checkpoint reserves is 15.48 GiB.
- [ ] Obtain an approved persistent runtime directory with at least 10 GiB free, or increase home quota to 20 GiB. No installation under the insufficient quota. See `docs/project-environment.md` and the capacity audit.
- [ ] Verify that activation works in `cluster/job.sbatch`, then run the 15-minute Chinese A/B/E smoke job. Only proceed after its Slurm exit status and diagnostic JSON both pass.
- [ ] Run Chinese 50M A, B, E sequentially, with validation-only periodic evaluation and `eval_batch=1`, then evaluate saved checkpoints, measure generation speed and plot actual logs.

## Cluster readiness and next steps

Before staging, measured home usage was 4.86 GiB (5,099,172 KiB), down from the earlier near-full state. Using the known 10 GiB planning limit, the conservative additional allowance was: assets 0.73 GiB; GPT-2 0.6; three checkpoints 1.8; checkpoint-save allowance 0.6; source/logs/headroom 0.55. Total planned peak about 9.14 GiB. After staging, reserve 2.9 GiB for checkpoints/save/logs; no new environment fits within this plan.

`DL_HW` still imports the dependencies, but remains torch 2.6.0+cu124 with no sm_120. The alternate shared environments inspected were ROCm builds. These are observations, not GPU execution tests. The user subsequently authorized a separate project environment; installation is now blocked by confirmed quota capacity. Do not repeat a quota-failing installation or switch the experiment to AMD/older GPUs.

The selected Chinese tokenizer is `wiki_zh70_gpt4_50k`; its actual union with GPT-2 has **84,717 rows**, so the previous 80k throughput benchmark is not a measurement for this run. `cluster/check_chinese_gpu.py` exercises exact A/B/E vocabularies, two AdamW steps with the real micro-batch/accumulation settings, validation evaluation at batch 1, and E pruning. It saves no checkpoint and reads no test data. CPU fixture tests passing do not establish GPU compatibility.

Evidence: `results/assets-verification-2026-11-09.json`, `results/chinese-transfer-verification-2026-11-09.json`, `results/gpt2-cache-verification-2026-11-09.json`. Summary figure: `results/milestone_summary.{png,pdf}`. It shows measured test compression and corpus capacity, explicitly labeled as 20 MB tokenizer pilots. Full 1 GB tokenizer studies, real model bpb, generation speed and training comparisons are still pending. No project GPU job has been submitted.

See `docs/data-handoff.md` for the pinned model path and conditional smoke/training commands, and `docs/cluster-preflight.md` for the storage calculation. Get useful A/B/E results and handoff materials ready by **11 November**, before the **12 November** departure and **17 November** presentation. 200M expansions and additional innovations remain lower priority.

## Authorized isolated runtime — later 9 November update

The user authorized a separate PyTorch/CUDA environment after Djam confirmed the
RTX 5050 requirement. The earlier “wait for Djamel’s runtime path” gate is
superseded. The current gate is **additional persistent storage**.

`results/project-environment-capacity-2026-11-09.json` records the resolved
package sizes and actual quota. `cluster/project-runtime-cp312-cu128.lock` pins
all 40 runtime wheels and hashes. No package installation was performed. The
pip dry-run downloaded wheels into temporary staging (despite dry-run mode);
it retained no cache and cleaned up those files. The original full resolver
report is kept privately outside Git. Shared environments remain unchanged.

The approved fresh-environment installation and existing Slurm forward/backward
test commands are ready in `docs/project-environment.md`. No GPU test, training
job ID or model result exists yet. Request either a 20 GiB home quota or an
approved persistent runtime path with at least 10 GiB free, visible to compute
nodes. Keep Chinese 50M A/B/E first once installation and GPU verification pass.

## GTX fallback — latest 9 November update

No extra path or quota increase is available. Job **757113** established CUDA
execution but hit the optional OpenCV/libGL import. Job **757153** completed
in **4m40s** after process-local text-only import isolation. All A/B/E pass
forward/backward, AdamW, validation, E pruning and real checkpoint saves. No lab
files/environments were changed. BF16 micro-batch 2 OOMs; use **micro 1 × accum
64 for all conditions**, preserving 65,536 tokens/step and BF16 training math.

Measured short-test training-rate estimates are 1,815 / 1,796 / 1,617 tokens/s
for A/B/E: projected 7.64 / 7.72 / 8.58 hours of training only, plus evaluation.
The three actual checkpoint files total **1.400 GiB**; the conservative 2.9 GiB
checkpoint/save/log reserve fits with current 6.10 GiB home usage. No new runtime
is installed. Both the full 6.284 GiB runtime and the optimistic 3.808 GiB
version-matched five-package subset fail the new-runtime/checkpoint budget.

The 36-hour sequential GTX allocation uses an immutable persistent release
worktree and `cluster/run_chinese_gtx50.py`. It first exercises the unmodified
trainer for two full accumulation steps per condition, then runs assigned
Chinese 50M A/B/E and evaluates/plots their real outputs automatically. The
training implementation remains unchanged. **41 offline tests pass**, including
quota refusal and batch/schedule invariants. Actual launch/job status is recorded
in the subsequent handoff update; no diagnostic weights are research results.

Read `docs/gtx1060-handoff.md` first for the current path. Preserve the active
release worktree. No 200M run, extra tokenizer fit or new environment is needed.
