# Chloé progress — 11 November 2026

## Latest audit outcome — 11 November, 15:41 Taipei

**Pilot results/figures DONE; corrected compatibility DONE; full corrected
training NOT STARTED (0 tokens, no Slurm ID).** Pilot 757238 and verifier 764463
completed and all three preserved hashes still match. Corrected gate 764704
initially failed E OOM; standalone E retry **766279** passed after a tensor-
lifetime-only fix. All 128 E gradient forwards/backwards and post-eval mode
restoration passed. Prior A/B passes are reused under unchanged protocol.

**70 tests pass**, including bit-identical paired A/B/C/E loss/weights against
4cc7bff. The gate supports 37 hours and the checkpoint/quota reserve fits. Full
continuation is currently blocked by automatic action approval review; no
continuation watcher was launched. It needs approval to submit the prepared
single corrected sequence. [The current audit](milestone-audit-2026-11-11.md)
contains the exact command, source, IDs, errors and presentation figure catalog.
Historical blocked/running snapshots below are superseded.


## Current verified audit — 11 November

**Pilot DONE; corrected full training NOT STARTED at today's initial audit.**
Pilot **757238** and independent verifier **764463** completed successfully.
Corrected GPU gate **764704** passed A/B but failed E backward with CUDA OOM;
the watcher stopped before submitting formal corrected A/B/E. Full corrected
training tokens are zero. Yesterday's armed plan was not a completed run.

Verified pilot bpb A/B/E: **1.864271 / 1.646345 / 1.714944**; generation characters/s:
**55.76 / 181.04 / 168.15**. Keep all model findings explicitly labelled
**dropout-disabled pilot / provisional for the intended dropout-0.1 study**.
All checkpoints and 49,938,432-token logs are preserved. Real bpb, speed and
labelled comparison/curve figures are now collected locally and hash verified.

See [today's audit](milestone-audit-2026-11-11.md) for DONE/BLOCKED/NOT STARTED,
actual IDs/errors and the exact presentation-ready figure/claim catalog. The
entries below describe earlier snapshots and are superseded by that audit.
No 200M extension or extra innovation was submitted.


Objective: compare GPT-2 small tokenizer switching on code and Chinese (A keep, B FVT swap, C random swap, E progressive/pruned). Compression gains are measured; quality gains from E remain hypotheses.

Milestone: **17 November 2026**, English, 5 minutes + 3 minutes Q&A. Deliver Chloé’s assets and presentation inputs by **11 November**; travel 12–21 November. Basic 50M comparisons precede rho ablations, 200M extensions or unspecified innovations.


## Current priority — preserve pilot; fresh corrected A/B/E authorized (10 November)

**All model scores from job 757238 are PROVISIONAL for the intended study:**
training used eval mode, so dropout 0.1 was disabled throughout A/B/E. Gradients
and updates still occurred. Evaluation itself correctly uses eval mode; valid
reevaluation describes the existing weights, but cannot repair their training.
Fresh training is necessary for the intended dropout-enabled comparison.


**Latest authorization:** finish/preserve this pilot, then prioritize fresh corrected
Chinese 50M A/B/E. A cluster-only watcher is now deployed and alive at revision
**4cc7bff**, waiting for pilot verification; no corrected Slurm job ID exists yet.
See [the corrected-run handoff](corrected-chinese-handoff.md) and the deployment
update below. Historical instructions against automatic corrected submission are
superseded. Full submissions require measured train-mode GPU/runtime and quota
checks; 200M work remains deferred.

The exact release/source hashes and six controlled A/B/E CLI traces establish
this diagnosis; see [train-mode-diagnosis.md](train-mode-diagnosis.md) and the
small synthetic report `results/train-mode-diagnosis-2026-11-10.json`. The fix
explicitly enables training and restores metric mode in `finally`. It was
implemented/tested in a separate local worktree. The active cluster release,
inputs, checkpoints, DL_HW and coursework remain unchanged.

**16:35 Taipei:** job **757238 RUNNING**, E logged step 410/762 (26,935,296 tokens).
A/B checkpoints are saved and controller-hashed. Their logged test bpb values
are **1.864271 / 1.646345**, respectively: provisional dropout-disabled pilot
measurements, not yet independently verified checkpoint results. Finishing E
is worthwhile for this consistent pilot. No cancellation, restart, corrected
50M run or 200M submission was made.

**Validation:** 49 full-suite tests passed in 99.57 seconds, followed by six
publication-guard tests (55 passing checks total). The first expanded run hit
local macOS ENOSPC; after disk headroom recovered, the complete rerun passed.
The paired six-CLI source reproduction passed independently. Slurm shell syntax
and branch whitespace checks passed. GPU post-verification remains pending.

The prior 42 tests did not assert actual forward-pass training mode and missed
this regression; corrected mode/exception/CLI tests and paired saved-checkpoint
proof now cover it. Earlier claims below that training behavior was unchanged
are superseded by this confirmed diagnosis. Old GPU gates also used dropout
disabled: retest corrected train-mode memory/speed before any fresh runs.

A separate, locked post-run watcher can submit read-only checkpoint verification
only after the original allocation ends and the account queue is empty. The
verifier checks all weight hashes, tokenizer behavior, all 1,000 held-out docs,
saved/logged score agreement, speed protocol and unchanged weights afterward.
It produces fresh clearly labelled provisional CSVs and PNG/PDF comparisons.
Its current deployment/job IDs are recorded below when verified; preparation is
not a claim that the pending results already exist.

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
- [x] 42 offline tests passed (9 November), including analytic bpb, split leakage, gzip logical checksums, checkpoint/speed CLIs, and tiny A/B/C/E training/pruning integration.
- [x] Separate 250 MB code model-training assets: verified; see capacity table.
- [x] Chinese 400 MB model-training assets + five-point tokenizer grid: verified; see capacity table.
- [ ] Real model bpb, generation speed and training plots: require actual checkpoints/logs; none invented.
- [x] GTX 1060 fallback validated in Slurm job 757153 using unchanged DL_HW. All Chinese A/B/E pass at BF16 micro-batch 1 / accumulation 64; checkpoint reserve fits. Separate RTX installation remains blocked, but no administrator help is needed for this fallback.
- [ ] Full Chinese 50M A/B/E: Slurm job **757238** submitted and running on cmpt053, all three exact trainer smoke gates passed; A/B have saved checkpoints and E is active (10 November). These are dropout-disabled pilots. See the latest execution update and `docs/gtx1060-handoff.md`; formal checkpoint results remain pending.

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
- [ ] Deferred RTX option: it would require extra persistent space or quota. Do not attempt installation under the insufficient quota; use the validated GTX fallback instead.
- [ ] Verify that activation works in `cluster/job.sbatch`, then run the 15-minute Chinese A/B/E smoke job. Only proceed after its Slurm exit status and diagnostic JSON both pass.
- [ ] Run Chinese 50M A, B, E sequentially, with validation-only periodic evaluation and `eval_batch=1`, then evaluate saved checkpoints, measure generation speed and plot actual logs.

## Cluster readiness and next steps

Before staging, measured home usage was 4.86 GiB (5,099,172 KiB), down from the earlier near-full state. Using the known 10 GiB planning limit, the conservative additional allowance was: assets 0.73 GiB; GPT-2 0.6; three checkpoints 1.8; checkpoint-save allowance 0.6; source/logs/headroom 0.55. Total planned peak about 9.14 GiB. After staging, reserve 2.9 GiB for checkpoints/save/logs; no new environment fits within this plan.

`DL_HW` still imports the dependencies, but remains torch 2.6.0+cu124 with no sm_120. The alternate shared environments inspected were ROCm builds. These are observations, not GPU execution tests. The user subsequently authorized a separate project environment; installation is now blocked by confirmed quota capacity. Do not repeat a quota-failing installation or switch the experiment to AMD/older GPUs.

The selected Chinese tokenizer is `wiki_zh70_gpt4_50k`; its actual union with GPT-2 has **84,717 rows**, so the previous 80k throughput benchmark is not a measurement for this run. `cluster/check_chinese_gpu.py` exercises exact A/B/E vocabularies, two AdamW steps with the real micro-batch/accumulation settings, validation evaluation at batch 1, and E pruning. It saves no checkpoint and reads no test data. CPU fixture tests passing do not establish GPU compatibility.

Evidence: `results/assets-verification-2026-11-09.json`, `results/chinese-transfer-verification-2026-11-09.json`, `results/gpt2-cache-verification-2026-11-09.json`. Summary figure: `results/milestone_summary.{png,pdf}`. It shows measured test compression and corpus capacity, explicitly labeled as 20 MB tokenizer pilots. Full 1 GB tokenizer studies, real model bpb, generation speed and training comparisons are still pending. That checkpoint preceded the later GTX diagnostics and active allocation documented below.

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
test commands are ready in `docs/project-environment.md`. At that checkpoint no GPU test or training job had been launched; the later
GTX diagnostics and allocation 757238 supersede that state. Request either a 20 GiB home quota or an
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
training implementation remains unchanged. **42 offline tests pass**, including
quota refusal and batch/schedule invariants. Actual launch/job status is recorded
in the subsequent handoff update; no diagnostic weights are research results.

Read `docs/gtx1060-handoff.md` first for the current path. Preserve the active
release worktree. No 200M run, extra tokenizer fit or new environment is needed.

## Active Chinese allocation — 9 November, 17:30 Taipei

Slurm job **757238** is running on **cmpt053**, using unchanged DL_HW and immutable
release **31dbce7f645018eb221d55f5b7756872396de9b5** at
`~/tokenizer-switch-releases/zh50-31dbce7f6450/`. It reserves one GTX 1060 for at
most 36 hours, runs validation-only actual-trainer smoke checks first, and then
assigned Chinese 50M A → B → E sequentially. All use micro-batch 1 / accumulation
64, BF16, 49,938,432 actual tokens and validation-only periodic evaluation.

The allocation automatically saves/hashes the real checkpoints, evaluates the
saved models, measures generation speed, collects results and plots real logs.
Its quota guard reserves 2.9 GiB for remaining checkpoints/save/log headroom;
preflight passed around 6.10 GiB used under the hard 10 GiB limit. Compute nodes
use a conservative `du` fallback against that confirmed limit when quota tools
are unavailable. No additional runtime, dataset upload or coursework deletion.

Monitor release `results/chinese_gtx50_757238.json` and canonical
`~/tokenizer-switch/logs/zh50_ABE_757238.out`. Checkpoints are canonical
`~/tokenizer-switch/runs/{A_keep_zh,B_swap_fvt_zh,E_progressive_zh}_50M/model`.
The detached release must not be changed or removed; the parent/main checkout
may synchronize handoff documents independently. Formal model quality/speed
results remain pending while this allocation runs. No 200M work is queued.

All **42 offline tests passed**, including quota/batch invariants and swapped-
vocabulary generation EOT regression. The generation benchmark now suppresses
each tokenizer’s actual EOT and uses a matching padding ID/explicit attention
mask. This is an evaluation correction; Mathieu’s training algorithms are
unchanged.

Verified after launch: **A/B/E exact trainer smoke gates all passed** (two full
accumulation steps each), and the controller advanced to **training_A_keep** at
17:36 Taipei. `results/chinese_gtx50_status_757238_2026-11-09.json` is a launch-day
snapshot; read the cluster controller for live status. No formal 50M checkpoint
has completed yet.


## Post-E verification armed — 10 November, 14:24 Taipei

The remote watcher is running independently (PID **4149846**) in new persistent
`~/tokenizer-switch-postcheck/0c375eedccad`, detached at **0c375ee**. Its durable
state is `results/postwatch757238.json`; stdout/stderr is
`logs/postwatch757238.log`. It was checked alive after SSH logout. Status is
**waiting**, so there is **no verification Slurm job ID yet**. It will submit
one read-only four-hour GTX 1060 job only after 757238 and any account lab jobs
leave the queue and all three checkpoint-save records exist.

The active release remains clean at **31dbce7**, and hashes of trainer, metric,
checkpoint scorer, generation benchmark and controller were identical before
and after staging. Quota before staging: **7,384,332 / 10,485,760 KiB** used;
new staging is only source, small logs and figures, with no model/data/runtime
copies. Checkpoint reserve remains available. The post-verifier retains a
1.1 GiB free-space guard. No coursework/environment changes were made.

Source/checkpoint scores remain provisional. After verification, the local
collector can publish only manifest-verified small outputs, requiring a clean
unchanged local/remote branch; it stops on authentication or concurrent edits.
Its durable state lives outside Git under `tokenizer-switch-local/diagnostics`.
See [the diagnosis](train-mode-diagnosis.md) for restart and handoff instructions.


## Independent corrected follow-up deployed — 10 November, 16:35 Taipei

| Stage | Actual Slurm job ID | Verified status |
|---|---|---|
| Dropout-disabled pilot A/B/E | **757238** | RUNNING, E logged step 410/762; A/B saved |
| Independent pilot verification | Not assigned yet | Original cluster watcher waiting for 757238/account queue to clear |
| Corrected training-mode GPU/memory/runtime gate | Not assigned yet | New cluster watcher waits for verified pilot/Slurm exit 0 |
| Fresh corrected Chinese 50M A/B/E | Not assigned yet | Conditional on measured gate/runtime/checkpoint reserve; no premature submission |

Pilot watcher **PID 4149846** and corrected watcher **PID 8980** were both alive
on hpclogin02, parent PID 1, separate session IDs, after their launching SSH
commands exited. **They survive laptop closure/SSH disconnect.** The new watcher
uses immutable persistent release
`~/tokenizer-switch-releases/zh50-corrected-4cc7bff72953/` (HEAD **4cc7bff**).
Read `results/corrected_followup.json` for future real job IDs, quotas, measured
runtime and accounting; logs are `logs/corrected_followup.log`. Slurm manages
submitted GPU jobs independently. A login-node reboot is a separate failure;
inspect durable state/locks before any restart.

**The Mac result collector is stopped** following DNS/SSH failure at 14:40.
Downloading/publishing results and future GitHub PR updates require reconnecting
the Mac; no GitHub credentials were copied to the cluster. Computation and
cluster result/state recording continue independently. Do not promise automatic
GitHub updates while the laptop is closed.

Measured quota after new source staging: **7,565,180,928 / 10,737,418,240 bytes**.
A/B checkpoint directories contain **501,332,991 / 500,956,042 bytes**. Approximating
E by B plus 16 MiB for pilot outputs, the initial corrected resource calculation
reserves **2.273 GiB** and projects a reserved peak of **9.801/10 GiB**. This is a
tight planning estimate, not a submission decision. Actual E size, quota and the
larger corrected gate saves are rechecked before full submission and each case.
Only final checkpoints are saved in a separate root; the pilot is never removed
or overwritten. No new runtime/cache copy or coursework changes.

**65 offline tests pass (60.32 s)**, including all A/B/C/E real toy CLI modes,
restoration after evaluation, existing-checkpoint refusal, matched settings,
quota/runtime failure gates, simulated Slurm sequence and pilot/corrected figure
scope labels. Shell syntax/whitespace checks pass. Corrected GPU memory/runtime
is pending the scheduled post-pilot gate; completed dropout-disabled A/B trainer
wall times are 9.05/7.95 h and are not corrected-mode benchmarks.

The new controller uses the pinned common GPT-2 base, model/data seeds 0/1234,
block 1024, BF16 micro 1 × accumulation 64, common optimizer/LR settings and
49,938,432 actual tokens. It verifies mode before every training step and after
every evaluation, then saves/hashes/evaluates corrected checkpoints and produces
separate correctly labelled CSV/PNG/PDF comparisons. 20 MB tokenizer fits remain
pilots, not the proposed 1 GB study. See the handoff for exact paths and gates.
