# Corrected Chinese 50M A/B/E — 10 November 2026

> **Verified completion, 12 November:** corrected job **766353 COMPLETED / 0:0**.
> All A/B/E checkpoints, final bpb, speed and figures pass verification. The mode
> fix held through all 762 steps/condition and every evaluation. See
> [final corrected results](chinese50m-corrected-results.md); older pending/blocked
> snapshots below are historical. The dropout-disabled pilot is preserved separately.


## Current corrected progress — 12 November, 14:19 Taipei

**Job 766353 RUNNING on cmpt053, stage E.** A and B each completed
**49,938,432 tokens** and saved controller-hashed checkpoints. E last logged
**21,692,416 / 49,938,432 tokens (43.4%)**. No errors were reported; all logged
losses/gradients are finite. No duplicate or extension job was submitted.

The mode fix is confirmed across the completed A/B runs: **762 per-step checks
each**, every logged training forward has mode true, and initial/periodic/final
evaluation restores training. E has the same positive mode records, including
after evaluations at steps 0/100/200/300. Source d11b8b5 remains immutable.

Preliminary trainer-logged test bpb: **A 1.9566465**, **B 1.6927551**. These are
corrected dropout-0.1 runs, **separate from pilot scores**. Saved-checkpoint
reevaluation, final E/pruning results, synchronized speed and full comparison
figures remain pending and will run automatically inside this allocation after
E finishes. Do not present these two scores as a verified complete comparison.

Latest checkpoint-space guard: **1.65 GB free** versus **1.42 GB reserved** for
remaining E checkpoint/save/log headroom; fits the 10 GiB quota. Current quota
also remains below the limit. The original dropout-disabled pilot and verified
figures remain preserved. Closing the laptop does not interrupt this job.

Evidence: `results/corrected_chinese_status_766353_2026-11-12.json` (token values
are last-logged lower bounds, not a live dashboard). Historical snapshots below
are superseded by this dated update.


> **Approved launch, 11 November 15:51:** corrected Chinese 50M A/B/E is now
> **RUNNING as job 766353 on cmpt053**, frozen source d11b8b5, 37-hour limit.
> Controller stage A startup/initial eval; zero gradient tokens logged at this
> snapshot, B/E waiting. Earlier approval block is resolved; exactly one job was
> submitted. [Current audit](milestone-audit-2026-11-11.md) has paths/progress.
> Slurm itself handles the sequence/evaluation independently of the laptop.


> **Latest 11 November:** E-only corrected memory gate **766279 COMPLETED / 0:0**;
> prior A/B compatibility is preserved. Formal corrected training is still
> **not submitted (0 tokens)**: automatic approval review blocked unattended
> full continuation. No new watcher is running. Pilot results/figures are verified
> and published. See [today's audit](milestone-audit-2026-11-11.md) for the exact
> prepared 37-hour command and presentation-ready claims.


> **11 November audit:** pilot 757238 and verifier 764463 COMPLETED. Corrected
> gate 764704 FAILED on E CUDA OOM after A/B passed; formal corrected A/B/E was
> **not submitted**. Read [the current audit](milestone-audit-2026-11-11.md),
> rather than treating the historical armed plan below as completed work.


The user authorized fresh corrected A/B/E after preserving job **757238** as a
dropout-disabled pilot. This supersedes the earlier instruction not to launch
corrected runs automatically. No 200M extension or innovation is queued.

## Diagnosis and preservation

Pilot release **31dbce7** never enabled training after GPT-2 loaded in eval mode.
Gradient updates happened, but dropout was disabled throughout A/B/E. Eval-mode
inference is correct for those weights; reevaluation cannot repair training.
Keep pilot weights, configs, logs and source unchanged. Its independently
verified figures/CSVs say **PROVISIONAL — eval-mode-trained / dropout disabled**.
Do not pool pilot and corrected comparisons. See [the diagnosis](train-mode-diagnosis.md).

Corrected code explicitly enables training, checks root/dropout mode before
**every optimizer step**, and verifies mode restoration after initial, periodic,
final and E post-pruning evaluation. The expected dropout probability is checked
as 0.1. A mismatch raises an error. Existing run directories are refused because
the trainer has no optimizer/RNG resume support.

## Matched fresh protocol

| Setting | All A/B/E |
|---|---|
| Initial weights | Pinned GPT-2 `607a30d783dfa663caf39e06633721c8d4cfcd7e`; never pilot weights |
| Data/tokenizers | Existing verified Chinese 400 MB corpus and encodings; original GPT-2 / frozen `wiki_zh70_gpt4_50k` |
| Budget | 50,000,000 requested; 762 × 65,536 = **49,938,432 actual tokens** |
| Context / effective batch | 1024; BF16 micro 1 × accumulation 64 |
| Seeds | Model 0, data 1234 |
| AdamW | LR 3e-4, betas (0.9, 0.95), decay 0.1, clip 1.0 |
| Schedule | Warmup 0.02, minimum LR ratio 0.1; E rho 0.5 |
| Dropout | Configured 0.1, enabled during every training phase |
| Evaluation | Same 200 validation docs every 100 steps, batch 1; all 1,000 test docs at completion |
| Environment/GPU | Unchanged DL_HW, GTX 1060; no new installation or dataset/weight upload |

Separate output root:
`~/tokenizer-switch/runs/chinese_corrected_dropout01_20261110/`.
Names are `A_keep_zh_dropout01_50M`, `B_swap_fvt_zh_dropout01_50M`,
`E_progressive_zh_dropout01_50M`. Only final checkpoints are saved; there are no
intermediate/optimizer/RNG snapshots. Tokenizers remain **20 MB fits**, not a
completed 1 GB study. One seed does not establish statistical significance.

## Cluster-only sequence and gates

1. Existing cluster watcher finishes pilot checkpoint verification on a separate
   GTX job after 757238 leaves the queue. It checks all weight hashes, tokenizer
   behavior, held-out coverage, bpb/log agreement and speed; labelled figures and
   a manifest remain in its persistent postcheck worktree.
2. New cluster watcher waits for that verification manifest, **Slurm COMPLETED /
   exit 0**, and an empty account queue. It recalculates quota/checkpoint reserves
   before submitting the corrected 30-minute GPU gate.
3. The gate runs the exact corrected CLI for **two full accumulation steps per
   A/B/E** using validation-only surrogate evaluation. Hooks require all 128
   gradient forwards per condition to have active dropout, scoring forwards to
   use eval mode, nonzero backward gradients and correctly restored training.
   E exercises both old/new paths and pruning. It measures peak CUDA memory,
   actual save sizes and trainer/CLI time. Only its newly created temporary
   diagnostic checkpoints are removed; they are not research results.
4. After successful gate Slurm accounting, the watcher recalculates space using
   the larger measured pilot/gate checkpoint sizes. It projects runtime from
   the two measured steps, adds **30% + one hour per condition + one allocation
   hour**, and requests the resulting wall limit (must fit the confirmed four-day
   QoS limit). These are projections until full runs finish.
5. It submits a **fresh sequential A → B → E allocation**, then verifies saved
   checkpoints, measures shared-prompt generation speed and produces separate
   corrected comparison figures. No main push, cancellation, overwrite or lab
   environment modification occurs. Errors stop the chain and preserve state.

QoS `dlp_qos`: MaxWall **4 days**, MaxSubmitPU **1** (verified 10 November).
Consequently jobs cannot all be queued with dependencies while 757238 runs.
An independent cluster process submits them one at a time. The submission lock
is retained after an error: inspect queue/accounting before any retry.

## Storage and runtime evidence

At 16:17 Taipei: **7,386,100 / 10,485,760 KiB** quota usage (~7.04/10 GiB),
with A/B saved and E still running. A/B model directories occupy 491,528 /
491,160 KiB. E's pruned checkpoint should be similar to B; actual size is required
after saving. Expected free space after E is about **2.5 GiB**.

Fresh final checkpoints are about **1.4 GiB total**. The controller reserves the
sum of measured sizes **plus 8 MiB each**, **one additional largest checkpoint**
for save workspace, and **384 MiB** for logs/cache/headroom: roughly **2.3 GiB**
additional allowance. It must fit actual quota both before submission and before
each condition. The gate separately reserves one temporary checkpoint plus
headroom. No pilot deletion, environment installation or duplicate model cache
is assumed. Corrected post-evaluation also retains 384 MiB headroom.

Completed pilot trainer wall times: A **32,566 s (9.05 h)**, B **28,614 s (7.95 h)**.
These include periodic/final evaluation but exclude initial evaluation/save;
they used dropout disabled. **Corrected GPU runtime/memory is not measured yet.**
Full corrected training is submitted only after the separate measured gate.
Completion before the 12 November departure cannot yet be promised.

## Monitoring, laptop closure and actual IDs

Both submission watchers run as detached Python processes on **hpclogin02**, with
stdin detached and stdout/stderr in persistent cluster logs. Their source and
state are on cluster storage. They survive closing the laptop or disconnecting
SSH. Once submitted, Slurm manages GPU jobs independently. This does not promise
survival of a login-node reboot; durable state/locks prevent a blind duplicate
restart.

The Mac collector is **separate**. It stopped at 14:40 on a DNS/SSH failure, while
the original cluster watcher remained alive at 16:17. Downloading results and
updating GitHub from that collector require an awake, connected Mac and valid
authentication. **No GitHub credentials were copied to the cluster; automatic
PR updates do not survive laptop disconnection.** Cluster training/evaluation
and state recording do not depend on the Mac collector.

Current and future real Slurm IDs are in the deployment update in
[chloe-progress.md](chloe-progress.md). A watcher PID is not a Slurm job ID. Fields
remain null until `sbatch` actually returns an ID.

After deployment, inspect the new worktree's:

```bash
cd ~/tokenizer-switch-releases/zh50-corrected-4cc7bff72953
cat results/corrected_followup.json
tail -n 30 logs/corrected_followup.log
squeue -u "$USER"
# After IDs exist:
sacct -j JOB_ID --format=JobID,State,ExitCode,Elapsed,MaxRSS
cat results/corrected_gate_JOB_ID.json
cat results/corrected_train_JOB_ID.json
```

The follow-up state stores pilot verification, corrected gate and corrected
training job IDs/accounting, storage calculations and measured runtime plan.
Final corrected outputs are `results/corrected_verified_JOB_ID/` in the new
release, with metadata/weight hashes, bpb, speed, logs, CSV and PNG/PDF figures.
Publish only manifest-verified small results to `chloe/data-eval`; never include
weights/data/environments or merge PR #4 automatically. Mathieu/Djam should
review the explicit mode policy, no-overwrite guard, matching settings and
resource/runtime gates.


Deployment checked at **16:35 Taipei**: corrected watcher **PID 8980**, parent PID
1, session 8980; pilot watcher **PID 4149846**, parent PID 1. Both alive. Source
**4cc7bff729537e17b471bef36e3a40cd2e11e5d3** is detached and unchanged.
`results/corrected-chinese-deployment-2026-11-10.json` records measured storage
and the initial estimate: reserved peak **9.801/10 GiB**. Actual future job IDs
remain unassigned while the pilot runs. **65 offline tests passed (60.32 s)**;
corrected GPU execution/runtime is still pending the gate.
