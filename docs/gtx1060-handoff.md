# Chinese 50M on GTX 1060 — verified 9 November 2026

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


> **Latest authorized sequence (10 November, 16:35):** preserve pilot job 757238,
> then run fresh corrected Chinese 50M A/B/E through the independent cluster
> watcher. Per-step mode checks, matched settings, separate outputs and measured
> GPU/quota/runtime gates are ready. See
> [corrected-chinese-handoff.md](corrected-chinese-handoff.md) for current paths,
> actual job IDs and laptop-independent monitoring. No corrected job is submitted
> until the post-pilot gates pass. Mac/GitHub publication remains separate.


> **10 November correction:** job 757238 trains in eval mode (dropout disabled).
> Keep its immutable release unchanged. All scores are provisional for the
> intended dropout-0.1 comparison; correct inference evaluation cannot repair
> the learned trajectory. A/B have saved checkpoints; E is running. Read
> [the confirmed diagnosis and separate post-run workflow](train-mode-diagnosis.md)
> before interpreting results or launching corrected training. Historical GPU
> gates do not establish corrected train-mode memory/speed. No automatic retrain.


The GTX 1060 fallback works with the **unchanged DL_HW environment**. No new
environment is needed. The user authorized assessing this fallback while
preserving coursework and staying below quota; the original assigned Chinese
50M A/B/E work continues with the verified configuration below.

## Actual Slurm assessments

| Job | Result |
|---|---|
| 757113 | COMPLETED, exit 0, 1m30s. CUDA matrix forward/backward passed; all GPT-2 trials failed at an optional OpenCV/libGL import. This was not model compatibility success. |
| 757153 | COMPLETED, exit 0, 4m40s, cmpt053. Process-local vision-import isolation fixed the headless import. All A/B/E model, optimizer, validation and checkpoint-save checks passed at BF16 micro-batch 1. |

Runtime: torch 2.6.0+cu124, CUDA 12.4, transformers 4.49.0. GPU compute
capability 6.1. Native BF16 hardware is absent, but the tested BF16 operations
and model paths execute successfully. `cluster/text_only.py` disables unused
OpenCV/torchvision discovery in that process; it modifies no installed file.

The current trainer’s BF16 math is preserved. All three conditions use
**micro_batch=1, grad_accum=64, block_size=1024**, keeping 65,536 tokens per
optimizer step, the original LR/ramp schedule and 762 steps / 49,938,432 actual
tokens. Use the same batch configuration across A/B/E to preserve matching
document sampling. BF16 micro-batch 2 ran out of memory for every condition.

| Condition | Estimated training tokens/s | 50M training-only projection | Peak allocated GiB | Saved checkpoint bytes |
|---|---:|---:|---:|---:|
| A | 1,815 | 7.64 h | 4.14 | 501,332,991 |
| B | 1,796 | 7.72 h | 4.13 | 500,956,042 |
| E, union 84,717 → pruned 50,000 | 1,617 | 8.58 h | 5.04 | 500,956,042 |

These are short diagnostics, not completed 50M training results. The rate
projection uses four steady microbatches plus measured optimizer-update cost;
it excludes full periodic/final evaluation, setup and saving. Total projected
training time is **23.95 hours**, with additional evaluation time. The original
8-hour job limit is insufficient, especially for E. The configured sequential
allocation allows 36 hours; account QoS permits up to four days and one submitted
job at a time. No 200M extension is scheduled.

FP32 also passed (A/B at micro-batch 2, E at micro-batch 1), but these are
different batch settings and the current trainer is BF16. No training precision
or algorithm was changed to use these faster diagnostic numbers.

The checkpoint sizes were measured by real `save_pretrained` writes after the
diagnostic updates; those new diagnostic directories were then removed. No
existing run, coursework or lab file was deleted. All three retained final
checkpoints together require **1,503,245,075 bytes = 1.400 GiB**. The original
conservative reserve remains 1.8 GiB checkpoints + 0.6 GiB saving allowance +
0.5 GiB logs/headroom = **2.9 GiB**. Existing home usage is about 6.10 GiB, so
this fits under the confirmed 10 GiB hard quota without a new environment.

Evidence: `results/gtx1060_assessment_757153.json` and
`results/gtx1060_feasibility.{csv,png,pdf}`. Regenerate the figure with
`scripts/plot_gtx_assessment.py`; it is explicitly labeled as short-test
projections and measured memory. Do not derive full evaluation throughput from
the old diagnostic validation token counter: E’s counter includes its additional
pruned check while the evaluation timer stops before pruning.

## Sequential allocation and reproducibility

`cluster/chinese_gtx50.sbatch` runs one GPU on tested node cmpt053, using
`cluster/run_chinese_gtx50.py`. It first validates quota, hashes, the actual GPU
and the unmodified trainer with two full accumulation steps for each condition.
The smoke uses a validation-only surrogate test file, and removes only the
diagnostic files it created. Full training starts only if all three pass.

Then it trains A, B and E sequentially into canonical `runs/*_zh_50M` directories.
It refuses existing run directories, never resumes/appends an old run, and
rechecks available quota before each condition. It records checkpoint hashes.
It subsequently evaluates saved models, measures generation speed, collects
real logs and plots actual training curves. Test text is used only for the
formal final/saved-checkpoint evaluations, not the smoke or tokenizer selection.

The allocation runs from a **detached persistent release worktree** under
`~/tokenizer-switch-releases/zh50-<commit>/`, with ignored links to canonical
data, artifacts, runs and logs. No large asset is duplicated. Do not edit or
delete that release while its job is active. The main checkout can synchronize
documentation without changing running source. Every run records the exact
release commit; the controller also refuses a changed release HEAD.

Status is in the release’s `results/chinese_gtx50_<jobid>.json`. Scheduler output
is at `~/tokenizer-switch/logs/zh50_ABE_<jobid>.out`. Checkpoint runs remain at
`~/tokenizer-switch/runs/`. Final metrics/speed/figures go to the release’s
`results/chinese_models_gtx50_<jobid>/`. A failure preserves completed real runs;
inspect the stage/error and never rerun blindly into those directories.

## Smaller RTX 5050 installation assessment

No additional installation was attempted. The full version-matched environment
expands to 6.284 GiB. Even optimistically reusing all other existing libraries,
torch 2.7.1+cu128 + new cuDNN + cuSPARSELt + NCCL + Triton total **3.808 GiB**.
Adding one 0.6 GiB checkpoint already requires 4.408 GiB, versus 3.898 GiB free,
before venv/log/save allowance. With all three checkpoint reserves, only about
0.802 GiB is available for an environment, less than torch alone (1.777 GiB).

Torch 2.7.0+cu128 is larger (1.870 GiB unpacked) and needs the same cuDNN,
cuSPARSELt and NCCL versions. Omitting dependencies or mixing incompatible lab
CUDA libraries is not a supported size reduction. No supported compact layout
was identified that fits this quota and the planned checkpoints. Evidence:
`results/smaller-runtime-assessment-2026-11-09.json` and
`results/torch270-size-2026-11-09.json`.

Additional storage would be required for the RTX 5050 option; **it is not
required for this validated GTX fallback**. Continue useful 50M results and
handoff before 11 November rather than repeatedly requesting a storage path.

## Active job and monitoring

**757238**, submitted 9 November at approximately 17:30 Taipei, is running on
cmpt053 with a 36-hour wall limit. Release: `31dbce7f645018eb221d55f5b7756872396de9b5`
at `~/tokenizer-switch-releases/zh50-31dbce7f6450/`. The parent checkout may advance
without changing this detached release. Do not remove or modify the release.

The controller first performs validation-only actual-trainer smoke checks. Its
current stage and passed gates are authoritative in:

```bash
cat ~/tokenizer-switch-releases/zh50-31dbce7f6450/results/chinese_gtx50_757238.json
squeue -j 757238
sacct -j 757238 --format=JobID,State,Elapsed,ExitCode
tail -30 ~/tokenizer-switch/logs/zh50_ABE_757238.out
```

After all three pass, A/B/E run sequentially without further submission or SSH
access. They preserve BF16 and use the same micro-batch 1 / accumulation 64.
Formal checkpoints stay at canonical `~/tokenizer-switch/runs/`. Final saved-
model bpb, generation speed CSV, run summary and PNG/PDF curves will be under
`~/tokenizer-switch-releases/zh50-31dbce7f6450/results/chinese_models_gtx50_757238/`.
There are no completed formal checkpoint results yet at this launch update.

The launch record is `results/chinese_gtx50_launch_757238.json`. All 42 offline
tests pass. The generation-speed CLI explicitly uses each tokenizer’s true EOT,
matching padding and an all-ones prompt attention mask; this prevents the old
GPT-2 EOT ID in a swapped model’s generation config from changing the protocol.

Verified launch-day status: all three exact trainer smoke gates passed and
**formal A_keep is active** (controller `training_A_keep`, 17:36 Taipei). The
committed status JSON is a snapshot, not live monitoring. No formal 50M model
quality/speed result is claimed yet.
