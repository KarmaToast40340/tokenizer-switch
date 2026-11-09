# Cluster preflight — 9 November 2026

**Latest: GTX fallback validated in job 757153; allocation 757238 now runs the assigned A/B/E sequence. All exact trainer gates passed and formal A_keep is active.** No new environment or
extra storage is needed for that path. Use `docs/gtx1060-handoff.md`; the isolated
RTX 5050 installation budget below remains blocked.

Chinese assets, source and pinned GPT-2 files are staged and verified. Djam and
the user now authorize a separate CUDA 12.8 project environment. **Installation
is blocked by the actual 10 GiB home quota**, not by missing permission. The
resolved runtime alone expands to 6.28 GiB, versus 3.90 GiB currently free.
No package installation has occurred. GTX diagnostic jobs 757113/757153 have
now run; coursework, course jobs and shared environments remain unchanged. See `docs/project-environment.md`.

## Storage verified before and after staging

Before staging, measured home usage was 5,099,172 KiB (**4.86 GiB**), down from
the 6 November near-full state. The queue was empty. The project directory did
not exist, so no existing runs/uploads were duplicated.

Conservative peak calculation against the known 10 GiB planning ceiling:

| Component | Allowance (GiB) |
|---|---:|
| Existing home use before staging | 4.86 |
| Chinese transfer assets (743.1 MiB actual) | 0.73 |
| Pinned GPT-2 cache | 0.60 |
| Three retained final A/B/E checkpoints | 1.80 |
| Checkpoint-save allowance | 0.60 |
| Source, logs and headroom | 0.55 |
| Planned peak | **9.14** |

After staging, measured home use is 6,405,356 KiB (**6.11 GiB**), leaving about
3.89 GiB against that ceiling. Reserve 2.9 GiB for the checkpoints, save allowance
and logs. Recheck consumption before each job. A private CUDA environment is
**not** included and must not be installed under this plan.

The default `quota -vs` shows the unrelated zero-usage gcs row. This was resolved
by `quota -vs -A --show-mntpoint` and `quota -v -f /net/course`: actual course-home
quota is 10,485,760 KiB, usage at the sizing check 6,398,908 KiB, free 3.90 GiB.
`df` remains filesystem-wide capacity, not account allowance. The earlier
checkpoint failure and failed installation remain historical facts. No package
installation was attempted under the insufficient quota.

## Runtime findings and remaining blocker

`/opt/conda/miniconda3/envs/DL_HW` imports torch, transformers, tokenizers, NumPy
and YAML. It remains torch **2.6.0+cu124**, CUDA **12.4**, transformers **4.49.0**,
tokenizers **0.21.0**. Compiled architectures are sm_50, sm_60, sm_70, sm_75,
sm_80, sm_86, sm_90; there is no sm_120. The scheduler still advertises RTX
5050 nodes. System CUDA 12.8 being installed does not change this torch build.

The other standard shared environment (`amd`) is
2.11.0.dev20260108+rocm7.1, with no CUDA build. The additional shared RL Lab5
environment inspected is 2.10.0+rocm7.1. These observations do not establish a
usable NVIDIA runtime. No alternate GPU experiment was substituted.

For the capability/support rationale recorded on 6 November, see
[NVIDIA capabilities](https://developer.nvidia.com/cuda/gpus) and
[PyTorch 2.7 release](https://pytorch.org/blog/pytorch-2-7/).
No RTX 5050 GPU execution test was performed. The later user authorization permits a
fresh separate project environment, preserving all these environments. Official
torch 2.7.1+cu128 is selected; additional persistent storage must be available
before installing it.

## What is staged

`~/tokenizer-switch` is a clone of `chloe/data-eval`. The source/diagnostic
checkpoint `bd44fc8` contains Djam’s reviewed download fix (`60a4e3e`,
cherry-picked as `7689a74`). `origin/main` at `4dfef36` was fetched and integrated;
it was already an ancestor. No `runs-v1` tag exists yet. Record the exact
feature-branch commit in every run and align it across accounts.

- Both Chinese encodings at `data/tokens/wiki_zh/`, matching document ordering.
- Chinese validation/test/stats at `data/raw/wiki_zh/`; no raw training text.
- GPT-2 and selected 50k tokenizer at `artifacts/tokenizers/`.
- Seven GPT-2 files pinned to revision
  `607a30d783dfa663caf39e06633721c8d4cfcd7e` under `data/hf-cache/`.
- No project checkpoints, logs or Slurm runs yet.

The 13 destination asset sizes/checksums and encoding structures pass
`scripts/verify_assets.py --layout destination --profiles chinese`.
All seven cached model files resolve offline. The 548,105,171-byte weights have
SHA-256 `248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707`,
matching their LFS content address. Their GPT-2 tokenizer hash equals the
tokenizer used for the frozen encodings; the embedding shape is 50,257 × 768.
This is a file/cache verification, not a GPU model-load test.

Portable evidence is in `results/chinese-transfer-verification-2026-11-09.json`
and `results/gpt2-cache-verification-2026-11-09.json`. Private account and course
diagnostics remain outside Git.

## RTX-specific gate (the current GTX path is in its dedicated handoff)

1. Obtain sufficient approved persistent storage, then install the separate
   hash-pinned project environment using `docs/project-environment.md`. Verify
   it with `pip check` and use its venv activation in `cluster/job.sbatch`. Do not
   change `DL_HW` or any course environment.
2. Recheck queue, storage, exact revision and asset hashes. Existing run
   directories must not be reused; the trainer has no optimizer/RNG resume and
   appends logs. Check the staged pinned model path explicitly.
3. Submit the 15-minute smoke command in `docs/data-handoff.md`, constrained to
   RTX 5050. `cluster/check_chinese_gpu.py` checks a real CUDA kernel, bf16,
   the exact Chinese A/B/E vocabularies, two AdamW steps with micro-batch 2 and
   accumulation 32, validation evaluation at batch 1, and E pruning. The actual
   union is **84,717** rows, not the earlier 80k benchmark. It reads no test text,
   writes no checkpoint and is excluded from model comparisons.
4. Require Slurm COMPLETED / exit 0 and diagnostic JSON `status=passed` for all
   conditions. The 38 passing offline tests include tiny CPU fixtures, which
   cannot validate GPU compatibility or memory capacity.
5. Run A, then B, then E at 50M on Chloé’s account, one at a time, using the pinned
   base path, selected 50k name, `eval_split=validation` and `eval_batch=1`.
   Evaluate real saved checkpoints, generation speed and real training logs.

200M B/E corpus expansions, a full 1 GB tokenizer study and extra innovations
remain pending and lower priority than useful A/B/E results before 11 November.

## Separate environment capacity audit

Resolved torch 2.7.1+cu128 with matching CUDA dependencies and project training
libraries for Python 3.12. The 40 wheels contain **6.284 GiB** uncompressed;
compressed downloads total **3.621 GiB**. These are archive-directory sizes, not
a completed installation measurement. Add 0.195 GiB for venv/filesystem overhead
and retain the existing 2.9 GiB checkpoint/save/log allowance.

Existing home 6.102 + runtime 6.284 + overhead 0.195 + training reserve 2.9 =
**15.482 GiB**. Even the environment without checkpoints exceeds the 10 GiB hard
limit. Thus a normal isolated installation cannot proceed safely in home.
Request 20 GiB total home quota, or an approved persistent directory with at
least 10 GiB free for the runtime (keeping training assets/checkpoints in home).
The latter must be accessible on compute nodes. No writable assigned shared
project location was identified; do not use another person’s directories.

Evidence: `results/project-environment-capacity-2026-11-09.json` and
`cluster/project-runtime-cp312-cu128.lock`. No installation, Slurm test or training
job was performed under this RTX installation plan. GTX assessment jobs are
757113 and 757153; see the latest fallback handoff.
