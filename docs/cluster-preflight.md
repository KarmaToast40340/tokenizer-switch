# Cluster preflight — 9 November 2026

Chinese assets, source and pinned GPT-2 files are staged and verified. **Wait for
Djamel’s working environment path and activation command before GPU submission.**
Chloé has requested those details. No new environment installation or further
open-ended environment search while waiting. No project GPU job has been
submitted; no coursework, course jobs or shared environments were modified.

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

`quota -vs` currently reports a zero-usage row for a different home filesystem;
it does not establish the effective quota on the mounted course home. `df`
reports filesystem-wide capacity, not this account’s allowance. The plan uses
the measured `du` consumption and conservative known 10 GiB ceiling. The earlier
write failure and failed installation remain historical facts; no deletion or
installation was attempted today.

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
No GPU execution test was performed today. Wait for Djamel’s already-working
environment details; do not modify any of these environments.

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

## Gate before Chinese 50M A/B/E

1. Receive Djamel’s environment and activation command. Verify activation inside
   `cluster/job.sbatch` (currently it sources `$VENV/bin/activate`; a conda
   activation command may require adapting this). Do not assume a path is enough.
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
