# Verified corrected Chinese 50M results — 12 November 2026

Job **766353 COMPLETED / exit 0** at **20:44:34 Bangkok / 21:44:34 Taipei**,
12 November. Allocation elapsed **29h54m07s**. Frozen source:
`d11b8b5e7188321b881101081c1119a89ec3276b`. No continuation or larger run was submitted.

## Final verified measurements

| Condition | Actual training tokens | Test bpb (lower) | Generation characters/s | Generation tokens/s | Reported trainer hours |
|---|---:|---:|---:|---:|---:|
| A keep | 49,938,432 | 1.9566465 | 55.66 | 112.37 | 9.22 |
| B FVT | 49,938,432 | 1.6927551 | 172.64 | 116.98 | 8.23 |
| E progressive, pruned | 49,938,432 | 1.7634909 | 165.08 | 116.52 | 11.08 |

**B has the lowest observed test bpb in this run. E does not demonstrate an
advantage over B.** These are valid corrected dropout-0.1 results from one seed;
not statistical significance or a completed 1 GB tokenizer study. Tokenizer fits
are **20 MB pilots**. Do not merge these results with the dropout-disabled pilot
or infer that dropout is generally harmful from the two protocols' scores.

Inference evaluation covers the same **1,000 held-out Chinese documents**,
**10,265,237 UTF-8 bytes**, batch 1, context 1024, with EOT targets included.
The saved-model bpb exactly matches the final trainer score (post-pruning for E).
Raw SHA-256: `3ab1840f01a07bfd5e36ea40e705d11151934794f503fa0fcd6bff6764e01c0f`.

Speed: GTX 1060, unchanged DL_HW, 20 shared 200-character prefixes, 128 greedy new
tokens, tokenizer-specific EOT/padding and explicit masks. CUDA synchronized;
warmup excluded. This is one fixed-token throughput benchmark, not generation
quality or a confidence interval. Equal token budgets do not imply equal raw-text
exposure, unique data coverage or wall-clock compute. Different characters/token account for much
of the characters/s difference; tokens/s are close. Trainer wall time includes
periodic/final evaluation but excludes initial evaluation, E pruning and saving.

## Completed correctness checks

- **762 training-phase checks per condition**, all 77 logged training events have
  `model_training=true`. Every initial/periodic/final evaluation restores training;
  E's post-pruning evaluation also restores it. All logged losses/gradients finite.
- All three saved weight sizes/hashes, vocabulary shapes and reference-tokenizer
  behavior verified; all 1,000 test documents have exact encoding/decoding checks.
- All test bytes/tokens covered, finite bpb and NLL arithmetic checked, saved/logged
  score agreement exact. Weight hashes unchanged after evaluation/speed.
- All **17 collected small files** match the cluster manifest. Both comparison
  and training figures were visually inspected; pilot/corrected labels are distinct.
- **72 current offline tests pass (130.01 s)**, including model/evaluation/mode
  workflow tests and explicit corrected-versus-pilot collection guards. All 17
  corrected publication-file hashes also match the cluster manifest.

Common training settings: pinned GPT-2 base, model/data seeds 0/1234, context
1024, BF16 micro 1 × accumulation 64, dropout 0.1, LR 3e-4, AdamW (0.9,0.95),
weight decay 0.1, clip 1.0, warmup 0.02/minimum LR ratio 0.1; E rho 0.5. Periodic
evaluation uses 200 validation docs every 100 steps; final evaluation uses test.
The lifetime-only memory fix preserves paired tiny A/B/C/E losses and weights
bit-for-bit against the preceding corrected trainer. No data/optimizer/schedule
change was used to pass the GPU gate.

## Figures ready for the presentation

- **Main corrected comparison:** `results/chinese766353_corrected_dropout01/comparison.png`
  and `.pdf`: test bpb, generation characters/s, reported trainer wall time.
- **Corrected learning curves:** `results/chinese766353_corrected_dropout01/training.png`
  and `.pdf`: measured training bpb and periodic validation, not final test curves.
- **Compression/capacity:** `results/milestone_summary.{png,pdf}`; selected test
  NSL code **0.5916**, Chinese **0.3035**, English **1.0813 / 1.0761**, lossless.
  The capacity chart describes code 250 MB / Chinese 400 MB corpora, not unique sampling.
- **Mini-grids:** `results/{milestone,chinese}/compression_validation.{png,pdf}`;
  selected held-out plots are `compression_test.{png,pdf}`. All say 20 MB pilot fits.
- **Historical dropout-disabled pilot:** `results/chinese757238_dropout_disabled/`.
  Its model plots retain **PROVISIONAL / dropout disabled**. Use them only when
  explaining the diagnosis; the corrected folder is the primary model comparison.

Small source CSV/JSON, numeric logs and `manifest.json` accompany corrected
figures. These are measured project outputs; diagnostic toy scores are excluded.

## Persistent checkpoints and reproduction

Cluster output root:
`~/tokenizer-switch/runs/chinese_corrected_dropout01_20261110/`
(the directory name reflects the original plan; actual training began 11 November).

| Model subdirectory | model.safetensors SHA-256 |
|---|---|
| `A_keep_zh_dropout01_50M/model` | `76e61af825fafdc8e7afcff8ecd000c4f01535f0e728a0bb9b7a634f7108e33b` |
| `B_swap_fvt_zh_dropout01_50M/model` | `6332cb6ba6a224961f3eb98eac26a90691a9fa36719accc0cf05dea658bef7f3` |
| `E_progressive_zh_dropout01_50M/model` | `1a5b39547b8804d03241de6c6d4dd992fe3e5d5269fa712c5a143e25d58ab79d` |

Immutable release: `~/tokenizer-switch-releases/zh50-memory-d11b8b5e7188/`.
Controller `results/corrected_train_766353.json`; original verified outputs
`results/corrected_verified_766353/`; stdout `logs/zh50_corrected_ABE_766353.out`.
Do not overwrite checkpoints or rerun the trainer into existing directories.
There is no optimizer/RNG resume state. No new environment was installed, and
no coursework/environment files or pilot assets were deleted.

To inspect evidence without recomputing experiments:

```bash
sacct -j 766353 --format=JobID,State,ExitCode,Elapsed
cat ~/tokenizer-switch-releases/zh50-memory-d11b8b5e7188/results/corrected_verified_766353/verification.json
```

Completion/step-mode evidence is published in
`results/corrected_chinese_completion_766353_2026-11-12.json`.
Weights, raw datasets, environments and credentials remain outside Git.

## Remaining work and scope

Chloé's assigned core Chinese 50M data/tokenizer/evaluation/figure deliverables
are ready for milestone review. Code-model checkpoint comparisons await
Mathieu/Djam's actual runs. No general E advantage, multi-seed significance,
full 1 GB fits or 200M extension is claimed; these remain deferred. The PR is
for teammate review, not automatic merging. No Slack/email message was sent.
