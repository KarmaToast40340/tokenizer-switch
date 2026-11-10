# Verified results and their scope

**10 November:** all checkpoint scores from job 757238 are provisional for the
intended study. Its A/B/E make updates with dropout disabled. See
[the exact-source diagnosis](../docs/train-mode-diagnosis.md). The small
`train-mode-diagnosis-2026-11-10.json` is synthetic CPU regression evidence,
not a Chinese model result. Independent real-checkpoint evaluation/speed and
labelled comparison figures remain pending E and the verification gate.


The tokenizer comparisons here are **20 MB fits per candidate: pilot results**,
smaller than the proposed 1 GB tokenizer study. Model-training corpus capacity,
tokenizer compression and model quality are separate measurements. A completed
preparation manifest does not mean model training has completed.

| Files | What they establish | Limits |
|---|---|---|
| `milestone/` | Code tokenizer pilot: validation selection, selected held-out compression, alignment and figures | Its 50 MB corpus is too small for 50M-token training with either encoding. |
| `code50/` | Separate 250 MB code corpus and aligned encodings: 113,886,429 GPT-2 / 63,154,284 selected-tokenizer tokens | Both cover 50M; neither covers 200M. The tokenizer remains the frozen 20 MB pilot selection. |
| `chinese/` | Chinese 20 MB tokenizer mini-grid, held-out compression, plus a 400 MB model corpus: 283,162,335 GPT-2 / 91,679,979 selected tokens | Both cover 50M; only GPT-2 covers 200M. These files do not establish model-quality gains. |
| `milestone_summary.{png,pdf}` | Measured pilot compression and encoded corpus capacity | Capacity is not unique coverage: training samples document starts with replacement. |
| `gtx1060_assessment_757113.json` | CUDA execution passed; model imports failed on missing optional `libGL` | A completed diagnostic job is not evidence that its model trials passed. |
| `gtx1060_assessment_757153.json`, `gtx1060_feasibility.{csv,png,pdf}` | Model, memory and checkpoint-save diagnostics passed; measured microbatch timings support runtime projections | Short diagnostics, not completed 50M runs. Projections exclude full evaluation/setup/save time. Diagnostic weights were removed. |
| `chinese_gtx50_launch_757238.json`, `chinese_gtx50_status_757238_2026-11-09.json` | Launch-day evidence: all three actual trainer smoke gates passed and formal A entered its training stage | Static snapshots, not live status or completed model comparisons. A/B/E execute sequentially in job 757238. |
| `project-environment-capacity-2026-11-09.json`, `smaller-runtime-assessment-2026-11-09.json`, `torch270-size-2026-11-09.json` | Wheel-size and quota analysis for the alternative RTX 5050 environment | Archive sizes and capacity bounds, not an installed or GPU-verified new environment. No new environment was installed. |

The selected tokenizers were frozen on validation before selected-tokenizer test
evaluation. Source revisions, split rules, package versions, checksums and fit
budgets are recorded in the profile manifests and
[data handoff](../docs/data-handoff.md). No full 1 GB tokenizer study or model
quality/speed improvement is claimed by these compression and diagnostic files.

`assets-manifest.json` lists physical transfer-file sizes and SHA-256 hashes.
The profile alignment reports verify document ordering and round trips; raw
training JSONL checksums refer to decompressed bytes. The dated verification
reports record checks at that time. With the existing local assets available,
`python3 scripts/verify_assets.py` from the repository root rechecks transfer
hashes, encoding structure and the 50M capacity without regenerating data.

Raw datasets, tokenizer binaries, environments, private archives, credentials,
training logs and model checkpoints are excluded from Git. Small measured CSVs,
figures, provenance and audit reports are included. See the
[GTX handoff](../docs/gtx1060-handoff.md) for the live controller and checkpoint
locations. Job 757238 uses a frozen release; publishing this catalog does not
change its source or inputs. Formal checkpoint metrics should be collected from
the real run after completion, never inferred from smoke-test losses.
