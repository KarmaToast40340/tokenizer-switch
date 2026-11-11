# Data and evaluation handoff — verified 9 November 2026

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


**Current execution path: validated GTX 1060 with unchanged DL_HW**, micro-batch
1 / accumulation 64, using the text-only launcher. See `docs/gtx1060-handoff.md`
for actual job results, checkpoint sizes and sequential-run status. Job 757238
has passed all exact trainer smoke gates and is running formal A_keep. The separate
RTX 5050 installation remains blocked; its conditional commands below are an
alternative, not the current GPU plan.

Use the persistent checkout `$HOME/tokenizer-switch-local/repo` on branch
`chloe/data-eval`. The old Desktop checkout remains a preservation copy.

## What is ready

| Profile | Selected tokenizer | GPT-2 tokens | New tokens | 50M both | 200M both |
|---|---|---:|---:|---|---|
| Code pilot, 50 MB | code70_gpt4_32k | 22,915,659 | 12,489,572 | No | No |
| Code model data, 250 MB | code70_gpt4_32k | 113,886,429 | 63,154,284 | Yes | No |
| Chinese model data, 400 MB | wiki_zh70_gpt4_50k | 283,162,335 | 91,679,979 | Yes | No |

Each model-data corpus has 1,000 validation and 1,000 test documents. Chinese
GPT-2 encoding alone is large enough for Mathieu's Chinese A at 200M. Code at
200M and Chinese B/E at 200M still require expanded, reverified corpora.
Capacity is not a promise of unique sampling: the existing sampler samples with
replacement. Full Chinese training is now active in job 757238, using the validated
GTX fallback; formal checkpoint comparisons are pending.

Both tokenizers were selected on validation using a predeclared rule: lossless on
target + English, English NSL <=1.25, minimum target NSL. All mini-grid fits use
20 MB text, smaller than the proposal's 1 GB study. Code pilot test NSL is 0.5916
on Python and 1.0813 on English. Chinese test NSL is 0.3035 on Chinese and 1.0761
on English. These measure compression, not model quality.

Chinese 50k beats Identity 32k by only 0.0024 validation NSL, while Identity keeps
English closer to GPT-2 and uses fewer vocabulary rows. The frozen rule selected
50k. Its CPU fit took 15.9 s versus Identity's 99.6 s; no GPU speed advantage is
claimed. See `results/tokenizer_runtime.csv` for measured fit times and
`results/*/preparation_timing.json` for preparation times.

## Files to transfer

`results/assets-manifest.json` is the exact source/destination/size/SHA-256 list.
It contains no raw documents, weights or credentials. Approximate transfer size:
code **364.2 MiB**, Chinese **743.1 MiB**.
The `.bin` is uint16; `.idx` is int64 document offsets. Every encoded document
round-tripped against the same raw source in `alignment.json`. Do not alter the
order, cut, tokenizer or offsets independently.

| Account | Needed assets |
|---|---|
| Djam | Code GPT-2 + selected encodings, validation/test/stats, selected tokenizer, original GPT-2 tokenizer/cache |
| Chloé | Both Chinese encodings, Chinese validation/test/stats, selected tokenizer, original GPT-2 tokenizer/cache |
| Mathieu | Both code encodings for code runs (200M expansion pending); Chinese GPT-2 encoding plus Chinese validation/test/stats and original GPT-2 tokenizer/cache for Chinese A |

Do not copy raw training JSONL to the cluster. Gzipped raw training text remains
local and is transparently readable by the preparation/encoding tools. Logical
raw checksums refer to decompressed JSONL; transfer manifest hashes refer to the
actual files being transferred.

For a new account only, clone the verified branch on that member’s own account.
Chloé already has this checkout and the Chinese assets; do not clone or upload
a second copy. The 9 November origin/main integration was already up to date,
and Djam’s download fix is included. The referenced `runs-v1` tag does not exist
yet; record the exact verified feature-branch commit instead:

```bash
git clone --branch chloe/data-eval https://github.com/KarmaToast40340/tokenizer-switch.git ~/tokenizer-switch
cd ~/tokenizer-switch
git rev-parse HEAD
```

For accounts still needing transfer, on the laptop after choosing that account
through its own SSH config (Chloé’s transfer is already verified):

```bash
cd "$HOME/tokenizer-switch-local/repo"
HOST=hpclogin02.cs.nycu.edu.tw
# Chinese example, only after enough quota has been confirmed:
ssh "$HOST" 'mkdir -p ~/tokenizer-switch/data/tokens ~/tokenizer-switch/data/raw/wiki_zh ~/tokenizer-switch/artifacts/tokenizers'
scp -r data/chinese/tokens/wiki_zh "$HOST:tokenizer-switch/data/tokens/"
scp data/chinese/raw/wiki_zh/validation.jsonl data/chinese/raw/wiki_zh/test.jsonl data/chinese/raw/wiki_zh/stats.json "$HOST:tokenizer-switch/data/raw/wiki_zh/"
scp -r artifacts/chinese/tokenizers/gpt2 artifacts/chinese/tokenizers/wiki_zh70_gpt4_50k "$HOST:tokenizer-switch/artifacts/tokenizers/"
```

For code, substitute `data/code50/...`, `code`, and
`artifacts/milestone/tokenizers/code70_gpt4_32k`. Each member must use their own
account. Compare destination SHA-256 with `results/assets-manifest.json` after
copying. Mathieu's Chinese A does not need the new encoding/tokenizer, but still
needs held-out text and GPT-2 weights/tokenizer.

## Conditional RTX 5050 Chinese 50M commands (not submitted)

Current blocker: **additional persistent installation storage**. Djam confirmed
the need to update PyTorch and the user authorized a separate project environment.
The resolved torch 2.7.1+cu128 runtime needs 6.28 GiB unpacked, versus 3.90 GiB free
under the actual 10 GiB home quota. Existing files + runtime + overhead + planned
checkpoint reserve total 15.48 GiB. Installation is not yet performed. Request
20 GiB home quota or an approved persistent runtime directory with at least
10 GiB free, visible from GPU nodes. The pinned lock and installation steps are
in `docs/project-environment.md`; preserve all shared/course environments.

Chloé’s account now has `~/tokenizer-switch` on the published branch, all Chinese
transfer assets at the canonical paths above, and the pinned GPT-2 cache below.
All 13 destination hashes and seven cached GPT-2 files are verified. Home usage
after staging is 6.11 GiB under the known 10 GiB planning limit; the remaining
3.89 GiB covers the 2.9 GiB checkpoint/save/log allowance. No raw training text,
private environment or project checkpoint is stored on the cluster.

Before staging the conservative peak calculation was 4.86 GiB existing + 0.73
assets + 0.6 GPT-2 + 1.8 three final checkpoints + 0.6 saving allowance + 0.55
source/logs/headroom = **9.14 GiB**. Recheck home use before each submission.
The default quota command reports the unrelated gcs row. Querying
`quota -vs -f /net/course` confirmed the actual 10 GiB hard limit. The earlier
9.14 GiB budget above excludes a private environment and cannot support the
new installation. The private-runtime plan now needs about 15.48 GiB total.

Evidence: `results/chinese-transfer-verification-2026-11-09.json` and
`results/gpt2-cache-verification-2026-11-09.json`. The pinned weights’ SHA-256 is
`248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707`.
No course jobs/files were changed. GTX diagnostic jobs 757113 and 757153 were
submitted and completed; the latter validates the current fallback.

Once environment, cache, data and GPU smoke verification pass, run **one** job,
then wait for completion before submitting the next. Check `squeue -u "$USER"`
and any existing `runs/*` directories first. Never rerun into an existing run
directory; the trainer appends logs and has no optimizer/RNG resume support.

```bash
cd ~/tokenizer-switch
export VENV="$HOME/tokenizer-switch-env-cu128"  # only AFTER storage approval and installation; see project-environment.md
export HF_HOME="$PWD/data/hf-cache"
export BASE_MODEL="$HF_HOME/hub/models--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e"
# Confirm the separate venv exists and job.sbatch activates it:
test -r "$VENV/bin/activate"
# Revalidate without uploading or regenerating completed files:
python3 scripts/verify_assets.py --layout destination --profiles chinese
test -f "$BASE_MODEL/model.safetensors"
squeue -u "$USER"
# Submit this smoke ONLY after the separate environment has been installed and verified:
sbatch --gres=gpu:nvidia_geforce_rtx_5050:1 --time=00:15:00 -J zh_gpu_check cluster/job.sbatch python -u cluster/check_chinese_gpu.py --base-model "$BASE_MODEL" --out results/chinese_gpu_check_2026-11-09.json
# Wait for COMPLETED / exit 0 AND JSON status=passed for all three conditions.
# Diagnostic output is excluded from model comparisons; use a new output name
# for a justified retry. Then run each sbatch separately with an empty queue,
# only when its run directory does not exist:
sbatch --gres=gpu:nvidia_geforce_rtx_5050:1 -J A_zh50 cluster/job.sbatch python -u scripts/train.py configs/train/A_keep.yaml run_name=A_keep_zh domain=wiki_zh base_model="$BASE_MODEL" old_tokenizer=artifacts/tokenizers/gpt2 token_budget=50000000 eval_split=validation eval_batch=1 account="$USER" commit="'$(git rev-parse --short HEAD)'"
sbatch --gres=gpu:nvidia_geforce_rtx_5050:1 -J B_zh50 cluster/job.sbatch python -u scripts/train.py configs/train/B_swap_fvt.yaml run_name=B_swap_fvt_zh domain=wiki_zh base_model="$BASE_MODEL" old_tokenizer=artifacts/tokenizers/gpt2 new_tokenizer=artifacts/tokenizers/wiki_zh70_gpt4_50k new_name=wiki_zh70_gpt4_50k token_budget=50000000 eval_split=validation eval_batch=1 account="$USER" commit="'$(git rev-parse --short HEAD)'"
sbatch --gres=gpu:nvidia_geforce_rtx_5050:1 -J E_zh50 cluster/job.sbatch python -u scripts/train.py configs/train/E_progressive.yaml run_name=E_progressive_zh domain=wiki_zh base_model="$BASE_MODEL" old_tokenizer=artifacts/tokenizers/gpt2 new_tokenizer=artifacts/tokenizers/wiki_zh70_gpt4_50k new_name=wiki_zh70_gpt4_50k token_budget=50000000 eval_split=validation eval_batch=1 account="$USER" commit="'$(git rev-parse --short HEAD)'"
```

The selected Chinese name is **50k**, overriding the 32k placeholder in
`cluster/RUNS.md`. Training paths above use the canonical destination layout,
not local pilot directories. The only training interface change is optional
`eval_split`; periodic eval reads validation, final eval reads test. Existing
GPU training algorithms remain Mathieu's. Micro-batch 2 / accumulation 32 yields
65,536 tokens/step; the trainer rounds down 50M to 49,938,432 actual tokens.
200M needs fresh LR/progressive schedules; it is not an unchanged continuation.
Historical cluster throughput figures used other vocabularies/batches; new
tokenizer GPU throughput is unmeasured. Do not assume equal wall-clock compute.

## Evaluate real saved models and figures

```bash
python scripts/evaluate_checkpoints.py runs/A_keep_zh_50M/model runs/B_swap_fvt_zh_50M/model runs/E_progressive_zh_50M/model --domain wiki_zh --raw data/raw --batch-size 1 --speed --out results/chinese_models_50M
python scripts/collect_results.py collected --out results/runs.csv
python scripts/plot_results.py --logs runs/A_keep_zh_50M/log.jsonl runs/B_swap_fvt_zh_50M/log.jsonl runs/E_progressive_zh_50M/log.jsonl --out results/chinese_training_50M
```

E/model is post-pruning; pre-pruning eval is in its real JSONL log only. No real
model-quality or speed results exist yet. Compression figures in
`results/milestone/` and `results/chinese/` come from measured held-out outputs.

For exact regeneration of the original 5 November English pilot ordering, use the recorded dataset revision/seed and `prepare_milestone.py --reader hf`; new Wikipedia data defaults to the sequential reader. Existing English files are reused. The legacy HF Parquet path may hang at shutdown after writing complete files; the new sequential path passed both Chinese smoke and 400 MB processing.

Tokenizer and corpus scales must be stated separately: both selected tokenizers
come from **20 MB pilot fits**, not the proposed 1 GB tokenizer study. Code
250 MB and Chinese 400 MB model corpora are **50M-ready**, with exact capacities
in the table. Neither profile covers 200M for both encodings; Chinese GPT-2
alone covers Mathieu’s 200M A. No model-quality or generation-speed claim follows
from these compression/capacity results.

Current presentation figure: `results/milestone_summary.{png,pdf}`, regenerated
with `.venv/bin/python scripts/plot_milestone_summary.py` only when its source
measurements change. The 9 November handoff and GPU gate are described in
`docs/cluster-preflight.md`. Prioritize A/B/E results before the 11 November
handoff target; do not begin 200M extensions while these comparisons are pending.
