# Data and evaluation handoff — verified 6 November 2026

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
replacement. Model training has not been launched.

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

After Djam provisions sufficient quota and a compatible project environment,
clone the verified branch on each member's own account:

```bash
git clone --branch chloe/data-eval https://github.com/KarmaToast40340/tokenizer-switch.git ~/tokenizer-switch
cd ~/tokenizer-switch
git rev-parse HEAD
```

On the laptop, after choosing the correct account through its own SSH config:

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

## Conditional Chinese 50M commands (not submitted)

Current blocker: school home quota is near 10 GB. A separate CUDA environment
installation failed with quota exceeded and was removed. Existing shared Lab5
torch is 2.6.0+cu124; do not modify it. The project target is a separately verified
CUDA 12.8 build for the planned RTX 5050 nodes. Official installation choices:
https://pytorch.org/get-started/previous-versions/ . There is no verified project
environment, cached GPT-2 weights or project data on this account yet.

Existing course jobs were checked without cancellation or modification. Resolve project storage with Djam before environment setup or transfer; do not delete unrelated coursework automatically. No tokenizer-project GPU jobs have been submitted.

Once environment, cache, data and GPU smoke verification pass, run **one** job,
then wait for completion before submitting the next. Check `squeue -u "$USER"`
and any existing `runs/*` directories first. Never rerun into an existing run
directory; the trainer appends logs and has no optimizer/RNG resume support.

```bash
cd ~/tokenizer-switch
export VENV="$HOME/APPROVED_PROJECT_ENV"  # Djam supplies the verified path
# GPT-2 model cache must be complete for offline compute nodes.
# Run each sbatch separately, with an empty account queue:
sbatch -J A_zh50 cluster/job.sbatch python -u scripts/train.py configs/train/A_keep.yaml run_name=A_keep_zh domain=wiki_zh old_tokenizer=artifacts/tokenizers/gpt2 token_budget=50000000 eval_split=validation eval_batch=1 account="$USER" commit="'$(git rev-parse --short HEAD)'"
sbatch -J B_zh50 cluster/job.sbatch python -u scripts/train.py configs/train/B_swap_fvt.yaml run_name=B_swap_fvt_zh domain=wiki_zh old_tokenizer=artifacts/tokenizers/gpt2 new_tokenizer=artifacts/tokenizers/wiki_zh70_gpt4_50k new_name=wiki_zh70_gpt4_50k token_budget=50000000 eval_split=validation eval_batch=1 account="$USER" commit="'$(git rev-parse --short HEAD)'"
sbatch -J E_zh50 cluster/job.sbatch python -u scripts/train.py configs/train/E_progressive.yaml run_name=E_progressive_zh domain=wiki_zh old_tokenizer=artifacts/tokenizers/gpt2 new_tokenizer=artifacts/tokenizers/wiki_zh70_gpt4_50k new_name=wiki_zh70_gpt4_50k token_budget=50000000 eval_split=validation eval_batch=1 account="$USER" commit="'$(git rev-parse --short HEAD)'"
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

For exact regeneration of yesterday’s English pilot ordering, use the recorded dataset revision/seed and `prepare_milestone.py --reader hf`; new Wikipedia data defaults to the sequential reader. Existing English files are reused. The legacy HF Parquet path may hang at shutdown after writing complete files; the new sequential path passed both Chinese smoke and 400 MB processing.

Storage clarification: 2 GiB assumes only one retained project checkpoint. Retaining all three Chinese 50M A/B/E checkpoints requires about 3 GiB of additional persistent project space with an approved shared environment; see `docs/cluster-preflight.md`.
