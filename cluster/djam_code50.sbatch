#!/bin/bash
#SBATCH -A dlp
#SBATCH -p class_nodes
#SBATCH -q dlp_qos
#SBATCH --gres=gpu:nvidia_geforce_rtx_5050:1
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -t 12:00:00
#SBATCH -J code50
#SBATCH -o logs/%x_%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
: "${DJAM_PYTHON:?Activate the working RTX environment, then export DJAM_PYTHON=$(command -v python)}"
: "${DJAM_BASE_MODEL:?Resolve the existing pinned local GPT-2 snapshot before submission}"
case "$DJAM_PYTHON" in /*) ;; *) echo 'DJAM_PYTHON must be an absolute executable path' >&2; exit 1;; esac
test -x "$DJAM_PYTHON"
test -f "$DJAM_BASE_MODEL/model.safetensors"
condition=${1:?Supply A_keep, B_swap_fvt, C_swap_random or E_progressive}
case "$condition" in A_keep|B_swap_fvt|C_swap_random|E_progressive) ;; *) echo 'Unsupported condition' >&2; exit 1;; esac
export LD_PRELOAD=/lib/x86_64-linux-gnu/libcuda.so.1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
"$DJAM_PYTHON" -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name())'
# No environment activation, installation or package mutation in this wrapper.
# All conditions use the same conservative microbatch/effective token batch.
exec "$DJAM_PYTHON" -u cluster/text_only.py scripts/train.py "configs/train/$condition.yaml" \
  "run_name=${condition}_code_dropout01" domain=code \
  "base_model=$DJAM_BASE_MODEL" old_tokenizer=artifacts/tokenizers/gpt2 old_name=gpt2 \
  new_tokenizer=artifacts/tokenizers/code70_gpt4_32k new_name=code70_gpt4_32k \
  tokens_dir=data/tokens raw_dir=data/raw out=runs/djam_code50_corrected \
  token_budget=50000000 block_size=1024 micro_batch=1 grad_accum=64 \
  seed=0 data_seed=1234 expected_dropout=0.1 eval_split=validation eval_batch=1 \
  eval_docs=200 eval_every=100 "account=$USER" "commit='$(git rev-parse HEAD)'"
