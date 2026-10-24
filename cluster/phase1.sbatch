#!/bin/bash
#SBATCH -A dlp
#SBATCH -p class_nodes
#SBATCH -q dlp_qos
#SBATCH --exclude=cmpt[047-050,053-055,066-069,082-085,096-099]
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH -t 08:00:00
#SBATCH -J phase1
#SBATCH -o logs/%x_%j.out

# Phase 1 runs A, B, C one after the other in a single job (the cluster allows one job per user).
#   sbatch cluster/phase1.sbatch 50000000
#   sbatch cluster/phase1.sbatch 200000000 code70_gpt4_32k
# A run whose model/ folder already exists is skipped, so resubmitting continues where it stopped.
BUDGET=${1:-50000000}
NEW=${2:-code70_gpt4_32k}
export LD_PRELOAD=/lib/x86_64-linux-gnu/libcuda.so.1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # without it, micro 4 x 1024 already fails on 8 GB
export PYTHONPATH=src
source "${VENV:-$HOME/venv_lab2}/bin/activate"
cd "$SLURM_SUBMIT_DIR"
set -euo pipefail
for cond in A_keep B_swap_fvt C_swap_random; do
  run="runs/${cond}_code_$((BUDGET / 1000000))M"
  if [ -d "$run/model" ]; then echo "[skip] $run"; continue; fi
  python -u scripts/train.py configs/train/$cond.yaml token_budget=$BUDGET \
      new_tokenizer=artifacts/tokenizers/$NEW new_name=$NEW
done
