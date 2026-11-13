#!/bin/bash
#SBATCH -A dlp
#SBATCH -p class_nodes
#SBATCH -q dlp_qos
#SBATCH --gres=gpu:nvidia_geforce_rtx_5050:1
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -t 02:00:00
#SBATCH -J eval_code50
#SBATCH -o logs/%x_%j.out
# Saved-checkpoint test bpb and generation speed for the four code 50M runs.
# Same settings as scripts/evaluate_checkpoints.py --batch-size 1 --speed --prompt-chars 200 --new-tokens 128
# (the Chinese protocol), called through cluster/text_only.py like the training wrapper.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
test -x "$DJAM_PYTHON"
export LD_PRELOAD=/lib/x86_64-linux-gnu/libcuda.so.1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=runs/djam_code50_corrected
O=results/djam_code50_checkpoints
mkdir -p "$O"
"$DJAM_PYTHON" -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name())'
git rev-parse HEAD
cks=""
for c in A_keep B_swap_fvt C_swap_random E_progressive; do
  ck=$R/${c}_code_dropout01_50M/model
  sha256sum "$ck/model.safetensors"
  "$DJAM_PYTHON" -u cluster/text_only.py scripts/eval_bpb.py "$ck" --raw data/raw --domain code \
    --batch-size 1 --block-size 1024 --out "$O/${c}_bpb.json"
  cks="$cks $ck"
done
"$DJAM_PYTHON" -u cluster/text_only.py scripts/gen_speed.py $cks --raw data/raw --domain code \
  --prompt-chars 200 --new-tokens 128 --out "$O/speed.csv"
for c in A_keep B_swap_fvt C_swap_random E_progressive; do sha256sum "$R/${c}_code_dropout01_50M/model/model.safetensors"; done
