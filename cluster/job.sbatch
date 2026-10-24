#!/bin/bash
#SBATCH -A dlp
#SBATCH -p class_nodes
#SBATCH -q dlp_qos
#SBATCH --exclude=cmpt[047-050,053-055,066-069,082-085,096-099]   # GTX 1060 nodes: driver too old for cu128 torch
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH -t 08:00:00               # the default is 5 min
#SBATCH -J toksw
#SBATCH -o logs/%x_%j.out

# Runs any command from the repository root on one GPU node, e.g.
#   sbatch cluster/job.sbatch python scripts/train.py configs/train/A_keep.yaml token_budget=50_000_000
#   sbatch -J bench cluster/job.sbatch python cluster/bench_throughput.py
# Compute nodes run offline: cache GPT-2 on the login node first (see cluster/README.md).
export LD_PRELOAD=/lib/x86_64-linux-gnu/libcuda.so.1   # requested by the NYCU guide
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # without it, micro 4 x 1024 already fails on 8 GB
export PYTHONPATH=src
source "${VENV:-$HOME/venv_lab2}/bin/activate"
cd "$SLURM_SUBMIT_DIR"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import torch; print('torch', torch.__version__, '| CUDA', torch.cuda.is_available())"
"$@"
