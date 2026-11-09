#!/bin/bash
#SBATCH -A dlp
#SBATCH -p class_nodes
#SBATCH -q dlp_qos
#SBATCH --gres=gpu:nvidia_geforce_gtx_1060_6gb:1
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -t 00:20:00
#SBATCH -J zh_gtx_assess
#SBATCH -o logs/%x_%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export LD_PRELOAD=/lib/x86_64-linux-gnu/libcuda.so.1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME="$PWD/data/hf-cache"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
exec /opt/conda/miniconda3/envs/DL_HW/bin/python -u cluster/assess_gtx1060.py --out "results/gtx1060_assessment_${SLURM_JOB_ID}.json"
