#!/usr/bin/env bash
# Phase 1 end to end: data -> tokenizer grid -> encoding -> runs A, B, C at 50M and 200M tokens.
# Steps do not check for existing outputs (except the tokenizer grid): comment out what you have already run.
set -euo pipefail
export PYTHONPATH=src

# 1. Data (streams from the HF hub; held-out test sets are written first)
python scripts/prepare_data.py --source code    --train-mb 1500
python scripts/prepare_data.py --source wiki_en --train-mb 1200 --min-chars 200

# 2. Tokenizer grid (CPU) and compression table
python scripts/tokenizer_grid.py configs/tokenizer_grid_code.yaml
python scripts/eval_tokenizers.py artifacts/tokenizers/code* --subsets code wiki_en \
    --out results/tokenizers_code.csv

# 3. Encode the code training set with GPT-2 and with the chosen tokenizer
NEW=code70_gpt4_32k
python scripts/encode_corpus.py --source code --tokenizer gpt2 --name gpt2
python scripts/encode_corpus.py --source code --tokenizer artifacts/tokenizers/$NEW --name $NEW

# 4. Training runs (GPU)
for budget in 50000000 200000000; do
  for cond in A_keep B_swap_fvt C_swap_random; do
    python scripts/train.py configs/train/$cond.yaml token_budget=$budget \
        new_tokenizer=artifacts/tokenizers/$NEW new_name=$NEW
  done
done
