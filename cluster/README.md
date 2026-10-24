# Running on the NYCU cluster

Constraints that shape this setup: one Slurm job per user at a time, a 10 GB home quota,
RTX 5050 nodes (8 GB) and compute nodes without internet.

## Where each step runs

| Step | Where | Why |
|---|---|---|
| `prepare_data.py`, `tokenizer_grid.py`, `eval_tokenizers.py`, `encode_corpus.py` | a laptop (CPU) | needs internet and disk, no GPU |
| `train.py`, `eval_bpb.py`, `gen_speed.py` | cluster, through `sbatch` | GPU |

Only the encoded corpora, the test sets and the chosen tokenizers go to the cluster.

## On a laptop

```bash
export PYTHONPATH=src
python scripts/prepare_data.py --source code    --train-mb 1000 --chunk-chars 3000
python scripts/prepare_data.py --source wiki_en --train-mb 1000 --min-chars 200
python scripts/prepare_data.py --source wiki_zh --train-mb 1000 --min-chars 200 --chunk-chars 1000
python scripts/tokenizer_grid.py configs/tokenizer_grid_code.yaml
python scripts/eval_tokenizers.py artifacts/tokenizers/code* --subsets code wiki_en --out results/tokenizers_code.csv
# encode the same first 700 MB of text with both tokenizers (enough for a 200M-token budget)
python scripts/encode_corpus.py --source code --tokenizer gpt2 --name gpt2 --max-mb 700
python scripts/encode_corpus.py --source code --tokenizer artifacts/tokenizers/code70_gpt4_32k --name code70_gpt4_32k --max-mb 700
```

`--chunk-chars` matters: training windows start at a document start, so without it only the first
~1024 tokens of each file (imports, license headers) would ever be trained on. Code files average
about 10 KB, i.e. ~4,700 GPT-2 tokens.

## Disk budget on the cluster (estimates)

| Item | Size |
|---|---|
| `~/venv_lab2` (reused, do not create a new venv) | ~5.5 GB |
| GPT-2 weights in `~/.cache/huggingface` | ~0.55 GB |
| `data/tokens/code/` (GPT-2 + new, uint16, 700 MB of text) | ~1.1 GB |
| `data/tokens/wiki_zh/` (same, Phase 2) | ~1.5 GB |
| `data/raw/*/test.jsonl`, tokenizers | < 50 MB |
| checkpoints: 0.5 GB per run | keep only the ones you need |

Storing the full 1.5 GB corpora as uint32, as in the first version of the pipeline, would take
about 10 GB of tokens alone.

## Setup (once, on the login node)

```bash
cd ~/tokenizer-switch
source ~/venv_lab2/bin/activate
pip install --no-cache-dir -e ".[zh]"      # keeps the venv's torch (cu128, needed for the RTX 5050)
python -c "from transformers import GPT2LMHeadModel; from tokenizers import Tokenizer; GPT2LMHeadModel.from_pretrained('gpt2'); Tokenizer.from_pretrained('gpt2')"
mkdir -p logs
```

Upload from the laptop, from the repository root (`USER` = your cluster login):

```bash
ssh USER@hpclogin01.cs.nycu.edu.tw "mkdir -p tokenizer-switch/data/raw/code tokenizer-switch/artifacts/tokenizers"
scp -r data/tokens USER@hpclogin01.cs.nycu.edu.tw:tokenizer-switch/data/
scp data/raw/code/test.jsonl data/raw/code/stats.json USER@hpclogin01.cs.nycu.edu.tw:tokenizer-switch/data/raw/code/
scp -r artifacts/tokenizers/code70_gpt4_32k USER@hpclogin01.cs.nycu.edu.tw:tokenizer-switch/artifacts/tokenizers/
```

## Jobs

```bash
sbatch -J bench cluster/job.sbatch python cluster/bench_throughput.py --micro 4 --vocab 80000
sbatch cluster/phase1.sbatch 50000000          # A, B, C at 50M tokens, one after the other
squeue -u $USER
tail -f logs/phase1_*.out
```

Each member can run one job at a time on their own account, so the three of us can run three
conditions in parallel with `cluster/job.sbatch python scripts/train.py ...`.
