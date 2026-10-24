# Can a small model switch tokenizers?

Code for our course project: a small-scale reproduction of
[Dagan et al. (2024), *Getting the most out of your tokenizer for pre-training and domain adaptation*](https://arxiv.org/abs/2402.01035),
and a **progressive tokenizer transition** for GPT-2 small (124M) on Python code and Chinese.

| Phase | What | Status |
|---|---|---|
| 1 | Tokenizer grid (data mix, pre-tokenization, vocab size) and abrupt swap A/B/C on code | code ready |
| 2 | Progressive transition E (union vocabulary, mixed tokenization, pruning) on code and Chinese | code ready |
| 3 | Shortest-path encoding at inference (optional) | not started |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu128   # pick your CUDA build
pip install -e ".[zh,dev]"
pytest                       # 14 CPU-only tests, no download needed
```

Scripts import the package `toksw` from `src/`. Either `pip install -e .` (above) or `export PYTHONPATH=src`.

## Data

| Name | Hugging Face dataset | Used for |
|---|---|---|
| `code` | [`codeparrot/codeparrot-clean`](https://huggingface.co/datasets/codeparrot/codeparrot-clean) | Python: tokenizer training, continued pre-training, test |
| `wiki_en` | [`wikimedia/wikipedia`](https://huggingface.co/datasets/wikimedia/wikipedia), config `20231101.en` | English share of the tokenizer mixes |
| `wiki_zh` | `wikimedia/wikipedia`, config `20231101.zh`, converted to simplified with OpenCC (`t2s`) | Chinese: tokenizer training, pre-training, test |

See [`docs/DATASETS.md`](docs/DATASETS.md) for the evaluation of both datasets and the size estimates.

`scripts/prepare_data.py` streams a dataset (no full download), shuffles it with a fixed seed,
removes exact duplicates and writes:

```
data/raw/<source>/test.jsonl    first 1,000 documents: held out before any tokenizer is trained
data/raw/<source>/train.jsonl   the following documents, up to --train-mb megabytes
data/raw/<source>/stats.json
```

For code, no file of a test repository enters `train.jsonl`.

## Pipeline

```bash
export PYTHONPATH=src
# 1. data
python scripts/prepare_data.py --source code    --train-mb 1500
python scripts/prepare_data.py --source wiki_en --train-mb 1200 --min-chars 200
python scripts/prepare_data.py --source wiki_zh --train-mb 1500 --min-chars 200

# 2. tokenizers (CPU): one lever at a time around code 70% / GPT-4 regex / 32k
python scripts/tokenizer_grid.py configs/tokenizer_grid_code.yaml --dry-run   # list the runs
python scripts/tokenizer_grid.py configs/tokenizer_grid_code.yaml
python scripts/eval_tokenizers.py artifacts/tokenizers/code* --subsets code wiki_en

# 3. encode the training set once per tokenizer (uint32 .bin + .idx document offsets)
python scripts/encode_corpus.py --source code --tokenizer gpt2 --name gpt2
python scripts/encode_corpus.py --source code --tokenizer artifacts/tokenizers/code70_gpt4_32k --name code70_gpt4_32k

# 4. training (GPU)
python scripts/train.py configs/train/A_keep.yaml        token_budget=50_000_000
python scripts/train.py configs/train/B_swap_fvt.yaml    token_budget=50_000_000
python scripts/train.py configs/train/C_swap_random.yaml token_budget=50_000_000
python scripts/train.py configs/train/E_progressive.yaml token_budget=50_000_000   # rho=0.1 for the ablation

# 5. evaluation
python scripts/eval_bpb.py runs/B_swap_fvt_code_50M/model --domain code
python scripts/gen_speed.py runs/A_keep_code_50M/model runs/B_swap_fvt_code_50M/model --domain code
```

`scripts/run_phase1.sh` chains steps 1 to 4 for Phase 1. Any config key can be overridden
on the command line as `key=value` (see `configs/train/base.yaml`).

## How the conditions are implemented

| | Model vocabulary | Init of new rows | Tokenization of training sequences |
|---|---|---|---|
| A keep | GPT-2 (50,257) | none | GPT-2 |
| B swap | new | FVT | new |
| C swap | new | random (mean/std of GPT-2 rows) | new |
| E progressive | union of GPT-2 and new | FVT | new with probability `p(t) = min(1, t / (rho T))` |

* **Same data, same order.** Every tokenizer encodes the same `train.jsonl`, so document *i* is the
  same text in every `.bin`. A batch draws documents and tokenizer coins from an RNG seeded by
  `(data_seed, step, micro_step)`, and each sequence starts at a document boundary. All conditions
  therefore start their sequences on the same texts; only the encoding changes.
* **Union vocabulary** (`toksw/vocab.py`): GPT-2 ids keep their rows, tokens of the new tokenizer
  that GPT-2 lacks are appended. Byte-level tokens are compared by their vocabulary string.
* **FVT**: a new token is the mean of the GPT-2 embeddings of its GPT-2 segmentation, obtained by
  running GPT-2's BPE model directly on the byte-level token.
* **Pruning**: at the end of E, the rows of the new vocabulary are gathered into a model with exactly
  the new vocabulary. The final eval is reported before and after pruning (pruning removes old-only
  tokens from the softmax, so bpb can only improve slightly).
* **Metrics**: bits per byte = total NLL in bits / UTF-8 bytes of the test documents, comparable across
  tokenizers. Training loss is logged in bpb too. NSL and bytes per token for the tokenizer grid.
* **Token budget** counts tokens of the tokenizer being trained on, as in the paper (fixed compute).
  A better tokenizer therefore sees more text for the same budget; `log.jsonl` records both.

## Known limitations

* The Identity pre-tokenizer turns whole documents into BPE "words", which makes the HF trainer slow
  and memory-hungry; the grid trains it on 300 MB instead of 1 GB (`identity_total_mb`).
* Chinese tokens are 3 UTF-8 bytes per character, so GPT-2 sees about 3x fewer Chinese characters per
  token than a Chinese tokenizer: at equal token budget, A sees much less Chinese text than B and E.

## Layout

```
configs/                 tokenizer grids and training configs (YAML)
scripts/                 command-line entry points
src/toksw/
  sources.py             streaming loaders (Wikipedia en/zh with OpenCC, codeparrot-clean)
  pretokenizers.py       GPT-2, GPT-4, Punct and Identity pre-tokenization
  tok.py                 BPE training and loading
  vocab.py               union vocabulary, FVT, pruning
  data.py                uint32 memmaps and the mixed-tokenization sampler
  model.py               GPT-2 setup for conditions A, B, C, E
  metrics.py             NSL, bytes/token, bits per byte
  schedule.py            p(t) and learning rate
tests/                   CPU tests on toy tokenizers and a tiny GPT-2
```
