# Who runs what

Runs are split by domain, not by condition: split by letter, C would take ~1 GPU-hour and E ~18,
and the account running E would hold up everyone. With this split each account runs ~13 hours and
only stores the data of its own domain. One job at a time per account; everyone submits on their
own account (no shared accounts or SSH keys).

| Account | Runs | GPU hours | Data on the cluster |
|---|---|---|---|
| Djam | code 50M A, B, C, E (validates the pipeline) · ρ ablation · innovation on code · code 200M E | ~14 h | code |
| Chloé | Chinese 50M A, B, E · Chinese 200M B, E · innovation on Chinese | ~13 h | Chinese |
| Mathieu | code 200M A, B · Chinese 200M A | ~12.5 h | code; Chinese in GPT-2 only |
| Spare | reruns, on the first free account | ~6 h | |

Order: Djam's code 50M runs come first; the 200M runs start once they look right (~19 Nov);
the Chinese runs start when the Chinese tokenizer is ready.

## Same code, same data everywhere

One comparison (say A vs E at 200M) runs on two accounts, so the only difference between them must
be the condition:

* the cluster copy is a `git clone`, checked out at the tag `runs-v1` (set on `main` once the
  smoke tests pass); every run records the commit it ran from (`commit=` below);
* the `.bin` files are encoded once, on the laptop, and copied as they are. Check after each copy:

```bash
sha256sum data/tokens/*/*.bin      # on the laptop and on the cluster: the lines must match
```

## Which encoded data for which budget

The sampler draws each training sequence from a random document start,
`rng.integers(0, n_docs)`, so **the draws depend on how many documents were encoded**. All the
conditions of one comparison (A, B, C, E at the same budget and domain) must therefore use `.bin`
files encoded with the **same `--max-mb`**. The 50M and 200M runs may use different encodings.

Chloé's current encodings (see `docs/data-handoff.md` on `chloe/data-eval`) are sized for 50M:

| Domain | New tokenizer | Text | GPT-2 tokens | New tokens | Use |
|---|---|---|---|---|---|
| code | `code70_gpt4_32k` | 250 MB | 113.9M | 63.2M | 50M runs |
| Chinese | `wiki_zh70_gpt4_50k` | 400 MB | 283.2M | 91.7M | 50M runs |

At 200M, these would not crash, but B and E would see each document ~3.2 times on code (~2.2 times
on Chinese) while A, whose GPT-2 encoding is longer, sees it ~1.8 times (~0.7 times on Chinese):
different amounts of repetition across the conditions we compare.
For the 200M runs, encode more text so that the new tokenizer has at least 200M tokens. Estimates
from the counts above:

| Domain | `--max-mb` | GPT-2 `.bin` | New `.bin` | On disk |
|---|---|---|---|---|
| code | ~800 | ~364M tokens, ~0.73 GB | ~202M tokens, ~0.40 GB | ~1.1 GB |
| Chinese | ~900 (close to the ~1 GB prepared) | ~637M tokens, ~1.27 GB | ~206M tokens, ~0.41 GB | ~1.7 GB |

Mathieu's Chinese A at 200M must use the same 900 MB `gpt2.bin` as Chloé's B and E (not a smaller
one), for the reason above. Check the free quota before copying (`bash cluster/setup_account.sh`
prints it).

E on Chinese uses a larger union vocabulary than the 80k measured on code (GPT-2 plus a 50k
tokenizer), so its speed and memory on the RTX 5050 are not measured yet: check the first log lines.

## Once per account

```bash
git clone https://github.com/KarmaToast40340/tokenizer-switch.git ~/tokenizer-switch
cd ~/tokenizer-switch && git checkout runs-v1
bash cluster/setup_account.sh
```

Copy your data from the laptop (repository root; `USER` = your cluster login):

```bash
# code (Djam, Mathieu)
scp -r data/tokens/code USER@hpclogin02.cs.nycu.edu.tw:tokenizer-switch/data/tokens/
ssh USER@hpclogin02.cs.nycu.edu.tw "mkdir -p tokenizer-switch/data/raw/code tokenizer-switch/data/raw/wiki_zh"
scp data/raw/code/test.jsonl data/raw/code/stats.json USER@hpclogin02.cs.nycu.edu.tw:tokenizer-switch/data/raw/code/
scp -r artifacts/tokenizers/code70_gpt4_32k USER@hpclogin02.cs.nycu.edu.tw:tokenizer-switch/artifacts/tokenizers/
# Chinese (Chloé: everything; Mathieu: only data/tokens/wiki_zh/gpt2.* and the test set)
scp -r data/tokens/wiki_zh USER@hpclogin02.cs.nycu.edu.tw:tokenizer-switch/data/tokens/
scp data/raw/wiki_zh/test.jsonl data/raw/wiki_zh/stats.json USER@hpclogin02.cs.nycu.edu.tw:tokenizer-switch/data/raw/wiki_zh/
scp -r artifacts/tokenizers/wiki_zh70_gpt4_50k USER@hpclogin02.cs.nycu.edu.tw:tokenizer-switch/artifacts/tokenizers/
```

Do not copy `data/raw/*/train.jsonl`: training reads the `.bin` files, and it would eat ~1 GB of quota.

## Submitting

Run every command from `~/tokenizer-switch` on the login node. First, in each new shell:

```bash
TAG="account=$USER commit='$(git rev-parse --short HEAD)'"
ZH="domain=wiki_zh new_tokenizer=artifacts/tokenizers/wiki_zh70_gpt4_50k new_name=wiki_zh70_gpt4_50k"
```

`account` and `commit` are extra config keys: `train.py` ignores them but writes them to the run's
`config.json`, where `scripts/collect_results.py` finds them. The quotes around the commit keep YAML
from reading a hash such as `0123456` as a number.

One run per job (`job.sbatch` allows 8 h; the longest run takes ~4.7 h). Submit the next run when
the previous one is done (`squeue -u $USER`).

If a run fails, rename its folder before resubmitting it, e.g.
`mv runs/B_swap_fvt_code_50M runs/B_swap_fvt_code_50M.failed1`. Today `train.py` appends to an
existing `log.jsonl`, which would mix the two attempts. Once the trainer hardening from
`chloe/data-eval` is merged, it refuses an existing run folder instead.

### Djam

```bash
sbatch -J A_code50 cluster/job.sbatch python -u scripts/train.py configs/train/A_keep.yaml token_budget=50_000_000 $TAG
sbatch -J B_code50 cluster/job.sbatch python -u scripts/train.py configs/train/B_swap_fvt.yaml token_budget=50_000_000 $TAG
sbatch -J C_code50 cluster/job.sbatch python -u scripts/train.py configs/train/C_swap_random.yaml token_budget=50_000_000 $TAG
sbatch -J E_code50 cluster/job.sbatch python -u scripts/train.py configs/train/E_progressive.yaml token_budget=50_000_000 $TAG
# ρ ablation (ρ = 0.5 is the main E run above; ρ = 0 is the union vocabulary with an abrupt swap)
sbatch -J E_rho0 cluster/job.sbatch python -u scripts/train.py configs/train/E_progressive.yaml run_name=E_rho0_code rho=0 token_budget=50_000_000 $TAG
sbatch -J E_rho025 cluster/job.sbatch python -u scripts/train.py configs/train/E_progressive.yaml run_name=E_rho025_code rho=0.25 token_budget=50_000_000 $TAG
sbatch -J E_rho075 cluster/job.sbatch python -u scripts/train.py configs/train/E_progressive.yaml run_name=E_rho075_code rho=0.75 token_budget=50_000_000 $TAG
sbatch -J E_code200 cluster/job.sbatch python -u scripts/train.py configs/train/E_progressive.yaml token_budget=200_000_000 $TAG
```

### Chloé

```bash
sbatch -J A_zh50 cluster/job.sbatch python -u scripts/train.py configs/train/A_keep.yaml run_name=A_keep_zh $ZH token_budget=50_000_000 $TAG
sbatch -J B_zh50 cluster/job.sbatch python -u scripts/train.py configs/train/B_swap_fvt.yaml run_name=B_swap_fvt_zh $ZH token_budget=50_000_000 $TAG
sbatch -J E_zh50 cluster/job.sbatch python -u scripts/train.py configs/train/E_progressive.yaml run_name=E_progressive_zh $ZH token_budget=50_000_000 $TAG
sbatch -J B_zh200 cluster/job.sbatch python -u scripts/train.py configs/train/B_swap_fvt.yaml run_name=B_swap_fvt_zh $ZH token_budget=200_000_000 $TAG
sbatch -J E_zh200 cluster/job.sbatch python -u scripts/train.py configs/train/E_progressive.yaml run_name=E_progressive_zh $ZH token_budget=200_000_000 $TAG
```

### Mathieu

```bash
sbatch -J A_code200 cluster/job.sbatch python -u scripts/train.py configs/train/A_keep.yaml token_budget=200_000_000 $TAG
sbatch -J B_code200 cluster/job.sbatch python -u scripts/train.py configs/train/B_swap_fvt.yaml token_budget=200_000_000 $TAG
sbatch -J A_zh200 cluster/job.sbatch python -u scripts/train.py configs/train/A_keep.yaml run_name=A_keep_zh $ZH token_budget=200_000_000 $TAG
```

The innovation runs (#3 or #4) get their commands once the variant is merged.

## Bringing results back

After each run, from the laptop, copy the run folder into `collected/<account>/` and free the
quota on the cluster (the weights take ~0.5 GB per run until they are saved in fp16):

```bash
scp -r USER@hpclogin02.cs.nycu.edu.tw:tokenizer-switch/runs/E_progressive_code_50M collected/USER/
python scripts/collect_results.py collected --out results/runs.csv
```

`collected/` stays out of git (it holds weights); share `results/runs.csv` with the group.
