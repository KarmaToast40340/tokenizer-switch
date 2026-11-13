# Code 50M checkpoint comparison (A, B, C, E)

Corrected code 50M; dropout 0.1; 20 MB tokenizer pilot; one seed

Four runs on Djam's account, 13 November 2026, RTX 5050, from commit
`6b9db6d36a6280bf520c2469bc42d45b0a4f41e0` with `cluster/djam_code50.sbatch`
(protocol in `docs/djam-code50-handoff.md`). Same protocol as the corrected Chinese
runs: 762 steps x 65,536 tokens = 49,938,432 training tokens, micro-batch 1 x
accumulation 64, seeds 0/1234, tokenizer `code70_gpt4_32k`, E with rho 0.5 and pruning.

| Condition | Slurm job | Test bpb (lower) | Generation characters/s | Generation tokens/s | Trainer hours | Training bytes read |
|---|---|---:|---:|---:|---:|---:|
| A keep GPT-2 | 774132 | 0.72643 | 374.9 | 134.2 | 1.18 | 114.7 MB |
| B FVT swap | 774666 | **0.69981** | 532.9 | 131.9 | 0.92 | 203.9 MB |
| C swap, new tokens random-init | 775336 | 0.78479 | 549.5 | 132.3 | 0.91 | 203.9 MB |
| E progressive, pruned | 775737 | 0.71643 | 500.8 | 135.5 | 1.19 | 181.6 MB |

## Main finding

On this run, the FVT swap (B) has the lowest observed test bpb (0.700), ahead of E (0.716),
A (0.726) and C (0.785). B, E and A come in the same order as in the corrected Chinese runs;
C was only run on code. E does not beat B, but it ends below A, and pruning leaves its score
unchanged. C ends 12% above B although the two differ only in how the 14,009 new tokens are
initialized, and it also ends above A: at this budget, the swap helps only with FVT.

## What was checked

- All four jobs ended `COMPLETED / 0:0`. Each log has 762 training-mode checks and
  `model_training=true` on all 77 logged training events.
- Saved-checkpoint evaluation (job 775937, `eval_code50.sbatch`) covers the same 1,000
  test documents, 10,891,645 UTF-8 bytes, batch 1, context 1024. The saved-model bpb
  equals the trainer's final score exactly for all four conditions (post-pruning for E).
- E before pruning scores 0.71664 on the new tokenizer and 0.87731 on the old one;
  after pruning 0.71643. Pruning does not change the score.
- Weight hashes are identical before and after evaluation and speed measurement, and
  between the cluster and the copy kept off-cluster. They are listed in `comparison.csv`.
- Speed: 20 shared 200-character test prefixes, 128 greedy new tokens, synchronized
  CUDA timing, warm-up excluded (`scripts/gen_speed.py`).

## Limits

- One seed, no confidence interval. Tokenizer fitted on a 20 MB pilot.
- Equal token budgets are not equal text: B and C read 1.78 times the bytes A reads,
  E 1.58 times. The advantage of B over A mixes a tokenizer effect and a data effect.
- C is not a from-scratch embedding matrix: the 17,991 tokens shared with GPT-2 keep
  their row in both B and C. Only the 14,009 new tokens are drawn at random.
- Speed is measured on an RTX 5050; the Chinese speeds were measured on a GTX 1060.
  Compare speeds across conditions of one domain, not across domains.
  Characters/s follows characters per token; tokens/s is the same in all conditions.
- Trainer hours include periodic and final evaluation, not the step-0 evaluation,
  pruning or saving. E evaluates two tokenizers, so its hours are not equal compute.
- The bpb axis of `comparison.png` starts at 0.6, which magnifies the differences.
- `training.png` plots validation bpb (200 documents) against training tokens, not bytes.

## Files

- `A/`, `B/`, `C/`, `E/`: each run's `config.json` and `log.jsonl`. The logs are
  unmodified. In the configs, the cluster login was removed from `base_model` (home
  path replaced by `~`) and `account` (set to `djam`), because this repository is public.
- `*_bpb.json`, `speed.csv`: unmodified outputs of `scripts/eval_bpb.py` and
  `scripts/gen_speed.py`. `eval_code50.sbatch` is the job that produced them.
- `runs.csv`: `python scripts/collect_results.py results/code50_djam_corrected_dropout01`.
- `comparison.{csv,png,pdf}`, `training.{png,pdf}` (`scripts/plot_results.py --logs`).
- `code_and_chinese_50M.csv`: these four rows next to the three corrected Chinese rows
  copied from `results/chinese766353_corrected_dropout01/comparison.csv`.
- `manifest.json`: size and SHA-256 of every file above.

Checkpoints stay outside Git. 200M runs, rho ablations and the innovation are not done.
