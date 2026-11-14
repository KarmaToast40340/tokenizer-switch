# Code 50M: rho ablation of the progressive transition (E)

Corrected code 50M; dropout 0.1; 20 MB tokenizer pilot; one seed

Three runs on Djam's account, 13 and 14 November 2026, RTX 5050, from commit
`6b9db6d36a6280bf520c2469bc42d45b0a4f41e0`, the same commit, data and protocol as the
four runs in `results/code50_djam_corrected_dropout01` (762 steps x 65,536 tokens,
micro-batch 1 x accumulation 64, seeds 0/1234, tokenizer `code70_gpt4_32k`, pruning at
the end). The job script `djam_code50_rho.sbatch` runs the command of
`cluster/djam_code50.sbatch E_progressive` with two changes only: `rho` and the run name.

In E, each sequence uses the new tokenizer with probability `min(1, step / (rho * T))`.
With rho 0 every sequence uses the new tokenizer from step 0: the union vocabulary of E
with the abrupt swap of B. A, B and E rho 0.5 are copied from the earlier folder as references.

| Run | Slurm job | Test bpb (lower) | Minus B | Old tokenizer, before pruning | Training bytes read |
|---|---|---:|---:|---:|---:|
| B FVT swap | 774666 | **0.69981** | | | 203.9 MB |
| E rho 0 | 776145 | 0.70015 | +0.0003 | 1.18750 | 203.9 MB |
| E rho 0.25 | 776550 | 0.70693 | +0.0071 | 1.04906 | 192.7 MB |
| E rho 0.5 | 775737 | 0.71643 | +0.0166 | 0.87731 | 181.6 MB |
| E rho 0.75 | 777515 | 0.72750 | +0.0277 | 0.79086 | 170.2 MB |
| A keep GPT-2 | 774132 | 0.72643 | +0.0266 | | 114.7 MB |

## Main finding

With rho 0, E ends 0.0003 bpb away from B (0.70015 against 0.69981). With one seed this
may be noise or a small cost of the union vocabulary; it cannot be a ramp cost, because
rho 0 has no ramp. The score then increases monotonically with rho, with no sudden jump
(0.70015, 0.70693, 0.71643, 0.72750). Finally, rho 0.75 ends at the level of A
(0.72750 against 0.72643).

## What was checked

- The three jobs ended `COMPLETED / 0:0`. Each `config.json` records the intended `rho`,
  and each log has 762 training-mode checks.
- Each log has one `final_eval` and one `final_eval_pruned` on the 1,000 test documents.
  Pruning changes the new-tokenizer score by at most 0.0006 bpb.
- The saved weights have the same SHA-256 on the cluster and on the copy kept off-cluster
  (listed in `rho_ablation.csv`). Unlike A, B, C and E rho 0.5, these three checkpoints
  were not re-evaluated by a separate job and their generation speed was not measured.

## Limits

- One seed per point, no confidence interval. The 0.0003 bpb between E rho 0 and B is
  not a measured difference.
- Equal token budgets are not equal text: the longer the ramp, the fewer bytes are read
  (170 MB at rho 0.75, 204 MB at rho 0). The middle panel of `rho_ablation.png` plots
  validation bpb against bytes read for this reason.
- In that panel, the bytes read at an evaluation step are taken from the training event
  logged at the same step. The curves use validation (200 documents); the table uses test.
- The right panel shows the old-tokenizer score, which E does not keep: it rises once
  all sequences use the new tokenizer, and the pruned model cannot use it at all.
- Code only. The Chinese runs have no rho ablation.

## Files

- `rho0/`, `rho025/`, `rho075/`: each run's `config.json` and `log.jsonl`. The logs are
  unmodified. In the configs, the cluster login was removed from `base_model` (home path
  replaced by `~`) and `account` (set to `djam`), because this repository is public.
- `djam_code50_rho.sbatch`: the job script, as submitted
  (`sbatch djam_code50_rho.sbatch 0 rho0`, `0.25 rho025`, `0.75 rho075`).
- `runs.csv`: `python scripts/collect_results.py results/code50_djam_rho_ablation_dropout01`.
- `plot_rho.py`: writes `rho_ablation.{csv,png,pdf}` from the logs of this folder and of
  `results/code50_djam_corrected_dropout01`.
- `manifest.json`: size and SHA-256 of every file above.

Checkpoints stay outside Git. 200M runs and the innovation are not done.
