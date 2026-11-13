# Djam code50 handoff — 13 November 2026

Use the completed PR #4 code on **main after merge**, in an isolated persistent
checkout. Do not launch from the older `djam/runs-update` branch without bringing
in the corrected trainer. That branch's relevant run-plan documentation has been
reviewed and integrated; its download fix was already present.

## Exact asset scope

The package is selected only from `results/assets-manifest.json` rows whose
profile is `code50`: **13 files, 381,891,424 uncompressed asset bytes**.
Source paths are resolved through the persistent repo's data/artifact links;
archive members are regular files at the manifest's **destination** paths.

| Contents | Manifest source | Transfer destination |
|---|---|---|
| Both encoded corpora (.bin/.idx/.json) | `data/code50/tokens/code/` | `data/tokens/code/` |
| validation/test/stats | `data/code50/raw/code/` | `data/raw/code/` |
| Frozen 32k tokenizer | `artifacts/milestone/tokenizers/code70_gpt4_32k/` | `artifacts/tokenizers/code70_gpt4_32k/` |
| Original GPT-2 tokenizer | `artifacts/milestone/tokenizers/gpt2/` | `artifacts/tokenizers/gpt2/` |

No raw training JSONL, Chinese data, environments, credentials, model weights,
cache directories or machine-specific symlinks are shipped. The README,
filtered manifest, `PACKAGE.json`, standalone verifier and SHA256SUMS are small
additional package metadata. The large archive remains outside Git.

The model corpus has **250,012,440 UTF-8 training bytes**, **95,761 encoded
chunks**, **113,886,429 GPT-2 tokens / 63,154,284 selected-tokenizer tokens**.
Validation and test each contain **1,000 documents**. Both encodings cover 50M;
neither covers 200M. Sampling is with replacement, not unique coverage.
The tokenizer was fitted on a **20 MB pilot mix**, not the 250 MB corpus and
not the proposed 1 GB study. Selection was frozen on validation.

## Obtain and verify the package

The generated package README pins the exact verified code commit; the final
handoff receipt records the archive path/hash/upload result. Do not describe the
archive as shared unless upload and access are actually verified.

On Djam's account, save the downloaded archive and matching `.sha256` sidecar
under `~/tokenizer-switch-handoff/`, then:

```bash
cd ~/tokenizer-switch-handoff
sha256sum -c code50-djam-2026-11-13.tar.gz.sha256
tar -xzf code50-djam-2026-11-13.tar.gz
cd code50-djam-2026-11-13
sha256sum -c SHA256SUMS
python3 tools/verify_assets.py --manifest assets-manifest.json --root . \
  --layout destination --profiles code50
```

Every asset size/hash must pass. The verification is read-only; it does not
refit or encode data. If extraction target exists, inspect it first rather than
overwriting another transfer.

## Corrected code and existing RTX environment

Do not change or install Djam's already working RTX 5050 environment. Activate
it with **his existing activation command**. The wrapper captures and uses that
exact Python executable; it does not source `venv_lab2` or install anything.

Use a separate persistent worktree so existing jobs/checkouts are untouched:

```bash
cd ~/tokenizer-switch
git fetch origin main
git worktree add --detach ~/tokenizer-switch-code50 origin/main
cd ~/tokenizer-switch-code50
git rev-parse HEAD
test ! -e data && test ! -e artifacts
ln -s "$HOME/tokenizer-switch-handoff/code50-djam-2026-11-13/data" data
ln -s "$HOME/tokenizer-switch-handoff/code50-djam-2026-11-13/artifacts" artifacts
python3 scripts/verify_assets.py --layout destination --profiles code50
mkdir -p logs
```

These are local ignored links to the single verified payload, avoiding another
381 MB copy. They are not archived or committed. Do not run account setup or
environment-install scripts for this handoff.

After activating the existing working environment:

```bash
export DJAM_PYTHON="$(command -v python)"
export DJAM_BASE_MODEL="$(HF_HUB_OFFLINE=1 "$DJAM_PYTHON" -c 'from huggingface_hub import snapshot_download; print(snapshot_download("gpt2", revision="607a30d783dfa663caf39e06633721c8d4cfcd7e", local_files_only=True))')"
test -x "$DJAM_PYTHON"
test -f "$DJAM_BASE_MODEL/model.safetensors"
squeue -u "$USER"
```

If the pinned cache is missing, stop and resolve only the missing model files
on the login node; **do not reinstall the working environment**. The cache is
not included in this asset archive. `local_files_only=True` prevents accidental
downloads in the commands above.

## Exact launches: one job at a time

Inspect existing Djam runs first; do not repeat completed comparable runs. The
account QoS permits one submitted job. Execute **one** line, wait for
`COMPLETED / 0:0` and inspect its outputs, then execute the next line:

```bash
sbatch --export=ALL -J A_code50_corrected cluster/djam_code50.sbatch A_keep
sbatch --export=ALL -J B_code50_corrected cluster/djam_code50.sbatch B_swap_fvt
sbatch --export=ALL -J C_code50_corrected cluster/djam_code50.sbatch C_swap_random
sbatch --export=ALL -J E_code50_corrected cluster/djam_code50.sbatch E_progressive
```

Do not paste all four at once. We did not submit any job to Djam's account.
The wrapper requests RTX 5050 and a **12-hour limit per condition**, not a
measured runtime promise. All four conditions use block 1024, BF16 micro 1 ×
accumulation 64, seeds 0/1234, the same optimizer/LR settings, dropout 0.1,
validation-only periodic eval (200 docs, batch 1, every 100 steps) and the full
1,000-document test set at completion. E uses rho 0.5 and pruning.

Budget is **50,000,000 requested tokens**, rounded to **762 × 65,536 =
49,938,432 actual tokens** per condition. Microbatch is deliberately conservative
and matched across conditions; do not silently mix these with old different-
batch or different-corpus runs. Code50 RTX throughput is not inferred from the
Chinese GTX measurements.

Outputs: `runs/djam_code50_corrected/{A_keep,B_swap_fvt,C_swap_random,E_progressive}_code_dropout01_50M/`.
The merged trainer explicitly enables training, checks mode/dropout before every
optimizer step and restores/checks the caller mode after each evaluation. The
wrapper passes `expected_dropout=0.1` and records the exact checkout commit.

## Failed runs and resubmission

The trainer refuses **any** existing run directory; there is no optimizer/RNG
resume state. A folder without `model/` is not resumable. Inspect Slurm/error
logs, then preserve a failed attempt under a new name before a justified retry:

```bash
# Example only, after verifying that this run failed and is not active:
mv runs/djam_code50_corrected/B_swap_fvt_code_dropout01_50M \
   runs/djam_code50_corrected/B_swap_fvt_code_dropout01_50M.failed1
```

Never rename a running/completed run or overwrite an existing `.failed1` target.
Legacy `phase1.sbatch` only skips completed `model/` folders and is not a resume
mechanism. Prefer the explicit per-condition wrapper above.

After saved checkpoints exist, use the existing inference tools to evaluate
and benchmark them, then plot the real logs. Keep checkpoints outside Git.
200M, rho ablations and extra innovations remain outside this handoff.

## Chinese result explanation to forward (not sent)

« Les trois entraînements chinois corrigés à 50M tokens sont terminés et
vérifiés, avec le dropout activé et le mode d'entraînement bien restauré après
chaque évaluation. Les bpb sont A = 1,95665, B = 1,69276 et E = 1,76349 : B est
le meilleur sur ce run, sans avantage démontré pour E. Les checkpoints, mesures
de vitesse et figures sont prêts. C'est un seul seed avec des tokenizers
entraînés sur des pilotes de 20 Mo, pas l'étude complète à 1 Go. L'ancien pilote
sans dropout est conservé séparément. »
