# Milestone audit — 11 November 2026

## Corrected milestone complete — 12 November, 20:44 Bangkok / 21:44 Taipei

**Job 766353 COMPLETED / exit 0**, elapsed **29h54m07s**. A/B/E each completed
**49,938,432 training tokens** and saved verified checkpoints. Final saved-model
bpb, shared-prompt generation speed, training curves and comparison PNG/PDF
figures are complete, collected locally and manifest-hash verified.

| Corrected condition | Test bpb | Generation characters/s | Reported trainer hours |
|---|---:|---:|---:|
| A keep | 1.956647 | 55.66 | 9.22 |
| B FVT | 1.692755 | 172.64 | 8.23 |
| E progressive, pruned | 1.763491 | 165.08 | 11.08 |

**B is best in this single run; an E advantage is not demonstrated.** These are
valid corrected dropout-0.1 results, with **20 MB tokenizer fits**, not the full
proposed 1 GB study. Do not pool them with the preserved dropout-disabled pilot
or claim statistical significance from one seed/benchmark.

The completed mode audit passes **762 training-phase checks per condition**,
all 77 logged training events per condition, and restoration after every
initial/periodic/final evaluation, including E post-pruning. All logged losses
and gradients are finite. Saved-checkpoint scores exactly match the appropriate
final/pruned log scores on all 1,000 test docs; weights are unchanged after
inference/speed verification. Both frozen source worktrees remain clean.

Presentation-ready corrected figures:
`results/chinese766353_corrected_dropout01/comparison.{png,pdf}` and
`results/chinese766353_corrected_dropout01/training.{png,pdf}`. CSVs, verified
bpb JSON, numeric logs and manifest are alongside them. No weight/data/env files
are committed. [Final results and handoff](chinese50m-corrected-results.md).

This completes the assigned Chinese 50M data/tokenizer/evaluation/figure core
milestone. Code-model checkpoint comparisons still depend on teammates' runs;
full 1 GB fits, additional seeds, 200M extensions and innovations remain deferred.
No further training job was submitted. Earlier running/pending snapshots below
are historical and superseded by this verified completion.


## Current corrected progress — 12 November, 14:19 Taipei

**Job 766353 RUNNING on cmpt053, stage E.** A and B each completed
**49,938,432 tokens** and saved controller-hashed checkpoints. E last logged
**21,692,416 / 49,938,432 tokens (43.4%)**. No errors were reported; all logged
losses/gradients are finite. No duplicate or extension job was submitted.

The mode fix is confirmed across the completed A/B runs: **762 per-step checks
each**, every logged training forward has mode true, and initial/periodic/final
evaluation restores training. E has the same positive mode records, including
after evaluations at steps 0/100/200/300. Source d11b8b5 remains immutable.

Preliminary trainer-logged test bpb: **A 1.9566465**, **B 1.6927551**. These are
corrected dropout-0.1 runs, **separate from pilot scores**. Saved-checkpoint
reevaluation, final E/pruning results, synchronized speed and full comparison
figures remain pending and will run automatically inside this allocation after
E finishes. Do not present these two scores as a verified complete comparison.

Latest checkpoint-space guard: **1.65 GB free** versus **1.42 GB reserved** for
remaining E checkpoint/save/log headroom; fits the 10 GiB quota. Current quota
also remains below the limit. The original dropout-disabled pilot and verified
figures remain preserved. Closing the laptop does not interrupt this job.

Evidence: `results/corrected_chinese_status_766353_2026-11-12.json` (token values
are last-logged lower bounds, not a live dashboard). Historical snapshots below
are superseded by this dated update.


## Approved corrected sequence — 11 November, 15:51 Taipei

The user explicitly approved **one corrected sequence**. It was submitted once
as Slurm job **766353**, **RUNNING on cmpt053**, start **15:50:27 Taipei**, wall
limit **37 hours** (scheduler limit 13 November 04:50:27, not a completion ETA).
The earlier approval block is resolved. No continuation/submission watcher is
needed or running: the single GPU allocation drives A → B → E → saved-checkpoint
verification → bpb/speed/figures itself, independently of laptop/SSH closure.

Source is immutable **d11b8b5e7188321b881101081c1119a89ec3276b**, passing gate
**766279**. Queue and absent corrected output root were rechecked before the
single `sbatch`; an exclusive persistent launch lock prevents an accidental
repeat. No duplicate training, 200M run, innovation or environment installation.
The pilot remains unchanged and separately labelled.

At **15:51:54**, the controller reports **training_A_keep**, configured dropout
0.1 enabled. A is starting/performing initial evaluation; **no gradient-token
record has been logged yet** (A/B/E last logged tokens 0/0/0). B/E are waiting
inside the same allocation. No corrected checkpoint or final result exists yet.
The source checks mode before every training step and restoration after eval;
actual mode/token evidence is read from the live logs as they appear.

Corrected outputs:
`~/tokenizer-switch/runs/chinese_corrected_dropout01_20261110/` (the date is the
original plan name; this actual sequence started 11 November). Frozen release:
`~/tokenizer-switch-releases/zh50-memory-d11b8b5e7188/`.
Controller: `results/corrected_train_766353.json`.
Slurm stdout/stderr: `logs/zh50_corrected_ABE_766353.out`.
Final verified outputs: `results/corrected_verified_766353/` after all stages pass.
No final metrics are claimed before that verification completes. Publication to
GitHub still requires a connected authenticated Mac; computation does not.

Quota recheck before submission: **2,665,304,064 bytes free** versus
**2,440,785,846 bytes reserved** for all fresh checkpoints/save workspace/headroom.
The controller rechecks reserve before each condition. It reuses existing
DL_HW and assets; no pilot/model-cache copies or lab-file changes.

Use the immutable release to monitor (read only):

```bash
squeue -j 766353
cat ~/tokenizer-switch-releases/zh50-memory-d11b8b5e7188/results/corrected_train_766353.json
tail -n 5 ~/tokenizer-switch/runs/chinese_corrected_dropout01_20261110/A_keep_zh_dropout01_50M/log.jsonl
```

Launch evidence: `results/corrected_chinese_launch_766353_2026-11-11.json`.
The previous approval-blocked / unsubmitted snapshots below are historical.


## Latest verified outcome — 15:41 Taipei

- **DONE:** data/splits/selected tokenizers/20 MB mini-grids, measured compression
  figures, preserved dropout-disabled pilot, all checkpoint hashes, final pilot
  bpb/speed and labelled comparisons. All three weight hashes were rechecked
  today and still match independent verifier 764463.
- **DONE:** corrected mode/memory compatibility after a routine lifetime fix.
  New E-only gate **766279 COMPLETED / 0:0, 2m41s**. E has two optimizer steps,
  **131,072 diagnostic tokens**, 128 active-dropout gradient forwards / backward
  calls and correctly restored training after every evaluation. Peak E allocation
  **6,018,094,592 bytes**. Prior A/B passes from 764704 were reused, not rerun.
- **BLOCKED:** fresh full corrected A/B/E submission, pending approval after
  automatic action review rejected the test-plus-full-continuation watcher.
  There is **no formal corrected job, 0 formal tokens**, and no automatic
  continuation process. This is now an operational approval block, not the
  previous E OOM. The standalone diagnostic did not start full training.
- **NOT STARTED:** corrected final checkpoints/bpb/speed/figures; full 1 GB fit
  study, 200M extensions and extra innovations.

The lifetime-only fix frees BF16 vocabulary logits after cross entropy is built,
then clears completed microbatch tensors. Exact paired tiny CLI runs against
`4cc7bff` produce **bit-identical losses and saved weights for A/B/C/E**. No
training settings, RNG, dropout, batch, context, data or schedule changed. Source
is frozen at **d11b8b5** in
`~/tokenizer-switch-releases/zh50-memory-d11b8b5e7188/`. Evidence reuse checks the
unchanged protocol files and restricts trainer differences to those deletions.

**70 current tests passed (76.11 s)**. E gate 766279 plus prior A/B measurements
reserve a **37-hour** Slurm allocation. Post-diagnostic quota has
**2,665,304,064 bytes free**, versus **2,440,785,846 bytes** reserved for all new
checkpoints/save workspace/headroom. The current reserved peak is below 10 GiB;
reserves are checked again before each condition. No pilot deletion or new
runtime is needed. These are runtime projections, not completed full training.

Prepared full-run command (not executed):

```bash
cd ~/tokenizer-switch-releases/zh50-memory-d11b8b5e7188
sbatch --job-name=zh50_corrected_ABE --time=37:00:00 \
  --output="$PWD/logs/%x_%j.out" cluster/corrected_chinese.sbatch \
  --phase train --gate-file "$PWD/results/corrected_gate_766279.json"
```

Automatic approval review rejected the earlier combined action because its
watcher could submit full A/B/E beyond the interpreted current E-only scope.
The safer standalone test was approved and completed. No full submission or
watcher was executed indirectly after that rejection.


## Initial audit: actual state, not the previous submission plan

| Deliverable | Status | Evidence |
|---|---|---|
| Data splits / encodings | DONE | Physical manifests, token capacities and structures reverified today for code and Chinese; prior full alignment applies to unchanged hashes. Code 250 MB and Chinese 400 MB both cover 50M in both encodings. |
| Selected tokenizers / mini-grids | DONE | `code70_gpt4_32k`, `wiki_zh70_gpt4_50k`; each candidate fitted on 20 MB. Validation selection, held-out compression and lossless round trips are recorded. |
| Dropout-disabled Chinese pilot | DONE | Job **757238 COMPLETED / 0:0**, 29h10m32s, all A/B/E saved at **49,938,432 tokens each**. Source/checkpoints preserved. |
| Pilot checkpoint verification / bpb / speed / figures | DONE | Job **764463 COMPLETED / 0:0**, 45m11s. All three hashes, vocab/tokenizer semantics, 1,000 test docs, logged/saved-score agreement and unchanged weights pass. Small outputs collected and hashes rechecked locally. |
| Corrected-mode compatibility | BLOCKED at initial audit | Gate **764704 FAILED / 1:0**, 5m03s. A/B pass two steps each (131,072 tokens), 128 active-dropout gradient forwards and restored training after evaluation. E backward OOM: 332 MiB requested, 279.62 MiB free. |
| Corrected full Chinese 50M A/B/E | NOT STARTED at initial audit | No full corrected Slurm job, no saved corrected model, **0 formal training tokens** per condition. Watcher stopped safely; an armed plan was not a completed run. |
| Corrected final quality / speed / comparisons | NOT STARTED | No formal corrected checkpoints exist. Diagnostic scores/weights are not model results. |
| Larger studies | NOT STARTED / deferred | Proposed 1 GB tokenizer study, both-encoding 200M expansion, extra innovations. Do not present them as completed. |

The mode fix was active in the real corrected A/B gate forwards and CPU
A/B/C/E integration, including after evaluation. There was no formal corrected
training trajectory to certify. The pilot was trained in eval mode throughout;
reevaluation validates those weights but cannot repair their learning trajectory.

## Verified dropout-disabled pilot results

| Condition | Actual training tokens | Test bpb | Generation characters/s | Trainer wall hours |
|---|---:|---:|---:|---:|
| A keep | 49,938,432 | 1.864271 | 55.76 | 9.05 |
| B FVT | 49,938,432 | 1.646345 | 181.04 | 7.95 |
| E progressive, pruned | 49,938,432 | 1.714944 | 168.15 | 10.80 |

**PROVISIONAL for the intended dropout-0.1 study; valid inference measurements
of dropout-disabled weights.** One seed, 20 MB tokenizer fits. B has the lowest
observed bpb in this pilot; there is no demonstrated E advantage or corrected
ranking. Do not claim statistical significance or repaired training.

Test bpb covers the same 1,000 Chinese documents and 10,265,237 UTF-8 bytes;
EOT targets are included. Speed is one synchronized GTX 1060 benchmark: 20 shared
200-character prefixes, 128 greedy new tokens, tokenizer-specific EOT/padding,
explicit masks and warmup excluded. It measures throughput, not output quality.
Trainer wall time includes periodic/final evaluation but excludes initial
evaluation, E pruning and saving. Do not interpret it as pure training-kernel time.

## Presentation-ready figures and claims

| Figure (PNG and PDF) | Safe claim / required label |
|---|---|
| `results/milestone_summary.*` | Measured selected-tokenizer test compression and 50M corpus capacity. **20 MB fits**, code 250 MB / Chinese 400 MB; capacity is not unique sampling coverage. |
| `results/milestone/compression_validation.*` | Code pilot mini-grid; selection used validation, including the English cap. |
| `results/milestone/compression_test.*` | Selected code tokenizer NSL **0.5916**, English **1.0813**, lossless. |
| `results/chinese/compression_validation.*` | Chinese pilot mini-grid and validation-only selection. |
| `results/chinese/compression_test.*` | Selected Chinese tokenizer NSL **0.3035**, English **1.0761**, lossless. |
| `results/chinese757238_dropout_disabled/comparison.*` | Actual pilot bpb, fixed-token generation speed and trainer wall times. Keep **PROVISIONAL / dropout disabled** in the title. |
| `results/chinese757238_dropout_disabled/presentation_training.*` | Readable real pilot training/validation curves, with the same dropout-disabled/provisional label. Original verified `training.*` remains preserved. |

The summary and pilot plots were visually inspected. The pilot CSV/JSON/plots
are preserved with the source verification manifest; the readable training
figure has a separate derivative manifest. No synthetic project results or
checkpoint binaries are committed.

**Not presentation-ready claims:** corrected 50M quality/ranking/speed, a full
1 GB design study, a 200M comparison, a general E advantage, or code model-quality
results without teammates' actual checkpoints. Compression is not model quality.

## Follow-up and monitoring

The overnight cluster processes worked independently of the laptop: pilot
verification ran and the corrected gate was submitted. The corrected watcher
then stopped on real E OOM; it is not still making progress. No queued/running
job was found at the initial 15:20 audit. Mac publication had stopped on SSH/DNS
failure, explaining why yesterday's PR still showed pending IDs.

Routine follow-up now: preserve/publish verified pilot outputs, shorten clipped
plot labels, update this audit/PR, and check a tensor-lifetime-only memory fix
for E. It releases vocabulary logits after constructing cross entropy, before
backward, and clears completed microbatch tensors. It changes no dropout,
context, batch, data, seed, schedule or optimizer setting. Exact tiny-CLI loss
and saved-weight equality against `4cc7bff` is checked before any GPU retry.
A new compatibility probe is justified by that change; it is not duplicate full
training. Any fresh full submission still requires all gate/resource checks.
Future real job IDs/status are appended below after verification; no success
or corrected results are assumed here.
