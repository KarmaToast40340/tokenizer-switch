# Milestone audit — 11 November 2026

## Actual state, not the previous submission plan

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
