# Training-mode diagnosis — 10 November 2026

**Job 757238 remains unchanged. Its Chinese A/B/E are dropout-disabled pilots,
not the intended dropout-0.1 training comparison.** Finishing E is useful for a
consistent three-condition pilot. Do not pool these weights with corrected runs
or claim the common defect cancels between conditions.

## Exact snapshot and cause

The live detached release is `31dbce7f645018eb221d55f5b7756872396de9b5`,
`~/tokenizer-switch-releases/zh50-31dbce7f6450`. On 10 November its tracked files
were clean; the running trainer/metric SHA-256 values match the archived source:

| File | SHA-256 |
|---|---|
| `scripts/train.py` | `3d705ca762f0e82bb34ecd6be6de27591fba329a95ab5623189b63180f6e2b7f` |
| `src/toksw/metrics.py` | `63a7e2e7f2dfedad457651bb47e3aeaa38f4cb8592c2339df789316ad49f42dc` |

Hugging Face `GPT2LMHeadModel.from_pretrained()` loads in evaluation mode.
The archived trainer never explicitly calls `model.train()`. Its step-zero and
periodic `bits_per_byte()` calls preserve the incoming mode. The older scorer
unconditionally called `model.train()` after evaluation; commit `4f76a6a`
changed that helper to restore the incoming mode without fixing the trainer's
dependency on the old side effect. This was a regression introduced during the
data/evaluation work; the earlier statement that training behavior was unchanged
was incorrect.

Consequently the *entire* A/B/E training loop, starting with the first gradient
forward pass, runs in eval mode. All three configured GPT-2 dropout rates are
0.1; effective dropout is disabled. This also applies to archived smoke gates.
It is not a problem that begins only after the first periodic evaluation.

## Consequences

| Question | Finding |
|---|---|
| Are updates happening? | Yes. Eval mode does not disable autograd. Backward, clipping and AdamW still run; real logs have nonzero gradient norms. The exact-CLI fixture confirms nonzero backward gradients. |
| Is training affected? | Yes. Missing dropout changes the objective's stochastic realization and learned weights; controlled paired runs produce different dense weights. Memory/runtime estimates from the old gates do not establish corrected training requirements. |
| Are scores calculated in the wrong mode? | No direct mode error: archived periodic/final bpb, saved-checkpoint evaluation and generation already use eval mode/no-grad as required for inference. Their numerical values describe the weights actually learned. They remain provisional for the intended study. |
| Are A/B/E affected equally? | All have the same structural mode defect. Its quantitative effect may differ with vocabulary, initialization and progressive schedule; there is no demonstrated cancellation or preserved ranking. |
| Can reevaluation repair it? | Reevaluation validates inference on the existing weights, not the training trajectory. A dropout-0.1 comparison requires fresh training from the pinned common base, same schedules/seeds/data, and new run names. Continuing these checkpoints cannot reproduce the missing steps; optimizer/RNG resume state was not saved. |
| Should E continue? | Yes, as an explicitly labelled dropout-disabled pilot completing the same protocol as A/B. The user requested keeping it unchanged. No cancellation, restart or corrected 50M submission was made. |

The existing bpb protocol includes EOT targets and uses nonoverlapping windows;
it is bits per UTF-8 byte, not perplexity. This diagnosis does not establish
quality gains or repair other protocol limitations. Tokenizer fits are **20 MB
pilots**, not the proposed 1 GB study.

## Isolated fix and evidence

Fix `6c9a0ee` explicitly enables training immediately after model construction;
the trainer now owns mode policy. `bits_per_byte()` preserves incoming mode in
`try/finally`, including exceptions. Training/evaluation mode is logged. Future
GPU diagnostics also explicitly enable training. No active release file,
environment, dataset or checkpoint was edited.

`results/train-mode-diagnosis-2026-11-10.json` is **synthetic CPU evidence, not
Chinese model quality**. `scripts/diagnose_training_mode.py` ran the actual
archived and fixed CLI four steps for A/B/E, with identical tiny GPT-2 fixtures,
dropout 0.1, synthetic data and seed 2026. Passive hooks observed:

- Archived: four gradient forwards per condition, root training false, zero
  active dropout modules, four nonzero backward gradients.
- Fixed: four gradient forwards per condition, root training true, four active
  dropout modules, four nonzero backward gradients.
- Both: all scoring forwards eval mode; same-checkpoint bpb is identical from
  train/eval entry modes and agrees with the saved final score, including E's
  pruned checkpoint (maximum differences zero).
- Paired dense-weight maximum absolute differences: A 0.00163633, B 0.00170687,
  E 0.00176999. These demonstrate a changed trajectory, not its quality impact.

To reproduce without touching cluster files:

```bash
# Archived checkout must be detached at 31dbce7; fixed checkout at 6c9a0ee or later.
PYTHONPATH=src python scripts/diagnose_training_mode.py \
  --old-root /persistent/path/to/archived-worktree \
  --fixed-root "$PWD" --scratch /persistent/path/to/diagnostics \
  --out /persistent/path/to/new-diagnosis.json
```

Validation on 10 November: **49 full-suite tests passed (99.57 s), plus six
result-publication guard tests (55 checks total)**. A prior expanded attempt
failed on local macOS ENOSPC; the subsequent full rerun passed after headroom
recovered. The paired six-CLI diagnosis passed separately. Slurm shell syntax
and whitespace checks passed. These are CPU/source checks; no corrected GPU
training or post-E model measurements have run yet.

Mode regression tests inspect actual forward/backward calls in A/B/C/E CLI
integration, not merely finite losses. Metric tests cover mode restoration on
success/failure and scorer independence from the caller's mode.

## After E: verify and measure without retraining

`cluster/watch_chinese_verification.py` waits for the entire original allocation
and an empty account queue, then requires all three controller records to say
`saved`. It submits **one separate four-hour GTX 1060 verification job** using
unchanged DL_HW. A persistent exclusive lock prevents duplicate submission.
It never cancels jobs, overwrites checkpoints or writes into the training
release. The current QoS permits only one submitted job, so a pending dependency
cannot be submitted while 757238 is active. A watcher is used instead.

The independent verifier checks controller weight hashes/sizes, tensor/vocab
shapes, reference-tokenizer vocabulary and exact encoding/decoding on all 1,000
test documents, raw SHA/byte count, actual 50M token budgets, and saved-score
agreement. It then reevaluates with explicit eval mode/batch 1, measures all
three models on the same 20 test prefixes (200 characters, 128 greedy tokens,
one warmup per model outside timing, CUDA synchronization), and verifies weights
again. Speed is a fixed-token throughput benchmark, not generation quality.
Generation uses each tokenizer's EOT/pad ID and an explicit attention mask.

Fresh outputs include `verification.json`, `manifest.json`, bpb JSON, speed and
comparison CSVs, real numeric logs and PNG/PDF figures. Every comparison figure
is labelled **PROVISIONAL — eval-mode-trained / dropout disabled**. A failed
check stops publication; numbers are not silently promoted to verified results.
The new job reuses checkpoints/data in place and needs only small logs/figures,
with a 1.1 GiB free-space guard; no environment or weight copies are needed.

The user subsequently authorized **fresh corrected Chinese 50M A/B/E**, after
preserving/validating the pilot. This supersedes the earlier no-auto-retrain
gate. The independent cluster watcher at `4cc7bff` runs a measured train-mode GPU
check, calculates actual checkpoint/quota/runtime reserves, then submits the
fresh matched sequence only if all checks pass. See
[corrected-chinese-handoff.md](corrected-chinese-handoff.md). Mathieu/Djam should
review mode ownership, restoration, no-overwrite guards, matching settings and
E post-pruning evaluation. 200M extensions remain deferred.

## Durable monitoring and publication

Run the watcher from a **new persistent detached worktree** containing the fix,
not from `zh50-31dbce7f6450`. For example, with that new worktree as `$PWD`:

```bash
# Start once, in a detached process with stdout/stderr redirected to a persistent log.
python3 cluster/watch_chinese_verification.py \
  --job-state "$HOME/tokenizer-switch-releases/zh50-31dbce7f6450/results/chinese_gtx50_757238.json" \
  --out "$PWD/results/postwatch757238.json"
```

The watcher state records its PID, exact source commit, training accounting and
actual submitted verification job ID. `waiting` means no new GPU job has yet
been submitted. Keep the `.lock` after errors and inspect the queue/accounting
before any justified retry. No credentials are needed by this cluster process.

`scripts/follow_chinese_verification.py` runs on the Mac with the existing
`/tmp/tok-%C` SSH connection. It stops on the **first** authentication failure
and retains its durable state, rather than retrying passwords. Once the remote
manifest exists, it collects only size-limited result files, verifies every
hash and retains a persistent local copy. Optional `--publish` requires an
unchanged clean `chloe/data-eval` checkout, matching remote HEAD and an open
PR #4; it then commits/pushes results normally and appends verified numbers to
the progress notes and PR. It never merges, force-pushes or touches main.

```bash
python scripts/follow_chinese_verification.py \
  --remote-root /absolute/path/to/new/cluster/worktree \
  --out /persistent/path/to/fresh/local-collection \
  --repo "$HOME/tokenizer-switch-local/repo" \
  --expected-head "$(git rev-parse HEAD)" --publish
```

The Mac must remain awake/connected for local collection/publication; remote
training and verification continue independently. A stopped collector is not
proof that the GPU job failed. Inspect its `collector.json`, the remote watcher
state and Slurm accounting before restarting. Do not publish partial or failed
verification output. No real E score or speed/figure is claimed before this gate.


The original Mac collector stopped at 14:40 on DNS/SSH failure. At 16:35 both
cluster watchers remained alive, parent PID 1 and independent session IDs.
The newly deployed corrected watcher does not depend on that collector. Closing
the laptop does not stop cluster submission, training or evaluation; downloading
results and GitHub publication require a connected authenticated Mac. No GitHub
credentials are stored on the cluster. See the corrected handoff for actual IDs.
