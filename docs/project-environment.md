# Separate RTX 5050 project runtime

The user authorized a separate project environment on 9 November after Djam
confirmed PyTorch needs updating. Preserve `DL_HW` and all lab environments.
The selected build is **PyTorch 2.7.1+cu128**, Python 3.12, with the existing
transformers 4.49.0 / tokenizers 0.21.0 interfaces. PyTorch documents Blackwell
support beginning with its CUDA 12.8 builds:
[release notes](https://pytorch.org/blog/pytorch-2-7/) and
[official installation versions](https://pytorch.org/get-started/previous-versions/).

## Current capacity gate — installation not performed

The actual quota was obtained with `quota -vs -A --show-mntpoint` and
`quota -v -f /net/course`. The default quota command only shows the unrelated
zero-usage gcs row; filesystem-wide `df` availability is not this account’s quota.

| Item | GiB |
|---|---:|
| Actual home hard quota | 10.000 |
| Home usage at capacity check | 6.102 |
| Free home quota | 3.898 |
| Resolved wheels, sum of uncompressed ZIP entries | 6.284 |
| Venv/filesystem allowance | 0.195 |
| Three checkpoints, save allowance and logs | 2.900 |
| Final home requirement with training reserve | **15.482** |

The environment alone exceeds available space. Even moving completed
checkpoints to the laptop after every run would not make the standard runtime
fit. No package installation or Slurm job has been performed. Ask for either
**20 GiB total home quota** or an **approved persistent runtime directory with
at least 10 GiB free**, accessible to compute nodes. With a separate runtime
location, the current home capacity can still cover project checkpoints.

The resolved wheel downloads total 3.621 GiB (largest 0.968 GiB). Download
staging is transient and separate from the required persistent environment;
allow enough installation workspace as well as final space. The sizing used
ZIP central-directory entries via bounded HTTP reads; it is not a measured
installed environment size. The preceding pip dry-run downloaded wheels despite
dry-run mode, installed nothing, retained no cache and cleaned up its staging.
The complete resolver report is preserved privately outside Git.

Evidence: `results/project-environment-capacity-2026-11-09.json`. The 40-wheel
lock `cluster/project-runtime-cp312-cu128.lock` contains exact public URLs and
SHA-256 hashes, including CUDA libraries and Triton. Do not omit CUDA
dependencies to force the install to fit, or mix the new torch with the lab
environment’s older libraries. No torchvision/torchaudio, datasets, duplicate
GPT-2 cache or plotting environment is needed for these training jobs.

## Install only after capacity and persistent location are approved

Run from the cluster checkout. `PROJECT_ENV` must be a new absolute path in the
approved persistent location. If home quota is increased, an example is
`$HOME/tokenizer-switch-env-cu128`. Do not use temporary storage for the venv.
Verify the real quota/capacity and checkpoint allowance before executing:

```bash
cd "$HOME/tokenizer-switch"
: "${PROJECT_ENV:?Set a NEW environment path in approved persistent storage}"
test ! -e "$PROJECT_ENV"
test -w "$(dirname "$PROJECT_ENV")"
# Use existing Python to CREATE an isolated venv; never install into its base.
/opt/conda/miniconda3/envs/DL_HW/bin/python -m venv "$PROJECT_ENV"
export PYTHONDONTWRITEBYTECODE=1
"$PROJECT_ENV/bin/python" -m pip --isolated install --no-cache-dir --no-compile --no-deps --require-hashes -r cluster/project-runtime-cp312-cu128.lock
"$PROJECT_ENV/bin/python" -m pip check
"$PROJECT_ENV/bin/python" - <<'PY'
from importlib.metadata import version
import sys
assert sys.prefix != sys.base_prefix
assert version('torch') == '2.7.1+cu128'
print(sys.prefix)
print({k: version(k) for k in ('torch', 'transformers', 'tokenizers', 'triton')})
PY
```

`--no-deps` is used with the complete resolved lock to avoid unpinned resolution;
it does not omit any listed dependency. `pip check` must pass. The venv is
isolated, without `--system-site-packages`; it does not inherit old torch,
torchvision or CUDA packages. Record actual installed versions and disk use.
Do not combine a login-node torch import with a large download: the login node
has an approximately 8 GiB virtual-memory cap. Test GPU imports/execution inside
Slurm instead.

## Slurm verification and Chinese runs

```bash
cd "$HOME/tokenizer-switch"
export VENV="$PROJECT_ENV"
export HF_HOME="$PWD/data/hf-cache"
export BASE_MODEL="$HF_HOME/hub/models--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e"
test -r "$VENV/bin/activate"
test -f "$BASE_MODEL/model.safetensors"
python3 scripts/verify_assets.py --layout destination --profiles chinese
squeue -u "$USER"
quota -vs -f /net/course
# Only with an empty account queue and no existing diagnostic output:
test ! -e results/chinese_gpu_check_2026-11-09.json
sbatch --gres=gpu:nvidia_geforce_rtx_5050:1 --time=00:15:00 -J zh_gpu_check cluster/job.sbatch python -u cluster/check_chinese_gpu.py --base-model "$BASE_MODEL" --out results/chinese_gpu_check_2026-11-09.json
```

Record the actual submitted job ID. Require Slurm `COMPLETED`, exit code 0 and
JSON `status=passed` for A/B/E. The test uses an actual RTX 5050, a CUDA kernel,
bf16 forward/backward and AdamW updates on the real old/new encodings, validation
evaluation and E pruning. It uses the actual 84,717-row Chinese union, saves no
checkpoint and reads no test text. Its outputs are diagnostics, not research
quality results.

Then use the Chinese 50M A/B/E commands in `docs/data-handoff.md`, one job at a
time. Set `VENV` to the actual approved path rather than the example. Keep the
pinned base-model path, validation periodic evaluation and `eval_batch=1`.
Recheck queue, quota and duplicate run directories before every submission.
Retain/save actual checkpoints in the budgeted persistent space. Evaluate the
completed checkpoints and generate figures from their real logs; record actual
job IDs, Slurm states and metrics. No job IDs exist yet for this installation.
