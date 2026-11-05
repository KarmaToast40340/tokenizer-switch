#!/bin/bash
# One-time setup of a cluster account, run by each member on their own account, on the login node:
#   cd ~/tokenizer-switch && bash cluster/setup_account.sh
# Light work only (checks, GPT-2 download, folders): no training or encoding on the login node.
# Safe to run again: every step checks first and skips what is already done.
set -euo pipefail

VENV="${VENV:-$HOME/venv_lab2}"
QUOTA_GB=10
cd "$(dirname "$0")/.."

echo "== account $USER, repository $(pwd)"
if git rev-parse --git-dir >/dev/null 2>&1; then
  echo "   commit $(git rev-parse --short HEAD) ($(git describe --tags --always 2>/dev/null))"
else
  echo "   WARNING: not a git clone; use 'git clone' (not scp) so runs can record their commit"
fi

echo "== Python environment: $VENV"
if [ ! -x "$VENV/bin/python" ]; then
  echo "   ERROR: $VENV not found. Reuse the Lab 2 venv (do not create a new one: 10 GB quota)," >&2
  echo "   or point VENV=... to another existing venv." >&2
  exit 1
fi
export PATH="$VENV/bin:$PATH"
missing=$(python - <<'EOF'
import importlib.util
mods = ["torch", "transformers", "tokenizers", "numpy", "yaml", "tqdm"]
print(" ".join(m for m in mods if importlib.util.find_spec(m) is None))
EOF
)
if [ -n "$missing" ]; then
  echo "   missing modules: $missing"
  echo "   installing the pure-Python ones (never torch: the venv's cu128 build is needed for the RTX 5050)"
  for m in $missing; do
    case $m in
      torch) echo "   ERROR: torch is missing from $VENV; ask Djam before installing it" >&2; exit 1 ;;
      yaml) pip install --no-cache-dir "pyyaml>=6.0" ;;
      *) pip install --no-cache-dir "$m" ;;
    esac
  done
fi
python -c "import torch, transformers; print('   torch', torch.__version__, '| transformers', transformers.__version__)"

echo "== GPT-2 in the Hugging Face cache (compute nodes are offline)"
python - <<'EOF'
from tokenizers import Tokenizer
from transformers import GPT2LMHeadModel
GPT2LMHeadModel.from_pretrained("gpt2")
Tokenizer.from_pretrained("gpt2")
print("   ok")
EOF

echo "== folders"
mkdir -p logs runs results data/tokens data/raw artifacts/tokenizers
echo "   logs/ runs/ results/ data/tokens/ data/raw/ artifacts/tokenizers/"

echo "== disk use of \$HOME (quota ${QUOTA_GB} GB)"
used_kb=$(du -sk "$HOME" 2>/dev/null | cut -f1)
awk -v kb="$used_kb" -v q="$QUOTA_GB" 'BEGIN {
  used = kb / 1024 / 1024
  printf "   used %.1f GB, free %.1f GB\n", used, q - used
  if (q - used < 2.0) print "   WARNING: less than 2 GB free; the encoded corpora of one domain take 1.1 to 1.5 GB"
}'

echo "== done. Next: copy your domain's data from the laptop (cluster/RUNS.md), then sbatch your runs."
