#!/usr/bin/env bash
# Prepared only. Submit after explicit approval of one H200 for up to 24 hours,
# eight CPUs, 120 GiB RAM, and no browser or paid API calls.
#SBATCH --job-name=arm-ag-sft-actions
#SBATCH --account=zixianma
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --output=/gpfs/scrubbed/zixianma/openwebrl-runtime/advantage-guidance-action-only-20261021/sft-head-common/logs/slurm-%j.out
set -euo pipefail

PILOT_ROOT=/gpfs/scrubbed/zixianma/openwebrl-runtime/advantage-guidance-action-only-20261021/sft-head-common
PILOT_PY=/gpfs/scrubbed/zixianma/openwebrl-runtime/venv/bin/python
export HF_HOME="$PILOT_ROOT/hf-cache"
export XDG_CACHE_HOME="$PILOT_ROOT/cache"
export TORCH_HOME="$PILOT_ROOT/torch-cache"
export TMPDIR="$PILOT_ROOT/tmp"
export PYTHONPATH="$PILOT_ROOT/python-deps${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1
mkdir -p "$PILOT_ROOT/logs" "$HF_HOME" "$XDG_CACHE_HOME" "$TORCH_HOME" "$TMPDIR"
sha256sum --check "$PILOT_ROOT/frozen-inputs.sha256"
"$PILOT_PY" -c 'import peft; assert peft.__version__ == "0.18.0"'
"$PILOT_PY" "$PILOT_ROOT/source/run_advantage_guidance_pilot.py" \
  --config "$PILOT_ROOT/config-proposed.json"
