#!/usr/bin/env bash
# Prepared only. Submit after explicit approval of 1 H200 x 12 h,
# 8 CPUs, 120 GiB RAM, zero browser/API calls.
#SBATCH --job-name=arm-advantage-guidance
#SBATCH --account=zixianma
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --output=/gpfs/scrubbed/zixianma/openwebrl-runtime/advantage-guidance-action-only-20261021/logs/slurm-%j.out
set -euo pipefail
cd /gpfs/projects/krishna/zixianma/OpenWebRL

PILOT_ROOT=/gpfs/scrubbed/zixianma/openwebrl-runtime/advantage-guidance-action-only-20261021
PILOT_PY=/gpfs/scrubbed/zixianma/openwebrl-runtime/venv/bin/python
export HF_HOME="$PILOT_ROOT/hf-cache"
export XDG_CACHE_HOME="$PILOT_ROOT/cache"
export TORCH_HOME="$PILOT_ROOT/torch-cache"
export TMPDIR="$PILOT_ROOT/tmp"
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1
mkdir -p "$PILOT_ROOT/logs" "$HF_HOME" "$XDG_CACHE_HOME" "$TORCH_HOME" "$TMPDIR"
sha256sum --check "$PILOT_ROOT/frozen-inputs.sha256"
"$PILOT_PY" "$PILOT_ROOT/source/run_advantage_guidance_pilot.py" --config "$PILOT_ROOT/config.json"
