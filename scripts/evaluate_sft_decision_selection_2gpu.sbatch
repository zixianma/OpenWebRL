#!/usr/bin/env bash
# Requires exact approval: 2 H200, 16 CPU, 240 GiB, 1h total including retries;
# 40 Browser Use sessions, 300 Jev calls, <=160 o4-mini calls, no text helper.
#SBATCH --job-name=sft-decision-select
#SBATCH --account=zixianma
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h200:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --time=01:00:00
#SBATCH --signal=B:TERM@90
#SBATCH --output=/gpfs/scrubbed/zixianma/openwebrl-runtime/logs/sft-decision-select-%j.out
set -euo pipefail
cd /gpfs/projects/krishna/zixianma/OpenWebRL
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export WANDB_PROJECT=openwebrl-evals
exec /gpfs/scrubbed/zixianma/openwebrl-runtime/venv/bin/python \
  scripts/evaluate_sft_decision_selection.py --execute
