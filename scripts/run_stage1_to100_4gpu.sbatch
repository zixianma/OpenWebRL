#!/usr/bin/env bash
# Baseline continuation extended to4H200x24h on2026-09-22; includes full-300 eval.
#SBATCH --job-name=arm-stage1-to100
#SBATCH --account=zixianma
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h200:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=480G
#SBATCH --time=24:00:00
#SBATCH --output=/gpfs/scrubbed/zixianma/openwebrl-runtime/logs/slurm-stage1-to100-%j.out
set -euo pipefail
cd /gpfs/projects/krishna/zixianma/OpenWebRL
: "${STAGE1_VARIANT:?Set baseline or additive in the approved submission}"
export WANDB_PROJECT=openwebrl
unset WANDB_SERVICE OPENWEBRL_ARM_TURN_BONUS_CONFIG OPENWEBRL_ARM_CONFIG
# The controller stays alive through GPU verification, training, and evaluation.
/gpfs/scrubbed/zixianma/openwebrl-runtime/venv/bin/python scripts/run_stage1_to100.py \
  --job-id "$SLURM_JOB_ID" --variant "$STAGE1_VARIANT" --execute
