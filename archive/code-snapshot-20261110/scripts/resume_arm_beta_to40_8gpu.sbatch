#!/usr/bin/env bash
# Submission requires exact approval:8 H200 x14h, including evaluations30/40.
#SBATCH --job-name=arm-beta1-to40-tp2
#SBATCH --account=zixianma
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h200:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=960G
#SBATCH --time=14:00:00
#SBATCH --output=/gpfs/projects/krishna/zixianma/OpenWebRL/logs/arm-beta-to40-20260925/slurm-%j.out
set -euo pipefail
cd /gpfs/projects/krishna/zixianma/OpenWebRL
: "${ARM_BETA40_APPROVED:?Exact resource approval required}"
source scripts/h200_env.sh
unset WANDB_SERVICE OPENWEBRL_ARM_TURN_BONUS_CONFIG OPENWEBRL_RECORD_EVAL_CONFIG
export WANDB_PROJECT=openwebrl
/gpfs/scrubbed/zixianma/openwebrl-runtime/venv/bin/python scripts/prepare_arm_beta_to40.py --job-id "$SLURM_JOB_ID"
