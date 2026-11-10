#!/usr/bin/env bash
# User-authorized replacement of unused332004; normal scheduler submission.
#SBATCH --job-name=arm-beta1-replacement
#SBATCH --account=zixianma
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h200:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=960G
#SBATCH --time=14:00:00
#SBATCH --output=/gpfs/projects/krishna/zixianma/OpenWebRL/logs/arm-beta-replacement-20260927/slurm-%j.out
set -euo pipefail
cd /gpfs/projects/krishna/zixianma/OpenWebRL
source scripts/h200_env.sh
unset WANDB_SERVICE OPENWEBRL_ARM_TURN_BONUS_CONFIG OPENWEBRL_RECORD_EVAL_CONFIG
export WANDB_PROJECT=openwebrl
/gpfs/scrubbed/zixianma/openwebrl-runtime/venv/bin/python -B scripts/prepare_arm_beta_replacement.py \
    --job-id "$SLURM_JOB_ID"
