#!/usr/bin/env bash
# One bounded attempt from the approved48h-per-variant continuation budget.
#SBATCH --job-name=arm-mixed-to60
#SBATCH --account=zixianma
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h200:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=960G
#SBATCH --time=24:00:00
#SBATCH --output=/gpfs/scrubbed/zixianma/openwebrl-runtime/logs/arm-mixed-to60-%j.out
set -euo pipefail
cd /gpfs/projects/krishna/zixianma/OpenWebRL
: "${ARM_MIXED_VARIANT:?Expected bonus or reweight}"
source scripts/h200_env.sh
unset WANDB_SERVICE OPENWEBRL_ARM_TURN_BONUS_CONFIG OPENWEBRL_ARM_FAILURE_AUX_MANIFEST OPENWEBRL_RECORD_EVAL_CONFIG
export WANDB_PROJECT=openwebrl
export PYTHONDONTWRITEBYTECODE=1
/gpfs/scrubbed/zixianma/openwebrl-runtime/venv/bin/python -B scripts/resume_arm_mixed_to60.py \
 --job-id "$SLURM_JOB_ID" --variant "$ARM_MIXED_VARIANT" --execute
