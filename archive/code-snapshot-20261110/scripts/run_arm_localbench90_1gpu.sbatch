#!/usr/bin/env bash
# First pass only; exact approval and registered remaining method budget required.
#SBATCH --job-name=arm-localbench90
#SBATCH --account=zixianma
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --time=12:00:00
#SBATCH --no-requeue
#SBATCH --output=/gpfs/scrubbed/zixianma/openwebrl-runtime/logs/arm-localbench90-%j.out
set -euo pipefail
cd /gpfs/projects/krishna/zixianma/OpenWebRL
source scripts/h200_env.sh
unset WANDB_SERVICE OPENWEBRL_ARM_TURN_BONUS_CONFIG OPENWEBRL_ARM_FAILURE_AUX_MANIFEST OPENWEBRL_RECORD_EVAL_CONFIG
export WANDB_PROJECT=openwebrl-evals PYTHONDONTWRITEBYTECODE=1
export SLIME_BROWSER_ENV_MODE=local_process
exec /gpfs/scrubbed/zixianma/openwebrl-runtime/venv/bin/python -B scripts/run_arm_localbench90.py \
  --execute --job-id "$SLURM_JOB_ID" --method "${1:?method required}" --repeat 1
