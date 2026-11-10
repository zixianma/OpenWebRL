#!/usr/bin/env bash
# Resource sizes and shortened remaining budgets are explicit sbatch overrides.
#SBATCH --job-name=arm-unfinished-recovery
#SBATCH --account=zixianma
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/gpfs/projects/krishna/zixianma/OpenWebRL/logs/arm-unfinished-20260925/slurm-%j.out
set -euo pipefail
cd /gpfs/projects/krishna/zixianma/OpenWebRL
: "${ARM_RECOVERY_KIND:?Set beta or coverage}"
source scripts/h200_env.sh
unset WANDB_SERVICE OPENWEBRL_ARM_TURN_BONUS_CONFIG OPENWEBRL_RECORD_EVAL_CONFIG
export WANDB_PROJECT=openwebrl-evals
if [[ "$ARM_RECOVERY_KIND" == coverage ]]; then export WANDB_PROJECT=openwebrl; fi
/gpfs/scrubbed/zixianma/openwebrl-runtime/venv/bin/python scripts/recover_arm_unfinished.py \
  --kind "$ARM_RECOVERY_KIND" --job-id "$SLURM_JOB_ID"
