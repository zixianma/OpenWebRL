#!/usr/bin/env bash
# Resource approval pending:8 H200 x16h; fresh iteration0->20, full300 eval10/20.
#SBATCH --job-name=arm-addrel-failureonly
#SBATCH --account=zixianma
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h200:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=960G
#SBATCH --time=16:00:00
#SBATCH --output=/gpfs/scrubbed/zixianma/openwebrl-runtime/logs/arm-additive-relaxed-failureonly-%j.out
set -euo pipefail
cd /gpfs/projects/krishna/zixianma/OpenWebRL
: "${ARM_ADDITIVE_FAILUREONLY_APPROVED:?Exact resource approval required}"
source scripts/h200_env.sh
unset WANDB_SERVICE OPENWEBRL_ARM_TURN_BONUS_CONFIG OPENWEBRL_RECORD_EVAL_CONFIG
export WANDB_PROJECT=openwebrl
/gpfs/scrubbed/zixianma/openwebrl-runtime/venv/bin/python scripts/prepare_arm_additive_relaxed_failureonly.py --job-id "$SLURM_JOB_ID" --execute
