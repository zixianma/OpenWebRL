#!/usr/bin/env bash
# Each new submission requires explicit approval of the five-job10-GPU-hour cap.
#SBATCH --job-name=arm-original-backfill
#SBATCH --account=zixianma
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h200:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=480G
#SBATCH --time=01:00:00
#SBATCH --output=/gpfs/scrubbed/zixianma/openwebrl-runtime/logs/slurm-arm-original-backfill-%j.out
set -euo pipefail
cd /gpfs/projects/krishna/zixianma/OpenWebRL
: "${ARM_ORIGINAL_BACKFILL_ITERATION:?Required completed iteration}"
unset WANDB_SERVICE OPENWEBRL_ARM_TURN_BONUS_CONFIG OPENWEBRL_RECORD_EVAL_CONFIG
export WANDB_PROJECT=openwebrl-evals
export MASTER_PORT=$((15000 + SLURM_JOB_ID % 1000))
export OPENWEBRL_EXPECTED_SCHEDULER_OFFSET=0
srun --jobid="$SLURM_JOB_ID" --nodes=1 --ntasks=1 --cpus-per-task=16 --gres=gpu:h200:2 --exact --cpu-bind=none \
 /gpfs/scrubbed/zixianma/openwebrl-runtime/venv/bin/python scripts/prepare_arm_original_backfill.py \
 --iteration "$ARM_ORIGINAL_BACKFILL_ITERATION" --job-id "$SLURM_JOB_ID" --execute
