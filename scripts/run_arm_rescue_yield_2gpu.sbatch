#!/usr/bin/env bash
# Proposed bounded pilot; exact allocation approval required before submission.
#SBATCH --job-name=arm-rescue-yield
#SBATCH --account=zixianma
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h200:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=480G
#SBATCH --time=02:00:00
#SBATCH --output=/gpfs/scrubbed/zixianma/openwebrl-runtime/logs/slurm-arm-rescue-yield-%j.out
set -euo pipefail
cd /gpfs/projects/krishna/zixianma/OpenWebRL
unset WANDB_SERVICE OPENWEBRL_ARM_TURN_BONUS_CONFIG OPENWEBRL_ARM_CONFIG
export WANDB_PROJECT=openwebrl-evals
export MASTER_PORT=$((15000 + SLURM_JOB_ID % 1000))
srun --jobid="$SLURM_JOB_ID" --nodes=1 --ntasks=1 --cpus-per-task=16 --gres=gpu:h200:2 --exact --cpu-bind=none \
 /gpfs/scrubbed/zixianma/openwebrl-runtime/venv/bin/python scripts/run_arm_rescue_yield.py \
 --job-id "$SLURM_JOB_ID" --execute
