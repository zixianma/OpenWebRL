#!/usr/bin/env bash
# Exact approval required: 2 H200 x 4h, terminal judging capped at $30 / 927 attempts.
# Approved data transfer must cover live rollout screenshots/history sent to api.openai.com.
#SBATCH --job-name=arm-task-success-audit
#SBATCH --account=zixianma
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h200:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=480G
#SBATCH --time=04:00:00
#SBATCH --output=/gpfs/scrubbed/zixianma/openwebrl-runtime/logs/slurm-arm-task-success-%j.out
set -euo pipefail
cd /gpfs/projects/krishna/zixianma/OpenWebRL
export WANDB_PROJECT=openwebrl-evals
unset WANDB_SERVICE OPENWEBRL_ARM_TURN_BONUS_CONFIG OPENWEBRL_ARM_CONFIG
/gpfs/scrubbed/zixianma/openwebrl-runtime/venv/bin/python scripts/run_arm_task_success_audit.py \
  --job-id "$SLURM_JOB_ID" --execute
