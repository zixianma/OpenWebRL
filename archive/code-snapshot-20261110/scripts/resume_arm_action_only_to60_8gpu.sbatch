#!/usr/bin/env bash
# Approved 2026-10-24: action-only 10->60, 8 H200 / 64 CPU / 960 GiB, 56 h total across all
# attempts (24 h + 24 h + 8 h afterok chain; --time is set per job at submission), judge cap $350.
#SBATCH --job-name=arm-action-only-to60
#SBATCH --account=zixianma
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h200:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=960G
#SBATCH --no-requeue
#SBATCH --output=/gpfs/scrubbed/zixianma/openwebrl-runtime/logs/arm-action-only-to60-%j.out
set -euo pipefail
cd /gpfs/projects/krishna/zixianma/OpenWebRL
source scripts/h200_env.sh
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}${no_proxy:+$no_proxy,}localhost,127.0.0.1,::1,$(hostname -s)"
export no_proxy="$NO_PROXY"
unset WANDB_SERVICE OPENWEBRL_ARM_TURN_BONUS_CONFIG OPENWEBRL_ARM_FAILURE_AUX_MANIFEST OPENWEBRL_RECORD_EVAL_CONFIG
export WANDB_PROJECT=openwebrl PYTHONDONTWRITEBYTECODE=1
exec /gpfs/scrubbed/zixianma/openwebrl-runtime/venv/bin/python -B scripts/resume_arm_action_only_to60.py --job-id "$SLURM_JOB_ID" --execute
