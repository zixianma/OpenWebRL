#!/usr/bin/env bash
# Proposed, not yet approved: one 8 H200 x16h continuation, including eval80/90.
#SBATCH --job-name=arm-gate-b-to90
#SBATCH --account=zixianma
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h200:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=960G
#SBATCH --time=16:00:00
#SBATCH --output=/gpfs/scrubbed/zixianma/openwebrl-runtime/logs/arm-gate-b90-%j.out
set -euo pipefail
cd /gpfs/projects/krishna/zixianma/OpenWebRL
source scripts/h200_env.sh
unset WANDB_SERVICE OPENWEBRL_ARM_TURN_BONUS_CONFIG OPENWEBRL_ARM_FAILURE_AUX_MANIFEST OPENWEBRL_RECORD_EVAL_CONFIG
export WANDB_PROJECT=openwebrl
/gpfs/scrubbed/zixianma/openwebrl-runtime/venv/bin/python -B scripts/resume_arm_gate_b90.py --job-id "$SLURM_JOB_ID" --execute
