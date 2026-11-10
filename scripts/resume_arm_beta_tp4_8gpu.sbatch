#!/usr/bin/env bash
#SBATCH --job-name=arm-beta1-tp4-recovery
#SBATCH --account=zixianma
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h200:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=960G
#SBATCH --time=13:31:00
#SBATCH --output=/gpfs/projects/krishna/zixianma/OpenWebRL/logs/arm-beta-tp4-recovery-20260927/slurm-%j.out
set -euo pipefail
cd /gpfs/projects/krishna/zixianma/OpenWebRL
source scripts/h200_env.sh
unset WANDB_SERVICE OPENWEBRL_ARM_TURN_BONUS_CONFIG OPENWEBRL_RECORD_EVAL_CONFIG
export WANDB_PROJECT=openwebrl
/gpfs/scrubbed/zixianma/openwebrl-runtime/venv/bin/python -B scripts/recover_arm_beta_tp4.py --job-id "$SLURM_JOB_ID"
