#!/usr/bin/env bash
# Approved September25: 1 H200 x 4h,8 CPUs,120GiB; LF SFT plus offline eval.
#SBATCH --job-name=arm-refresh-lf-sft
#SBATCH --account=zixianma
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --output=/gpfs/scrubbed/zixianma/openwebrl-runtime/logs/slurm-arm-refresh-training-%j.out
set -euo pipefail
cd /gpfs/projects/krishna/zixianma/OpenWebRL
export WANDB_PROJECT=openwebrl
unset WANDB_SERVICE OPENWEBRL_ARM_TURN_BONUS_CONFIG OPENWEBRL_ARM_CONFIG
/gpfs/scrubbed/zixianma/openwebrl-runtime/venv/bin/python scripts/run_arm_refresh_training.py
