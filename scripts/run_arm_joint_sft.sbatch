#!/usr/bin/env bash
# Proposal only: submit after explicit approval of 1 H200 x 4 hours.
#SBATCH --job-name=arm-joint-v2-sft
#SBATCH --account=zixianma
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --signal=B:TERM@300
#SBATCH --output=/gpfs/scrubbed/zixianma/openwebrl-runtime/logs/arm-joint-v2-sft-%j.out
set -euo pipefail
cd /gpfs/projects/krishna/zixianma/OpenWebRL
source scripts/h200_env.sh
# Exec makes the batch process own training, validation, and final saving.
exec /gpfs/scrubbed/zixianma/openwebrl-runtime/arm-reproduction/venv/bin/python \
  scripts/train_arm_joint_sft.py "$@"
