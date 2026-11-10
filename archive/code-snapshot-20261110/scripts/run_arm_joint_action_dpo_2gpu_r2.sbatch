#!/usr/bin/env bash
#SBATCH --job-name=arm-action-dpo-r2
#SBATCH --account=zixianma
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h200:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --time=03:30:00
#SBATCH --signal=B:TERM@300
#SBATCH --output=/gpfs/scrubbed/zixianma/openwebrl-runtime/logs/arm-joint-action-dpo-r2-%j.out
# Requires exact approval: 2 H200 x 3.5 hours, including full-300 endpoint evaluation.
set -euo pipefail
cd /gpfs/projects/krishna/zixianma/OpenWebRL
source scripts/h200_env.sh
exec /gpfs/scrubbed/zixianma/openwebrl-runtime/arm-reproduction/venv/bin/python \
  scripts/run_arm_joint_pipeline.py --objective dpo_action \
  --config openwebrl/docs/arm_results/joint_data_v2/dpo_action-2gpu-r2-config.json
