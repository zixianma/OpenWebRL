#!/usr/bin/env bash
# Replacement allocation: requires explicit approval before submission.
#SBATCH --job-name=arm-joint-sft-r2-om2w
#SBATCH --account=zixianma
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h200:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --time=03:00:00
#SBATCH --signal=B:TERM@300
#SBATCH --output=/gpfs/scrubbed/zixianma/openwebrl-runtime/logs/arm-joint-sft-r2-%j.out
set -euo pipefail
cd /gpfs/projects/krishna/zixianma/OpenWebRL
source scripts/h200_env.sh
exec /gpfs/scrubbed/zixianma/openwebrl-runtime/arm-reproduction/venv/bin/python \
  scripts/supervise_arm_joint_pipeline.py \
  --config openwebrl/docs/arm_results/joint_data_v2/sft-2gpu-r2-config.json
