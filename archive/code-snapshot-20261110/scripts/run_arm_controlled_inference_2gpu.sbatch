#!/usr/bin/env bash
# Submit only after exact resources and judge cap are explicitly approved.
#SBATCH --job-name=arm-controlled-inference
#SBATCH --account=zixianma
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h200:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --time=08:00:00
#SBATCH --output=/gpfs/scrubbed/zixianma/openwebrl-runtime/logs/slurm-arm-controlled-inference-%j.out
set -euo pipefail
cd /gpfs/projects/krishna/zixianma/OpenWebRL
export WANDB_PROJECT=openwebrl-evals
exec /gpfs/scrubbed/zixianma/openwebrl-runtime/venv/bin/python scripts/run_arm_controlled_inference.py \
  --job-id "$SLURM_JOB_ID" --shard "${1:?pass shard0 or shard1}" --execute
