#!/usr/bin/env bash
# Eight independent submissions, explicit SHARD=0..7. Exact approval required.
#SBATCH --job-name=arm-task-screen-shard
#SBATCH --account=zixianma
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=08:00:00
#SBATCH --output=/gpfs/scrubbed/zixianma/openwebrl-runtime/logs/slurm-arm-task-screen-shard-%j.out
set -euo pipefail
cd /gpfs/projects/krishna/zixianma/OpenWebRL
export WANDB_PROJECT=openwebrl-evals
unset WANDB_SERVICE OPENWEBRL_ARM_TURN_BONUS_CONFIG OPENWEBRL_ARM_CONFIG
/gpfs/scrubbed/zixianma/openwebrl-runtime/venv/bin/python -B scripts/run_arm_task_screen_shards.py --job-id "$SLURM_JOB_ID" --shard "${SHARD:?Set SHARD=0..7}" --execute
