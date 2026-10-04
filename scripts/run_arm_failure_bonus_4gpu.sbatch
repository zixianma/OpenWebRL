#!/usr/bin/env bash
# Proposed resources only. Obtain explicit approval before submitting this job.
#SBATCH --job-name=arm-all-failure-bonus
#SBATCH --account=zixianma
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h200:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=480G
#SBATCH --time=04:00:00
#SBATCH --output=/gpfs/scrubbed/zixianma/openwebrl-runtime/logs/slurm-arm-allfailure-%j.out
set -euo pipefail
cd /gpfs/projects/krishna/zixianma/OpenWebRL
srun --jobid="$SLURM_JOB_ID" --nodes=1 --ntasks=1 --cpus-per-task=32 \
    --gres=gpu:h200:4 --cpu-bind=none --exact \
    /gpfs/scrubbed/zixianma/openwebrl-runtime/venv/bin/python \
    scripts/run_arm_failure_bonus.py --execute --check-before-execute --job-id "$SLURM_JOB_ID" --minutes "${ARM_TRAIN_MINUTES:-240}"
