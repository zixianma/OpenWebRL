#!/usr/bin/env bash
# Each submission needs exact budget approval. Default 4h; approved 8h uses
# sbatch --time=08:00:00 --export=ALL,ARM_TRAIN_MINUTES=480,ARM_RESUME_FROM=...
#SBATCH --job-name=arm-turn-bonus-resume
#SBATCH --account=zixianma
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h200:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=480G
#SBATCH --time=04:00:00
#SBATCH --output=/gpfs/scrubbed/zixianma/openwebrl-runtime/logs/slurm-arm-turn-resume-%j.out
set -euo pipefail
cd /gpfs/projects/krishna/zixianma/OpenWebRL
: "${ARM_RESUME_FROM:?Set the verified ARM run root}"
srun --jobid="$SLURM_JOB_ID" --nodes=1 --ntasks=1 --cpus-per-task=32 \
    --gres=gpu:h200:4 --cpu-bind=none --exact \
    /gpfs/scrubbed/zixianma/openwebrl-runtime/venv/bin/python \
    scripts/resume_arm_turn_bonus.py --execute --job-id "$SLURM_JOB_ID" \
    --resume-from "$ARM_RESUME_FROM" --minutes "${ARM_TRAIN_MINUTES:-240}"
