#!/usr/bin/env bash
# Approved 2026-09-13: 4 H200 x 4 hours, existing 32 CPU / 480 GiB profile, 48 browsers.
#SBATCH --job-name=sol-turn-bonus-fresh
#SBATCH --account=zixianma
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h200:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=480G
#SBATCH --time=04:00:00
#SBATCH --output=/gpfs/scrubbed/zixianma/openwebrl-runtime/logs/slurm-sol-turn-fresh-%j.out
set -euo pipefail
cd /gpfs/projects/krishna/zixianma/OpenWebRL
srun --jobid="$SLURM_JOB_ID" --nodes=1 --ntasks=1 --cpus-per-task=32 \
    --gres=gpu:h200:4 --cpu-bind=none --exact \
    /gpfs/scrubbed/zixianma/openwebrl-runtime/venv/bin/python \
    scripts/run_arm_turn_bonus_cycles.py --execute --job-id "$SLURM_JOB_ID" --provider sol --minutes "${ARM_TRAIN_MINUTES:-240}"
