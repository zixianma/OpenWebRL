#!/usr/bin/env bash
# New submission requires exact approval: 4 H200 x 2h, 32 CPUs, 480 GiB.
# Maximum 8 GPU-hours, prior project estimate $7.20 plus GPT-4.1 judge usage.
# TP4, 32 browser slots and a matching 32-task gate.
# Calibrate, train one native batch if gates pass, save and stop.
# No further collection, evaluation, or allocation is submitted automatically.
#SBATCH --job-name=arm-turn-bonus-pilot-4gpu
#SBATCH --account=zixianma
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h200:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=480G
#SBATCH --time=02:00:00
#SBATCH --output=/gpfs/scrubbed/zixianma/openwebrl-runtime/logs/slurm-arm-turn-pilot-%j.out
set -euo pipefail
cd /gpfs/projects/krishna/zixianma/OpenWebRL
srun --jobid="$SLURM_JOB_ID" --nodes=1 --ntasks=1 --cpus-per-task=32 \
    --gres=gpu:h200:4 --cpu-bind=none --exact \
    /gpfs/scrubbed/zixianma/openwebrl-runtime/venv/bin/python \
    scripts/run_arm_turn_bonus_calibration.py --execute --job-id "$SLURM_JOB_ID" \
    --phase calibrate-train --beta 0.5 --gpus 4
