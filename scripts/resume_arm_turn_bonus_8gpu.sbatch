#!/usr/bin/env bash
# Each submission needs exact budget approval. Eight GPUs for eight hours; approved submission uses
# sbatch --time=08:00:00 --export=ALL,ARM_TRAIN_MINUTES=480,ARM_RESUME_FROM=...
#SBATCH --job-name=arm-turn-bonus-resume8
#SBATCH --account=zixianma
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h200:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=960G
#SBATCH --time=08:00:00
#SBATCH --output=/gpfs/scrubbed/zixianma/openwebrl-runtime/logs/slurm-arm-turn-resume-%j.out
set -euo pipefail
cd /gpfs/projects/krishna/zixianma/OpenWebRL
: "${ARM_RESUME_FROM:?Set the verified ARM run root}"
srun --jobid="$SLURM_JOB_ID" --nodes=1 --ntasks=1 --cpus-per-task=64 \
    --gres=gpu:h200:8 --cpu-bind=none --exact \
    /gpfs/scrubbed/zixianma/openwebrl-runtime/venv/bin/python \
    scripts/start_arm_resume_8gpu.py
