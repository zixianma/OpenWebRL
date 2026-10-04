#!/usr/bin/env bash
# Each submission needs exact budget approval. Four GPUs for sixteen hours; approved submission uses
# sbatch --time=16:00:00 --export=ALL,ARM_TRAIN_MINUTES=480,ARM_RESUME_FROM=...
#SBATCH --job-name=arm-turn-bonus-resume4
#SBATCH --account=zixianma
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h200:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=480G
#SBATCH --time=16:00:00
#SBATCH --output=/gpfs/scrubbed/zixianma/openwebrl-runtime/logs/slurm-arm-turn-resume-%j.out
set -euo pipefail
cd /gpfs/projects/krishna/zixianma/OpenWebRL
: "${ARM_RESUME_FROM:?Set the verified ARM run root}"
srun --jobid="$SLURM_JOB_ID" --nodes=1 --ntasks=1 --cpus-per-task=32 \
    --gres=gpu:h200:4 --cpu-bind=none --exact \
    /gpfs/scrubbed/zixianma/openwebrl-runtime/venv/bin/python \
    scripts/start_arm_resume_8gpu.py
