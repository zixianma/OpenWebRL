#!/usr/bin/env bash
#SBATCH --job-name=openwebrl-qcq7i4ug-tp4
#SBATCH --account=zixianma
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h200:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=480G
#SBATCH --time=08:00:00
#SBATCH --output=/gpfs/scrubbed/zixianma/openwebrl-runtime/logs/slurm-qcq7i4ug-%j.out
# Template budget: four H200 GPUs for eight hours (32 H200 GPU-hours).
# Obtain explicit approval for every submission; this file grants no authorization.
set -euo pipefail
cd /gpfs/projects/krishna/zixianma/OpenWebRL
export OPENWEBRL_BATCH_DRIVER_JOB="${SLURM_JOB_ID:?Run this driver through the approved sbatch allocation}"
resume_source=/gpfs/scrubbed/zixianma/openwebrl-runtime/reference-stage1-tp4-cache-release-20260909
python3 scripts/resume_baseline.py --job-id "$SLURM_JOB_ID" --gpus 4 \
    --source "$resume_source" --wandb-run-id qcq7i4ug --verify-resume-only --launch
python3 scripts/resume_baseline.py --job-id "$SLURM_JOB_ID" --gpus 4 \
    --source "$resume_source" --wandb-run-id qcq7i4ug --launch
