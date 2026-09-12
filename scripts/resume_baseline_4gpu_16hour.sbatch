#!/usr/bin/env bash
#SBATCH --job-name=openwebrl-qcq7i4ug-16h
#SBATCH --account=zixianma
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h200:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=480G
#SBATCH --time=16:00:00
#SBATCH --output=/gpfs/scrubbed/zixianma/openwebrl-runtime/logs/slurm-qcq7i4ug-%j.out
# Budget: 64 H200 GPU-hours, estimated $57.60 plus judge usage.
# Every submission requires explicit user approval; this template grants none.
set -euo pipefail
cd /gpfs/projects/krishna/zixianma/OpenWebRL
export OPENWEBRL_BATCH_DRIVER_JOB="${SLURM_JOB_ID:?Run through an approved allocation}"
resume_source=/gpfs/scrubbed/zixianma/openwebrl-runtime/reference-stage1-browsers32-16hour-20260912
python3 scripts/resume_baseline.py --job-id "$SLURM_JOB_ID" --gpus 4 \
    --source "$resume_source" --wandb-run-id qcq7i4ug --verify-resume-only --launch
python3 scripts/resume_baseline.py --job-id "$SLURM_JOB_ID" --gpus 4 \
    --source "$resume_source" --wandb-run-id qcq7i4ug --launch
