#!/usr/bin/env bash
# Prepared budget: 2 H200 x 6 hours = at most 12 H200-hours, plus judge API usage.
# Obtain explicit approval for this exact request before every submission.
#SBATCH --job-name=openwebrl-finish90-tp2
#SBATCH --account=zixianma
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h200:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=480G
#SBATCH --time=06:00:00
#SBATCH --output=/gpfs/scrubbed/zixianma/openwebrl-runtime/logs/slurm-qcq7i4ug-%j.out
set -euo pipefail
cd /gpfs/projects/krishna/zixianma/OpenWebRL
export OPENWEBRL_BATCH_DRIVER_JOB="${SLURM_JOB_ID:?Run through the approved sbatch allocation}"
export FLASHINFER_WORKSPACE_BASE=/gpfs/scrubbed/zixianma/openwebrl-runtime
resume_source=/gpfs/scrubbed/zixianma/openwebrl-runtime/reference-stage1-browsers32-20260911
# The preserved reference profile stops at 90 and evaluates every 10 iterations.
# Keep its final scheduled evaluation; do not set STOP_AFTER_SAVED_ROLLOUT.
python3 scripts/resume_baseline.py --job-id "$SLURM_JOB_ID" --gpus 2 \
    --source "$resume_source" --wandb-run-id qcq7i4ug --verify-resume-only --launch
python3 scripts/resume_baseline.py --job-id "$SLURM_JOB_ID" --gpus 2 \
    --source "$resume_source" --wandb-run-id qcq7i4ug --launch
