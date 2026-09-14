#!/usr/bin/env bash
#SBATCH --job-name=openwebrl-stage2
#SBATCH --account=zixianma
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h200:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=480G
#SBATCH --time=08:00:00
#SBATCH --output=/gpfs/scrubbed/zixianma/openwebrl-runtime/logs/slurm-stage2-%j.out
# First stage-2 allocation only: 32 H200 GPU-hours. Explicit approval required.
# Further continuations use resume_baseline.py and the activated current pointer.
set -euo pipefail
cd /gpfs/projects/krishna/zixianma/OpenWebRL
export OPENWEBRL_BATCH_DRIVER_JOB="${SLURM_JOB_ID:?Requires an explicitly approved batch allocation}"
export WANDB_PROJECT=openwebrl
stage2_state=/gpfs/scrubbed/zixianma/openwebrl-runtime/baseline_stage2_prepared_state.json
python3 scripts/resume_baseline.py --job-id "$SLURM_JOB_ID" --gpus 4 \
    --state "$stage2_state" --wandb-run-id qcq7i4ug --verify-resume-only --launch
python3 scripts/prepare_baseline_stage2.py --activate --job-id "$SLURM_JOB_ID"
python3 scripts/resume_baseline.py --job-id "$SLURM_JOB_ID" --gpus 4 \
    --wandb-run-id qcq7i4ug --launch
