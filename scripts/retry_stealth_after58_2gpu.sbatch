#!/usr/bin/env bash
# Each submission requires explicit approval: 2 H200 x 1 hour, 2 GPU-hours.
# Estimated GPU cost $1.80; Browser Use and o4-mini usage are additional.
#SBATCH --job-name=openwebrl-after58-retry
#SBATCH --account=zixianma
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h200:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=480G
#SBATCH --time=01:00:00
#SBATCH --output=/gpfs/scrubbed/zixianma/openwebrl-runtime/logs/slurm-after58-retry-%j.out
set -euo pipefail
cd /gpfs/projects/krishna/zixianma/OpenWebRL
srun --jobid="$SLURM_JOB_ID" --nodes=1 --ntasks=1 --cpus-per-task=16 \
    --gres=gpu:h200:2 --cpu-bind=none --exact \
    /gpfs/scrubbed/zixianma/openwebrl-runtime/venv/bin/python scripts/retry_stealth_invalid.py \
    --execute --job-id "$SLURM_JOB_ID" \
    --source /gpfs/scrubbed/zixianma/openwebrl-runtime/reference-after58-invalid-retry-20260914 \
    --output "/gpfs/scrubbed/zixianma/openwebrl-runtime/evaluations/qcq7i4ug-after58-invalid-retry-$SLURM_JOB_ID"
