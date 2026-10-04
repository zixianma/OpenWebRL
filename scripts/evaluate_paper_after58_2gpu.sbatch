#!/usr/bin/env bash
# Approved: 2 H200 x 2 h (4 GPU-h, est. $3.60), plus benchmark API usage.
#SBATCH --job-name=openwebrl-paper-after58
#SBATCH --account=zixianma
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h200:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=480G
#SBATCH --time=02:00:00
#SBATCH --output=/gpfs/scrubbed/zixianma/openwebrl-runtime/logs/slurm-paper-after58-%j.out
set -euo pipefail
cd /gpfs/projects/krishna/zixianma/OpenWebRL
srun --jobid="$SLURM_JOB_ID" --nodes=1 --ntasks=1 --cpus-per-task=16 \
    --gres=gpu:h200:2 --cpu-bind=none --exact \
    /gpfs/scrubbed/zixianma/openwebrl-runtime/venv/bin/python scripts/evaluate_baseline_checkpoint.py \
    --execute --job-id "$SLURM_JOB_ID" --gpus 2 --browser-env browser-use --protocol benchmark \
    --source /gpfs/scrubbed/zixianma/openwebrl-runtime/reference-paper-om2w-20260912 \
    --checkpoint /gpfs/scrubbed/zixianma/openwebrl-runtime/runs/openwebrl-4b-reference-288861-20260912T040027/iter_0000057 \
    --output "/gpfs/scrubbed/zixianma/openwebrl-runtime/evaluations/qcq7i4ug-paper-$SLURM_JOB_ID-after58"
