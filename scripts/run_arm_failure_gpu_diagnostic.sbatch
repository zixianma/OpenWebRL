#!/usr/bin/env bash
# Approved 2026-09-22: 2 H200 x 1 hour, 16 CPUs, 480 GiB. No follow-on allocation.
#SBATCH --job-name=arm-failure-gpu-diagnostic
#SBATCH --account=zixianma
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h200:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=480G
#SBATCH --time=01:00:00
#SBATCH --output=/gpfs/scrubbed/zixianma/openwebrl-runtime/logs/slurm-arm-failure-diagnostic-%j.out
set -euo pipefail
cd /gpfs/projects/krishna/zixianma/OpenWebRL
exec srun --jobid="$SLURM_JOB_ID" --nodes=1 --ntasks=1 --cpus-per-task=16 \
    --gres=gpu:h200:2 --cpu-bind=none --exact \
    /gpfs/scrubbed/zixianma/openwebrl-runtime/venv/bin/python \
    scripts/run_arm_failure_gpu_diagnostic.py --execute --job-id "$SLURM_JOB_ID"
