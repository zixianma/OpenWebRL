#!/usr/bin/env bash
# Do not submit without exact user approval: 4 H200 x 3 hours, 32 CPU, 480 GiB.
#SBATCH --job-name=arm-failure-coverage
#SBATCH --account=zixianma
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h200:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=480G
#SBATCH --time=03:00:00
#SBATCH --output=/gpfs/scrubbed/zixianma/openwebrl-runtime/logs/slurm-arm-failure-coverage-%j.out
set -euo pipefail
cd /gpfs/projects/krishna/zixianma/OpenWebRL
unset WANDB_SERVICE
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
srun --jobid="$SLURM_JOB_ID" --ntasks=1 --nodes=1 --cpus-per-task=32 --gres=gpu:h200:4 --exact --cpu-bind=none \
 /gpfs/scrubbed/zixianma/openwebrl-runtime/venv/bin/python scripts/prepare_arm_failure_coverage.py \
 --execute --job-id "$SLURM_JOB_ID"
