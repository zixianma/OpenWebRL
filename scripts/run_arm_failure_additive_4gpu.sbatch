#!/usr/bin/env bash
# TEMPLATE ONLY. A new submission requires explicit approval of 4 H200 x 4 hours.
#SBATCH --job-name=arm-additive-failure
#SBATCH --account=zixianma
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h200:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=480G
#SBATCH --time=04:00:00
#SBATCH --output=/gpfs/scrubbed/zixianma/openwebrl-runtime/logs/slurm-arm-additive-%j.out
set -euo pipefail
cd /gpfs/projects/krishna/zixianma/OpenWebRL
unset WANDB_SERVICE
srun --jobid="$SLURM_JOB_ID" --nodes=1 --ntasks=1 --cpus-per-task=32 --gres=gpu:h200:4 --exact --cpu-bind=none \
 /gpfs/scrubbed/zixianma/openwebrl-runtime/venv/bin/python scripts/run_arm_failure_additive.py --execute --job-id "$SLURM_JOB_ID" --minutes "${ARM_ADDITIVE_MINUTES:-240}"
