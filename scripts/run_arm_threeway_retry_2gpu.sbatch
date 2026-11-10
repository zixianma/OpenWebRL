#!/usr/bin/env bash
#SBATCH --job-name=arm-threeway-retry
#SBATCH --account=zixianma
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --gpus=h200:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=480G
#SBATCH --time=01:30:00
#SBATCH --output=/gpfs/scrubbed/zixianma/openwebrl-runtime/logs/slurm-arm-threeway-retry-%j.out
set -euo pipefail
cd /gpfs/projects/krishna/zixianma/OpenWebRL
unset WANDB_SERVICE
exec /gpfs/scrubbed/zixianma/openwebrl-runtime/venv/bin/python scripts/run_arm_threeway_eval.py --retry-controller --job-id "$SLURM_JOB_ID"
