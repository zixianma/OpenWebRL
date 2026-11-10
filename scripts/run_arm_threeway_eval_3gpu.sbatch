#!/usr/bin/env bash
#SBATCH --job-name=arm-threeway-100
#SBATCH --account=zixianma
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --gpus-per-task=h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=720G
#SBATCH --time=02:00:00
#SBATCH --output=/gpfs/scrubbed/zixianma/openwebrl-runtime/logs/slurm-arm-threeway-%j.out
set -euo pipefail
cd /gpfs/projects/krishna/zixianma/OpenWebRL
exec /gpfs/scrubbed/zixianma/openwebrl-runtime/venv/bin/python scripts/run_arm_threeway_eval.py --controller --job-id "$SLURM_JOB_ID"
