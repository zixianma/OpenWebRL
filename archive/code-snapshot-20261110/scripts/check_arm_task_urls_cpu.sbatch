#!/usr/bin/env bash
# Prepared only: exact approval required before submission; zero GPUs/API calls.
#SBATCH --job-name=arm-live-urls
#SBATCH --account=zixianma
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=none
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/gpfs/scrubbed/zixianma/openwebrl-runtime/logs/arm-live-urls-%j.out
set -euo pipefail
cd /gpfs/projects/krishna/zixianma/OpenWebRL
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export PLAYWRIGHT_BROWSERS_PATH=/gpfs/scrubbed/zixianma/openwebrl-runtime/browsers
/gpfs/scrubbed/zixianma/openwebrl-runtime/venv/bin/python -B scripts/check_arm_task_urls.py --execute
