#!/usr/bin/env bash
# Four separately accounted CPU-only shards; explicit resource approval required.
#SBATCH --job-name=arm-live-full
#SBATCH --account=zixianma
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=none
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=13:00:00
#SBATCH --output=/gpfs/scrubbed/zixianma/openwebrl-runtime/logs/arm-live-full-%j.out
set -euo pipefail
cd /gpfs/projects/krishna/zixianma/OpenWebRL
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export PLAYWRIGHT_BROWSERS_PATH=/gpfs/scrubbed/zixianma/openwebrl-runtime/browsers
exec /gpfs/scrubbed/zixianma/openwebrl-runtime/venv/bin/python -B \
 /gpfs/scrubbed/zixianma/openwebrl-runtime/arm-turn-bonus-preparation/task-pool-expansion-20260922/curation-v3-20260929/live-browser-full-20260929/frozen/controller.py \
 --execute --shard "${1:?shard index required}"
