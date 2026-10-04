#!/usr/bin/env bash
# Prepared pilot only. Submit after exact approval: 0 GPU, 4 CPU, 8 GiB, 2 hours,
# 10 Browser Use sessions (12 minutes each), and bounded Jev/text/judge API calls.
#SBATCH --job-name=jev-om2w-pilot
#SBATCH --account=zixianma
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=none
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --signal=B:TERM@180
#SBATCH --output=/gpfs/scrubbed/zixianma/openwebrl-runtime/logs/jev-om2w-%j.out
set -euo pipefail
cd /gpfs/projects/krishna/zixianma/OpenWebRL
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export WANDB_PROJECT=openwebrl-evals
export PLAYWRIGHT_BROWSERS_PATH=/gpfs/scrubbed/zixianma/openwebrl-runtime/browsers
exec /gpfs/scrubbed/zixianma/openwebrl-runtime/jev-eval-venv/bin/python \
  scripts/evaluate_jev_ultrafast.py --execute "$@"
