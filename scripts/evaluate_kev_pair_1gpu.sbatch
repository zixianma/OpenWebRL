#!/usr/bin/env bash
# Submit after exact approval: 1 H200, 8 CPUs, 120 GiB, 2 hours total;
# 20 Browser Use sessions; <=3600 text and 80 judge API attempts including retries.
#SBATCH --job-name=kev-pair-om2w
#SBATCH --account=zixianma
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --signal=B:TERM@180
#SBATCH --output=/gpfs/scrubbed/zixianma/openwebrl-runtime/logs/kev-pair-om2w-%j.out
set -euo pipefail
cd /gpfs/projects/krishna/zixianma/OpenWebRL
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export WANDB_PROJECT=openwebrl-evals
export CPATH="/gpfs/scrubbed/zixianma/openwebrl-runtime/src/python-headers/Include:/gpfs/scrubbed/zixianma/openwebrl-runtime/src/python-headers${CPATH:+:$CPATH}"
export PLAYWRIGHT_BROWSERS_PATH=/gpfs/scrubbed/zixianma/openwebrl-runtime/browsers
exec /gpfs/scrubbed/zixianma/openwebrl-runtime/jev-eval-venv/bin/python \
  scripts/evaluate_kev_pair.py --execute "$@"
