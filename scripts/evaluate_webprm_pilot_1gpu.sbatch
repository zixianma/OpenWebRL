#!/usr/bin/env bash
# Approved 2026-10-09: 1 H200 x 2h, 8 CPUs, 120GiB, API $30 incl retries.
#SBATCH --job-name=webprm-om2w-pilot
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
#SBATCH --output=/gpfs/scrubbed/zixianma/openwebrl-runtime/logs/webprm-pilot-%j.out
set -euo pipefail
cd /gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-baselines-20261009/code-v2
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export WANDB_PROJECT=openwebrl-evals
export WEBPRM_ALLOCATION_SECONDS=${WEBPRM_ALLOCATION_SECONDS:-7200}
export CUDA_HOME=/gpfs/scrubbed/zixianma/openwebrl-runtime/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
exec /gpfs/scrubbed/zixianma/openwebrl-runtime/venv/bin/python scripts/launch_webprm_pilot.py \
  --root /gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-baselines-20261009 \
  --browser-python /gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-baselines-20261009/browser-env/bin/python \
  --engine-python /gpfs/scrubbed/zixianma/openwebrl-runtime/venv/bin/python \
  --env-file /gpfs/projects/krishna/zixianma/OpenWebRL/.env
