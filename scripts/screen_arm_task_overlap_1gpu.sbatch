#!/usr/bin/env bash
# Prepared only. Submit after exact 1 H200 x 30 min approval.
#SBATCH --job-name=arm-task-overlap
#SBATCH --account=zixianma
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=00:30:00
#SBATCH --output=/gpfs/scrubbed/zixianma/openwebrl-runtime/logs/arm-task-overlap-%j.out
set -euo pipefail
cd /gpfs/projects/krishna/zixianma/OpenWebRL
source scripts/h200_env.sh
export PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
export HF_HOME=/gpfs/scrubbed/zixianma/openwebrl-runtime/hf-cache
/gpfs/scrubbed/zixianma/openwebrl-runtime/venv/bin/python -B scripts/screen_arm_task_overlap.py --execute
