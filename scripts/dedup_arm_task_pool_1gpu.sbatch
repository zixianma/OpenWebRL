#!/usr/bin/env bash
# Prepared request: 1 H200 x3 hours,8 CPUs,120GiB; includes all retry time.
# Submit only after exact approval and recording this job's remaining budget.
#SBATCH --job-name=arm-task-curation
#SBATCH --account=zixianma
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=03:00:00
#SBATCH --output=/gpfs/scrubbed/zixianma/openwebrl-runtime/logs/arm-task-curation-%j.out
set -euo pipefail
cd /gpfs/projects/krishna/zixianma/OpenWebRL
source scripts/h200_env.sh
export PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
export HF_HOME=/gpfs/scrubbed/zixianma/openwebrl-runtime/hf-cache
/gpfs/scrubbed/zixianma/openwebrl-runtime/venv/bin/python -B scripts/dedup_arm_task_pool.py --execute
