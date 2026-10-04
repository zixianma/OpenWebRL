#!/usr/bin/env bash
#SBATCH --job-name=arm-actions-only-om2w
#SBATCH --account=zixianma
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=02:30:00
#SBATCH --signal=B:TERM@180
#SBATCH --output=/gpfs/scrubbed/zixianma/openwebrl-runtime/logs/arm-actions-only-%j.out
# Exact GPU allocation approval required before sbatch.
set -euo pipefail
cd /gpfs/projects/krishna/zixianma/OpenWebRL
source scripts/h200_env.sh
exec /gpfs/scrubbed/zixianma/openwebrl-runtime/arm-reproduction/venv/bin/python scripts/run_arm_compact_eval.py
