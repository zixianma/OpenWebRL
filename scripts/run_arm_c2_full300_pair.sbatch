#!/usr/bin/env bash
#SBATCH --job-name=arm-c2-u500-u700-full300
#SBATCH --account=zixianma
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --output=/gpfs/scrubbed/zixianma/openwebrl-runtime/logs/arm-c2-full300-%j.out
# Approved budget must be exactly one H200 for four hours (4 H200-hours).
# This file is a prepared artifact and never grants its own submission approval.
set -euo pipefail
cd /gpfs/projects/krishna/zixianma/OpenWebRL
source scripts/h200_env.sh
run_root=/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-reproduction/runs/c2-full-282782-20260908T075414Z
exec /gpfs/scrubbed/zixianma/openwebrl-runtime/arm-reproduction/venv/bin/python \
  scripts/run_arm_c2_full300_pair.py --job-id "$SLURM_JOB_ID" --run-root "$run_root"
