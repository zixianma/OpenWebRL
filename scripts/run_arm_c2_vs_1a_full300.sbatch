#!/usr/bin/env bash
#SBATCH --job-name=arm-c2-vs-1a-holdout200
#SBATCH --account=zixianma
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h200:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --time=01:30:00
#SBATCH --output=/gpfs/scrubbed/zixianma/openwebrl-runtime/logs/arm-c2-vs-1a-holdout200-%j.out
# Approved 2026-09-09: two H200s for 1.5 hours (three H200-hours maximum).
set -euo pipefail
cd /gpfs/projects/krishna/zixianma/OpenWebRL
source scripts/h200_env.sh
exec /gpfs/scrubbed/zixianma/openwebrl-runtime/arm-reproduction/venv/bin/python \
  scripts/run_arm_c2_vs_1a_full300.py --job-id "$SLURM_JOB_ID"
