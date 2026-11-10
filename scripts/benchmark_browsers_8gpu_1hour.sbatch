#!/usr/bin/env bash
#SBATCH --job-name=openwebrl-browser-scale
#SBATCH --account=zixianma
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h200:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=960G
#SBATCH --time=01:00:00
#SBATCH --output=/gpfs/scrubbed/zixianma/openwebrl-runtime/logs/slurm-browser-scale-%j.out
# Approved: eight H200 GPU-hours, estimated $7.20 plus judge usage.
# Two independent SFT starts; no writes to the existing RL lineage.
set -euo pipefail
cd /gpfs/projects/krishna/zixianma/OpenWebRL
python3 scripts/scaling_benchmark.py --controller "${SLURM_JOB_ID:?Approved allocation required}" --browser-pair
