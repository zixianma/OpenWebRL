#!/usr/bin/env bash
#SBATCH --job-name=openwebrl-topology-8gpu
#SBATCH --account=zixianma
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h200:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=960G
#SBATCH --time=02:00:00
#SBATCH --output=/gpfs/scrubbed/zixianma/openwebrl-runtime/logs/slurm-topology-%j.out
# Requires explicit approval: 16 H200 GPU-hours, estimated $14.40.
# Saved-batch optimizer replay only; no fresh browser collection or judge calls.
set -euo pipefail
cd /gpfs/projects/krishna/zixianma/OpenWebRL
python3 scripts/scaling_benchmark.py --controller "${SLURM_JOB_ID:?Approved allocation required}" --topology-only
