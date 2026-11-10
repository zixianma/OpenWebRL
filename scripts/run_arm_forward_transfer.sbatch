#!/usr/bin/env bash
# Submit only after exact approval and complete, validated teacher labels.
#SBATCH --job-name=arm-offline-refresh
#SBATCH --account=zixianma
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --output=/gpfs/scrubbed/zixianma/openwebrl-runtime/logs/slurm-arm-refresh-%j.out
set -euo pipefail
cd /gpfs/projects/krishna/zixianma/OpenWebRL
source scripts/h200_env.sh
export PYTHONPATH="$PWD/scripts:$PYTHONPATH"
export OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
# This foreground worker owns and awaits every evaluation/training stage.
exec /gpfs/scrubbed/zixianma/openwebrl-runtime/arm-reproduction/venv/bin/python \
    scripts/run_arm_forward_transfer.py --run
