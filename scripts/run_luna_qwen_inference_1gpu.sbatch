#!/usr/bin/env bash
# Exact four-hour allocation and $15 Luna / $5 judge caps require approval.
#SBATCH --job-name=luna-qwen-pilot
#SBATCH --account=zixianma
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --output=/gpfs/scrubbed/zixianma/openwebrl-runtime/logs/slurm-luna-qwen-%j.out
set -euo pipefail
cd /gpfs/projects/krishna/zixianma/OpenWebRL
export WANDB_PROJECT=openwebrl-evals
exec /gpfs/scrubbed/zixianma/openwebrl-runtime/venv/bin/python scripts/prepare_luna_qwen_inference.py --execute "$SLURM_JOB_ID"
