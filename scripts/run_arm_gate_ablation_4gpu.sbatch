#!/usr/bin/env bash
# Submit once per B/C variant only after exact resource/budget approval.
#SBATCH --job-name=arm-min2-ablation
#SBATCH --account=zixianma
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h200:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=480G
#SBATCH --time=08:00:00
#SBATCH --output=/gpfs/scrubbed/zixianma/openwebrl-runtime/logs/slurm-arm-gate-%j.out
set -euo pipefail
cd /gpfs/projects/krishna/zixianma/OpenWebRL
unset WANDB_SERVICE
: "${ARM_GATE_VARIANT:?Choose B or C}"
srun --jobid="$SLURM_JOB_ID" --ntasks=1 --nodes=1 --cpus-per-task=32 --gres=gpu:h200:4 --exact --cpu-bind=none \
 /gpfs/scrubbed/zixianma/openwebrl-runtime/venv/bin/python scripts/run_arm_gate_ablation.py \
 --execute --job-id "$SLURM_JOB_ID" --variant "$ARM_GATE_VARIANT"
