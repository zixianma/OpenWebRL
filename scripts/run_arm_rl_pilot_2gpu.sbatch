#!/usr/bin/env bash
# Each submission requires approval: 2 H200 x 2 h, 16 CPU, 480 GiB.
# Est. $3.60 GPU cost per track; GPT-4.1 API usage is additional.
# Submit separately with ARM_RL_TRACK=rescue or local_preference after approval.
#SBATCH --job-name=arm-rl-mechanics
#SBATCH --account=zixianma
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h200:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=480G
#SBATCH --time=02:00:00
#SBATCH --output=/gpfs/scrubbed/zixianma/openwebrl-runtime/logs/slurm-arm-rl-mechanics-%j.out
set -euo pipefail
cd /gpfs/projects/krishna/zixianma/OpenWebRL
srun --jobid="$SLURM_JOB_ID" --nodes=1 --ntasks=1 --cpus-per-task=16 \
    --gres=gpu:h200:2 --cpu-bind=none --exact \
    /gpfs/scrubbed/zixianma/openwebrl-runtime/venv/bin/python scripts/run_arm_rl_pilot.py \
    --execute --job-id "$SLURM_JOB_ID" --track "${ARM_RL_TRACK:?Set the approved track}"
