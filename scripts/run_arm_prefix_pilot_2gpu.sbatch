#!/usr/bin/env bash
# Requires exact approval: 2 H200 x2h, 16 CPUs, 480 GiB; GPT-4.1 cap $20 /900 calls.
#SBATCH --job-name=arm-prefix-pilot
#SBATCH --account=zixianma
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h200:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=480G
#SBATCH --time=02:00:00
#SBATCH --output=/gpfs/scrubbed/zixianma/openwebrl-runtime/logs/slurm-arm-prefix-pilot-%j.out
set -euo pipefail
cd /gpfs/projects/krishna/zixianma/OpenWebRL
export WANDB_PROJECT=openwebrl-evals
unset WANDB_SERVICE OPENWEBRL_ARM_TURN_BONUS_CONFIG OPENWEBRL_ARM_CONFIG
/gpfs/scrubbed/zixianma/openwebrl-runtime/venv/bin/python scripts/run_arm_prefix_pilot.py --job-id "$SLURM_JOB_ID" --execute
