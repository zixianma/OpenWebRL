#!/usr/bin/env bash
# Awaiting explicit approval: one2-H200 x24h allocation plus browser/judge usage.
#SBATCH --job-name=arm-stealth90-threeway-r3
#SBATCH --account=zixianma
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h200:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=480G
#SBATCH --time=24:00:00
#SBATCH --output=/gpfs/scrubbed/zixianma/openwebrl-runtime/logs/arm-stealth90-repeats-%j.out
set -euo pipefail
cd /gpfs/projects/krishna/zixianma/OpenWebRL
source scripts/h200_env.sh
unset WANDB_SERVICE OPENWEBRL_ARM_TURN_BONUS_CONFIG OPENWEBRL_ARM_FAILURE_AUX_MANIFEST OPENWEBRL_RECORD_EVAL_CONFIG
export WANDB_PROJECT=openwebrl-evals
export PYTHONDONTWRITEBYTECODE=1
/gpfs/scrubbed/zixianma/openwebrl-runtime/venv/bin/python -B scripts/run_arm_stealth90_repeats.py --job-id "$SLURM_JOB_ID" --execute
