#!/usr/bin/env bash
# Approved reduced subset: one7h job each for Additive90 and GateB90.
#SBATCH --job-name=arm-stealth90-tp1
#SBATCH --account=zixianma
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --time=07:00:00
#SBATCH --output=/gpfs/scrubbed/zixianma/openwebrl-runtime/logs/arm-stealth90-tp1-%j.out
set -euo pipefail
cd /gpfs/projects/krishna/zixianma/OpenWebRL
source scripts/h200_env.sh
unset WANDB_SERVICE OPENWEBRL_ARM_TURN_BONUS_CONFIG OPENWEBRL_ARM_FAILURE_AUX_MANIFEST OPENWEBRL_RECORD_EVAL_CONFIG
export WANDB_PROJECT=openwebrl-evals
export PYTHONDONTWRITEBYTECODE=1
/gpfs/scrubbed/zixianma/openwebrl-runtime/venv/bin/python -B scripts/run_arm_stealth90_parallel.py --execute --job-id "$SLURM_JOB_ID" --method "${1:?method required}" --repeat "${2:?repeat required}"
