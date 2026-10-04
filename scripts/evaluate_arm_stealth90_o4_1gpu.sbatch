#!/usr/bin/env bash
# Three methods; each new allocation requires the exact recorded resource approval.
#SBATCH --job-name=arm-stealth90-o4-t06
#SBATCH --account=zixianma
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --time=07:00:00
#SBATCH --output=/gpfs/scrubbed/zixianma/openwebrl-runtime/logs/arm-stealth90-o4-t06-%j.out
set -euo pipefail
cd /gpfs/projects/krishna/zixianma/OpenWebRL
source scripts/h200_env.sh
unset WANDB_SERVICE OPENWEBRL_ARM_TURN_BONUS_CONFIG OPENWEBRL_ARM_FAILURE_AUX_MANIFEST OPENWEBRL_RECORD_EVAL_CONFIG
export WANDB_PROJECT=openwebrl-evals PYTHONDONTWRITEBYTECODE=1
/gpfs/scrubbed/zixianma/openwebrl-runtime/venv/bin/python -B scripts/run_arm_stealth90_o4.py --execute --job-id "$SLURM_JOB_ID" --method "${1:?method required}" "${@:2}"
