#!/usr/bin/env bash
# Submission requires exact approval:unused8 H200 x19:18:00, including evaluations70/80/90.
#SBATCH --job-name=arm-b69-recovery
#SBATCH --account=zixianma
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h200:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=960G
#SBATCH --time=19:18:00
#SBATCH --output=/gpfs/projects/krishna/zixianma/OpenWebRL/logs/arm-b-recovery-20260926/slurm-%j.out
set -euo pipefail
cd /gpfs/projects/krishna/zixianma/OpenWebRL
: "${ARM_B90_APPROVED:?Exact resource approval required}"
source scripts/h200_env.sh
unset WANDB_SERVICE OPENWEBRL_ARM_TURN_BONUS_CONFIG OPENWEBRL_RECORD_EVAL_CONFIG
export WANDB_PROJECT=openwebrl
/gpfs/scrubbed/zixianma/openwebrl-runtime/venv/bin/python scripts/recover_arm_b69.py --job-id "$SLURM_JOB_ID"
