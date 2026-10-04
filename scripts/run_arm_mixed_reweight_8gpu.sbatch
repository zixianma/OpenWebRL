#!/usr/bin/env bash
# Proposed exact budget:8 H200 x8h per variant, including full300 at iteration10.
#SBATCH --job-name=arm-mixed-relaxed
#SBATCH --account=zixianma
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h200:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=960G
#SBATCH --time=08:00:00
#SBATCH --output=/gpfs/scrubbed/zixianma/openwebrl-runtime/logs/arm-mixed-relaxed-%j.out
set -euo pipefail
cd /gpfs/projects/krishna/zixianma/OpenWebRL
: "${ARM_MIXED_PAIR_APPROVED:?Exact resource approval required}"
: "${ARM_MIXED_VARIANT:?Expected bonus or reweight}"
source scripts/h200_env.sh
unset WANDB_SERVICE OPENWEBRL_ARM_TURN_BONUS_CONFIG OPENWEBRL_ARM_FAILURE_AUX_MANIFEST OPENWEBRL_RECORD_EVAL_CONFIG
export WANDB_PROJECT=openwebrl
/gpfs/scrubbed/zixianma/openwebrl-runtime/venv/bin/python scripts/prepare_arm_mixed_reweight.py --job-id "$SLURM_JOB_ID" --variant "$ARM_MIXED_VARIANT" --execute
