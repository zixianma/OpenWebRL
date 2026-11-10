#!/usr/bin/env bash
# Profile time limit and resource approval must be supplied at submission.
#SBATCH --job-name=arm-ablation-to60
#SBATCH --account=zixianma
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h200:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=960G
#SBATCH --output=/gpfs/projects/krishna/zixianma/OpenWebRL/logs/arm-ablation-to60-20260926/slurm-%j.out
set -euo pipefail
cd /gpfs/projects/krishna/zixianma/OpenWebRL
: "${ARM_ABLATION_PROFILE:?Required continuation profile}"
: "${ARM_ABLATION_RESUME:?Required preserved training root}"
source scripts/h200_env.sh
unset WANDB_SERVICE OPENWEBRL_ARM_TURN_BONUS_CONFIG OPENWEBRL_RECORD_EVAL_CONFIG
export WANDB_PROJECT=openwebrl
/gpfs/scrubbed/zixianma/openwebrl-runtime/venv/bin/python scripts/prepare_arm_ablation_to60.py \
    --job-id "$SLURM_JOB_ID" --profile "$ARM_ABLATION_PROFILE" --root "$ARM_ABLATION_RESUME"
