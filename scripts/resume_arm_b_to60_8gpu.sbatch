#!/usr/bin/env bash
# Submission requires explicit approval of this new 64 GPU-hour budget.
#SBATCH --job-name=arm-b-to60-tp2
#SBATCH --account=zixianma
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h200:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=960G
#SBATCH --time=08:00:00
#SBATCH --output=/gpfs/scrubbed/zixianma/openwebrl-runtime/logs/slurm-arm-b-to60-%j.out
set -euo pipefail
cd /gpfs/projects/krishna/zixianma/OpenWebRL
: "${ARM_B_CONTINUATION_APPROVED:?Explicit approval token required}"
source scripts/h200_env.sh
export WANDB_PROJECT=openwebrl
unset WANDB_SERVICE ARM_CONTINUATION_LABEL_GUARD ARM_VARIANT_PORT_SOURCE ARM_VARIANT_OUTPUT_STAGE
/gpfs/scrubbed/zixianma/openwebrl-runtime/venv/bin/python scripts/prepare_arm_b_to60.py \
    --job-id "$SLURM_JOB_ID" --execute
