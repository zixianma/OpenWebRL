#!/usr/bin/env bash
# Approved eight-GPU substitution: one24h allocation per variant,384 GPU-hours total.
#SBATCH --job-name=arm-gate-to60-8gpu
#SBATCH --account=zixianma
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h200:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=960G
#SBATCH --time=24:00:00
#SBATCH --output=/gpfs/scrubbed/zixianma/openwebrl-runtime/logs/slurm-arm-gate-to60-%j.out
set -euo pipefail
cd /gpfs/projects/krishna/zixianma/OpenWebRL
: "${ARM_GATE_VARIANT:?Set B or C}"
: "${ARM_GATE_RESUME_FROM:?Set completed iteration20 training root}"
source scripts/h200_env.sh
export WANDB_PROJECT=openwebrl
unset WANDB_SERVICE ARM_CONTINUATION_LABEL_GUARD ARM_VARIANT_PORT_SOURCE ARM_VARIANT_OUTPUT_STAGE
/gpfs/scrubbed/zixianma/openwebrl-runtime/venv/bin/python scripts/prepare_arm_gate_to60.py \
 --variant "$ARM_GATE_VARIANT" --target 60 \
 --resume-from "$ARM_GATE_RESUME_FROM" --job-id "$SLURM_JOB_ID" --execute
