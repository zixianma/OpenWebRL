#!/usr/bin/env bash
# Proposal only: each variant needs two dependent 4-H200 x24h stages, approval first.
#SBATCH --job-name=arm-gate-stage
#SBATCH --account=zixianma
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h200:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=480G
#SBATCH --time=24:00:00
#SBATCH --output=/gpfs/scrubbed/zixianma/openwebrl-runtime/logs/slurm-arm-gate-stage-%j.out
set -euo pipefail
cd /gpfs/projects/krishna/zixianma/OpenWebRL
: "${ARM_GATE_VARIANT:?Set B or C}"
: "${ARM_GATE_TARGET_ITERATION:?Set40 or60}"
: "${ARM_GATE_RESUME_FROM:?Set preceding completed training root}"
source scripts/h200_env.sh
export WANDB_PROJECT=openwebrl
unset WANDB_SERVICE ARM_CONTINUATION_LABEL_GUARD ARM_VARIANT_PORT_SOURCE
/gpfs/scrubbed/zixianma/openwebrl-runtime/venv/bin/python scripts/prepare_arm_gate_to60.py \
 --variant "$ARM_GATE_VARIANT" --target "$ARM_GATE_TARGET_ITERATION" \
 --resume-from "$ARM_GATE_RESUME_FROM" --job-id "$SLURM_JOB_ID" --execute
