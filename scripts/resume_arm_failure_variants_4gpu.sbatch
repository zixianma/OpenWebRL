#!/usr/bin/env bash
# Each submission requires exact resource/budget approval.
#SBATCH --job-name=arm-failure-to90
#SBATCH --account=zixianma
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h200:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=480G
#SBATCH --time=16:00:00
#SBATCH --output=/gpfs/scrubbed/zixianma/openwebrl-runtime/logs/slurm-arm-failure-resume-%j.out
set -euo pipefail
cd /gpfs/projects/krishna/zixianma/OpenWebRL
unset WANDB_SERVICE
: "${ARM_VARIANT_RESUME_FROM:?}" "${ARM_VARIANT_HOURS:?}" "${ARM_VARIANT_TARGET_ITERATION:?}"
# Accept either the run root or its runtime/checkpoint subdirectory.  The
# validator consumes the run root, while checkpoint paths are conventionally
# reported with the runtime suffix.
if [[ "$ARM_VARIANT_RESUME_FROM" == */runtime ]]; then
  ARM_VARIANT_RESUME_FROM="${ARM_VARIANT_RESUME_FROM%/runtime}"
  export ARM_VARIANT_RESUME_FROM
fi
srun --jobid="$SLURM_JOB_ID" --ntasks=1 --nodes=1 --cpus-per-task=32 --gres=gpu:h200:4 --exact --cpu-bind=none \
 /gpfs/scrubbed/zixianma/openwebrl-runtime/venv/bin/python scripts/resume_arm_failure_variants.py \
 --execute --job-id "$SLURM_JOB_ID" --resume-from "$ARM_VARIANT_RESUME_FROM" --hours "$ARM_VARIANT_HOURS"
