#!/usr/bin/env bash
# C continuation uses the established 4 H200 x 7h profile.
#SBATCH --job-name=arm-gate-c-continue
#SBATCH --account=zixianma
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h200:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=480G
#SBATCH --time=07:00:00
#SBATCH --output=/gpfs/scrubbed/zixianma/openwebrl-runtime/logs/slurm-arm-gate-c-continue-%j.out
set -euo pipefail
cd /gpfs/projects/krishna/zixianma/OpenWebRL
unset WANDB_SERVICE ARM_CONTINUATION_LABEL_GUARD
export ARM_VARIANT_TARGET_ITERATION=10
export ARM_VARIANT_MULTIMODAL_STORAGE=shared
srun --jobid="$SLURM_JOB_ID" --ntasks=1 --nodes=1 --cpus-per-task=32 --gres=gpu:h200:4 --exact --cpu-bind=none \
 /gpfs/scrubbed/zixianma/openwebrl-runtime/venv/bin/python scripts/resume_arm_failure_variants.py \
 --execute --job-id "$SLURM_JOB_ID" \
 --resume-from /gpfs/scrubbed/zixianma/openwebrl-runtime/evaluations/arm-failure-additive-311203 --hours 7
