#!/usr/bin/env bash
# Prepared request only: requires approval for 4 H200 x 24h before submission.
#SBATCH --job-name=arm-allfailure-to100
#SBATCH --account=zixianma
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h200:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=480G
#SBATCH --time=24:00:00
#SBATCH --output=/gpfs/scrubbed/zixianma/openwebrl-runtime/logs/slurm-arm-allfailure-to100-%j.out
set -euo pipefail
cd /gpfs/projects/krishna/zixianma/OpenWebRL
unset WANDB_SERVICE
export ARM_VARIANT_RESUME_FROM=/gpfs/scrubbed/zixianma/openwebrl-runtime/evaluations/arm-turn-bonus-fresh-allfailure-301272
export ARM_VARIANT_TARGET_ITERATION=100
export ARM_VARIANT_HOURS=24
export ARM_CONTINUATION_LABEL_GUARD=adjacent_batch_v1
# The controller verifies GPU restoration, owns and awaits all workers, and
# exits once iteration 100 is saved or a time/health gate stops the run.
srun --jobid="$SLURM_JOB_ID" --ntasks=1 --nodes=1 --cpus-per-task=32 --gres=gpu:h200:4 --exact --cpu-bind=none \
 /gpfs/scrubbed/zixianma/openwebrl-runtime/venv/bin/python scripts/resume_arm_failure_variants.py \
 --execute --job-id "$SLURM_JOB_ID" --resume-from "$ARM_VARIANT_RESUME_FROM" --hours "$ARM_VARIANT_HOURS"
