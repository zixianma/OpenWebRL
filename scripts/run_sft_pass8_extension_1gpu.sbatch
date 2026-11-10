#!/usr/bin/env bash
# Four independent submissions, SHARD=0..3, extending only the 707 pass@5 misses.
# Each shard's 10 hours is TOTAL across all scheduler attempts, including failures.
#SBATCH --job-name=sft-pass8-misses
#SBATCH --account=zixianma
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=10:00:00
#SBATCH --output=/gpfs/scrubbed/zixianma/openwebrl-runtime/logs/slurm-sft-pass8-misses-%j.out
set -euo pipefail
export WANDB_PROJECT=openwebrl-evals
unset WANDB_SERVICE OPENWEBRL_ARM_TURN_BONUS_CONFIG OPENWEBRL_ARM_CONFIG
exec /gpfs/scrubbed/zixianma/openwebrl-runtime/venv/bin/python -B \
    /gpfs/scrubbed/zixianma/openwebrl-runtime/reference-sft-pass8-misses-20261006-v1/scripts/run_sft_pass8_extension.py \
    --job-id "$SLURM_JOB_ID" --shard "${SHARD:?Set SHARD=0..3}" --execute
