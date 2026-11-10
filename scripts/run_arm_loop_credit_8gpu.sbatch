#!/usr/bin/env bash
# Idea-4 loop-credit pair, one job per variant: proposed 8 H200 x 12 h each, including full-300 at iteration 10.
# Submit with --export=ALL,ARM_LOOP_APPROVED=8gpu_12h,ARM_LOOP_VARIANT=<loop-bonus|repeat-penalty>.
#SBATCH --job-name=arm-loop-credit
#SBATCH --account=zixianma
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h200:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=960G
#SBATCH --time=12:00:00
#SBATCH --output=/gpfs/scrubbed/zixianma/openwebrl-runtime/logs/arm-loop-credit-%j.out
set -euo pipefail
cd /gpfs/projects/krishna/zixianma/OpenWebRL
: "${ARM_LOOP_APPROVED:?Exact resource approval required}" "${ARM_LOOP_VARIANT:?Variant required}"
source scripts/h200_env.sh
# Since 2026-10-15 /etc/profile.d sets http_proxy with no_proxy lacking 127.0.0.1;
# keep local selector/model/browser traffic off the cluster proxy (379243 died on this).
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}${no_proxy:+$no_proxy,}localhost,127.0.0.1,::1,$(hostname -s)"
export no_proxy="$NO_PROXY"
unset WANDB_SERVICE OPENWEBRL_ARM_TURN_BONUS_CONFIG OPENWEBRL_ARM_FAILURE_AUX_MANIFEST OPENWEBRL_RECORD_EVAL_CONFIG
export WANDB_PROJECT=openwebrl
/gpfs/scrubbed/zixianma/openwebrl-runtime/venv/bin/python scripts/prepare_arm_loop_credit.py --job-id "$SLURM_JOB_ID" --variant "$ARM_LOOP_VARIANT" --execute
