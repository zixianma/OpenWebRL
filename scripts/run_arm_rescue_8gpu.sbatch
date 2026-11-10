#!/usr/bin/env bash
# ARM-guided in-group rescue (approved 2026-11-03, zixianma/urgent): smoke 8 H200 x 1.5 h (second smoke approved 2026-11-04); main 24 h + 8 h.
# Submit with --time matching the stage and --export=ALL,ARM_RESCUE_APPROVED=<stage>,ARM_RESCUE_STAGE=<stage>.
#SBATCH --job-name=arm-rescue
#SBATCH --account=zixianma
#SBATCH --partition=gpu-h200
#SBATCH --qos=urgent
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h200:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=960G
#SBATCH --no-requeue
#SBATCH --output=/gpfs/scrubbed/zixianma/openwebrl-runtime/logs/arm-rescue-%j.out
set -euo pipefail
cd /gpfs/projects/krishna/zixianma/OpenWebRL
: "${ARM_RESCUE_APPROVED:?Exact resource approval required}"
source scripts/h200_env.sh
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}${no_proxy:+$no_proxy,}localhost,127.0.0.1,::1,$(hostname -s)"
export no_proxy="$NO_PROXY"
unset WANDB_SERVICE OPENWEBRL_ARM_TURN_BONUS_CONFIG OPENWEBRL_ARM_FAILURE_AUX_MANIFEST OPENWEBRL_RECORD_EVAL_CONFIG
# Refresh CPU readiness (frozen source, native parse, frozen tests) inside the allocation first.
/gpfs/scrubbed/zixianma/openwebrl-runtime/venv/bin/python scripts/prepare_arm_rescue.py --prepare
/gpfs/scrubbed/zixianma/openwebrl-runtime/venv/bin/python scripts/prepare_arm_rescue.py --job-id "$SLURM_JOB_ID" --stage "$ARM_RESCUE_STAGE" --execute
