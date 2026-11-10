#!/usr/bin/env bash
# Prepared proposal; each submission requires explicit budget approval.
# B: --time=18:00:00 after job 311964 ends. C: --time=12:00:00.
#SBATCH --job-name=arm-gate-to20
#SBATCH --account=zixianma
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h200:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=480G
#SBATCH --time=12:00:00
#SBATCH --output=/gpfs/scrubbed/zixianma/openwebrl-runtime/logs/slurm-arm-gate-to20-%j.out
set -euo pipefail
cd /gpfs/projects/krishna/zixianma/OpenWebRL
: "${ARM_GATE_VARIANT:?Set B or C}"
case "$ARM_GATE_VARIANT" in
 B) prior=arm-failure-additive-311964; hours=18 ;;
 C) prior=arm-failure-additive-311965; hours=12 ;;
 *) exit 2 ;;
esac
export ARM_VARIANT_TARGET_ITERATION=20
export ARM_VARIANT_MULTIMODAL_STORAGE=shared
export WANDB_PROJECT=openwebrl
unset WANDB_SERVICE ARM_CONTINUATION_LABEL_GUARD
srun --jobid="$SLURM_JOB_ID" --nodes=1 --ntasks=1 --cpus-per-task=32 --gres=gpu:h200:4 --exact --cpu-bind=none \
 /gpfs/scrubbed/zixianma/openwebrl-runtime/venv/bin/python scripts/resume_arm_failure_variants.py \
 --job-id "$SLURM_JOB_ID" --hours "$hours" \
 --resume-from "/gpfs/scrubbed/zixianma/openwebrl-runtime/evaluations/$prior" --execute
