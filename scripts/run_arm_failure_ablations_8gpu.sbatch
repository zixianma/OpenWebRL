#!/usr/bin/env bash
# Independent weight/coverage from iteration0: approved up to8H200x24h each.
#SBATCH --job-name=arm-failure-ablation
#SBATCH --account=zixianma
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h200:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=960G
#SBATCH --time=24:00:00
#SBATCH --output=/gpfs/scrubbed/zixianma/openwebrl-runtime/logs/slurm-arm-failure-ablation-%j.out
set -euo pipefail
cd /gpfs/projects/krishna/zixianma/OpenWebRL
: "${ARM_FAILURE_ABLATION:?Set weight or coverage}"
source scripts/h200_env.sh
export WANDB_PROJECT=openwebrl
unset WANDB_SERVICE ARM_CONTINUATION_LABEL_GUARD ARM_VARIANT_PORT_SOURCE ARM_VARIANT_OUTPUT_STAGE
/gpfs/scrubbed/zixianma/openwebrl-runtime/venv/bin/python scripts/prepare_arm_failure_ablations.py \
 --variant "$ARM_FAILURE_ABLATION" --job-id "$SLURM_JOB_ID" --execute
