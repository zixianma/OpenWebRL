#!/usr/bin/env bash
# Established full-300 profile; every new submission needs budget approval.
# Override --time and ARM_EVAL_COMPLETED_ITERATION for an approved new checkpoint.
#SBATCH --job-name=arm-iter80-eval
#SBATCH --account=zixianma
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h200:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=480G
#SBATCH --time=02:00:00
#SBATCH --output=/gpfs/scrubbed/zixianma/openwebrl-runtime/logs/slurm-arm-iter80-eval-%j.out
set -euo pipefail
cd /gpfs/projects/krishna/zixianma/OpenWebRL
: "${ARM_EVAL_VARIANT:?Set original, additive, or allfailure}"
evaluation_args=(--completed-iteration "${ARM_EVAL_COMPLETED_ITERATION:-80}")
if [[ -n "${ARM_EVAL_TRAINING_ROOT:-}" ]]; then
 evaluation_args+=(--training-root "$ARM_EVAL_TRAINING_ROOT")
fi
unset WANDB_SERVICE OPENWEBRL_ARM_TURN_BONUS_CONFIG
export WANDB_PROJECT=openwebrl-evals
export MASTER_PORT=$((15000 + SLURM_JOB_ID % 1000))
export OPENWEBRL_EXPECTED_SCHEDULER_OFFSET=0
# This batch process owns and awaits its worker. A failure exits and releases
# the allocation; durable task archives are retained for supervised recovery.
srun --jobid="$SLURM_JOB_ID" --nodes=1 --ntasks=1 --cpus-per-task=16 --gres=gpu:h200:2 --cpu-bind=none --exact \
 /gpfs/scrubbed/zixianma/openwebrl-runtime/venv/bin/python scripts/run_arm_iteration80_eval.py \
 --variant "$ARM_EVAL_VARIANT" --job-id "$SLURM_JOB_ID" "${evaluation_args[@]}" --execute
