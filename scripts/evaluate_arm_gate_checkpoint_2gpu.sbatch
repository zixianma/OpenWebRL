#!/usr/bin/env bash
# One full-300 B/C evaluation; explicit resource approval required before sbatch.
#SBATCH --job-name=arm-gate-iter20-eval
#SBATCH --account=zixianma
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h200:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=480G
#SBATCH --time=01:00:00
#SBATCH --output=/gpfs/scrubbed/zixianma/openwebrl-runtime/logs/slurm-arm-gate-eval-%j.out
set -euo pipefail
cd /gpfs/projects/krishna/zixianma/OpenWebRL
: "${ARM_GATE_VARIANT:?Set B or C}"
: "${ARM_EVAL_TRAINING_ROOT:?Set the continuation run root}"
unset WANDB_SERVICE OPENWEBRL_ARM_TURN_BONUS_CONFIG
export WANDB_PROJECT=openwebrl-evals
export MASTER_PORT=$((15000 + SLURM_JOB_ID % 1000))
export OPENWEBRL_EXPECTED_SCHEDULER_OFFSET=0
srun --jobid="$SLURM_JOB_ID" --nodes=1 --ntasks=1 --cpus-per-task=16 --gres=gpu:h200:2 --exact --cpu-bind=none \
 /gpfs/scrubbed/zixianma/openwebrl-runtime/venv/bin/python scripts/run_arm_gate_checkpoint_eval.py \
 --variant "$ARM_GATE_VARIANT" --training-root "$ARM_EVAL_TRAINING_ROOT" \
 --completed-iteration "${ARM_EVAL_COMPLETED_ITERATION:-20}" --job-id "$SLURM_JOB_ID" --execute
