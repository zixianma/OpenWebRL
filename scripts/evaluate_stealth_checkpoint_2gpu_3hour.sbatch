#!/usr/bin/env bash
# Each submission needs explicit approval: 2 H200 x 3 hours = 6 GPU-hours.
# Browser Use hosting and o4-mini judge usage are additional.
# Submit checkpoint jobs sequentially to share the eight-session browser limit.
#SBATCH --job-name=openwebrl-stealth-o4
#SBATCH --account=zixianma
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h200:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=480G
#SBATCH --time=03:00:00
#SBATCH --output=/gpfs/scrubbed/zixianma/openwebrl-runtime/logs/slurm-stealth-o4-%j.out
set -euo pipefail
cd /gpfs/projects/krishna/zixianma/OpenWebRL
benchmark_checkpoint="${OPENWEBRL_STEALTH_CHECKPOINT:?Set the selected checkpoint directory}"
benchmark_iteration="${OPENWEBRL_STEALTH_ITERATION:?Set the completed training iteration}"
benchmark_expected=$(printf 'iter_%07d' "$((benchmark_iteration - 1))")
[[ "${benchmark_checkpoint##*/}" == "$benchmark_expected" ]] || {
    printf 'Checkpoint path and completed iteration disagree\n' >&2
    exit 1
}
srun --jobid="$SLURM_JOB_ID" --nodes=1 --ntasks=1 --cpus-per-task=16 \
    --gres=gpu:h200:2 --cpu-bind=none --exact \
    /gpfs/scrubbed/zixianma/openwebrl-runtime/venv/bin/python scripts/evaluate_baseline_checkpoint.py \
    --execute --job-id "$SLURM_JOB_ID" --gpus 2 --browser-env browser-use --protocol benchmark \
    --source /gpfs/scrubbed/zixianma/openwebrl-runtime/reference-paper-om2w-20260912 \
    --checkpoint "$benchmark_checkpoint" \
    --output "/gpfs/scrubbed/zixianma/openwebrl-runtime/evaluations/qcq7i4ug-stealth-o4-$SLURM_JOB_ID-after$benchmark_iteration"
