#!/usr/bin/env bash
# Approved: 2 H200 x 1 hour, standard 16-CPU / 480-GiB training test profile.
# No subsequent allocation is submitted by this job.
#SBATCH --job-name=arm-sol-gpu-diagnostic
#SBATCH --account=zixianma
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h200:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=480G
#SBATCH --time=01:00:00
#SBATCH --output=/gpfs/scrubbed/zixianma/openwebrl-runtime/logs/slurm-arm-sol-diagnostic-%j.out
set -euo pipefail
cd /gpfs/projects/krishna/zixianma/OpenWebRL
diagnostic_args=(--execute --job-id "$SLURM_JOB_ID")
if [[ "${ARM_GPU_DIAGNOSTIC_TRAINING_ONLY:-0}" == 1 ]]; then diagnostic_args+=(--training-only); fi
srun --jobid="$SLURM_JOB_ID" --nodes=1 --ntasks=1 --cpus-per-task=16 \
    --gres=gpu:h200:2 --cpu-bind=none --exact \
    /gpfs/scrubbed/zixianma/openwebrl-runtime/venv/bin/python \
    scripts/run_arm_sol_gpu_diagnostic.py "${diagnostic_args[@]}"
