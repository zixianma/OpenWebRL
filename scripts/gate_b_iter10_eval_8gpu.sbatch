#!/usr/bin/env bash
# Approved 2026-10-31: Gate B iteration-10 full-300 evaluation, 8 H200 / 64 CPU / 960 GiB, 1.5 h,
# account vlm, GPT-4.1 judge cap $10.
#SBATCH --job-name=gate-b-iter10-eval
#SBATCH --account=vlm
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h200:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=960G
#SBATCH --time=01:30:00
#SBATCH --no-requeue
#SBATCH --output=/gpfs/scrubbed/zixianma/openwebrl-runtime/logs/gate-b-iter10-eval-%j.out
set -euo pipefail
cd /gpfs/projects/krishna/zixianma/OpenWebRL/scripts
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}${no_proxy:+$no_proxy,}localhost,127.0.0.1,::1,$(hostname -s)"
export no_proxy="$NO_PROXY" PYTHONDONTWRITEBYTECODE=1
exec /gpfs/scrubbed/zixianma/openwebrl-runtime/venv/bin/python -B run_gate_b_iter10_eval.py --execute --job-id "$SLURM_JOB_ID"
