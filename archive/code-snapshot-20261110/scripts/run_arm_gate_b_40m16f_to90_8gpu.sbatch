#!/usr/bin/env bash
# Approved 2026-10-31: Gate B 40M+16F continuation 60 -> 90, 8 H200 / 64 CPU / 960 GiB, 28 h total
# across attempts (at most 24 h per job), account vlm, additional GPT-4.1 judge cap $175. Register each
# job in the ledger (--register-job) before it starts.
#SBATCH --job-name=arm-gate-b-40m16f-to90
#SBATCH --account=vlm
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h200:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=960G
#SBATCH --no-requeue
#SBATCH --output=/gpfs/scrubbed/zixianma/openwebrl-runtime/logs/arm-gate-b-40m16f-to90-%j.out
set -euo pipefail
cd /gpfs/projects/krishna/zixianma/OpenWebRL/scripts
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}${no_proxy:+$no_proxy,}localhost,127.0.0.1,::1,$(hostname -s)"
export no_proxy="$NO_PROXY" PYTHONDONTWRITEBYTECODE=1
exec /gpfs/scrubbed/zixianma/openwebrl-runtime/venv/bin/python -B run_arm_gate_b_40m16f_to90.py --execute --job-id "$SLURM_JOB_ID"
