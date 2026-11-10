#!/usr/bin/env bash
# Approved 2026-11-04: Gate B 48M+16F continuation 60 -> 90, 8 H200 / 64 CPU / 960 GiB, 28 h total
# across attempts (24 h job + 4 h backup, at most 24 h per job), account zixianma, urgent QoS, additional
# GPT-4.1 judge cap $175. Register each job in the ledger (--register-job) before it starts.
#SBATCH --job-name=arm-gate-b-48m16f-to90
#SBATCH --account=zixianma
#SBATCH --partition=gpu-h200
#SBATCH --qos=urgent
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h200:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=960G
#SBATCH --no-requeue
#SBATCH --output=/gpfs/scrubbed/zixianma/openwebrl-runtime/logs/arm-gate-b-48m16f-to90-%j.out
set -euo pipefail
cd /gpfs/projects/krishna/zixianma/OpenWebRL/scripts
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}${no_proxy:+$no_proxy,}localhost,127.0.0.1,::1,$(hostname -s)"
export no_proxy="$NO_PROXY" PYTHONDONTWRITEBYTECODE=1
exec /gpfs/scrubbed/zixianma/openwebrl-runtime/venv/bin/python -B run_arm_gate_b_48m16f_to90.py --execute --job-id "$SLURM_JOB_ID"
