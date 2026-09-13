#!/usr/bin/env bash
#SBATCH --job-name=openwebrl-browser-diagnostic
#SBATCH --account=zixianma
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h200:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=960G
#SBATCH --time=01:00:00
#SBATCH --output=/gpfs/scrubbed/zixianma/openwebrl-runtime/logs/slurm-browser-diagnostic-%j.out
# Pending explicit approval: 8 GPU-hours, $7.20 maximum estimated GPU charge.
# Tillicum requires eight full GPU units to allocate 64 CPUs.
# Browser-only diagnosis first. If a burst-launch profile passes >=95% of
# resets and >=30 minutes remain, attempt one isolated fresh-SFT RL iteration.
set -euo pipefail
cd /gpfs/projects/krishna/zixianma/OpenWebRL
srun --nodes=1 --ntasks=1 --cpus-per-task=64 --gres=gpu:h200:8 --cpu-bind=none --exact \
  /gpfs/scrubbed/zixianma/openwebrl-runtime/venv/bin/python \
  scripts/diagnose_browser_scaling.py --controller "${SLURM_JOB_ID:?Approved allocation required}" --confirm-rl
