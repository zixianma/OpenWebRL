#!/usr/bin/env bash
# Proposed validation-only budget. Requires explicit approval before submission.
#SBATCH --job-name=arm-serial-eval-gates
#SBATCH --account=zixianma
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h200:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --time=01:00:00
#SBATCH --signal=B:TERM@300
#SBATCH --output=/gpfs/scrubbed/zixianma/openwebrl-runtime/logs/arm-serial-eval-gates-%j.out
set -euo pipefail
cd /gpfs/projects/krishna/zixianma/OpenWebRL
source scripts/h200_env.sh
exec /gpfs/scrubbed/zixianma/openwebrl-runtime/arm-reproduction/venv/bin/python scripts/check_arm_serial_eval.py
