#!/usr/bin/env bash
# Proposed quick all5 validation. Requires approval before submission.
# Runs parity, free generation, then five browser tasks only if generation passes.
# Never launches the full evaluation set; deadline handling is in the worker.
#SBATCH --job-name=arm-serial-quick-check
#SBATCH --account=zixianma
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=00:30:00
#SBATCH --signal=B:TERM@300
#SBATCH --output=/gpfs/scrubbed/zixianma/openwebrl-runtime/logs/arm-serial-quick-check-%j.out
set -euo pipefail
cd /gpfs/projects/krishna/zixianma/OpenWebRL
source scripts/h200_env.sh
exec /gpfs/scrubbed/zixianma/openwebrl-runtime/arm-reproduction/venv/bin/python scripts/check_arm_serial_eval.py \
  --config openwebrl/docs/arm_results/serial_sft/recovery-v2/all5-config.json
