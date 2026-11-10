#!/usr/bin/env bash
# Approved: two H200s x four hours, training plus fixed-100 evaluation.
#SBATCH --job-name=arm-serial-all5
#SBATCH --account=zixianma
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h200:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --time=04:00:00
#SBATCH --signal=B:TERM@300
#SBATCH --output=/gpfs/scrubbed/zixianma/openwebrl-runtime/logs/arm-serial-all5-%j.out
set -euo pipefail
cd /gpfs/projects/krishna/zixianma/OpenWebRL
source scripts/h200_env.sh
exec /gpfs/scrubbed/zixianma/openwebrl-runtime/arm-reproduction/venv/bin/python scripts/run_arm_serial_pipeline.py --config openwebrl/docs/arm_results/serial_sft/all5-config.json
