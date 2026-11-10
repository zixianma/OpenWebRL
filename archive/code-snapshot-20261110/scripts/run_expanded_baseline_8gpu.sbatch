#!/usr/bin/env bash
# Submit only after exact allocation/API approval and ledger registration.
#SBATCH --job-name=outcome-expanded4102
#SBATCH --account=zixianma
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h200:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=960G
#SBATCH --time=24:00:00
#SBATCH --no-requeue
#SBATCH --output=/gpfs/scrubbed/zixianma/openwebrl-runtime/logs/outcome-expanded4102-%j.out
set -euo pipefail
cd /gpfs/projects/krishna/zixianma/OpenWebRL
exec /gpfs/scrubbed/zixianma/openwebrl-runtime/venv/bin/python -B scripts/run_expanded_baseline.py \
  --job-id "$SLURM_JOB_ID" --execute
