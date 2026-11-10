#!/usr/bin/env bash
# Requires explicit approval: one H200 for one hour, eight CPUs, 80 GiB RAM.
# Archived-state replay only: no browser rollouts, actor generation, or API calls.
#SBATCH --job-name=arm-confidence-audit
#SBATCH --account=zixianma
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=01:00:00
#SBATCH --output=/gpfs/scrubbed/zixianma/openwebrl-runtime/logs/slurm-arm-confidence-%j.out
set -euo pipefail
cd /gpfs/projects/krishna/zixianma/OpenWebRL
export WANDB_PROJECT=openwebrl-evals
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
audit_root=/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-turn-bonus-preparation/confidence-audit-20260919
audit_python=/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-reproduction/venv/bin/python
srun --nodes=1 --ntasks=1 --cpus-per-task=8 --gres=gpu:h200:1 --cpu-bind=none --exact \
    "$audit_python" scripts/replay_arm_confidence.py score \
    --inputs "$audit_root/inputs.jsonl" --output "$audit_root/scores-${SLURM_JOB_ID}"
python3 scripts/replay_arm_confidence.py report --output "$audit_root/scores-${SLURM_JOB_ID}"
