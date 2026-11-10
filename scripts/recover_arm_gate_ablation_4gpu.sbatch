#!/usr/bin/env bash
# Do not submit without exact replacement-allocation and guard-change approval.
#SBATCH --job-name=arm-gate-recovery
#SBATCH --account=zixianma
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h200:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=480G
#SBATCH --time=07:00:00
#SBATCH --output=/gpfs/scrubbed/zixianma/openwebrl-runtime/logs/slurm-arm-gate-recovery-%j.out
set -euo pipefail
cd /gpfs/projects/krishna/zixianma/OpenWebRL
unset WANDB_SERVICE
: "${ARM_GATE_VARIANT:?Choose B or C}"
: "${ARM_GATE_RECOVERY_APPROVED:?Explicit guard and replacement budget approval required}"
srun --jobid="$SLURM_JOB_ID" --ntasks=1 --nodes=1 --cpus-per-task=32 --gres=gpu:h200:4 --exact --cpu-bind=none \
 /gpfs/scrubbed/zixianma/openwebrl-runtime/venv/bin/python scripts/recover_arm_gate_ablation.py \
 --execute --job-id "$SLURM_JOB_ID" --variant "$ARM_GATE_VARIANT"
