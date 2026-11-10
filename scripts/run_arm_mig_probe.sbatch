#!/usr/bin/env bash
# Requires exact approval: two 18-GB MIG slices x one hour, 2 CPUs, 60 GiB; judge <= $1.
#SBATCH --job-name=arm-mig-probe
#SBATCH --account=zixianma
#SBATCH --partition=gpu-h200-mig
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200_1g.18gb:2
#SBATCH --cpus-per-task=2
#SBATCH --mem=60G
#SBATCH --time=01:00:00
#SBATCH --output=/gpfs/scrubbed/zixianma/openwebrl-runtime/logs/slurm-arm-mig-probe-%j.out
set -euo pipefail
cd /gpfs/projects/krishna/zixianma/OpenWebRL
source scripts/h200_env.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 WANDB_PROJECT=openwebrl-evals
unset WANDB_SERVICE OPENWEBRL_ARM_TURN_BONUS_CONFIG OPENWEBRL_ARM_CONFIG
/gpfs/scrubbed/zixianma/openwebrl-runtime/venv/bin/python scripts/run_arm_mig_probe.py --execute --job-id "$SLURM_JOB_ID"
