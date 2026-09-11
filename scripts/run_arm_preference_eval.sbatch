#!/usr/bin/env bash
#SBATCH --job-name=arm-pref-u131-eval100
#SBATCH --account=zixianma
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=01:00:00
#SBATCH --output=/gpfs/scrubbed/zixianma/openwebrl-runtime/logs/arm-pref-eval-%j.out
set -euo pipefail

cd /gpfs/projects/krishna/zixianma/OpenWebRL
source scripts/h200_env.sh

run_root=/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-reproduction/runs/arm-preference-calibrated-286384-r4
checkpoint="$run_root/student/endpoint"
sample="$run_root/evaluation/checkpoint-scaling-100/sample.json"

exec /gpfs/scrubbed/zixianma/openwebrl-runtime/arm-reproduction/venv/bin/python \
    scripts/run_arm_checkpoint_eval.py \
    --job-id "$SLURM_JOB_ID" \
    --run-root "$run_root" \
    --checkpoint "$checkpoint" \
    --label endpoint-000131 \
    --sample "$sample" \
    --actor-port 19110 \
    --browser-port-start 19200 \
    --browser-port-end 19399 \
    --parallel 12
