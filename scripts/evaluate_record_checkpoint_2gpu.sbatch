#!/usr/bin/env bash
# Requires explicit approval: 2 H200 x 2 hours = 4 GPU-hours, estimated $3.60.
# GPT-4.1 judge API usage is additional. Local browsers only.
#SBATCH --job-name=openwebrl-record-eval
#SBATCH --account=zixianma
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h200:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=480G
#SBATCH --time=02:00:00
#SBATCH --output=/gpfs/scrubbed/zixianma/openwebrl-runtime/logs/slurm-record-eval-%j.out
set -euo pipefail
cd /gpfs/projects/krishna/zixianma/OpenWebRL
browser_eval_source="${OPENWEBRL_RECORD_EVAL_SOURCE:?Set the preserved evaluation source}"
browser_eval_checkpoint="${OPENWEBRL_RECORD_EVAL_CHECKPOINT:?Set the selected checkpoint}"
browser_eval_iteration="${OPENWEBRL_RECORD_EVAL_ITERATION:?Set the completed training iteration}"
browser_eval_root=/gpfs/scrubbed/zixianma/openwebrl-runtime/evaluations
browser_eval_deadline=$(python3 - "$SLURM_JOB_ID" <<'PY'
from datetime import datetime
import re, subprocess, sys
text = subprocess.check_output(['scontrol', 'show', 'job', sys.argv[1], '-o'], text=True)
print(int(datetime.fromisoformat(re.search(r'EndTime=(\S+)', text)[1]).timestamp()) - 180)
PY
)
    browser_eval_attempt=0
while true; do
    browser_eval_output="$browser_eval_root/qcq7i4ug-record-$SLURM_JOB_ID-after$browser_eval_iteration"
    if (( browser_eval_attempt > 0 )); then browser_eval_output="$browser_eval_output-retry$browser_eval_attempt"; fi
    if srun --jobid="$SLURM_JOB_ID" --overlap --nodes=1 --ntasks=1 --cpus-per-task=16 --gres=gpu:h200:2 --exact \
        /gpfs/scrubbed/zixianma/openwebrl-runtime/venv/bin/python scripts/evaluate_baseline_checkpoint.py \
        --execute --job-id "$SLURM_JOB_ID" --source "$browser_eval_source" \
        --checkpoint "$browser_eval_checkpoint" --output "$browser_eval_output" \
        --browser-env local_process --gpus 2 --attempt "$browser_eval_attempt"; then
        break
    fi
    printf 'Record-checkpoint evaluation stopped. Awaiting supervised repair marker: %s\n' "$browser_eval_output/retry_after_repair"
    while [[ ! -f "$browser_eval_output/retry_after_repair" ]]; do
        if (( $(date +%s) >= browser_eval_deadline )); then exit 1; fi
        sleep 15
    done
    browser_eval_attempt=$((browser_eval_attempt+1))
done
