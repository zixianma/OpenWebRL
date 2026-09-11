#!/usr/bin/env bash
# Requires explicit approval: 2 H200 x 3 hours = 6 GPU-hours, estimated $5.40.
# Browser Use hosting and GPT-4.1 judge API usage are additional; proxy disabled.
#SBATCH --job-name=openwebrl-browseruse-20-30
#SBATCH --account=zixianma
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h200:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=480G
#SBATCH --time=03:00:00
#SBATCH --output=/gpfs/scrubbed/zixianma/openwebrl-runtime/logs/slurm-browseruse-eval-%j.out
set -euo pipefail
cd /gpfs/projects/krishna/zixianma/OpenWebRL
browser_eval_source=/gpfs/scrubbed/zixianma/openwebrl-runtime/reference-browseruse-eval-20260910
browser_eval_root=/gpfs/scrubbed/zixianma/openwebrl-runtime/evaluations
browser_eval_deadline=$(python3 - "$SLURM_JOB_ID" <<'PY'
from datetime import datetime
import re, subprocess, sys
text = subprocess.check_output(['scontrol', 'show', 'job', sys.argv[1], '-o'], text=True)
print(int(datetime.fromisoformat(re.search(r'EndTime=(\S+)', text)[1]).timestamp()) - 180)
PY
)
for browser_eval_iteration in 20 30; do
    case "$browser_eval_iteration" in
        20) browser_eval_checkpoint=/gpfs/scrubbed/zixianma/openwebrl-runtime/runs/openwebrl-4b-reference-285546-20260909T235114/iter_0000019 ;;
        30) browser_eval_checkpoint=/gpfs/scrubbed/zixianma/openwebrl-runtime/runs/openwebrl-4b-reference-286382-20260910T164338/iter_0000029 ;;
    esac
    browser_eval_attempt=0
while true; do
    browser_eval_output="$browser_eval_root/qcq7i4ug-browseruse-$SLURM_JOB_ID-after$browser_eval_iteration"
    if (( browser_eval_attempt > 0 )); then browser_eval_output="$browser_eval_output-retry$browser_eval_attempt"; fi
    if srun --jobid="$SLURM_JOB_ID" --overlap --nodes=1 --ntasks=1 --cpus-per-task=16 --gres=gpu:h200:2 --exact \
        /gpfs/scrubbed/zixianma/openwebrl-runtime/venv/bin/python scripts/evaluate_baseline_checkpoint.py \
        --execute --job-id "$SLURM_JOB_ID" --source "$browser_eval_source" \
        --checkpoint "$browser_eval_checkpoint" --output "$browser_eval_output" \
        --browser-env browser-use --gpus 2 --attempt "$browser_eval_attempt"; then
        break
    fi
    printf 'Browser Use evaluation stopped. Awaiting supervised repair marker: %s\n' "$browser_eval_output/retry_after_repair"
    while [[ ! -f "$browser_eval_output/retry_after_repair" ]]; do
        if (( $(date +%s) >= browser_eval_deadline )); then exit 1; fi
        sleep 15
    done
    browser_eval_attempt=$((browser_eval_attempt+1))
done
done
