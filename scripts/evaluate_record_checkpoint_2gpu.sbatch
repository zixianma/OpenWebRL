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
# Concurrent 2-GPU evaluations can share a node. Give each allocation a
# deterministic rendezvous port instead of inheriting the same default.
export MASTER_PORT=$((15000 + SLURM_JOB_ID % 1000))
browser_eval_source="${OPENWEBRL_RECORD_EVAL_SOURCE:?Set the preserved evaluation source}"
browser_eval_checkpoint="${OPENWEBRL_RECORD_EVAL_CHECKPOINT:?Set the selected checkpoint}"
browser_eval_iteration="${OPENWEBRL_RECORD_EVAL_ITERATION:?Set the completed training iteration}"
export OPENWEBRL_EXPECTED_SCHEDULER_OFFSET="${OPENWEBRL_EXPECTED_SCHEDULER_OFFSET:-1}"
browser_eval_root=/gpfs/scrubbed/zixianma/openwebrl-runtime/evaluations
if [[ -n "${OPENWEBRL_RECORD_EVAL_TASK_FILE:-}" ]]; then
    export OPENWEBRL_RECORD_EVAL_EXPECTED_TASK_COUNT="$(wc -l < "$OPENWEBRL_RECORD_EVAL_TASK_FILE")"
    browser_eval_config="/tmp/arm-eval-${SLURM_JOB_ID}.yaml"
    python3 - "$browser_eval_config" "$OPENWEBRL_RECORD_EVAL_TASK_FILE" <<'PY'
import sys
from pathlib import Path
out, tasks = map(Path, sys.argv[1:])
out.write_text("""eval:\n  defaults:\n    n_samples_per_eval_prompt: 1\n    temperature: 0.0\n    top_p: 1.0\n    top_k: 1\n    max_response_len: 4096\n  datasets:\n    - name: online-mind2web-monitor\n      path: %s\n      input_key: prompt\n      metadata_key: metadata\n      custom_generate_function_path: openwebrl.eval_monitor.generate\n""" % tasks)
PY
    export OPENWEBRL_RECORD_EVAL_CONFIG="$browser_eval_config"
fi
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
    if srun --jobid="$SLURM_JOB_ID" --overlap --nodes=1 --ntasks=1 --cpus-per-task=16 --gres=gpu:h200:2 --cpu-bind=none --exact \
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
