#!/usr/bin/env bash
# Requires separate explicit approval: 4 H200 x 3 hours = 12 GPU-hours.
# Estimated cluster charge $10.80; judge API usage is additional.
#SBATCH --job-name=openwebrl-eval-after21-22
#SBATCH --account=zixianma
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h200:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=480G
#SBATCH --time=03:00:00
#SBATCH --output=/gpfs/scrubbed/zixianma/openwebrl-runtime/logs/slurm-baseline-eval-%j.out
set -euo pipefail
cd /gpfs/projects/krishna/zixianma/OpenWebRL
eval_source=/gpfs/scrubbed/zixianma/openwebrl-runtime/reference-stage1-tp4-eval-cache-20260909
eval_root=/gpfs/scrubbed/zixianma/openwebrl-runtime/runs/openwebrl-4b-reference-285546-20260909T235114
eval_output=/gpfs/scrubbed/zixianma/openwebrl-runtime/evaluations
eval_deadline=$(python3 - "$SLURM_JOB_ID" <<'PY'
from datetime import datetime
import re, subprocess, sys
text = subprocess.check_output(['scontrol', 'show', 'job', sys.argv[1], '-o'], text=True)
print(int(datetime.fromisoformat(re.search(r'EndTime=(\S+)', text)[1]).timestamp()) - 180)
PY
)
for eval_index in 20 21; do
    printf -v eval_directory 'iter_%07d' "$eval_index"
    eval_attempt=0
    while true; do
      eval_work="$eval_output/qcq7i4ug-$SLURM_JOB_ID-after$((eval_index+1))"
      if (( eval_attempt > 0 )); then eval_work="$eval_work-retry$eval_attempt"; fi
      if srun --jobid="$SLURM_JOB_ID" --overlap --nodes=1 --ntasks=1 \
        --cpus-per-task=16 --gres=gpu:h200:4 --exact \
        /gpfs/scrubbed/zixianma/openwebrl-runtime/venv/bin/python \
        scripts/evaluate_baseline_checkpoint.py --execute --job-id "$SLURM_JOB_ID" \
        --source "$eval_source" --checkpoint "$eval_root/$eval_directory" \
        --output "$eval_work" --attempt "$eval_attempt"; then
          break
      fi
      # Keep the approved allocation available for supervised repair. A new
      # attempt needs an explicit local marker after diagnosis, never blind retry.
      eval_retry_marker="$eval_work/retry_after_repair"
      printf 'Evaluation stopped. Awaiting supervised repair marker: %s\n' "$eval_retry_marker"
      while [[ ! -f "$eval_retry_marker" ]]; do
          if (( $(date +%s) >= eval_deadline )); then exit 1; fi
          sleep 15
      done
      eval_attempt=$((eval_attempt+1))
    done
done
