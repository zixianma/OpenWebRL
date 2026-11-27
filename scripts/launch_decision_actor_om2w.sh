#!/usr/bin/env bash
# Approved 2026-11-27: three same-day full-300 Online-Mind2Web jobs, 1 H200 / 8 CPUs / 120 GiB each, at most 8 h each
# (requested here: 04:00:00): sft1 control, projected_sft and projected_sft_strict. Judges inline (gpt-5.6-terra,
# AgentTrek, final 3 screenshots) for all three, so the cohorts are directly comparable. See openwebrl/docs/ARM_DECISION_ACTOR.md.
#
#   bash scripts/launch_decision_actor_om2w.sh            # prepare, write sbatch files, submit, record job ids
#   DRY=1 bash scripts/launch_decision_actor_om2w.sh      # prepare and write sbatch files only
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
PY=${PYTHON:-/gpfs/scrubbed/zixianma/openwebrl-runtime/venv/bin/python}
R=${ROOT:-/gpfs/scrubbed/zixianma/openwebrl-runtime/evaluations/decision-actor-$(date +%Y%m%d)}
TIME=${TIME:-04:00:00}
cd "$REPO"
$PY -m pytest -q tests/test_decision_actor.py
mkdir -p "$R"
CONDITIONS=(sft1 decision:$REPO/openwebrl/decision_actor/configs/projected_sft.json decision:$REPO/openwebrl/decision_actor/configs/projected_sft_strict.json)
for C in "${CONDITIONS[@]}"; do
  OMP_NUM_THREADS=1 $PY scripts/run_decision_actor_eval.py prepare --root "$R" --condition "$C" --benchmarks online_mind2web \
      --n 300 --seed 20261018 --temperature 1.0 --top-p 0.9 > /dev/null
  out=$(OMP_NUM_THREADS=1 $PY scripts/run_decision_actor_eval.py launch-script --root "$R" --condition "$C" --benchmarks online_mind2web --time "$TIME")
  script=$(echo "$out" | $PY -c "import json,sys; print(json.load(sys.stdin)['sbatch'])")
  if [ -n "${DRY:-}" ]; then echo "prepared $C -> $script"; continue; fi
  j=$(sbatch --parsable --account=${ACCOUNT:-vlm} --qos=${QOS:-normal} ${EXCLUDE:+--exclude=$EXCLUDE} "$script")
  echo "$(date -Is) online_mind2web $C -> $(dirname "$script") job $j" | tee -a "$R/jobs.log"
done
echo "report: $PY scripts/decision_actor_report.py --dir $R/decision-projected-sft-*/online_mind2web --control $R/sft1/online_mind2web"
