#!/usr/bin/env bash
# Canonical live 3-benchmark eval (protocol of the November 6-7 selector evals; see openwebrl/docs/EVAL.md).
#   run_crossbench_eval.sh prepare ROOT CONDITION      write plans: ROOT/<benchmark>/<condition slug>/<benchmark>/plan.json
#   run_crossbench_eval.sh submit  ROOT CONDITION      one judge-deferred GPU job per benchmark (1 H200, 8 CPUs, 120 GiB)
# CONDITION: sft1 | head5:<head.pt> | head5x:<head.pt> | arm5:<merged SelectionARM dir> | piotr5
# Use a separate ROOT per repeated run (run1, run2, ...); the seed stays fixed. Judge afterwards with rejudge.sbatch.
# Env: BENCHMARKS (default all three), ACCOUNT (vlm), QOS (normal), EXCLUDE (nodes to avoid), DEPENDENCY (sbatch dependency).
set -euo pipefail
MODE=$1; ROOT=$2; C=$3
REPO=$(cd "$(dirname "$0")/../.." && pwd)
PY=${PYTHON:-/gpfs/scrubbed/zixianma/openwebrl-runtime/venv/bin/python}
declare -A N=([online_mind2web]=300 [webvoyager]=595 [deepshop]=150) T=([online_mind2web]=03:30:00 [webvoyager]=05:00:00 [deepshop]=02:30:00)
cd $REPO
for B in ${BENCHMARKS:-webvoyager online_mind2web deepshop}; do
  if [ "$MODE" = prepare ]; then
    OMP_NUM_THREADS=1 $PY scripts/run_crossbench_selector_eval.py prepare --root $ROOT/$B --condition $C --benchmarks $B --n ${N[$B]} \
        --seed 20261018 --temperature 1.0 --top-p 0.9
  elif [ "$MODE" = submit ]; then
    out=$(OMP_NUM_THREADS=1 $PY scripts/run_crossbench_selector_eval.py launch-script --root $ROOT/$B --condition $C --benchmarks $B --time ${T[$B]})
    script=$(echo "$out" | $PY -c "import json,sys; print(json.load(sys.stdin)['submit'].split()[1])")
    script=$(scripts/eval/defer_launch.sh $(dirname $script) $B)
    j=$(sbatch --parsable --account=${ACCOUNT:-vlm} --qos=${QOS:-normal} ${EXCLUDE:+--exclude=$EXCLUDE} ${DEPENDENCY:+--dependency=$DEPENDENCY} $script)
    echo "$(date -Is) $B $C -> $(dirname $script) job $j"
  else
    echo "usage: $0 prepare|submit ROOT CONDITION" >&2; exit 2
  fi
done
