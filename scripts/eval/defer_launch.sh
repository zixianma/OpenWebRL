#!/usr/bin/env bash
# defer_launch.sh <condition dir> <benchmark>: copy launch-<benchmark>.sbatch (written by run_crossbench_selector_eval.py
# launch-script) to launch-<benchmark>-defer.sbatch, which runs scripts/eval/crossbench_defer.py exec (roll out now, judge
# later) instead of run_crossbench_selector_eval.py execute. Prints the new path. See openwebrl/docs/EVAL.md.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
C=$1; B=$2; S=$C/launch-$B.sbatch; O=$C/launch-$B-defer.sbatch
grep -q 'scripts/run_crossbench_selector_eval.py execute' $S
sed "s#[^ ]*scripts/run_crossbench_selector_eval.py execute#$REPO/scripts/eval/crossbench_defer.py exec#; s#--job-name=xbench-#--job-name=xdefer-#" $S > $O
grep -q 'crossbench_defer.py exec' $O; echo $O
