#!/usr/bin/env bash
# CPU-only: build + test + review + stage 2 of the v12 smoke packages (never submits).
set -euo pipefail
D=/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-webarena-20261011/next-stage-preparation/webarena-subset-review-20261021; T=$D/v12-tools
PY=/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-baselines-20261009/browser-env/bin/python
build() { (cd /tmp && V12_NAME=$1 V12_SITE=$2 V12_SECONDS=$3 V12_API_USD=$4 V12_MAX_WORKERS=2 V12_BLOCKS="$5" TMPDIR=/tmp PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 $PY -B $T/build_v12.py | tail -12); }
build reddit-smoke reddit 5400 8 "no_search:14,97,132;piotr_selectionarm:14,97,132"
build cms-smoke shopping_admin 7200 8 "no_search:0,40,101;piotr_selectionarm:0,40,101"
cd $T && V12_NAMES="reddit-smoke cms-smoke" bash run_tests_v12.sbatch
for n in reddit-smoke cms-smoke; do (cd /tmp && $PY -B $T/make_review_v12.py $n $T/tests-v12-$n.log); done
build reddit-smoke reddit 5400 8 "no_search:14,97,132;piotr_selectionarm:14,97,132"
build cms-smoke shopping_admin 7200 8 "no_search:0,40,101;piotr_selectionarm:0,40,101"
