#!/usr/bin/env bash
# CPU-only: build + test + review + stage 2 of the v12 r2 CMS smoke replacement (never submits).
set -euo pipefail
D=/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-webarena-20261011/next-stage-preparation/webarena-subset-review-20261021; T=$D/v12-tools; R=$T/r2
PY=/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-baselines-20261009/browser-env/bin/python
build() { (cd /tmp && V12_NAME=$1 V12_SITE=$2 V12_SECONDS=$3 V12_API_USD=$4 V12_MAX_WORKERS=2 V12_BLOCKS="$5" TMPDIR=/tmp PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 $PY -B $R/build_v12r2.py | tail -6); }
build cms-smoke3 shopping_admin 7200 7.99 "no_search:0,40,101;piotr_selectionarm:0,40,101"
V12_NAMES="cms-smoke3" bash $R/run_tests_v12r2.sbatch
(cd /tmp && $PY -B $R/make_review_v12r2.py cms-smoke3 $R/tests-v12-cms-smoke3.log)
build cms-smoke3 shopping_admin 7200 7.99 "no_search:0,40,101;piotr_selectionarm:0,40,101"
