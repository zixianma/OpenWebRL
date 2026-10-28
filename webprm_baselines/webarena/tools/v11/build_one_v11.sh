#!/usr/bin/env bash
# Build one v11 package (stage chosen by the builder). Usage: build_one_v11.sh NAME SITE SECONDS API_USD WORKERS TASKS
set -euo pipefail
D=/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-webarena-20261011/next-stage-preparation/webarena-subset-review-20261021
PY=/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-baselines-20261009/browser-env/bin/python
cd /tmp && V11_NAME=$1 V11_SITE=$2 V11_SECONDS=$3 V11_API_USD=$4 V11_MAX_WORKERS=$5 V11_TASKS="$6" TMPDIR=/tmp PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 $PY -B $D/v11-tools/build_v11.py | tail -12
