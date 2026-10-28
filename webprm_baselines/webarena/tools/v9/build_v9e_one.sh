#!/usr/bin/env bash
# Build one v9e package (stage chosen by the builder). Usage: build_v9d_one.sh NAME SITE SECONDS API_USD WORKERS BLOCKS
set -euo pipefail
D=/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-webarena-20261011/next-stage-preparation/webarena-subset-review-20261021
PY=/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-baselines-20261009/browser-env/bin/python
b=build_v9e.py; [ "$2" = gitlab ] && b=build_v9e_gitlab.py
cd /tmp && V9_NAME=$1 V9_SITE=$2 V9_SECONDS=$3 V9_API_USD=$4 V9_MAX_WORKERS=$5 V9_BLOCKS="$6" TMPDIR=/tmp PYTHONNOUSERSITE=1 $PY -B $D/v9b-tools/$b | tail -3
