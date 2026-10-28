#!/usr/bin/env bash
# Build one v11c replacement package (stage chosen by the builder).
# Usage: build_one_v11c.sh NAME SITE SECONDS API_USD WORKERS BLOCKS REPLACES(comma package dirs)
set -euo pipefail
D=/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-webarena-20261011/next-stage-preparation/webarena-subset-review-20261021
PY=/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-baselines-20261009/browser-env/bin/python
cd /tmp && V11_NAME=$1 V11_SITE=$2 V11_SECONDS=$3 V11_API_USD=$4 V11_MAX_WORKERS=$5 V11_BLOCKS="$6" V11C_REPLACES="$7" TMPDIR=/tmp PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 $PY -B $D/v11b-tools/build_v11c.py | tail -12
