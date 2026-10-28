#!/usr/bin/env bash
# Build v9b packages with env derived from the matching v9 proposals (same shards/caps). Stage chosen by build_v9*.py.
set -euo pipefail
D=/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-webarena-20261011/next-stage-preparation/webarena-subset-review-20261021
PY=/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-baselines-20261009/browser-env/bin/python
cd /tmp
while read name site secs usd workers tasks; do
  [[ "$name" == \#* || -z "$name" ]] && continue
  for want in "$@"; do [ "$want" = "$name" ] || continue
    blocks=$(for s in no_search arbiter7b shepherd8b; do printf '%s:%s;' $s $tasks; done); blocks=${blocks%;}
    b=build_v9c.py; [ "$site" = gitlab ] && b=SKIP
    V9_NAME=$name V9_SITE=$site V9_BLOCKS="$blocks" V9_API_USD=$usd V9_SECONDS=$secs V9_MAX_WORKERS=$workers \
      TMPDIR=/tmp PYTHONNOUSERSITE=1 $PY -B $D/v9b-tools/$b | tail -3
  done
done < <(cat $D/v9b-tools/shards.sh $D/v9b-tools/shards-gitlab.sh)
