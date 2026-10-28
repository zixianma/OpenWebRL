#!/usr/bin/env bash
# Replace a crashed v9 job with a v9e package running only cells lacking a valid (or fuzzy-invalid) outcome.
# Usage: replace_v9e.sh NEW_NAME SITE SECONDS API_USD OLD_JOB_ID OLD_PACKAGE [OLDER_PACKAGES...]
# Steps: ledger accounting for OLD_JOB_ID -> blocks -> stage 1 -> CPU tests -> review -> stage 2 -> submit -> ledger.
set -euo pipefail
D=/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-webarena-20261011/next-stage-preparation/webarena-subset-review-20261021
B=$D/v9b-tools; G=$D/gpu-budget-v9-20261026; PY=/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-baselines-20261009/browser-env/bin/python
name=$1 site=$2 secs=$3 usd=$4 old=$5; shift 5
used=$(sacct -X -n -j $old -o ElapsedRaw | tr -d ' ')
python3 - "$G/ledger.json" "$old" "$used" <<'P'
import json,sys
p,old,used=sys.argv[1],sys.argv[2],int(sys.argv[3]); d=json.load(open(p))
for a in d['allocations']:
    if a.get('job_id')==old and a['status']=='SUBMITTED':
        a['status']='FAILED_replaced_v9e'; a['actual_gpu_seconds']=used
        d['unreserved_gpu_seconds']+=a['reserved_gpu_seconds']-used; a['released_gpu_seconds']=a['reserved_gpu_seconds']-used; a['reserved_gpu_seconds']=used
json.dump(d,open(p,'w'),indent=1); print('ledger unreserved',d['unreserved_gpu_seconds'])
P
python3 $B/replacement_blocks.py "$@" > $B/blocks-$name.json; cat $B/blocks-$name.json
BL=$(python3 -c "import json;print(json.load(open('$B/blocks-$name.json'))['blocks'])")
$B/build_v9e_one.sh $name $site $secs $usd 2 "$BL" >/dev/null
if [ $site = gitlab ]; then T=run_tests_v9e_gitlab.sbatch; R=make_review_v9e_gitlab.py; else T=run_tests_v9e.sbatch; R=make_review_v9e.py; fi
j=$(cd $B && V9D_NAMES=$name sbatch --parsable --account=vlm --export=ALL $T 2>/dev/null)
until s=$(sacct -X -n -j $j -o State | tr -d ' '); [ -n "$s" ] && ! echo "$s" | grep -qE "PENDING|RUNNING"; do sleep 15; done
grep -A4 "same-cell" $B/tests-v9e-$name.log | grep -E "^Ran|^OK|FAIL" | tr '\n' ' '; echo
(cd /tmp && $PY -B $B/$R $name $B/tests-v9e-$name.log)
$B/build_v9e_one.sh $name $site $secs $usd 2 "$BL" >/dev/null
out=$(cd $B && ./submit_v9e.sh $name); echo "$out"
jid=$(echo "$out" | grep -o '"job_id": "[0-9]*"' | grep -o '[0-9]*')
python3 - "$G/ledger.json" "$name" "$jid" <<'P'
import json,sys
p,name,jid=sys.argv[1:]; d=json.load(open(p))
for a in d['allocations']:
    if a['reservation_id']==f'native-table5-v9e-{name}-20261026': a['job_id']=jid; a['status']='SUBMITTED'
json.dump(d,open(p,'w'),indent=1); print('submitted',jid,'unreserved',d['unreserved_gpu_seconds'])
P
