#!/usr/bin/env bash
# Prepare v11c replacement package(s) for a halted/finished v11b (or v11c) job; never submits (prints the submit command).
# Usage: replace_v11c.sh NEW_NAME SITE SECONDS API_USD OLD_JOB_ID OLD_PACKAGE [EARLIER_PACKAGES...]
#   SECONDS / API_USD are per replacement job. Pending cells (no outcome, or infra-invalid; fuzzy-judge / agent
#   'Invalid action' / 'List requirements...' are final) are split into chunks of <= 20 cells, interleaved per task;
#   one package per chunk: NEW_NAME (one chunk) or NEW_NAME-1, NEW_NAME-2, ...
# Steps: close OLD_JOB in the v11 ledger at sacct elapsed (release unused GPU-s / API $) -> blocks -> funding check
#        (only released v11b/v11c reservations; same 104,400 s / $280 caps and "yes" receipt) -> stage 1 -> CPU-only tests
#        (sbatch, waited) -> review -> stage 2 -> print the submit command.
set -euo pipefail
D=/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-webarena-20261011/next-stage-preparation/webarena-subset-review-20261021
B=$D/v11b-tools; G=$D/gpu-budget-v11-20261027; PY=/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-baselines-20261009/browser-env/bin/python
name=$1 site=$2 secs=$3 usd=$4 old=$5; shift 5
pkgs=("$@"); replaces=$(IFS=,; echo "${pkgs[*]}")
python3 -B $B/ledger_close_v11.py $old replaced_v11c | tail -8
python3 -B $B/replacement_blocks_v11.py --max 20 "${pkgs[@]}" > $B/blocks-v11c-$name.json; cat $B/blocks-v11c-$name.json
n=$(python3 -c "import json;print(len(json.load(open('$B/blocks-v11c-$name.json'))['chunks']))")
[ "$n" -gt 0 ] || { echo "nothing pending for $name"; exit 0; }
python3 - "$G/ledger.json" "$n" "$secs" "$usd" <<'P'
import json,sys
d=json.load(open(sys.argv[1])); n,secs,usd=int(sys.argv[2]),int(sys.argv[3]),float(sys.argv[4])
assert n*secs<=d['unreserved_gpu_seconds'], ('insufficient released GPU-seconds', n*secs, d['unreserved_gpu_seconds'])
assert n*usd<=d['unreserved_api_usd']+1e-9, ('insufficient released API USD', n*usd, d['unreserved_api_usd'])
print('funding ok:', n, 'x', secs, 's /', usd, 'USD; unreserved', d['unreserved_gpu_seconds'], 's /', d['unreserved_api_usd'], 'USD')
P
names=()
for i in $(seq 0 $((n-1))); do
  nm=$name; [ "$n" -gt 1 ] && nm=$name-$((i+1)); names+=($nm)
  bl=$(python3 -c "import json;print(json.load(open('$B/blocks-v11c-$name.json'))['chunks'][$i]['blocks'])")
  echo "$nm|$site|$secs|$usd|$bl|$replaces" > $B/shard-v11c-$nm.txt
  $B/build_one_v11c.sh $nm $site $secs $usd 2 "$bl" "$replaces" >/dev/null
done
j=$(cd $B && V11C_NAMES="${names[*]}" sbatch --parsable --account=vlm --export=ALL run_tests_v11c.sbatch 2>/dev/null)
echo "CPU tests job $j"
until s=$(sacct -X -n -j $j -o State | tr -d ' '); [ -n "$s" ] && ! echo "$s" | grep -qE "PENDING|RUNNING"; do sleep 15; done
for nm in "${names[@]}"; do
  (cd /tmp && $PY -B $B/make_review_v11c.py $nm $B/tests-v11c-$nm.log)
  IFS='|' read -r _ st sc us bl rp < $B/shard-v11c-$nm.txt
  $B/build_one_v11c.sh $nm $st $sc $us 2 "$bl" "$rp" >/dev/null
  python3 -c "import json;p=json.load(open('$D/launch-preparation-v11c-$nm/proposal.json'));assert p['status']=='prepared_v11c_pending_approval';print('$nm',p['status'],p['resources']['total_seconds'],'s',p['limits']['api_usd'],'USD',[(b['selector'],len(b['task_ids'])) for b in p['v8_cell_blocks']])"
done
echo "SUBMIT (run yourself): cd /tmp && Q='yes' $B/submit_v11c.sh ${names[*]}"
