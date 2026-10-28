#!/usr/bin/env bash
# Submit v12 packages (OpenWebRL-4B-SFT no_search + Piotr SelectionARM). REFUSES unless the exact user approval receipt exists:
#   gpu-budget-v12-20261028/user-approval-v12-20261028.json  (written only after the user approves the exact request).
# Each package: 1 H200 / 8 CPUs / 240 GiB / its own seconds, account vlm, its own API cap. Flow (copy of v11b-tools/submit_v11c.sh):
# reserve in the v12 GPU ledger -> shared-gpu-reservation.json -> approved-request.json (validate_native_approval)
# -> submit_webprm_table5_campaign.py --submit (sbatch --hold, register, scontrol release).
# Usage: Q=<exact user quote> submit_v12.sh reddit-smoke ...
set -euo pipefail
D=/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-webarena-20261011/next-stage-preparation/webarena-subset-review-20261021
G=$D/gpu-budget-v12-20261028; R=$G/user-approval-v12-20261028.json
[ -f "$R" ] || { echo "no v12 approval receipt: $R" >&2; exit 2; }
: "${Q:?set Q to the exact user approval quote}"
PY=/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-baselines-20261009/browser-env/bin/python
for name in "$@"; do
  P=$D/launch-preparation-v12-$name
  (cd /tmp && TMPDIR=/tmp PYTHONNOUSERSITE=1 P="$P" R="$R" G="$G" Q="$Q" NAME=$name $PY -B - <<'PYEOF'
import copy, datetime, hashlib, json, os, sys
from pathlib import Path
P = Path(os.environ['P']); sys.path.insert(0, str(P/'source'))
from scripts import launch_webprm_table5_campaign as L
name = os.environ['NAME']
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest(); now = lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
def put(path, value):
    with Path(path).open('x') as out: json.dump(value, out, sort_keys=True, indent=2); out.write('\n')
    os.chmod(path, 0o600)
proposal = L.read(P/'proposal.json'); request = L.read(P/'resource-request.json'); L.validate_proposal(proposal)
assert proposal['status'] == 'prepared_v12_pending_approval'
secs = proposal['resources']['total_seconds']
receipt = Path(os.environ['R']); lp = Path(os.environ['G'])/'ledger.json'; ledger = json.loads(lp.read_text())
assert ledger.get('approval') == str(receipt), 'ledger not bound to the v12 approval receipt'
rid = 'native-table5-v12-'+name+'-20261028'
assert all(a['reservation_id'] != rid for a in ledger['allocations'])
ledger['allocations'].append(dict(reservation_id=rid, job_id=None, status='RESERVED_pending_submission', gpu_count=1,
    maximum_seconds=secs, reserved_gpu_seconds=secs, package=str(P), account='vlm', api_usd_cap=proposal['limits']['api_usd']))
ledger['unreserved_gpu_seconds'] -= secs; assert ledger['unreserved_gpu_seconds'] >= 0
assert sum(a.get('api_usd_cap', 0) for a in ledger['allocations']) <= ledger['api_usd_cap'], 'v12 API cap exceeded'
lp.write_text(json.dumps(ledger, indent=1)+'\n')
reservation = P/'shared-gpu-reservation.json'
put(reservation, dict(status='reserved', reserved_utc=now(), reservation_id=rid, account='vlm', gpu_seconds=secs,
    proposal_sha256=L.digest(proposal), ledger=dict(path=str(lp), sha256=sha(lp)),
    approval_receipt=dict(path=str(receipt), sha256=sha(receipt)), maximum_allocations=1))
approval = copy.deepcopy(request['approval_template'])
approval.update(status='approved', user_approval_quote=os.environ['Q'], user_approval_receipt=str(receipt),
                user_approval_receipt_sha256=sha(receipt), shared_gpu_reservation=dict(path=str(reservation), sha256=sha(reservation)))
L.validate_native_approval(proposal, approval); put(P/'approved-request.json', approval)
PYEOF
  )
  out=$(cd $P/source && TMPDIR=/tmp PYTHONNOUSERSITE=1 $PY -B scripts/submit_webprm_table5_campaign.py --proposal $P/proposal.json \
      --approval $P/approved-request.json --submit 2>&1 | grep -o '"job_id": "[0-9]*"\|"status": "[a-z_]*"\|rror.*' | tr '\n' ' ' || true)
  echo "$out <- $name"
  jid=$(echo "$out" | grep -o '"job_id": "[0-9]*"' | grep -o '[0-9]*' | head -1 || true)
  python3 - "$G/ledger.json" "$name" "$jid" <<'P2'
import json,sys
p,name,jid=sys.argv[1:]; d=json.load(open(p))
for a in d["allocations"]:
    if a["reservation_id"]==f"native-table5-v12-{name}-20261028" and jid: a["job_id"]=jid; a["status"]="SUBMITTED"
json.dump(d,open(p,"w"),indent=1); print("ledger: v12",name,"job",jid or "UNKNOWN (record manually)")
P2
done
# The job id is recorded in $G/ledger.json above; if it printed UNKNOWN, record it manually.
