#!/usr/bin/env bash
# Submit approved WebArena-Lite v9 packages (user 2026-10-26: "ok yes I approve ..."; receipt user-approval-v9-20261026.json).
# "yes I approve anything else u need"). Each: 1 H200 / 8 CPUs / 240 GiB / 14,280 s, account vlm, USD 50 API cap.
# Flow per package (as v7k): reserve in the v8 GPU ledger -> shared-gpu-reservation.json -> approved-request.json
# (validate_native_approval) -> submit_webprm_table5_campaign.py --submit (sbatch --hold, register, scontrol release).
# Usage: submit_v8.sh [j1-ns-shep j2-arb-a j3-arb-b j4-shep]
set -euo pipefail
D=/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-webarena-20261011/next-stage-preparation/webarena-subset-review-20261021
G=$D/gpu-budget-v9-20261026; R=$G/user-approval-v9-20261026.json; Q='ok yes I approve. also let'"'"'s keep track of latency and api cost (with our selection ARM we'"'"'ll still just do N=5)'
PY=/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-baselines-20261009/browser-env/bin/python
for name in "$@"; do
  P=$D/launch-preparation-v9e-$name
  (cd /tmp && TMPDIR=/tmp PYTHONNOUSERSITE=1 P="$P" R="$R" G="$G" Q="$Q" NAME=$name $PY -B - <<'EOF'
import copy, datetime, hashlib, json, os, sys
from pathlib import Path
P = Path(os.environ['P']); sys.path.insert(0, str(P/'source'))
from scripts import launch_webprm_table5_campaign as L
secs = None; name = os.environ['NAME']
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest(); now = lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
def put(path, value):
    with Path(path).open('x') as out: json.dump(value, out, sort_keys=True, indent=2); out.write('\n')
    os.chmod(path, 0o600)
proposal = L.read(P/'proposal.json'); request = L.read(P/'resource-request.json'); L.validate_proposal(proposal)
secs = proposal['resources']['total_seconds']
receipt = Path(os.environ['R']); lp = Path(os.environ['G'])/'ledger.json'; ledger = json.loads(lp.read_text())
rid = 'native-table5-v9e-'+name+'-20261026'
assert all(a['reservation_id'] != rid for a in ledger['allocations'])
ledger['allocations'].append(dict(reservation_id=rid, job_id=None, status='RESERVED_pending_submission', gpu_count=1,
    maximum_seconds=secs, reserved_gpu_seconds=secs, package=str(P), account='vlm'))
ledger['unreserved_gpu_seconds'] -= secs; assert ledger['unreserved_gpu_seconds'] >= 0
lp.write_text(json.dumps(ledger, indent=1)+'\n')
reservation = P/'shared-gpu-reservation.json'
put(reservation, dict(status='reserved', reserved_utc=now(), reservation_id=rid, account='vlm', gpu_seconds=secs,
    proposal_sha256=L.digest(proposal), ledger=dict(path=str(lp), sha256=sha(lp)),
    approval_receipt=dict(path=str(receipt), sha256=sha(receipt)), maximum_allocations=1))
approval = copy.deepcopy(request['approval_template'])
approval.update(status='approved', user_approval_quote=os.environ['Q'], user_approval_receipt=str(receipt),
                user_approval_receipt_sha256=sha(receipt), shared_gpu_reservation=dict(path=str(reservation), sha256=sha(reservation)))
L.validate_native_approval(proposal, approval); put(P/'approved-request.json', approval)
EOF
  )
  (cd $P/source && TMPDIR=/tmp PYTHONNOUSERSITE=1 $PY -B scripts/submit_webprm_table5_campaign.py --proposal $P/proposal.json \
      --approval $P/approved-request.json --submit 2>&1 | grep -o '"job_id": "[0-9]*"\|"status": "[a-z_]*"\|rror.*' | tr '\n' ' ')
  echo " <- $name"
done
# Afterwards: record each job_id in $G/ledger.json (status SUBMITTED) and in the supervisor pointer
# /gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-webarena-20261011/current_table5_reproduction.json next_stage_jobs.
