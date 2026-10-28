"""Close one finished v12 allocation in the v12 ledger at its actual sacct elapsed time and actual API charge,
releasing the unused GPU-seconds and API dollars for replacement packages. Refuses a job that is still PENDING/RUNNING
or already closed. Usage: ledger_close_v11.py JOB_ID [REASON]   (prints the ledger totals)"""
import json, subprocess, sys
from pathlib import Path
G = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-webarena-20261011/next-stage-preparation/webarena-subset-review-20261021/gpu-budget-v12-20261028')
job = sys.argv[1]; reason = sys.argv[2] if len(sys.argv) > 2 else ''
lp = G/'ledger.json'; d = json.loads(lp.read_text())
state, elapsed = subprocess.run(['sacct', '-X', '-n', '-P', '-j', job, '-o', 'State,ElapsedRaw'], capture_output=True, text=True,
                                check=True).stdout.strip().splitlines()[0].split('|')
assert not state.startswith(('PENDING', 'RUNNING', 'REQUEUED')), ('job not finished', job, state)
elapsed = int(elapsed)
rows = [a for a in d['allocations'] if a.get('job_id') == job]
assert len(rows) == 1, ('one ledger allocation per job', job, len(rows))
a = rows[0]
if a.get('status') != 'SUBMITTED':
    print('already closed:', job, a['status'])
else:
    events = Path(a['package'])/'campaign/shared-budget.json.events.jsonl'  # actual spend = sum of settle 'actual' (v11 lesson)
    charged = sum(float(e.get('actual', 0)) for e in (json.loads(l) for l in events.read_text().splitlines() if l.strip())
                  if e.get('event', e.get('type')) == 'settle') if events.exists() else 0.0
    a.update(status=state.split()[0] + ('_' + reason if reason else ''), elapsed_seconds=elapsed, elapsed_source='sacct ElapsedRaw',
             released_gpu_seconds=a['reserved_gpu_seconds'] - elapsed, reserved_gpu_seconds=elapsed,
             api_usd_charged=round(charged, 6), api_usd_cap_released=round(a['api_usd_cap'] - charged, 6), api_usd_cap=round(charged, 6))
    assert a['released_gpu_seconds'] >= 0 and a['api_usd_cap_released'] >= 0
reserved = sum(x['reserved_gpu_seconds'] for x in d['allocations'])
d['unreserved_gpu_seconds'] = d['gpu_seconds_cap'] - reserved
d['used_gpu_seconds'] = sum(x.get('elapsed_seconds', 0) for x in d['allocations'] if x.get('status') != 'SUBMITTED')
d['api_usd_reserved_or_charged'] = round(sum(x['api_usd_cap'] for x in d['allocations']), 6)
d['unreserved_api_usd'] = round(d['api_usd_cap'] - d['api_usd_reserved_or_charged'], 6)
assert d['unreserved_gpu_seconds'] >= 0 and d['unreserved_api_usd'] >= 0
lp.write_text(json.dumps(d, indent=1)+'\n')
open_ = [x for x in d['allocations'] if x.get('status') == 'SUBMITTED']
print(json.dumps(dict(closed=job, used_gpu_seconds=d['used_gpu_seconds'], reserved_open_gpu_seconds=sum(x['reserved_gpu_seconds'] for x in open_),
                      unreserved_gpu_seconds=d['unreserved_gpu_seconds'], api_usd_reserved_or_charged=d['api_usd_reserved_or_charged'],
                      unreserved_api_usd=d['unreserved_api_usd'], open_jobs=[x['job_id'] for x in open_]), indent=1))
