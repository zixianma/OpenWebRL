"""Isolate metered native HTTP500/503/520 cells without resending any failed call."""
from pathlib import Path
import json,os,fcntl,hashlib,collections,re,time

def isolate_provider_failure(root,item,row,budgets):
 if not (row.get('physical_browser_attempt') and row.get('valid') is False and row.get('reward') is None and row.get('error_type')=='InternalServerError' and re.match(r'^Error code: (500|503|520) -',row.get('error',''))):return row
 if row.get('browser_cleanup',{}).get('returncode')!=0:return row
 job=os.environ['SLURM_JOB_ID'];ep=root/'episodes'/(item['task_id']+'-browsercode-'+row['mode']);unknown=[]
 for p in (ep/'requests').glob('luna-*.json'):
  d=json.loads(p.read_text())
  if d.get('status')=='usage_unknown':
   if d.get('error_type')!='InternalServerError':return row
   unknown.append((p,d))
 if not unknown:return row
 with (root/'provider-5xx-review.lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  pending=json.loads((budgets['actor'].root/'outstanding-reservations.json').read_text());available=collections.Counter(round(x,12) for x in pending);required=collections.Counter(round(d['reserved_usd'],12) for p,d in unknown)
  if required-available:raise AssertionError('Unknown provider charge missing from retained reservations')
  errors=[]
  for p in (root/'episodes').glob('*/result.json'):
   d=json.loads(p.read_text())
   if d.get('job_id')==job and d.get('physical_browser_attempt') and d.get('error_type')=='InternalServerError' and re.match(r'^Error code: (500|503|520) -',d.get('error','')):errors.append(d['task_id'])
  if row.get('error','').startswith('Error code: 520 -'):
   (root/'provider-admission-not-before.json').write_text(json.dumps({'epoch':time.time(),'not_before_epoch':time.time()+60,'job_id':job,'reason':'Native API edge HTTP520 explicitly requests at least60s backoff; no failed request is resent.'},indent=2)+'\n')
  halt=budgets['actor'].root/'halt.json'
  if len(errors)>=3:
   (root/('provider-circuit-breaker-'+job+'.json')).write_text(json.dumps({'epoch':time.time(),'job_id':job,'failed_tasks_this_attempt':errors,'threshold':3,'action':'Keep actor halt; no further dispatch. In-flight work remains accounted.'},indent=2)+'\n');return row
  if not halt.exists():return row
  state=json.loads(halt.read_text())
  if state!={'reason':'API failure; inspect before dispatching further calls','error_type':'InternalServerError'}:return row
  target=ep/'reviewed-provider-5xx-halt.json';assert not target.exists();halt.rename(target)
  (ep/'provider-5xx-isolation-review.json').write_text(json.dumps({'epoch':time.time(),'job_id':job,'failed_task_id':item['task_id'],'physical_outcome_unchanged':True,'failed_call_resent':False,'unknown_reservations_preserved':[d['reserved_usd'] for p,d in unknown],'unknown_receipt_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p,d in unknown},'circuit_breaker_threshold':3,'failures_this_attempt':len(errors)},indent=2)+'\n')
 return row

def wait_for_provider_backoff(root):
 marker=root/'provider-admission-not-before.json'
 while marker.exists():
  remaining=json.loads(marker.read_text())['not_before_epoch']-time.time()
  if remaining<=0:return
  time.sleep(min(2,remaining))
