"""Durable ID-based all-attempt budgets and a process-shared model HTTP gate.
No automatic retries, refunds, orphan reclamation, PID-based slot recovery, or
clock reset. Root owns explicit reconciliation of any interrupted owner.
"""
from pathlib import Path
from contextlib import contextmanager
from decimal import Decimal, InvalidOperation
import fcntl,json,os,time,math,uuid

def need(value,message):
 if not value:raise ValueError(message)
def money(value):
 need(type(value)in(int,float,str,Decimal)and type(value)is not bool,'Typed dollar amount required')
 try:d=Decimal(str(value))
 except InvalidOperation:raise ValueError('Malformed dollar amount')from None
 need(d.is_finite()and d>=0,'Finite nonnegative dollars required');return d

def atomic_json(path,data):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.tmp-'+uuid.uuid4().hex)
 with tmp.open('x')as f:json.dump(data,f,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(tmp,path)

def exclusive_json(path,data):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('x')as f:json.dump(data,f,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())

class ReservationBudget:
 def __init__(self,root,max_usd,max_calls):
  self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True);self.max_usd=money(max_usd);need(type(max_calls)is int and max_calls>0,'Positive integer call cap');self.max_calls=max_calls
 @contextmanager
 def locked(self):
  with(self.root/'owner.lock').open('a+')as f:
   fcntl.flock(f,fcntl.LOCK_EX);yield
 def _state(self):
  p=self.root/'ledger.json'
  if not p.exists():return {'schema':'physical-id-reservations-v1','max_usd':str(self.max_usd),'max_calls':self.max_calls,'calls':0,'charged_or_reserved_usd':'0','requests':{}}
  d=json.loads(p.read_text());need(d['max_usd']==str(self.max_usd)and d['max_calls']==self.max_calls,'Budget caps changed');need(d['calls']==len(d['requests']),'Budget ID/call count mismatch');need(money(d['charged_or_reserved_usd'])==sum((money(x['charged_or_reserved_usd'])for x in d['requests'].values()),Decimal(0)),'Budget total mismatch');return d
 def _save(self,state,event):
  # Persist the conservative state before journal. A crash cannot erase a charge.
  atomic_json(self.root/'ledger.json',state)
  with(self.root/'events.jsonl').open('a')as f:json.dump(event,f,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 def halted(self):return(self.root/'halt.json').exists()
 def halt(self,reason):
  with self.locked():
   if not self.halted():exclusive_json(self.root/'halt.json',{'epoch':time.time(),'reason':str(reason)})
 def reserve(self,physical_id,reserved_usd,metadata=None):
  need(type(physical_id)is str and bool(physical_id),'Physical ID required');amount=money(reserved_usd)
  with self.locked():
   d=self._state();need(not self.halted(),'Budget halted; explicit diagnosis required');need(physical_id not in d['requests'],'Duplicate physical ID cannot be retried');need(d['calls']<self.max_calls,'All-attempt call cap exhausted');need(money(d['charged_or_reserved_usd'])+amount<=self.max_usd,'All-attempt model/judge USD cap exhausted')
   row={'physical_id':physical_id,'reserved_usd':str(amount),'charged_or_reserved_usd':str(amount),'status':'reserved','reserve_epoch':time.time(),'metadata':metadata or {}};d['requests'][physical_id]=row;d['calls']+=1;d['charged_or_reserved_usd']=str(money(d['charged_or_reserved_usd'])+amount);self._save(d,{'event':'reserve',**row});return dict(row)
 def reserve_when_available(self,physical_id,reserved_usd,metadata=None,wait_seconds=120,poll_seconds=.05,deadline_monotonic=None):
  """Wait for temporary full reservations to settle, never for new funds.

  No call/ID/charge is admitted while waiting. Settled and unknown usage remains
  fully charged. Money and the call cap are rechecked atomically under flock;
  sleep holds neither that lock nor a model HTTP slot. No orphan reclamation.
  """
  need(type(physical_id)is str and bool(physical_id),'Physical ID required');amount=money(reserved_usd)
  need(type(wait_seconds)in(int,float)and math.isfinite(wait_seconds)and 0<wait_seconds<=120,'Admission wait must be finite and at most120seconds')
  need(type(poll_seconds)in(int,float)and math.isfinite(poll_seconds)and 0<poll_seconds<=1,'Finite positive admission polling interval required')
  start=time.monotonic();deadline=start+wait_seconds
  if deadline_monotonic is not None:
   need(type(deadline_monotonic)in(int,float)and math.isfinite(deadline_monotonic),'Finite absolute monotonic deadline required');deadline=min(deadline,deadline_monotonic)
  while True:
   with self.locked():
    d=self._state();need(not self.halted(),'Budget halted; explicit diagnosis required');need(physical_id not in d['requests'],'Duplicate physical ID cannot be retried');need(d['calls']<self.max_calls,'All-attempt call cap exhausted')
    # Even if every live reservation returned zero usage, these immutable
    # settled/unknown charges cannot make room: fail immediately and honestly.
    committed=sum((money(row['charged_or_reserved_usd'])for row in d['requests'].values()if row['status']!='reserved'),Decimal(0))
    need(committed+amount<=self.max_usd,'Settled/unknown usage plus full request reserve exceeds frozen USD cap')
    now=time.monotonic()
    if now>=deadline:raise TimeoutError('Bounded reservation admission deadline exhausted; no call admitted')
    total=money(d['charged_or_reserved_usd'])
    if total+amount<=self.max_usd:
     row={'physical_id':physical_id,'reserved_usd':str(amount),'charged_or_reserved_usd':str(amount),'status':'reserved','reserve_epoch':time.time(),'metadata':metadata or{},'admission_wait_seconds':now-start}
     d['requests'][physical_id]=row;d['calls']+=1;d['charged_or_reserved_usd']=str(total+amount);self._save(d,{'event':'reserve',**row});return dict(row)
    # Only unsettled reservations can make the currently blocked amount fit.
    need(any(row['status']=='reserved'and money(row['charged_or_reserved_usd'])>0 for row in d['requests'].values()),'No in-flight charge can settle to permit admission')
   time.sleep(min(poll_seconds,max(0,deadline-time.monotonic())))
 def finish(self,physical_id,actual_usd,usage=None):
  actual=money(actual_usd)
  with self.locked():
   d=self._state();need(physical_id in d['requests'],'Finish has no durable matching physical reservation');row=d['requests'][physical_id];need(row['status']=='reserved','Duplicate/unknown physical settlement forbidden');reserve=money(row['reserved_usd']);d['charged_or_reserved_usd']=str(money(d['charged_or_reserved_usd'])-reserve+actual);row.update(status='known',charged_or_reserved_usd=str(actual),actual_usd=str(actual),usage=usage,finish_epoch=time.time());self._save(d,{'event':'finish',**row})
   if actual>reserve or money(d['charged_or_reserved_usd'])>self.max_usd:
    if not self.halted():exclusive_json(self.root/'halt.json',{'epoch':time.time(),'reason':'Actual usage exceeded conservative reservation','physical_id':physical_id})
    raise ValueError('Actual usage exceeded reservation; charge retained and shard halted')
   return dict(row)
 def unknown(self,physical_id,reason):
  with self.locked():
   d=self._state();need(physical_id in d['requests'],'Unknown has no durable reservation');row=d['requests'][physical_id];need(row['status']=='reserved','Known/unknown record cannot be replaced');row.update(status='usage_unknown',unknown_epoch=time.time(),error_type=str(reason));self._save(d,{'event':'usage_unknown',**row})
   if not self.halted():exclusive_json(self.root/'halt.json',{'epoch':time.time(),'reason':'Unknown physical usage; full reserve retained','physical_id':physical_id})
 def snapshot(self):
  with self.locked():
   d=self._state();return {**d,'charged_or_reserved_usd_decimal':d['charged_or_reserved_usd'],'charged_or_reserved_usd':float(money(d['charged_or_reserved_usd'])),'unknown_calls':sum(x['status']=='usage_unknown'for x in d['requests'].values()),'reserved_calls':sum(x['status']=='reserved'for x in d['requests'].values())}

class SharedHTTPGate:
 """Never stale-clears slots. An interrupted process leaves a visible blocker."""
 def __init__(self,root,limit=8,wait_seconds=180):
  need(type(limit)is int and 0<limit<=8,'Model HTTP gate must be<=8');need(type(wait_seconds)in(int,float)and math.isfinite(wait_seconds)and wait_seconds>=0,'Finite gate timeout');self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True);self.limit=limit;self.wait_seconds=wait_seconds
 @contextmanager
 def locked(self):
  with(self.root/'lock').open('a+')as f:fcntl.flock(f,fcntl.LOCK_EX);yield
 def _read(self):
  p=self.root/'state.json';d=json.loads(p.read_text())if p.exists()else{'limit':self.limit,'active':{},'peak':0};need(d['limit']==self.limit and len(d['active'])<=self.limit,'HTTP shared gate changed/corrupt');return d
 @contextmanager
 def slot(self,physical_id):
  start=time.monotonic()
  while True:
   with self.locked():
    d=self._read();need(physical_id not in d['active'],'Duplicate active HTTP ID')
    if len(d['active'])<self.limit:
     d['active'][physical_id]={'pid':os.getpid(),'started_epoch':time.time()};d['peak']=max(d['peak'],len(d['active']));atomic_json(self.root/'state.json',d);break
   if time.monotonic()-start>=self.wait_seconds:raise TimeoutError('Shared HTTP slots unavailable; no automatic orphan reclamation')
   time.sleep(.02)
  try:yield
  finally:
   with self.locked():
    d=self._read();need(physical_id in d['active'],'HTTP ID disappeared');d['active'].pop(physical_id);atomic_json(self.root/'state.json',d)
