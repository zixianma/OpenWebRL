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
 """Eight fixed GPFS atomic-directory slots; no shared JSON mutation/flock.

 A dead owner leaves its directory occupied. No PID/liveness-based reclamation.
 Release unlinks only the exact unique token filename it created, then removes
 that empty slot directory once. A stale owner can never unlink a successor's
 differently named owner file. Durable identity/acquire/release receipts remain.
 """
 def __init__(self,root,limit=8,wait_seconds=180):
  need(type(limit)is int and 0<limit<=8,'Model HTTP gate must be<=8');need(type(wait_seconds)in(int,float)and math.isfinite(wait_seconds)and wait_seconds>=0,'Finite gate timeout');self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True);self.limit=limit;self.wait_seconds=wait_seconds
  (self.root/'physical-identities').mkdir(exist_ok=True);(self.root/'leases').mkdir(exist_ok=True)
  self._publish(self.root/'capacity.json',{'schema':'atomic-directory-http-gate-v1','limit':limit},same_ok=True)
 @staticmethod
 def _encoded(value):return (json.dumps(value,sort_keys=True,allow_nan=False)+'\n').encode()
 def _publish(self,path,data,same_ok=False):
  """Publish fully fsynced immutable evidence atomically using no-overwrite link."""
  path=Path(path);encoded=self._encoded(data);temporary=path.with_name('.'+path.name+'.'+uuid.uuid4().hex+'.partial')
  with temporary.open('xb')as stream:stream.write(encoded);stream.flush();os.fsync(stream.fileno())
  try:
   try:os.link(temporary,path)
   except FileExistsError:
    need(same_ok and path.read_bytes()==encoded,'Duplicate/changed immutable gate evidence '+str(path))
  finally:temporary.unlink()
 def snapshot(self):
  """Occupied slots are authoritative, even an incompletely published owner."""
  active={};slots=[]
  for index in range(self.limit):
   slot=self.root/f'slot-{index:02d}'
   try:names=list(slot.iterdir())
   except FileNotFoundError:continue
   owners=[p for p in names if p.name.startswith('owner-')and p.suffix=='.json'];row={'slot':index,'occupied':True,'owner_pending_or_unknown':True}
   if len(owners)==1:
    try:row=json.loads(owners[0].read_text());row.update(slot=index,occupied=True,owner_pending_or_unknown=False)
    except(FileNotFoundError,json.JSONDecodeError):pass
   key=row.get('physical_id','occupied-slot-'+str(index));active[key]=row;slots.append(index)
  return {'schema':'atomic-directory-http-gate-v1','limit':self.limit,'active':active,'occupied_slot_indices':slots,'occupied_count':len(slots),'capacity_proven_by_fixed_atomic_directories':True,'peak':self.limit,'peak_is_capacity_bound_not_measured_peak':True}
 def _release(self,slot,owner_path,identity,lease):
  need(owner_path.parent==slot and owner_path.name=='owner-'+identity['lease_token']+'.json','Release token/path mismatch')
  need(owner_path.read_bytes()==self._encoded(identity),'Owned gate lease identity changed; do not reclaim')
  # This token-specific unlink cannot remove another lease's differently named
  # file, even if a corrupted/stale caller references a reused slot directory.
  owner_path.unlink();slot.rmdir()
  self._publish(lease/'released.json',{'physical_id':identity['physical_id'],'lease_token':identity['lease_token'],'slot':int(slot.name.split('-')[1]),'released_epoch':time.time(),'node':identity['node'],'verified_owned_release':True})
 @contextmanager
 def slot(self,physical_id):
  import socket,threading
  need(type(physical_id)is str and len(physical_id)==32 and all(c in'0123456789abcdef'for c in physical_id),'Physical gate ID must be exact UUID hex')
  token=uuid.uuid4().hex;start=time.monotonic();identity={'physical_id':physical_id,'lease_token':token,'pid':os.getpid(),'thread_id':threading.get_ident(),'node':socket.gethostname(),'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'created_epoch':time.time()};lease=self.root/'leases'/token;lease.mkdir(exist_ok=False)
  self._publish(self.root/'physical-identities'/(physical_id+'.json'),identity)
  slot=None;owner_path=None
  try:
   while slot is None:
    for index in range(self.limit):
     candidate=self.root/f'slot-{index:02d}'
     try:candidate.mkdir()
     except FileExistsError:continue
     slot=candidate;owner_path=slot/('owner-'+token+'.json');self._publish(owner_path,identity);break
    if slot is not None:break
    if time.monotonic()-start>=self.wait_seconds:raise TimeoutError('Fixed HTTP slots occupied; no automatic dead-owner reclamation')
    time.sleep(.02)
   self._publish(lease/'acquired.json',{'identity':identity,'slot':int(slot.name.split('-')[1]),'acquired_epoch':time.time(),'admission_wait_seconds':time.monotonic()-start,'slot_directory':str(slot),'owner_path':str(owner_path)})
   try:yield
   finally:self._release(slot,owner_path,identity,lease)
  except BaseException as exc:
   self._publish(lease/'error.json',{'physical_id':physical_id,'lease_token':token,'error_type':type(exc).__name__,'error':str(exc),'epoch':time.time(),'slot_owned':str(slot)if slot is not None else None})
   raise
