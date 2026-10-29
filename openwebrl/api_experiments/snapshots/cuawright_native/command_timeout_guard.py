"""Mechanical240s command ownership; actor uses outside broker identity receipts."""
import ctypes,hashlib,inspect,json,os,signal,socket,subprocess,time,uuid
from pathlib import Path
EXECUTE_SHA256='d0fbd0e6d31f6764732d8aa5e302753b2d31845e389bcf9759bed5bd193d88ff'

class CommandCleanupUnverified(BaseException):pass
class CommandInterrupted(BaseException):pass

def identity(pid):
 try:
  fields=Path('/proc/'+str(pid)+'/stat').read_text().rsplit(')',1)[1].split()
  return {'pid':int(pid),'parent':int(fields[1]),'process_group':int(fields[2]),'session':int(fields[3]),'start_ticks':fields[19],'state':fields[0]}
 except (OSError,ValueError):return None

def group_members(original):
 rows=[]
 for p in Path('/proc').iterdir():
  if p.name.isdigit():
   now=identity(int(p.name))
   if now and now['process_group']==original['pid'] and int(now['start_ticks'])>=int(original['start_ticks']):rows.append(now)
 return rows

class CommandOwnership:
 def __init__(self,parent_pid,evidence):
  import threading
  self.parent_pid=int(parent_pid);self.evidence=Path(evidence);self.commands={};self.lock=threading.RLock();self.owner_identity=identity(self.parent_pid);assert self.owner_identity;self.owner_uid=os.getuid()
 def record(self,row):
  with self.lock,(self.evidence/'command-timeout-events.jsonl').open('a') as f:f.write(json.dumps({'time':time.time(),**row})+'\n')
 def authorize(self,peer_pid,peer_uid,token,operation):
  now=identity(self.parent_pid)
  if int(peer_pid)!=self.parent_pid or int(peer_uid)!=self.owner_uid or not now or now['start_ticks']!=self.owner_identity['start_ticks']:
   raise PermissionError('Command RPC caller is not the original native owner')
  result={'peer_pid':int(peer_pid),'peer_uid':int(peer_uid),'owner_start_ticks':self.owner_identity['start_ticks']}
  self.record({'event':'command_peer_verified','operation':operation,'token':token,**result});return result
 def register(self,token,pid,command_sha256):
  assert len(token)==32 and all(c in '0123456789abcdef' for c in token)
  assert len(command_sha256)==64 and all(c in '0123456789abcdef' for c in command_sha256)
  original=identity(int(pid));assert original and original['parent']==self.parent_pid and original['pid']==original['process_group']==original['session'] and int(pid) not in (os.getpid(),self.parent_pid),'Command is not an owned direct-child session'
  with self.lock:
   assert token not in self.commands;self.commands[token]=original
  self.record({'event':'command_registered','token':token,'owner_pid':self.parent_pid,'owner_start_ticks':self.owner_identity['start_ticks'],'owner_uid':self.owner_uid,'identity':original,'command_sha256':command_sha256})
  return {'verified':True,'identity':original,'token':token}
 def cleanup(self,token,grace=.15,wait=3):
  with self.lock:original=self.commands[token]
  # Parent retains this shell unreaped until this response. Any changed start
  # identity invalidates the request; never infer ownership from a reused PGID.
  current=identity(original['pid'])
  if not current or current['start_ticks']!=original['start_ticks'] or current['process_group']!=original['pid']:
   result={'verified':False,'reason':'leader_identity_missing_or_reused','members':[],'remaining':[],'signals':[]};self.record({'event':'command_cleanup','token':token,**result});return result
  captured={};signals=[];deadline=time.monotonic()+wait
  def capture():
   for row in group_members(original):captured[(row['pid'],row['start_ticks'])]=row
  def send(signum):
   capture()
   for row in list(captured.values()):
    now=identity(row['pid'])
    if now and now['start_ticks']==row['start_ticks'] and now['state']!='Z':
     try:os.kill(row['pid'],signum);signals.append({'pid':row['pid'],'start_ticks':row['start_ticks'],'signal':int(signum)})
     except ProcessLookupError:pass
  send(signal.SIGTERM);time.sleep(grace);send(signal.SIGKILL)
  while True:
   capture();remaining=[row for row in captured.values() if (now:=identity(row['pid'])) and now['start_ticks']==row['start_ticks'] and now['state']!='Z']
   if not remaining or time.monotonic()>=deadline:break
   send(signal.SIGKILL);time.sleep(.03)
  result={'verified':not remaining,'identity':original,'members':list(captured.values()),'remaining':remaining,'signals':signals,'bounded_wait_seconds':wait}
  self.record({'event':'command_cleanup','token':token,**result})
  if result['verified']:
   with self.lock:self.commands.pop(token,None)
  return result
 def finish(self,token):
  with self.lock:original=self.commands.pop(token)
  self.record({'event':'command_finished','token':token,'identity':original});return {'verified':True}


def rpc(cfg,op,**fields):
 connection=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM);connection.settimeout(10)
 try:
  connection.connect(cfg['browser_broker_socket']);connection.sendall(json.dumps({'op':op,**fields}).encode()+b'\n');row=json.loads(connection.makefile('rb').readline())
  if row.get('error'):raise RuntimeError(row['error'])
  return row
 finally:connection.close()

def reap_group(group,timeout=2):
 deadline=time.monotonic()+timeout;reaped=[]
 while time.monotonic()<deadline:
  try:
   pid,status=os.waitpid(-group,os.WNOHANG)
   if pid:reaped.append({'pid':pid,'wait_status':status});continue
  except ChildProcessError:return {'verified':True,'reaped':reaped}
  time.sleep(.02)
 return {'verified':False,'reaped':reaped}

class RunProxy:
 def __init__(self,real,cfg,evidence,rpc_function=None):self.real=real;self.cfg=cfg;self.evidence=evidence;self.rpc_function=rpc_function or (lambda op,**fields:rpc(cfg,op,**fields))
 def __getattr__(self,name):return getattr(self.real,name)
 def note(self,row):
  with (Path(self.cfg['output'])/'command-timeout-observations.jsonl').open('a') as f:f.write(json.dumps({'time':time.time(),**row})+'\n')
 def run(self,*args,**kwargs):
  # Avoid Popen's KeyboardInterrupt path reaping the leader before the host can
  # validate it. Preserve handlers outside this one selected-command interval.
  import threading
  previous={}
  if threading.current_thread() is threading.main_thread():
   def interrupted(signum,frame):raise CommandInterrupted('Owned command interrupted by signal '+str(signum))
   for signum in (signal.SIGTERM,signal.SIGINT):
    previous[signum]=signal.getsignal(signum);signal.signal(signum,interrupted)
  try:return self._run_owned(*args,**kwargs)
  finally:
   for signum,handler in previous.items():signal.signal(signum,handler)
 def _run_owned(self,*popenargs,input=None,capture_output=False,timeout=None,check=False,**kwargs):
  assert timeout is not None and timeout>0
  assert not kwargs.get('preexec_fn') and not kwargs.get('process_group')
  if input is not None:
   if kwargs.get('stdin') is not None:raise ValueError('stdin and input arguments may not both be used')
   kwargs['stdin']=self.real.PIPE
  if capture_output:
   if kwargs.get('stdout') is not None or kwargs.get('stderr') is not None:raise ValueError('stdout/stderr with capture_output')
   kwargs['stdout']=self.real.PIPE;kwargs['stderr']=self.real.PIPE
  kwargs['start_new_session']=True;deadline=time.monotonic()+timeout;proc=self.real.Popen(*popenargs,**kwargs);token=uuid.uuid4().hex;registered=False
  command=popenargs[0] if popenargs else kwargs.get('args');digest=hashlib.sha256(command.encode() if isinstance(command,str) else json.dumps(command).encode()).hexdigest()
  try:
   registered_row=self.rpc_function('command_register',token=token,pid=proc.pid,command_sha256=digest);assert registered_row['verified'];registered=True
   original=registered_row['identity'];assert original['pid']==original['process_group']==original['session']==proc.pid
   with (Path(self.cfg['output'])/'owned-child-processes.jsonl').open('a') as f:f.write(json.dumps(original)+'\n')
   try:stdout,stderr=proc.communicate(input,timeout=max(.001,deadline-time.monotonic()))
   except self.real.TimeoutExpired as error:
    # communicate timeout does not reap/kill the child. Host validates retained
    # PID/startticks before touching any descendant; browser broker is another PG.
    cleanup=self.rpc_function('command_cleanup',token=token);self.note({'event':'timeout_cleanup','token':token,'command_sha256':digest,'cleanup':cleanup,'configured_timeout_seconds':timeout})
    for member in cleanup.get('members',[]):
     with (Path(self.cfg['output'])/'owned-child-processes.jsonl').open('a') as f:f.write(json.dumps(member)+'\n')
    if not cleanup.get('verified'):raise CommandCleanupUnverified('Timed-out command descendants not verified closed')
    stdout,stderr=proc.communicate(timeout=2);reaped=reap_group(proc.pid);self.note({'event':'timeout_reaped','token':token,'reaping':reaped})
    if not reaped['verified']:raise CommandCleanupUnverified('Timed-out command descendants not reaped')
    error.output=stdout;error.stderr=stderr;error.timeout=timeout;raise
   self.rpc_function('command_finish',token=token)
   if check and proc.returncode:raise self.real.CalledProcessError(proc.returncode,proc.args,output=stdout,stderr=stderr)
   return self.real.CompletedProcess(proc.args,proc.returncode,stdout,stderr)
  except (self.real.TimeoutExpired,self.real.CalledProcessError):raise
  except CommandCleanupUnverified:raise
  except BaseException:
   # Registration/transport failures stop the episode, never start another model
   # turn with a potentially live command. Retain child for host/parent cleanup.
   if registered:
    try:
     cleanup=self.rpc_function('command_cleanup',token=token)
     self.note({'event':'interrupted_cleanup','token':token,'cleanup':cleanup})
     for member in cleanup.get('members',[]):
      with (Path(self.cfg['output'])/'owned-child-processes.jsonl').open('a') as f:f.write(json.dumps(member)+'\n')
     if cleanup.get('verified'):
      proc.communicate(timeout=2);self.note({'event':'interrupted_reaped','token':token,'reaping':reap_group(proc.pid)})
    except BaseException:pass
   else:
    # No communicate/wait/poll has occurred: the new session leader is our
    # unreaped direct child, so its numeric PID cannot have been recycled.
    if proc.returncode is None and os.getpgid(proc.pid)==proc.pid:
     for signum in (signal.SIGTERM,signal.SIGKILL):
      try:os.killpg(proc.pid,signum)
      except ProcessLookupError:pass
     proc.wait(timeout=2);reap_group(proc.pid)
   raise CommandCleanupUnverified('Command ownership/transport failed closed')

def install(cfg,rpc_function=None):
 from cuawright.webwright.environments import local_workspace
 original=local_workspace.LocalWorkspaceEnvironment.execute
 if hashlib.sha256(inspect.getsource(original).encode()).hexdigest()!=EXECUTE_SHA256:raise RuntimeError('Pinned native execute changed')
 if isinstance(local_workspace.subprocess,RunProxy):return local_workspace.subprocess.evidence
 libc=ctypes.CDLL(None,use_errno=True)
 if libc.prctl(36,1,0,0,0):raise RuntimeError('Cannot own orphan command descendants')
 evidence={'adapter':'owned-command-timeout-descendants-v1','upstream_execute_sha256':EXECUTE_SHA256,'native_timeout_unchanged':True,'actor_subreaper':True,'cleanup_identity_source':'outside-actor-browser-broker'}
 local_workspace.subprocess=RunProxy(local_workspace.subprocess,cfg,evidence,rpc_function)
 return evidence


def registered_shard_scope(root,output):
 import re
 root=Path(root).resolve();output=Path(output).resolve();parts=output.relative_to(root).parts
 assert parts and re.fullmatch(r'shard(?:0|[1-9][0-9]*)',parts[0]),'Malformed registered shard path'
 shard=int(parts[0][5:]);plan=json.loads((root/'plan.json').read_text());caps=plan['shards']
 assert 0<=shard<len(caps) and caps[shard]['shard']==shard and output.name in caps[shard]['task_ids'],'Native task/shard outside frozen local plan'
 centralpath=root.parents[1]/'request.json';centralraw=centralpath.read_bytes();assert hashlib.sha256(centralraw).hexdigest()==plan['central_plan_sha256'],'Central request binding differs'
 arms=[a for a in json.loads(centralraw)['arms'] if a['id']==plan['arm_id']];assert len(arms)==1 and len(arms[0]['shards'])==len(caps) and arms[0]['shards'][shard]['task_ids']==caps[shard]['task_ids'],'Native shard outside exact approved arm'
 return shard

def bind_runtime_inventory(root,output):
 """Select a source version registered BEFORE installation, including live originals.

 The original shard registration's prospective history can authorize a future
 child without mutating an immutable replacement registration already queued.
 """
 import re
 root=Path(root).resolve();output=Path(output).resolve();shard=registered_shard_scope(root,output)
 digest=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
 source_hashes={name:digest(root/name) for name in ('native_episode.py','browser_broker.py','command_timeout_guard.py','native_proposal_adapter.py','native_group_transport.py','native_group_relay.py','native_capture.py')}
 registrations=[root/f'shard{shard}/registration.json']
 active=root/f'shard{shard}/active-attempt.json'
 if active.exists():
  entry=json.loads(active.read_text());registration=Path(entry['registration_path']).resolve();assert registration.is_relative_to(root)
  if entry.get('registration_sha256'):assert digest(registration)==entry['registration_sha256']
  registrations.append(registration)
 matches=[]
 for registration in dict.fromkeys(registrations):
  registration_raw=registration.read_bytes();data=json.loads(registration_raw);assert data['shard']==shard
  entries=list(data.get('execution_preflight_history',[]))
  if data.get('execution_preflight_path'):entries.append({'path':data['execution_preflight_path'],'sha256':data['execution_preflight_sha256']})
  for entry in entries:
   path=Path(entry['path']).resolve();inventory_raw=path.read_bytes();assert path.is_relative_to(root) and hashlib.sha256(inventory_raw).hexdigest()==entry['sha256'];inventory=json.loads(inventory_raw)
   if inventory.get('passed') is not True or inventory.get('frozen_plan_sha256')!=digest(root/'plan.json'):continue
   if not all(inventory['source_hashes'].get(name)==sha for name,sha in source_hashes.items()):continue
   if not all((root/name).resolve().is_relative_to(root) and digest(root/name)==sha for name,sha in inventory['source_hashes'].items()):continue
   matches.append((path,entry['sha256'],registration,registration_raw,inventory_raw))
 assert matches,'Native child source was not prospectively registered'
 path,sha,registration,registration_raw,inventory_raw=matches[-1];snapshot=output/'runtime-execution-inventory.json';assert not snapshot.exists();snapshot.write_bytes(inventory_raw)
 registration_snapshot=output/'runtime-registration-snapshot.json';assert not registration_snapshot.exists();registration_snapshot.write_bytes(registration_raw)
 binding={'schema':'cuawright-runtime-child-source-v1','shard':shard,'pid':os.getpid(),'parent_pid':os.getppid(),'source_hashes':source_hashes,'original_plan_sha256':digest(root/'plan.json'),'registered_inventory_path':str(path),'registered_inventory_sha256':sha,'registration_path':str(registration),'registration_sha256':hashlib.sha256(registration_raw).hexdigest(),'registration_snapshot_path':str(registration_snapshot),'registration_snapshot_sha256':digest(registration_snapshot),'snapshot_path':str(snapshot),'snapshot_sha256':digest(snapshot)}
 (output/'runtime-child-source.json').write_text(json.dumps(binding,indent=2)+'\n');return binding
