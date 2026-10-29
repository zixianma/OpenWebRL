"""Owned local browser and native executor adapters; no model calls here."""
from bootstrap import ROOT,CONFIG
from pathlib import Path
import base64,json,os,re,selectors,signal,stat,struct,subprocess,sys,time,threading
from cuawright.webwright.environments.local_workspace import LocalWorkspaceEnvironment

def process_identity(pid):
 """Read one Linux identity; command names may contain spaces/parentheses."""
 try:
  data=Path(f'/proc/{int(pid)}/stat').read_text();parts=data[data.rfind(')')+2:].split()
  return {'pid':int(pid),'state':parts[0],'parent_pid':int(parts[1]),'process_group':int(parts[2]),'session':int(parts[3]),'start_ticks':int(parts[19])}
 except (FileNotFoundError,ProcessLookupError):return None

def record_owned(proc,directory,component):
 """Bind a newly spawned private session before it can run useful work."""
 import socket
 identity=process_identity(proc.pid)
 if identity is None:raise RuntimeError('Owned process exited before identity registration')
 if identity['process_group']!=proc.pid or identity['session']!=proc.pid:raise RuntimeError('Owned process must start a private process group/session')
 identity.update(node=socket.gethostname(),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),registered_epoch=time.time(),component=component)
 path=Path(directory)/('owned-identity-'+component+'.json')
 with path.open('x')as stream:
  json.dump(identity,stream,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())
 proc._owned_identity=identity;proc._owned_identity_path=str(path)
 return identity

def owned_group_members(identity):
 import socket
 if identity['node']!=socket.gethostname() or identity['boot_id']!=Path('/proc/sys/kernel/random/boot_id').read_text().strip():raise RuntimeError('Owned process belongs to another host or boot')
 root=process_identity(identity['pid'])
 if root is not None and root['start_ticks']!=identity['start_ticks']:raise RuntimeError('Owned root PID reused; refusing all signals')
 members=[]
 for path in Path('/proc').iterdir():
  if not path.name.isdigit():continue
  row=process_identity(int(path.name))
  if row and row['process_group']==identity['process_group'] and row['session']==identity['session'] and row['start_ticks']>=identity['start_ticks']:
   members.append(row)
 return members

def stop(proc,grace_seconds=30):
 """Terminate only registered identities, including surviving group descendants.

 Group/session IDs remain reserved while their original members survive. A dead
 leader therefore does not hide its children; each signal still rechecks the
 individual PID/start tick. Zombies are reported as dead, never signaled.
 """
 if proc is None:return {'verified':True,'remaining_live_group_members':[],'root_returncode':None}
 cached=getattr(proc,'_owned_close_receipt',None)
 if cached is not None:return cached
 identity=getattr(proc,'_owned_identity',None)
 if identity is None:
  # Generic callers must register immediately after Popen. Capture is safe only
  # while the Popen leader remains alive; an unregistered dead group is unknown.
  if proc.poll() is not None:raise RuntimeError('Cannot prove descendants of unregistered exited process')
  import socket
  identity=process_identity(proc.pid)
  if identity is None:raise RuntimeError('Owned process vanished before identity capture')
  if identity['process_group']!=proc.pid or identity['session']!=proc.pid:raise RuntimeError('Refusing nonprivate process group')
  identity.update(node=socket.gethostname(),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),registered_epoch=time.time(),component='generic-controller-child')
  proc._owned_identity=identity
 started=time.monotonic();signals=[];term_sent=set()
 def send(row,sig):
  fresh=process_identity(row['pid'])
  if fresh is None or fresh['state']=='Z':return
  if any(fresh[key]!=row[key] for key in ('start_ticks','process_group','session')):return
  try:os.kill(row['pid'],sig);signals.append({'pid':row['pid'],'start_ticks':row['start_ticks'],'signal':int(sig)})
  except ProcessLookupError:pass
 deadline=time.monotonic()+grace_seconds
 while True:
  rows=owned_group_members(identity);live=[r for r in rows if r['state']!='Z']
  if not live:break
  for row in live:
   key=(row['pid'],row['start_ticks'])
   if key not in term_sent:send(row,signal.SIGTERM);term_sent.add(key)
  if time.monotonic()>=deadline:break
  proc.poll();time.sleep(.1)
 for row in owned_group_members(identity):send(row,signal.SIGKILL)
 try:proc.wait(timeout=5)
 except subprocess.TimeoutExpired:pass
 settle=time.monotonic()+5
 while True:
  remaining=owned_group_members(identity);live=[r for r in remaining if r['state']!='Z']
  if not live or time.monotonic()>=settle:break
  for row in live:send(row,signal.SIGKILL)
  time.sleep(.05)
 receipt={'verified':not live and proc.poll()is not None,'identity':identity,'signals':signals,'root_returncode':proc.returncode,
          'wait_complete':proc.poll()is not None,'remaining_live_group_members':live,'zombie_group_members':[r for r in remaining if r['state']=='Z'],
          'cleanup_seconds':time.monotonic()-started,'verified_scope':'No live members remain in the recorded private process group/session; dead zombies reported separately'}
 if getattr(proc,'_owned_identity_path',None):
  path=Path(proc._owned_identity_path).with_name('owned-close-'+identity['component']+'.json');temp=path.with_suffix('.tmp-'+str(os.getpid()));temp.write_text(json.dumps(receipt,indent=2)+'\n');temp.replace(path)
 if not receipt['verified']:raise RuntimeError('Owned process-group cleanup did not verify empty')
 proc._owned_close_receipt=receipt
 return receipt

def cleanup_registered_groups(root,grace_seconds=30):
 """Reap every durably registered private session after worker termination.

 Nonchild exit codes cannot be recovered with waitpid. They remain unknown;
 process absence/zombie state and an empty live group are verified separately.
 All registrations are attempted before any unresolved ownership/cleanup raises.
 """
 root=Path(root).resolve();rows=[];errors=[]
 class RegisteredProcess:
  def __init__(self,identity,path):
   self.pid=identity['pid'];self._owned_identity=identity;self.returncode=None
  def poll(self):
   current=process_identity(self.pid)
   if current is not None and current['start_ticks']!=self._owned_identity['start_ticks']:raise RuntimeError('Registered root PID reused; refusing cleanup')
   return 0 if current is None or current['state']=='Z' else None
  def wait(self,timeout=None):
   deadline=time.monotonic()+(timeout or 0)
   while self.poll()is None:
    if time.monotonic()>=deadline:raise subprocess.TimeoutExpired('registered nonchild identity',timeout)
    time.sleep(.05)
   return None
 for path in sorted(root.rglob('owned-identity-*.json')):
  row={'identity_path':str(path),'verified':False}
  try:
   if path.is_symlink() or not path.resolve().is_relative_to(root):raise RuntimeError('Ownership receipt escapes owned attempt')
   identity=json.loads(path.read_text())
   required={'pid','start_ticks','process_group','session','node','boot_id','component','registered_epoch'}
   if not required.issubset(identity):raise RuntimeError('Incomplete durable owned identity')
   if not all(type(identity[k])is int and identity[k]>0 for k in ('pid','start_ticks','process_group','session')):raise RuntimeError('Invalid durable process identity')
   if not(identity['pid']==identity['process_group']==identity['session']):raise RuntimeError('Registered group is not a private session')
   if path.name!='owned-identity-'+identity['component']+'.json':raise RuntimeError('Ownership component/path mismatch')
   proxy=RegisteredProcess(identity,path)
   receipt=stop(proxy,grace_seconds=grace_seconds)
   receipt.update(nonchild_observation=True,root_returncode=None,root_returncode_known=False,waitpid_applicable=False,
                  verified_scope='No live members remain in registered private process group/session; nonchild exit status unavailable')
   close_path=path.with_name('registered-close-'+identity['component']+'.json');temp=close_path.with_suffix('.tmp-'+str(os.getpid()));temp.write_text(json.dumps(receipt,indent=2)+'\n');temp.replace(close_path)
   row.update(verified=receipt['verified'],cleanup=receipt)
  except Exception as exc:
   row.update(error_type=type(exc).__name__,error=str(exc));errors.append(row.copy())
  rows.append(row)
 receipt={'epoch':time.time(),'verified':not errors and all(row['verified'] for row in rows),'registered_groups':rows,'errors':errors}
 path=root/'registered-group-teardown.json';temp=path.with_suffix('.tmp-'+str(os.getpid()));temp.write_text(json.dumps(receipt,indent=2)+'\n');temp.replace(path)
 if errors:raise RuntimeError('Registered owned group cleanup unresolved: '+str(len(errors))+'; inspect registered-group-teardown.json')
 return receipt

def child_env(workspace,executable):
 env=os.environ.copy();env.update(WORKSPACE_DIR=str(workspace),CUA_CHROME_EXECUTABLE=str(executable),TMPDIR='/tmp',PYTHONDONTWRITEBYTECODE='1',BUN_RUNTIME_TRANSPILER_CACHE_PATH=str(workspace/'.cache/bun-transpiler'))
 for k in ['BASH_ENV','ENV','V4_RUN_ID','BU_CDP_WS','BU_CDP_URL']:env.pop(k,None)
 for k in ['XDG_CACHE_HOME','XDG_CONFIG_HOME','XDG_DATA_HOME']:env[k]=str(workspace/'.cache'/k.lower())
 return env

class StableWorkspace(LocalWorkspaceEnvironment):
 def _recent_workspace_files(self):
  root=self._workspace_dir();rows=[]
  for p in root.rglob('*'):
   try:s=p.stat()
   except FileNotFoundError:continue
   if stat.S_ISREG(s.st_mode):rows.append((s.st_mtime,p))
  rows.sort(key=lambda v:v[0],reverse=True)
  return [str(p.relative_to(root)) for _,p in rows[:self.config.recent_files_limit]]
 def close(self):
  cmd=[sys.executable,'-m','cuawright.webwright.tools.browser_session','close','--workspace-dir',str(self._workspace_dir())]
  try:
   output=self.execute({'command':__import__('shlex').join(cmd)});(self._workspace_dir()/'controller-cleanup.json').write_text(json.dumps(output,indent=2)+'\n')
  except Exception as exc:(self._workspace_dir()/'controller-cleanup-error.json').write_text(json.dumps({'error_type':type(exc).__name__,'error':str(exc)})+'\n')

def encode_browser_world_output(output,raw_reply,workspace,step_index):
 """Preserve world-side invalid UTF16 units as literal escapes before UTF8 I/O.

 Only malformed surrogate code points change. Valid Unicode, including real
 non-BMP characters, and every native assistant/thinking block stay untouched.
 The original bridge reply remains losslessly recoverable from ASCII JSON.
 """
 import hashlib
 invalid=[i for i,char in enumerate(output)if 0xD800<=ord(char)<=0xDFFF]
 if not invalid:return output,None
 escaped=output.encode('utf-8',errors='backslashreplace').decode('utf-8')
 directory=Path(workspace)/'world-output-encoding';directory.mkdir(parents=True,exist_ok=True)
 raw_path=directory/f'step_{step_index:04d}-raw-bridge-reply.json'
 raw_json=json.dumps(raw_reply,ensure_ascii=True,indent=2)+'\n'
 with raw_path.open('x',encoding='ascii')as stream:
  stream.write(raw_json);stream.flush();os.fsync(stream.fileno())
 receipt={'kind':'browsercode_world_output_utf8_encoding','algorithm':'Invalid surrogate code points become literal \\uXXXX; valid Unicode unchanged','altered_code_units':len(invalid),'altered_positions':invalid,
          'raw_world_utf8_surrogatepass_sha256':hashlib.sha256(output.encode('utf-8',errors='surrogatepass')).hexdigest(),
          'escaped_world_utf8_sha256':hashlib.sha256(escaped.encode('utf-8')).hexdigest(),'raw_bridge_reply_path':str(raw_path),
          'raw_bridge_reply_sha256':hashlib.sha256(raw_json.encode('ascii')).hexdigest(),'native_assistant_or_thinking_modified':False}
 receipt_path=directory/f'step_{step_index:04d}-encoding-receipt.json'
 with receipt_path.open('x',encoding='ascii')as stream:
  json.dump(receipt,stream,ensure_ascii=True,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())
 return escaped,receipt

class BrowserCodeWorkspace(StableWorkspace):
 def start(self,ws,target):
  self.counter=0;workspace=self._workspace_dir();bun=ROOT/'javascript-runtime/bun';log=(workspace/'browsercode-stderr.log').open('ab')
  self.proc=subprocess.Popen([sys.executable,str(ROOT/'sandbox.py'),'--browser',str(bun),str(ROOT/'browser_bridge.ts')],cwd=workspace,env=child_env(workspace,bun),stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,bufsize=1,start_new_session=True)
  record_owned(self.proc,workspace,'browsercode-bridge')
  result=self.call({'op':'init','wsUrl':ws,'targetId':target},30)
  if not result['ok']:raise RuntimeError('Native BrowserCode init failed: '+result.get('error',''))
 def call(self,request,timeout=125):
  self.counter+=1;request=dict(request,id=self.counter);self.proc.stdin.write(json.dumps(request)+'\n');self.proc.stdin.flush()
  with selectors.DefaultSelector() as waiter:
   waiter.register(self.proc.stdout,selectors.EVENT_READ)
   if not waiter.select(timeout):raise TimeoutError('Native BrowserCode bridge command deadline')
  line=self.proc.stdout.readline()
  if not line:raise RuntimeError('Native BrowserCode bridge exited')
  reply=json.loads(line)
  if reply.get('id')!=self.counter:raise RuntimeError('Native bridge response identity mismatch')
  return reply
 def execute(self,action,cwd=''):
  self._step_index+=1;command=str(action.get('command') or action.get('bash_command') or '');self._persist_step_command(command);args=json.loads(command)
  reply=self.call({'op':'execute','args':args});result=reply.get('result',{});output=result.get('output','') if reply['ok'] else reply.get('error','')
  output,encoding=encode_browser_world_output(output,reply,self._workspace_dir(),self._step_index)
  log=self._write_step_log(output)
  obs=self._capture_observation(command=command,cwd=self._workspace_dir(),output=output,returncode=0 if reply['ok'] else 1,exception_info='' if reply['ok'] else output,log_path=log)
  obs['image_attachments']=result.get('image_attachments',[])
  if encoding is not None:obs['world_output_encoding']=encoding
  return {'output':output,'returncode':obs['returncode'],'exception_info':obs['exception'],'observation':obs}
 def capture(self,path):
  reply=self.call({'op':'capture'},30)
  if not reply['ok']:raise RuntimeError('Native BrowserCode final capture failed: '+reply.get('error',''))
  Path(path).write_bytes(base64.b64decode(reply['result']['data']))
 def close(self):
  if not hasattr(self,'proc'):return
  try:self.call({'op':'close'},10)
  except Exception:pass
  finally:self.cleanup_receipt=stop(self.proc)

class OwnedBrowser:
 def __init__(self,directory):self.directory=directory;self.proc=None
 def start(self):
  d=self.directory;profile=d/'browser-profile';env=child_env(d,CONFIG['chromium']);log=(d/'chromium.log').open('ab')
  self.started=time.monotonic();self.proc=subprocess.Popen([sys.executable,str(ROOT/'sandbox.py'),'--browser',CONFIG['chromium'],'--headless=new','--no-sandbox','--disable-dev-shm-usage','--disable-http2','--remote-debugging-port=0','--remote-allow-origins=*','--user-data-dir='+str(profile),'about:blank'],env=env,stdout=log,stderr=log,start_new_session=True)
  record_owned(self.proc,d,'chromium')
  deadline=time.monotonic()+45;port=profile/'DevToolsActivePort'
  while True:
   if self.proc.poll() is not None or time.monotonic()>deadline:raise RuntimeError('Owned Chromium startup failed')
   lines=port.read_text().splitlines() if port.exists() else []
   if len(lines)==2 and lines[0].isdigit() and lines[1].startswith('/devtools/browser/'):
    self.ws='ws://127.0.0.1:'+lines[0]+lines[1];return self.ws
   time.sleep(.1)
 def close(self):
  cleanup=stop(self.proc);return {'pid':self.proc.pid if self.proc else None,'returncode':self.proc.returncode if self.proc else None,'owned_browser_seconds':time.monotonic()-self.started if hasattr(self,'started') else 0,'owned_group_cleanup':cleanup,'verified':cleanup['verified']}

def live_page(ws,navigate=None):
 from playwright.sync_api import sync_playwright
 manager=sync_playwright().start();browser=manager.chromium.connect_over_cdp(ws,timeout=30000);pages=[p for c in browser.contexts for p in c.pages];page=pages[-1];page.set_viewport_size({'width':1280,'height':1800})
 try:
  if navigate is not None:page.goto(navigate,wait_until='domcontentloaded',timeout=30000)
  target=page.context.new_cdp_session(page).send('Target.getTargetInfo')['targetInfo']['targetId']
  return target
 finally:browser.close();manager.stop()

def capture_cdp(ws,path):
 from playwright.sync_api import sync_playwright
 with sync_playwright() as p:
  browser=p.chromium.connect_over_cdp(ws,timeout=30000);page=[page for c in browser.contexts for page in c.pages][-1];session=page.context.new_cdp_session(page);raw=session.send('Page.captureScreenshot',{'format':'png','captureBeyondViewport':False,'fromSurface':True});Path(path).write_bytes(base64.b64decode(raw['data']));browser.close()

def verify_png(path):
 data=Path(path).read_bytes()
 if data[:8]!=b'\x89PNG\r\n\x1a\n' or struct.unpack('>II',data[16:24])!=(1280,1800):raise ValueError('Final evidence is not a raw1280x1800 PNG')
 from PIL import Image
 with Image.open(path) as image:
  if image.format!='PNG' or image.size!=(1280,1800):raise ValueError('Invalid final image geometry')
  image.load()

class ResourceSampler:
 def __init__(self):self.halt=threading.Event();self.cpu={};self.peak=0;self.samples=0;self.thread=threading.Thread(target=self.loop,daemon=True)
 def start(self):self.thread.start()
 def loop(self):
  ticks=os.sysconf('SC_CLK_TCK');page=os.sysconf('SC_PAGE_SIZE')
  while not self.halt.is_set():
   procs={}
   for p in Path('/proc').iterdir():
    if not p.name.isdigit():continue
    try:
     if p.stat().st_uid!=os.getuid():continue
     text=(p/'stat').read_text();fields=text[text.rfind(')')+2:].split();procs[int(p.name)]=(int(fields[1]),fields[19],(int(fields[11])+int(fields[12]))/ticks,int(fields[21])*page)
    except (FileNotFoundError,ProcessLookupError,PermissionError,IndexError):continue
   owned={os.getpid()};changed=True
   while changed:
    new={pid for pid,v in procs.items() if v[0] in owned};changed=not new.issubset(owned);owned.update(new)
   rss=0
   for pid in owned:
    if pid in procs:
     _,start,cpu,mem=procs[pid];key=(pid,start)
     if key not in self.cpu:self.cpu[key]=[cpu,cpu]
     else:self.cpu[key][1]=max(cpu,self.cpu[key][1])
     rss+=mem
   self.peak=max(self.peak,rss);self.samples+=1;self.halt.wait(.25)
 def close(self):
  self.halt.set();self.thread.join(timeout=5)
  return {'owned_process_cpu_seconds_observed':sum(v[1]-v[0] for v in self.cpu.values()),'owned_process_peak_rss_mib_sampled':self.peak/2**20,'sample_interval_seconds':.25,'samples':self.samples,'scope':'Sampled worker descendant process tree; CPU is an observed lower bound. Peak RSS sums sampled resident sets and may double-count shared pages.'}
