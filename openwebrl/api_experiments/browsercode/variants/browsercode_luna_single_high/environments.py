"""Owned local browser and native executor adapters; no model calls here."""
from bootstrap import ROOT,CONFIG
from pathlib import Path
import base64,json,os,re,selectors,signal,stat,struct,subprocess,sys,time,threading
from cuawright.webwright.environments.local_workspace import LocalWorkspaceEnvironment

def stop(proc):
 if proc is not None and proc.poll() is None:
  os.killpg(proc.pid,signal.SIGTERM)
  try:proc.wait(timeout=10)
  except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()

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

class BrowserCodeWorkspace(StableWorkspace):
 def start(self,ws,target):
  self.counter=0;workspace=self._workspace_dir();bun=ROOT/'javascript-runtime/bun';log=(workspace/'browsercode-stderr.log').open('ab')
  self.proc=subprocess.Popen([sys.executable,str(ROOT/'sandbox.py'),'--browser',str(bun),str(ROOT/'browser_bridge.ts')],cwd=workspace,env=child_env(workspace,bun),stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,bufsize=1,start_new_session=True)
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
  reply=self.call({'op':'execute','args':args});result=reply.get('result',{});output=result.get('output','') if reply['ok'] else reply.get('error','');log=self._write_step_log(output)
  obs=self._capture_observation(command=command,cwd=self._workspace_dir(),output=output,returncode=0 if reply['ok'] else 1,exception_info='' if reply['ok'] else output,log_path=log)
  obs['image_attachments']=result.get('image_attachments',[])
  return {'output':output,'returncode':obs['returncode'],'exception_info':obs['exception'],'observation':obs}
 def capture(self,path):
  reply=self.call({'op':'capture'},30)
  if not reply['ok']:raise RuntimeError('Native BrowserCode final capture failed: '+reply.get('error',''))
  Path(path).write_bytes(base64.b64decode(reply['result']['data']))
 def close(self):
  if not hasattr(self,'proc'):return
  try:self.call({'op':'close'},10)
  except Exception:pass
  finally:stop(self.proc)

class OwnedBrowser:
 def __init__(self,directory):self.directory=directory;self.proc=None
 def start(self):
  d=self.directory;profile=d/'browser-profile';env=child_env(d,CONFIG['chromium']);log=(d/'chromium.log').open('ab')
  self.started=time.monotonic();self.proc=subprocess.Popen([sys.executable,str(ROOT/'sandbox.py'),'--browser',CONFIG['chromium'],'--headless=new','--no-sandbox','--disable-dev-shm-usage','--disable-http2','--remote-debugging-port=0','--remote-allow-origins=*','--user-data-dir='+str(profile),'about:blank'],env=env,stdout=log,stderr=log,start_new_session=True)
  deadline=time.monotonic()+45;port=profile/'DevToolsActivePort'
  while True:
   if self.proc.poll() is not None or time.monotonic()>deadline:raise RuntimeError('Owned Chromium startup failed')
   lines=port.read_text().splitlines() if port.exists() else []
   if len(lines)==2 and lines[0].isdigit() and lines[1].startswith('/devtools/browser/'):
    self.ws='ws://127.0.0.1:'+lines[0]+lines[1];return self.ws
   time.sleep(.1)
 def close(self):
  stop(self.proc);return {'pid':self.proc.pid if self.proc else None,'returncode':self.proc.returncode if self.proc else None,'owned_browser_seconds':time.monotonic()-self.started if hasattr(self,'started') else 0}

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
