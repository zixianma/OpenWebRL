from pathlib import Path
import argparse,fcntl,hashlib,json,os,re,signal,subprocess,sys,time
ROOT=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text())
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def verify(shard,job=None):
 p=read(ROOT/'campaign-plan.json');science={k:v for k,v in p.items() if k!='plan_sha256'}
 if hashlib.sha256(json.dumps(science,sort_keys=True,separators=(',',':')).encode()).hexdigest()!=p['plan_sha256']:raise ValueError('Plan hash mismatch')
 for n,w in p['runtime_files'].items():
  if digest(ROOT/n)!=w:raise ValueError('Pinned runtime file mismatch: '+n)
 for n,w in p['vendor_files'].items():
  if digest(ROOT/n)!=w:raise ValueError('Pinned BrowserCode mismatch: '+n)
 for n,w in p['javascript_runtime_files'].items():
  if digest(ROOT/n)!=w:raise ValueError('Pinned JS runtime mismatch: '+n)
 upstream=Path(json.loads((ROOT/'runtime-config.json').read_text())['cuawright_source'])
 for n,w in p['cuawright_source_files'].items():
  if digest(upstream/n)!=w:raise ValueError('Pinned CUAWright source mismatch: '+n)
 a=read(ROOT/f'shard{shard}/approval.json');sp=read(ROOT/f'shard{shard}/plan.json')
 if sp['plan_sha256']!=p['plan_sha256'] or a['plan_sha256']!=p['plan_sha256'] or not a['approved'] or a['resources']!=p['resources_per_shard'] or a['limits']!=p['limits_per_shard']:raise ValueError('Exact approved shard binding absent')
 if job:
  if len([x for x in a['attempts'] if x['job_id']==job])!=1:raise ValueError('Unregistered allocation')
  prior_seconds=0
  for old_attempt in a['attempts']:
   if old_attempt['job_id']==job:continue
   acct=subprocess.check_output(['sacct','-X','-n','-P','-j',old_attempt['job_id'],'--format=JobIDRaw,State,ElapsedRaw'],text=True)
   rows=[line.split('|') for line in acct.splitlines() if line.split('|')[0]==old_attempt['job_id']]
   if len(rows)!=1 or rows[0][1].split()[0] not in ('COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL'):raise ValueError('Prior attempt accounting unresolved')
   prior_seconds+=int(rows[0][2])
  raw=subprocess.check_output(['scontrol','show','job','-o',job],text=True);f=dict(re.findall(r'(\w+)=(\S+)',raw))
  if f.get('JobState')!='RUNNING' or f.get('Account')!='vlm' or f.get('NumCPUs')!='4' or f.get('NumNodes')!='1' or f.get('UserId','').split('(')[-1].rstrip(')')!=str(os.getuid()):raise ValueError('Scheduler allocation mismatch')
  tres=dict(x.split('=',1) for x in f.get('AllocTRES','').split(',') if '=' in x)
  if tres.get('mem') not in ('16G','16384M') or any(k.startswith('gres/gpu') and float(v)!=0 for k,v in tres.items()):raise ValueError('Memory/GPU allocation mismatch')
  tl=f.get('TimeLimit','00:00:00');parts=list(map(int,tl.split(':')));seconds=sum(v*60**i for i,v in enumerate(reversed(parts)))
  if seconds>7200-prior_seconds:raise ValueError('All-attempt time cap exceeded')
  if not re.search(r'(?:^|/)job[_-]'+re.escape(job)+r'(?:/|$)',Path('/proc/self/cgroup').read_text()):raise ValueError('Controller outside owned job cgroup')
 return p,a
def has_verified_preflight(shard, harness, approval):
 for attempt in reversed(approval['attempts']):
  directory=ROOT/f'shard{shard}/startup-preflights/{attempt["job_id"]}'
  try:
   receipt=read(directory/'receipt.json')
   setup=read(directory/f'{harness}-setup.json')
   step=read(directory/f'{harness}-step.json')
  except (OSError,ValueError,KeyError):
   continue
  setup_ok=setup.get('observation',{}).get('success') is True
  step_ok=step.get('observation',{}).get('success') is True and 'native-startup-ready' in step.get('output','')
  cleanup=receipt.get('browser_cleanup',{})
  if receipt.get('browser_sessions')==1 and cleanup.get('returncode')==0 and setup_ok and step_ok:
   return attempt['job_id']
 return None

def main():
 q=argparse.ArgumentParser();q.add_argument('--shard',type=int,required=True);q.add_argument('--execute');z=q.parse_args();p,a=verify(z.shard,z.execute)
 if not z.execute:print(json.dumps({'valid':True,'shard':z.shard,'episodes':30,'cell':read(ROOT/f'shard{z.shard}/plan.json')['cell']}));return
 root=ROOT/f'shard{z.shard}';children=[];started=time.monotonic();os.environ.update(PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',WANDB_PROJECT='openwebrl-evals',TMPDIR='/tmp',PYTHONPATH=os.pathsep.join([str(ROOT),str(ROOT/'upstream/src'),p['support_source']]))
 for k in ['XDG_CACHE_HOME','XDG_CONFIG_HOME','XDG_DATA_HOME','WANDB_DIR','WANDB_CONFIG_DIR','WANDB_CACHE_DIR','WANDB_DATA_DIR','HF_HOME','TORCH_HOME','MPLCONFIGDIR']:
  v=root/'runtime-cache'/k.lower();v.mkdir(parents=True,exist_ok=True);os.environ[k]=str(v)
 from worker import preflight
 from environments import stop
 from openwebrl.luna_qwen_policy import Budget
 from openwebrl.luna_qwen_metrics import write_json
 signal.signal(signal.SIGTERM,lambda *unused:(_ for _ in ()).throw(SystemExit(143)))
 with (root/'owner.lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  try:
   verified=has_verified_preflight(z.shard,read(root/'plan.json')['cell']['harness'],a)
   if not verified:raise RuntimeError('No verified matching native preflight in preserved attempts; fail closed without consuming another browser session')
   write_json(root/'current.json',{'stage':'collecting','job_id':z.execute,'preflight_reused_from_job':verified,'epoch':time.time()})
   for slot in range(2):
    log=(root/f'worker{slot}.log').open('ab');children.append(subprocess.Popen([sys.executable,str(ROOT/'worker.py'),'--shard',str(z.shard),'--slot',str(slot)],stdout=log,stderr=log,start_new_session=True))
   deadline=started+7200-180
   while any(c.poll() is None for c in children):
    if any(c.poll() not in [None,0] for c in children) or time.monotonic()>deadline:raise RuntimeError('Worker failure/deadline; preserve attempts, no automatic retry')
    write_json(root/'current.json',{'stage':'collecting','job_id':z.execute,'recorded':len(list((root/'episodes').rglob('result.json'))),'epoch':time.time()});time.sleep(8)
   if any(c.returncode for c in children):raise RuntimeError('Worker exit nonzero')
   write_json(root/'current.json',{'stage':'worker_collection_finished_pending_independent_audit','job_id':z.execute,'complete':False,'epoch':time.time()})
  finally:
   for c in children:stop(c)
   write_json(root/f'allocation-{z.execute}-worker-exits.json',[{'pid':c.pid,'returncode':c.returncode} for c in children])
if __name__=='__main__':main()
