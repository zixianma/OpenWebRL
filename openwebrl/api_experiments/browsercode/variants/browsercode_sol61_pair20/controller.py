"""Registered shard allocation owns preflight and both paired episode workers."""
from bootstrap import ROOT,CONFIG
from pathlib import Path
import argparse,fcntl,hashlib,json,os,re,signal,subprocess,sys,time
from preparation_gate import validate

def read(path):return json.loads(path.read_text())
def verify(shard,job=None):
 validate(require_approval=True);plan=read(ROOT/'campaign-plan.json');root=ROOT/f'shard{shard}';sp=read(root/'plan.json');approval=read(root/'approval.json')
 if sp!={**plan['shards'][shard],'plan_sha256':plan['plan_sha256']} or not approval['approved'] or approval['plan_sha256']!=plan['plan_sha256'] or approval['resources']!=plan['resources_per_shard'] or approval['limits']!=plan['limits_per_shard']:raise ValueError('Wrong exact shard approval/manifest')
 manifest=ROOT/'execution-source-manifest.json'
 if hashlib.sha256(manifest.read_bytes()).hexdigest()!=approval.get('execution_source_manifest_sha256'):raise ValueError('Execution source is not bound to registered approval')
 for name,want in read(manifest)['files'].items():
  if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=want:raise ValueError('Execution source changed: '+name)
 if not os.access(ROOT/'sandbox.py',os.X_OK):raise ValueError('Sandbox executable preflight failed')
 for name,want in CONFIG['support_source_hashes'].items():
  if hashlib.sha256((Path(CONFIG['frozen_support_source'])/name).read_bytes()).hexdigest()!=want:raise ValueError('Metering support changed')
 # Pin the original native CUAWright runtime; the controlled adapters are separate files.
 old=read(ROOT.parent/'cuawright30-preparation-v1/campaign-plan.json')
 for name,want in old['files'].items():
  if name.startswith('upstream/') and hashlib.sha256((ROOT.parent/'cuawright30-preparation-v1'/name).read_bytes()).hexdigest()!=want:raise ValueError('Native CUAWright source changed')
 if job is not None:
  if not job:raise ValueError('Registered Slurm allocation required')
  active=[a for a in approval['attempts'] if a['job_id']==job]
  if len(active)!=1:raise ValueError('Unregistered allocation')
  prior=0
  for a in approval['attempts']:
   if a['job_id']==job:continue
   raw=subprocess.check_output(['sacct','-X','-n','-P','-j',a['job_id'],'--format=JobIDRaw,State,ElapsedRaw'],text=True);rows=[s.split('|') for s in raw.splitlines() if s.split('|')[0]==a['job_id']]
   if len(rows)!=1 or rows[0][1] not in ['COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL']:raise ValueError('Unresolved prior accounting')
   prior+=int(rows[0][2])
  if prior+active[0]['time_limit_seconds']>7200:raise ValueError('All-attempt time cap exceeded')
  raw=subprocess.check_output(['scontrol','show','job','-o',job],text=True);fields=dict(re.findall(r'(\w+)=(\S+)',raw));tres=dict(v.split('=',1) for v in fields['AllocTRES'].split(','))
  if fields['JobState']!='RUNNING' or fields['NumCPUs']!='4' or tres.get('mem')!='16G' or any(k.startswith('gres/gpu') and float(v) for k,v in tres.items()):raise ValueError('Wrong allocated resources')
  clock=fields['TimeLimit'];days,clock=clock.split('-') if '-' in clock else ('0',clock);seconds=int(days)*86400+sum(int(v)*60**i for i,v in enumerate(reversed(clock.split(':'))))
  if seconds>active[0]['time_limit_seconds'] or fields.get('Account')!='vlm':raise ValueError('Wrong scheduler time/account')
  if not fields['UserId'].endswith(f'({os.getuid()})'):raise ValueError('Wrong job owner')
  cg=Path('/proc/self/cgroup').read_text()
  if not re.search(r'(?:^|/)job[_-]'+re.escape(job)+r'(?:/|$)',cg):raise ValueError('Controller/worker outside owned allocation')
 return plan,approval

def environment(root):
 env=os.environ.copy();env.update(PYTHONPATH=os.pathsep.join([str(ROOT),CONFIG['cuawright_source'],CONFIG['frozen_support_source']]),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='',WANDB_PROJECT='openwebrl-evals',TMPDIR='/tmp',DO_NOT_TRACK='1',PATH=str(Path(CONFIG['python']).parent)+os.pathsep+env.get('PATH','/usr/bin:/bin'))
 for k in ['XDG_CACHE_HOME','XDG_CONFIG_HOME','XDG_DATA_HOME','WANDB_DIR','WANDB_CONFIG_DIR','WANDB_CACHE_DIR','WANDB_DATA_DIR','HF_HOME','TORCH_HOME','MPLCONFIGDIR']:
  p=root/'runtime-cache'/k.lower();p.mkdir(parents=True,exist_ok=True);env[k]=str(p)
 return env

def main():
 p=argparse.ArgumentParser();p.add_argument('--shard',type=int,required=True);p.add_argument('--execute');a=p.parse_args();plan,approval=verify(a.shard,a.execute)
 if a.execute is None:print(json.dumps({'valid':True,'approved':True,'shard':a.shard,'episodes':20}));return
 started=time.monotonic();root=ROOT/f'shard{a.shard}';os.environ.update(environment(root));children=[]
 from worker import preflight
 from environments import stop
 from openwebrl.luna_qwen_policy import Budget
 from openwebrl.luna_qwen_metrics import write_json
 signal.signal(signal.SIGTERM,lambda *unused:(_ for _ in ()).throw(SystemExit(143)))
 with (root/'owner.lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  try:
   write_json(root/'current.json',{'epoch':time.time(),'stage':'native_startup_preflight','job_id':a.execute});preflight(root,Budget(root/'budget/browser',30,30))
   for slot in range(2):
    log=(root/f'worker{slot}.log').open('ab');children.append(subprocess.Popen([sys.executable,str(ROOT/'worker.py'),'--shard',str(a.shard),'--slot',str(slot)],env=environment(root),stdout=log,stderr=log,start_new_session=True))
   deadline=started+next(x['time_limit_seconds'] for x in approval['attempts'] if x['job_id']==a.execute)-240
   while any(c.poll() is None for c in children):
    if any(c.poll() not in [None,0] for c in children) or time.monotonic()>deadline:raise RuntimeError('Worker/deadline failure; preserve all attempts and diagnose before retry')
    write_json(root/'current.json',{'epoch':time.time(),'stage':'collecting','job_id':a.execute,'recorded':len(list((root/'episodes').glob('*/result.json')))});time.sleep(10)
   if any(c.returncode for c in children):raise RuntimeError('Worker failed')
   write_json(root/'current.json',{'epoch':time.time(),'stage':'worker_collection_finished_pending_independent_audit','job_id':a.execute,'complete':False})
  finally:
   for c in children:stop(c)
   write_json(root/('allocation-'+a.execute+'-worker-exits.json'),[{'pid':c.pid,'returncode':c.returncode} for c in children])
if __name__=='__main__':main()
