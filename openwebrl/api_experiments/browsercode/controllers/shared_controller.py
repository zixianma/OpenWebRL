from bootstrap import ROOT,CONFIG
from pathlib import Path
import argparse,json,os,subprocess,time,re,hashlib,signal,fcntl

def read(p):return json.loads(Path(p).read_text())
def write(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp-'+str(os.getpid()));q.write_text(json.dumps(d,indent=2)+'\n');q.replace(p)
def sha(p):
    digest=hashlib.sha256()
    with Path(p).open("rb") as stream:
        while block:=stream.read(4*1024*1024):digest.update(block)
    return digest.hexdigest()
def validate_sources():
 m=read(ROOT/'source-manifest.json');assert m['plan_sha256']==sha(ROOT/'campaign-plan.json')
 for p,h in m['source_hashes'].items():assert sha(p)==h,'Source changed: '+p
 assert sha(ROOT/'tasks.json')==read(ROOT/'campaign-plan.json')['task_manifest_sha256']
 return m

def verify(arm,shard,job_id):
 validate_sources();reg=read(ROOT/'orchestration/registry.json');entries=reg.get('attempts',reg.get('jobs',[]));matches=[x for x in entries if str(x['job_id'])==str(job_id)and x['arm']==arm and int(x['shard'])==shard];assert len(matches)==1,'Unregistered worker'
 e=matches[0]
 from financial_plan import verify_enrollment
 verify_enrollment(e,ROOT)
 assert e['plan_sha256']==sha(ROOT/'campaign-plan.json')and e['source_manifest_sha256']==sha(ROOT/'source-manifest.json'),'Enrollment source/plan changed'
 assert type(e['shard'])is int and type(e['prior_elapsed_seconds'])is int and type(e['requested_seconds'])is int,'Typed allocation fields required'
 prior=[x for x in entries if x['arm']==arm and x['shard']==shard and x is not e]
 from orchestration.launch_campaign import scheduler,account_prior
 consumed=account_prior(prior,scheduler([x['job_id']for x in prior]));assert consumed==e['prior_elapsed_seconds'],'Prior elapsed accounting changed'
 raw=subprocess.check_output(['scontrol','show','job','-o',str(job_id)],text=True);fields=dict(re.findall(r'(\w+)=(\S+)',raw));tres=dict(x.split('=',1)for x in fields['AllocTRES'].split(','));assert fields['JobState']=='RUNNING'and fields['NumCPUs']=='4'and tres.get('mem')=='16G'and fields['Account']=='vlm'and fields['UserId'].endswith('('+str(os.getuid())+')');assert not any(k.startswith('gres/gpu')and float(v)for k,v in tres.items())
 t=fields['TimeLimit'];days,clock=t.split('-')if '-'in t else('0',t);seconds=int(days)*86400+sum(int(v)*60**i for i,v in enumerate(reversed(clock.split(':'))));assert seconds==e['requested_seconds']and seconds+e['prior_elapsed_seconds']<=14400,'Live/all-attempt allocation cap mismatch'
 return e

def env(root):
 e=os.environ.copy();e.update(PYTHONPATH=os.pathsep.join([str(ROOT),CONFIG['cuawright_source'],CONFIG['frozen_support_source']]),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='',WANDB_PROJECT='openwebrl-evals',TMPDIR='/tmp',DO_NOT_TRACK='1',PATH=str(Path(CONFIG['python']).parent)+os.pathsep+e.get('PATH','/usr/bin:/bin'))
 for k in ['XDG_CACHE_HOME','XDG_CONFIG_HOME','XDG_DATA_HOME','XDG_STATE_HOME','WANDB_DIR','WANDB_CONFIG_DIR','WANDB_CACHE_DIR','WANDB_DATA_DIR','HF_HOME','TORCH_HOME','MPLCONFIGDIR','BUN_RUNTIME_TRANSPILER_CACHE_PATH','BUN_INSTALL_CACHE_DIR','PIP_CACHE_DIR']:
  p=root/'runtime-cache'/k.lower();p.mkdir(parents=True,exist_ok=True);e[k]=str(p)
 return e

def canary_gate(arm):
 p=ROOT/'orchestration/gates'/f'{arm}-canary.json'
 if not p.exists():return False
 from orchestration.launch_campaign import gate
 gate(arm,'canary');return True

def wait_canary_gate(arm,root):
 while not canary_gate(arm):
  write(root/'current.json',{'epoch':time.time(),'stage':'awaiting_independent_two_canary_audit','arm':arm});time.sleep(5)

def stop_children(children,root,stopper):
 rows=[];errors=[]
 for child in children:
  try:
   cleanup=stopper(child,grace_seconds=90);rows.append({'pid':child.pid,'returncode':child.returncode,'wait_complete':child.poll()is not None,'cleanup':cleanup})
  except BaseException as exc:
   errors.append({'pid':child.pid,'error_type':type(exc).__name__,'error':str(exc)})
 write(root/'owned-worker-teardown.json',{'epoch':time.time(),'workers':rows,'cleanup_errors':errors,'verified_complete':not errors and len(rows)==len(children)})
 if errors:raise RuntimeError('Unresolved owned worker cleanup; all children attempted')

def main():
 p=argparse.ArgumentParser();p.add_argument('--arm',required=True);p.add_argument('--shard',required=True,type=int);p.add_argument('--job-id',required=True);a=p.parse_args();entry=verify(a.arm,a.shard,a.job_id);root=Path(entry['attempt_dir']);root.mkdir(parents=True,exist_ok=True);os.environ.update(env(root));children=[];started=time.monotonic();logical=root.parent.parent
 from financial_plan import verify_enrollment
 sp=verify_enrollment(entry,ROOT)
 from worker import preflight
 from environments import stop,record_owned,cleanup_registered_groups
 from openwebrl.luna_qwen_policy import Budget
 signal.signal(signal.SIGTERM,lambda *unused:(_ for _ in ()).throw(SystemExit(143)))
 with (logical/'allocation-owner.lock').open('a+')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  try:
   write(root/'current.json',{'epoch':time.time(),'stage':'mechanical_browser_preflight','job_id':a.job_id,'arm':a.arm,'shard':a.shard})
   preflight(root,Budget(logical/'budget/browser',sp['browser_starts_all_attempts'],sp['browser_starts_all_attempts']))
   for slot in range(2):
    log=(root/f'worker{slot}.log').open('ab');children.append(subprocess.Popen([CONFIG['python'],'-B',str(ROOT/'worker.py'),'--arm',a.arm,'--shard',str(a.shard),'--slot',str(slot),'--attempt-dir',str(root)],env=env(root),stdout=log,stderr=log,start_new_session=True));record_owned(children[-1],root,f'worker{slot}')
   deadline=started+int(entry.get('requested_seconds',14400))-180
   while any(c.poll()is None for c in children):
    if any(c.poll()not in(None,0)for c in children):raise RuntimeError('Worker failed; diagnose before retry')
    if time.monotonic()>deadline:raise TimeoutError('Allocation remaining time exhausted')
    n=len(list((root/'episodes').glob('*/result.json')));write(root/'heartbeat.json',{'epoch':time.time(),'stage':'running','outcomes':n,'arm':a.arm,'shard':a.shard,'job_id':a.job_id});time.sleep(5)
   assert all(c.returncode==0 for c in children)
   write(root/'current.json',{'epoch':time.time(),'stage':'collection_finished_pending_independent_audit','outcomes':len(list((root/'episodes').glob('*/result.json'))),'arm':a.arm,'shard':a.shard,'job_id':a.job_id})
  except BaseException as exc:
   write(root/'current.json',{'epoch':time.time(),'stage':'failed_requires_diagnosis','error_type':type(exc).__name__,'error':str(exc),'job_id':a.job_id});raise
  finally:
   try:stop_children(children,root,stop)
   finally:cleanup_registered_groups(root)
if __name__=='__main__':main()
