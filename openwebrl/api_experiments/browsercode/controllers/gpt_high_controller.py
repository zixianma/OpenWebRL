from bootstrap import ROOT,CONFIG
from pathlib import Path
import argparse,fcntl,hashlib,json,os,re,signal,subprocess,sys,time
from preparation_gate import validate
def read(p): return json.loads(p.read_text())
def seconds(limit):
 days,clock=limit.split('-') if '-' in limit else ('0',limit)
 return int(days)*86400+sum(int(v)*60**i for i,v in enumerate(reversed(clock.split(':'))))
def verify(shard,job=None):
 validate(require_approval=True); plan=read(ROOT/'campaign-plan.json'); r=ROOT/f'shard{shard}'; sp=read(r/'plan.json'); a=read(r/'approval.json')
 if sp!={**plan['shards'][shard],'plan_sha256':plan['plan_sha256']}: raise ValueError('Shard manifest mismatch')
 if job is None: return plan,a
 active=[x for x in a['attempts'] if x['job_id']==job]
 if len(active)!=1: raise ValueError('Job is not registered exactly once')
 consumed=0
 for prior in a['attempts']:
  if prior['job_id']==job: continue
  raw=subprocess.check_output(['sacct','-X','-n','-P','-j',prior['job_id'],'--format=JobIDRaw,State,ElapsedRaw'],text=True)
  rows=[x.split('|') for x in raw.splitlines() if x.split('|')[0]==prior['job_id']]
  if len(rows)!=1 or not rows[0][1].startswith(('COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL')): raise ValueError('Prior scheduler accounting unresolved')
  if int(rows[0][2])!=prior.get('accounted_elapsed_seconds'): raise ValueError('Prior time mismatch')
  consumed+=int(rows[0][2])
 raw=subprocess.check_output(['scontrol','show','job','-o',job],text=True); f=dict(re.findall(r'(\w+)=(\S+)',raw)); tres=dict(x.split('=',1) for x in f['AllocTRES'].split(','))
 if f.get('JobState')!='RUNNING' or f.get('NumCPUs')!='4' or tres.get('mem')!='16G' or f.get('Account')!='vlm' or any(k.startswith('gres/gpu') and float(v) for k,v in tres.items()): raise ValueError('Live scheduler resources differ from approval')
 if not f.get('UserId','').endswith(f'({os.getuid()})'): raise ValueError('Slurm owner mismatch')
 if consumed+seconds(f['TimeLimit'])>a['resources']['total_seconds_all_attempts']: raise ValueError('All-attempt time cap exceeded')
 return plan,a
def env(root):
 e=os.environ.copy(); e.update(PYTHONPATH=os.pathsep.join([str(ROOT),CONFIG['cuawright_source'],CONFIG['frozen_support_source']]),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='',WANDB_PROJECT='openwebrl-evals',TMPDIR='/tmp',DO_NOT_TRACK='1',PATH=str(Path(CONFIG['python']).parent)+os.pathsep+e.get('PATH','/usr/bin:/bin'))
 for k in ['XDG_CACHE_HOME','XDG_CONFIG_HOME','XDG_DATA_HOME','WANDB_DIR','WANDB_CONFIG_DIR','WANDB_CACHE_DIR','WANDB_DATA_DIR','HF_HOME','TORCH_HOME','MPLCONFIGDIR']:
  q=root/'runtime-cache'/k.lower();q.mkdir(parents=True,exist_ok=True);e[k]=str(q)
 return e
def main():
 q=argparse.ArgumentParser();q.add_argument('--shard',type=int,required=True);q.add_argument('--execute',required=True);a=q.parse_args();plan,approval=verify(a.shard,a.execute); root=ROOT/f'shard{a.shard}'; started=time.monotonic();children=[];os.environ.update(env(root))
 from worker import preflight
 from environments import stop
 from openwebrl.luna_qwen_policy import Budget
 from openwebrl.luna_qwen_metrics import write_json
 signal.signal(signal.SIGTERM,lambda *unused:(_ for _ in ()).throw(SystemExit(143)))
 with (root/'owner.lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  try:
   write_json(root/'current.json',{'epoch':time.time(),'stage':'native_browsercode_preflight','job_id':a.execute,'shard':a.shard,'complete':False})
   limits=read(root/'plan.json')['limits'];preflight(root,Budget(root/'budget/browser',limits['browser_sessions_all_attempts'],limits['browser_sessions_all_attempts']))
   for slot in range(2):
    log=(root/f'worker{slot}.log').open('ab'); children.append(subprocess.Popen([sys.executable,str(ROOT/'worker.py'),'--shard',str(a.shard),'--slot',str(slot)],env=env(root),stdout=log,stderr=log,start_new_session=True))
   deadline=started+next(x['time_limit_seconds'] for x in approval['attempts'] if x['job_id']==a.execute)-180
   while any(c.poll() is None for c in children):
    if any(c.poll() not in [None,0] for c in children): raise RuntimeError('Worker failed; preserve attempts and diagnose')
    if time.monotonic()>deadline: raise TimeoutError('Approaching all-attempt shard deadline')
    n=len(list((root/'episodes').glob('*/result.json')));write_json(root/'current.json',{'epoch':time.time(),'stage':'collecting','job_id':a.execute,'shard':a.shard,'episodes_recorded':n,'complete':False});time.sleep(5)
   if any(c.returncode for c in children): raise RuntimeError('Worker failed')
   write_json(root/'current.json',{'epoch':time.time(),'stage':'worker_collection_finished_pending_independent_audit','job_id':a.execute,'shard':a.shard,'episodes_recorded':len(list((root/'episodes').glob('*/result.json'))),'complete':False})
  except BaseException as e:
   write_json(root/'current.json',{'epoch':time.time(),'stage':'worker_or_preflight_failed','job_id':a.execute,'shard':a.shard,'error_type':type(e).__name__,'error':str(e),'complete':False});raise
  finally:
   for c in children: stop(c)
   write_json(root/f'allocation-{a.execute}-worker-exits.json',[{'pid':c.pid,'returncode':c.returncode} for c in children])
if __name__=='__main__': main()
