"""One approved Slurm allocation owns preflight and both pilot workers."""
from bootstrap import ROOT,CONFIG
from pathlib import Path
import argparse,fcntl,hashlib,json,os,re,signal,subprocess,sys,time
from preparation_gate import validate

def read(p):return json.loads(p.read_text())
def verify(job):
 validate(require_approval=True);p=read(ROOT/'campaign-plan.json');a=read(ROOT/'approval.json');
 if a.get('resources')!=p['resources'] or a.get('limits')!=p['limits']: raise ValueError('approval differs from campaign resource/API limits')
 if not job or sum(x['job_id']==job for x in a['attempts'])!=1:raise ValueError('job is not registered to approved attempt')
 consumed=0
 for prior in a['attempts']:
  if prior['job_id']==job: continue
  raw_prior=subprocess.check_output(['sacct','-X','-n','-P','-j',prior['job_id'],'--format=JobIDRaw,State,ElapsedRaw'],text=True)
  rows=[x.split('|') for x in raw_prior.splitlines() if x.split('|')[0]==prior['job_id']]
  if len(rows)!=1 or rows[0][1] not in ['COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL']: raise ValueError('Prior attempt has unresolved scheduler accounting')
  elapsed=int(rows[0][2]);
  if elapsed!=prior.get('accounted_elapsed_seconds'): raise ValueError('Prior attempt elapsed accounting mismatch')
  consumed+=elapsed
 raw=subprocess.check_output(['scontrol','show','job','-o',job],text=True);f=dict(re.findall(r'(\w+)=(\S+)',raw));t=dict(v.split('=',1) for v in f['AllocTRES'].split(','))
 if f.get('JobState')!='RUNNING' or f.get('NumCPUs')!='16' or t.get('mem')!='64G' or f.get('Account')!='vlm' or any(k.startswith('gres/gpu') and float(v) for k,v in t.items()):raise ValueError('live scheduler resources differ from approved allocation')
 if not f.get('UserId','').endswith(f'({os.getuid()})'):raise ValueError('Slurm owner mismatch')
 lim=f['TimeLimit'];days,clock=lim.split('-') if '-' in lim else ('0',lim);secs=int(days)*86400+sum(int(v)*60**i for i,v in enumerate(reversed(clock.split(':'))))
 if consumed+secs>p['resources']['time_limit_seconds']:raise ValueError('Remaining all-attempt allocation time cap exceeded')
 return p,a

def env(root):
 e=os.environ.copy();e.update(PYTHONPATH=os.pathsep.join([str(ROOT),CONFIG['cuawright_source'],CONFIG['frozen_support_source']]),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='',WANDB_PROJECT='openwebrl-evals',TMPDIR='/tmp',DO_NOT_TRACK='1',PATH=str(Path(CONFIG['python']).parent)+os.pathsep+e.get('PATH','/usr/bin:/bin'))
 for k in ['XDG_CACHE_HOME','XDG_CONFIG_HOME','XDG_DATA_HOME','WANDB_DIR','WANDB_CONFIG_DIR','WANDB_CACHE_DIR','WANDB_DATA_DIR','HF_HOME','TORCH_HOME','MPLCONFIGDIR']:
  q=root/'runtime-cache'/k.lower();q.mkdir(parents=True,exist_ok=True);e[k]=str(q)
 return e

def main():
 q=argparse.ArgumentParser();q.add_argument('--execute',required=True);a=q.parse_args();plan,approval=verify(a.execute);root=ROOT/'shard0';started=time.monotonic();children=[]
 from worker import preflight
 from environments import stop
 from openwebrl.luna_qwen_policy import Budget
 from openwebrl.luna_qwen_metrics import write_json
 signal.signal(signal.SIGTERM,lambda *unused:(_ for _ in ()).throw(SystemExit(143)))
 with (root/'owner.lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  try:
   write_json(root/'current.json',{'epoch':time.time(),'stage':'native_browsercode_preflight','job_id':a.execute,'complete':False})
   preflight(root,Budget(root/'budget/browser',plan['limits']['browser_sessions_all_attempts'],plan['limits']['browser_sessions_all_attempts']))
   for slot in range(2):
    log=(root/f'worker{slot}.log').open('ab');children.append(subprocess.Popen([sys.executable,str(ROOT/'worker.py'),'--shard','0','--slot',str(slot)],env=env(root),stdout=log,stderr=log,start_new_session=True))
   deadline=started+plan['resources']['time_limit_seconds']-180
   while any(c.poll() is None for c in children):
    if any(c.poll() not in [None,0] for c in children):raise RuntimeError('Worker failed; preserve attempts and diagnose before any rerun')
    if time.monotonic()>deadline:raise TimeoutError('Approaching approved allocation deadline')
    write_json(root/'current.json',{'epoch':time.time(),'stage':'collecting','job_id':a.execute,'episodes_recorded':len(list((root/'episodes').glob('*/result.json'))),'complete':False});time.sleep(10)
   if any(c.returncode for c in children):raise RuntimeError('Worker failed')
   write_json(root/'current.json',{'epoch':time.time(),'stage':'worker_collection_finished_pending_independent_audit','job_id':a.execute,'episodes_recorded':len(list((root/'episodes').glob('*/result.json'))),'complete':False})
  finally:
   for c in children:stop(c)
   write_json(root/f'allocation-{a.execute}-worker-exits.json',[{'pid':c.pid,'returncode':c.returncode} for c in children])
if __name__=='__main__':main()
