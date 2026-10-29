"""Shard worker: two owned native episodes, independent grading, durable dispositions."""
import concurrent.futures,hashlib,json,os,signal,subprocess,sys,threading,time
from pathlib import Path
R=Path(__file__).resolve().parent
PYTHON='/gpfs/scrubbed/zixianma/openwebrl-runtime/venv/bin/python'

def read(p):return json.loads(p.read_text())
def write(p,d):
 q=p.with_name(p.name+'.'+str(threading.get_ident())+'.partial');q.write_text(json.dumps(d,indent=2)+'\n');q.replace(p)
def clean_env(output):
 env=dict(os.environ)
 for k in list(env):
  if any(x in k.upper() for x in ['API_KEY','TOKEN','SECRET','CREDENTIAL']):env.pop(k,None)
 env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(R/'upstream/src'),PATH=str(Path(PYTHON).parent)+':/usr/bin:/bin',NO_PROXY='127.0.0.1,localhost,::1',no_proxy='127.0.0.1,localhost,::1')
 for name in ['TMPDIR','XDG_CACHE_HOME','XDG_CONFIG_HOME','XDG_DATA_HOME','XDG_STATE_HOME','PIP_CACHE_DIR','UV_CACHE_DIR','WANDB_CACHE_DIR','WANDB_CONFIG_DIR','WANDB_DIR']:
  p=output/'.cache'/name.lower();p.mkdir(parents=True,exist_ok=True);env[name]=str(p)
 return env

def invoke(cmd,output,deadline,log):
 remaining=deadline-time.time()-30
 if remaining<=0:raise TimeoutError('Owned allocation time exhausted before subprocess')
 with log.open('ab') as f:
  p=subprocess.Popen(cmd,cwd=output,env=clean_env(output),stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  write(output/'active-process.json',{'pid':p.pid,'pgid':p.pid,'command_kind':Path(cmd[1]).stem,'started_epoch':time.time()})
  try:return p.wait(timeout=remaining)
  except subprocess.TimeoutExpired:
   os.killpg(p.pid,signal.SIGTERM)
   try:p.wait(timeout=15)
   except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait(timeout=10)
   raise TimeoutError('Owned allocation deadline reached')
  finally:
   write(output/'active-process.json',{'pid':p.pid,'exit_code':p.poll(),'finished_epoch':time.time()})
   if Path(cmd[1]).name in ['native_episode.py','diagnostic_browser.py']:
    from confinement import owned_browser_members,terminate_recorded_members
    receipt_name='startup-diagnostic.json' if Path(cmd[1]).name=='diagnostic_browser.py' else 'result.json'
    receipt=read(output/receipt_name) if (output/receipt_name).exists() else {}
    broker=(receipt.get('owned_teardown') or {}).get('broker_verification') or {}
    if broker.get('evidence'):
     evidence=Path(broker['evidence']).resolve();expected=(output.parent/'.broker-evidence').resolve()
     assert evidence.is_relative_to(expected) and evidence!=expected,'Broker evidence outside owned parent scope'
     original=evidence/'owned-browser-processes.jsonl'
     if original.exists():
      target=output/'owned-browser-processes.jsonl'
      assert not target.exists(),'Never replace earlier physical browser identity records'
      target.write_bytes(original.read_bytes())
     write(output/'broker-evidence-binding.json',{'evidence':str(evidence),'owned_records_sha256':hashlib.sha256(original.read_bytes()).hexdigest() if original.exists() else None,'broker_verified':broker.get('verified',False),'parent_collected_epoch':time.time()})
    cleanup=terminate_recorded_members(owned_browser_members(output))
    write(output/'parent-owned-teardown.json',cleanup)
    if not cleanup['verified']:raise RuntimeError('Parent owned browser teardown failed')

def episode_directory(shard):
 active_path=R/f'shard{shard}'/'active-attempt.json'
 if not active_path.exists():return R/f'shard{shard}'/'episodes'
 a=read(active_path);registration=Path(a['registration_path']);assert registration.resolve().is_relative_to((R/f'shard{shard}').resolve())
 assert a['registration_sha256']==hashlib.sha256(registration.read_bytes()).hexdigest()
 assert a['job_id']==os.environ['SLURM_JOB_ID'] and read(registration)['attempt']==a['attempt']
 out=Path(a['episode_root']).resolve();assert out.is_relative_to((R/f'shard{shard}').resolve());return out

def approved_browser_remaining(shard,usage):
 from budget_gateway import effective_browser_cap
 remaining=effective_browser_cap(R,shard)-usage['browser_starts']
 if remaining<=0:raise PermissionError('No approved all-attempt browser starts remain')
 return remaining

def startup_probe(shard,baseurl,deadline,usage):
 d=R/f'shard{shard}';active=read(d/'active-attempt.json') if (d/'active-attempt.json').exists() else {};recovery=Path(active['recovery_root']) if active.get('recovery_root') else d/'startup-check';recovery.mkdir(parents=True,exist_ok=True)
 localplan=read(R/'plan.json');ids=localplan['shards'][shard]['task_ids'];canary_here=[c for c in localplan['canary_task_ids'] if c in ids];case=(canary_here or ids)[0];probe=0;prior_signature=None
 while True:
  source_names=['diagnostic_browser.py','native_episode.py','confinement.py','confinement_support/sitecustomize.py','browser_broker.py','browser_broker_support.py','browser_launcher.py']
  signature=hashlib.sha256(''.join(hashlib.sha256((R/n).read_bytes()).hexdigest() for n in source_names).encode()).hexdigest()
  if probe:
   request=read(recovery/'diagnostic-retry.json') if (recovery/'diagnostic-retry.json').exists() else {}
   if not request.get('authorized_mechanical_retry') or request.get('source_signature')!=signature or signature==prior_signature:
    if time.time()>deadline-60:raise TimeoutError('Startup repair not provided before allocation deadline')
    write(d/'heartbeat.json',{'epoch':time.time(),'stage':'startup_diagnostic_failed_waiting_tested_repair','probe_count':probe,'usage':usage(R,shard)});time.sleep(5);continue
  output=recovery/f'browser-preflight-{probe}';output.mkdir(exist_ok=False);before=usage(R,shard);started=time.time()
  remaining=approved_browser_remaining(shard,before)
  code=invoke([PYTHON,str(R/'diagnostic_browser.py'),'--output',str(output),'--endpoint',baseurl+'/browser/'+case+'/reserve','--approved-browser-remaining',str(remaining)],output,deadline,output/'probe.log')
  finished=time.time();after=usage(R,shard);receipt=read(output/'startup-diagnostic.json');parent=read(output/'parent-owned-teardown.json')
  rows=[json.loads(x) for x in (output/'owned-browser-processes.jsonl').read_text().splitlines()] if (output/'owned-browser-processes.jsonl').exists() else []
  good=code==0 and receipt.get('actor_release_permitted') is True and receipt.get('startup_healthy') is True and receipt.get('offline_mock') is False and receipt.get('paid_model_calls')==0 and receipt.get('model_dispatches')==0 and receipt.get('owned_teardown',{}).get('process_verification',{}).get('verified') is True and parent.get('verified') is True
  assert before['model']['calls']==after['model']['calls'] and before['judge']['calls']==after['judge']['calls']
  gate={'verified':good,'diagnostic_path':str(output/'startup-diagnostic.json'),'diagnostic_sha256':hashlib.sha256((output/'startup-diagnostic.json').read_bytes()).hexdigest(),'browser_reservation_id':rows[0]['reservation']['id'] if len(rows)==1 else None,'case_id':case,'started_epoch':started,'finished_epoch':finished,'model_calls_before':before['model']['calls'],'model_calls_after':after['model']['calls'],'judge_calls_before':before['judge']['calls'],'judge_calls_after':after['judge']['calls'],'browser_starts_before':before['browser_starts'],'browser_starts_after':after['browser_starts'],'target_url':'about:blank','closed':parent.get('verified',False),'source_signature':signature,'probe_index':probe}
  write(recovery/f'browser-preflight-{probe}-gate.json',gate)
  if good:
   assert after['browser_starts']==before['browser_starts']+1 and len(rows)==1
   write(d/'startup-diagnostic-gate.json',gate)
   write(d/'operational-halt.json',{'mode':'cleared','startup_diagnostic_path':gate['diagnostic_path'],'sha256':gate['diagnostic_sha256']})
   return
  prior_signature=signature;probe+=1

def require_dispatch_allowed(shard):
 from budget_gateway import _halted
 if _halted(R,shard,'model'):
  raise RuntimeError('Operational halt blocks new native episode dispatch; preserve pending tasks for diagnosed recovery')

def execute_cases(batch,shard,baseurl,deadline,completed,on_progress,on_result):
 # Only two submitted futures: no executor backlog can drain after a shard halt.
 pending=iter(batch);exhausted=False;halt_error=None
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
  futures={}
  while futures or not exhausted:
   if halt_error is None:
    try:require_dispatch_allowed(shard)
    except RuntimeError as e:halt_error=e;exhausted=True
   while not exhausted and len(futures)<2:
    try:t=next(pending)
    except StopIteration:exhausted=True;break
    try:require_dispatch_allowed(shard)
    except RuntimeError as e:halt_error=e;exhausted=True;break
    futures[pool.submit(run_case,t,shard,baseurl,deadline)]=t
   if not futures:break
   done,_=concurrent.futures.wait(futures,timeout=10,return_when=concurrent.futures.FIRST_COMPLETED)
   on_progress(len(futures))
   for future in done:
    futures.pop(future)
    try:result=future.result()
    except RuntimeError as e:
     if 'Operational halt blocks' not in str(e):raise
     halt_error=e;exhausted=True;continue
    completed.append(result);on_result(result)
  if halt_error is not None:raise halt_error

def run_case(task,shard,baseurl,deadline):
 require_dispatch_allowed(shard)
 d=episode_directory(shard)/task['task_id'];d.mkdir(parents=True,exist_ok=False)
 result_path=d/'disposition.json'
 if result_path.exists():raise RuntimeError('Never overwrite an existing disposition or retry an unchanged episode')
 taskfile=d/'task-input.json';write(taskfile,task);started=time.time();native_code=None;judge_code=None;error=None
 arm=read(R/'plan.json')['arm_id'];native_cmd=[PYTHON,str(R/'native_episode.py'),'--task-json',str(taskfile),'--output',str(d),'--endpoint',baseurl+'/model/'+task['task_id']+'/responses','--arm',arm,'--absolute-deadline',str(deadline)]
 if read(R/'plan.json')['candidate_count']>1:native_cmd+=['--group-config',str(R/f'shard{shard}/.host'/('group-config-'+os.environ['SLURM_JOB_ID']+'.json'))]
 try:
  require_dispatch_allowed(shard)
  native_code=invoke(native_cmd,d,deadline,d/'native.log')
  actor_finished=time.time()
  if not (d/'trajectory.json').exists():raise RuntimeError('Native episode did not save trajectory')
  nr=read(d/'result.json')
  if not nr.get('owned_teardown',{}).get('process_verification',{}).get('verified'):raise RuntimeError('Owned native browser teardown unverified')
  from native_capture import collect_host_capture_inventory
  collect_host_capture_inventory(R,shard,d)
  judge_code=invoke([PYTHON,str(R/'external_judge.py'),'--task-json',str(taskfile),'--episode-dir',str(d),'--endpoint',baseurl+'/judge/'+task['task_id']+'/chat/completions','--output',str(d/'external-judge.json')],d,deadline,d/'judge.log')
 except Exception as e:
  error={'type':type(e).__name__,'message':str(e)};actor_finished=locals().get('actor_finished',time.time())
 native=read(d/'result.json') if (d/'result.json').exists() else {}
 judge=read(d/'external-judge.json') if (d/'external-judge.json').exists() else {}
 trajectory=read(d/'trajectory.json') if (d/'trajectory.json').exists() else {};info=trajectory.get('info',{})
 status=info.get('exit_status') or native.get('exit_status') or (native.get('native_result') or {}).get('exit_status')
 execution_valid=native_code==0 and status in ['Submitted','LimitsExceeded'] and not native.get('run_exception') and not native.get('error') and native.get('owned_teardown',{}).get('process_verification',{}).get('verified',False) and read(d/'parent-owned-teardown.json').get('verified',False)
 grading_valid=judge_code==0 and bool(judge.get('valid'))
 result={'task_id':task['task_id'],'shard':shard,'job_id':os.environ['SLURM_JOB_ID'],'native_exit_code':native_code,'judge_exit_code':judge_code,'native_exit_status':status,'execution_valid':execution_valid,'grading_valid':grading_valid,'valid':execution_valid and grading_valid,'official_label':judge.get('official_label'),'strict_label':judge.get('strict_label'),'reward':judge.get('strict_label') if execution_valid and grading_valid else None,'task_latency_seconds':actor_finished-started,'external_judge_seconds':time.time()-actor_finished,'decisions':info.get('api_calls',native.get('decisions')),'error':error,'finished_epoch':time.time()}
 write(result_path,result);return result

def release_verified():
 path=R/'release-all.json'
 if not path.exists():return False
 d=read(path);a=R/'canary-independent-audit.json'
 return d.get('approved_release') is True and a.exists() and d.get('canary_independent_audit_sha256')==hashlib.sha256(a.read_bytes()).hexdigest() and read(a).get('passed') is True

def run_shard(shard,deadline):
 from dotenv import dotenv_values
 from budget_gateway import start_gateway,totals
 import secrets as private_secrets
 plan=read(R/'plan.json');cap=plan['shards'][shard];arm=plan['arm_id']
 credentials=dotenv_values('/gpfs/projects/krishna/zixianma/OpenWebRL/.env');key=credentials.get('OPENAI_API_KEY');assert key
 d=R/f'shard{shard}';d.mkdir(exist_ok=True)
 for name in ['TMPDIR','XDG_CACHE_HOME','XDG_CONFIG_HOME','XDG_DATA_HOME','XDG_STATE_HOME','WANDB_CACHE_DIR','WANDB_CONFIG_DIR','WANDB_DIR']:
  cache=d/'.cache'/name.lower();cache.mkdir(parents=True,exist_ok=True);os.environ[name]=str(cache)
 os.environ['NETRC']=str(d/'.cache/netrc');os.environ['PYTHONDONTWRITEBYTECODE']='1';os.environ['WANDB_API_KEY']=credentials['WANDB_API_KEY']
 import wandb
 identity=plan.get('wandb_run_prefix','cua-native-three-arm-20261023-')+arm+'-s'+str(shard)
 run=wandb.init(entity='zixianma',project='openwebrl-evals',id=identity,resume='allow',name=identity,config={'arm':arm,'shard':shard,'model':plan['model'],'reasoning_effort':'high','candidate_count':plan['candidate_count'],'selector_reasoning_effort':'high'if plan['candidate_count']>1 else None,'decisions':100,'actor_max_output_tokens':20000,'browser':'local','tasks':10,'model_usd_cap':cap['model_usd_all_attempts'],'judge_usd_cap':cap['external_judge_usd_all_attempts'],'all_attempt_seconds_cap':28800,'model_physical_call_cap':cap['model_calls_all_attempts'],'browser_start_cap':60,'maximum_model_inflight':8})
 secret=private_secrets.token_hex(32)
 server,baseurl=start_gateway(R,shard,deadline,key,group_secret=secret)
 host=d/'.host';host.mkdir(exist_ok=True);groupfile=host/('group-config-'+os.environ['SLURM_JOB_ID']+'.json');assert not groupfile.exists();write(groupfile,{'group_base_url':baseurl,'group_secret':secret,'arm':arm,'shard':shard,'deadline_epoch':deadline})
 sys.path.insert(0,str(R.parents[1]/'orchestration'));from task_phases import phases
 tasks=[x for x in read(R/'tasks.json')if x['shard']==shard];stages=phases(plan,shard,tasks);completed=[]
 # Completed dispositions are never silently rerun. Recovery must provide a separately audited selected-task manifest.
 existing=list(episode_directory(shard).glob('*/disposition.json'))
 assert not existing,'Existing disposition requires explicit mechanical recovery selection'
 def execute(batch):
  def progress(active):write(d/'heartbeat.json',{'epoch':time.time(),'stage':'episodes','active':active,'completed':len(completed),'usage':totals(R,shard),'arm':arm,'job_id':os.environ['SLURM_JOB_ID']})
  def result_log(result):run.log({'completed':len(completed),'valid':sum(x['valid']for x in completed),'successes':sum(x.get('reward')==1 for x in completed),'last_task_latency_seconds':result['task_latency_seconds']})
  execute_cases(batch,shard,baseurl,deadline,completed,progress,result_log)
 try:
  startup_probe(shard,baseurl,deadline,totals)
  if stages['before_own_arm_independent_release']:
   execute(stages['before_own_arm_independent_release']);write(d/'canary-finished.json',{'shard':shard,'job_id':os.environ['SLURM_JOB_ID'],'tasks':len(completed),'task_ids':[t['task_id'] for t in stages['before_own_arm_independent_release']],'epoch':time.time(),'arm':arm})
  while not release_verified():
   if time.time()>deadline-60:raise TimeoutError('Independent canary release not granted before remaining allocation deadline')
   write(d/'heartbeat.json',{'epoch':time.time(),'stage':'waiting_independent_canary_release','completed':len(completed),'usage':totals(R,shard),'job_id':os.environ['SLURM_JOB_ID'],'arm':arm});time.sleep(10)
  execute(stages['after_own_arm_independent_release']);assert len(completed)==len(tasks) and len({x['task_id'] for x in completed})==len(tasks)
  summary={'recorded':len(completed),'valid':sum(x['valid']for x in completed),'successes':sum(x.get('reward')==1 for x in completed),'usage':totals(R,shard),'job_id':os.environ['SLURM_JOB_ID'],'verified_complete':False,'epoch':time.time(),'arm':arm};write(d/'worker-finished.json',summary);run.summary.update(summary)
 finally:
  server.shutdown();server.server_close();run.finish()
