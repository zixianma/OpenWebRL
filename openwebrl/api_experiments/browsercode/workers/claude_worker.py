"""Fresh four-cell episodes, shared native actor/selector/judge accounting."""
from bootstrap import ROOT,CONFIG
import argparse,json,os,signal,sys,time,shlex,hashlib
from pathlib import Path

from openwebrl.luna_qwen_policy import Budget
from openwebrl.luna_qwen_metrics import write_json
from cuawright.webwright.agents.default import DefaultAgent
from environments import StableWorkspace,BrowserCodeWorkspace,OwnedBrowser,live_page,capture_cdp,verify_png,ResourceSampler
from claude_model import ClaudeModel as PairedModel
from claude_budget import ReservationBudget
from common_judge import judge
from episode_metrics import receipt_metrics
PLAN=json.loads((ROOT/'campaign-plan.json').read_text())

def environment(workspace,ws,kind):
 cls=StableWorkspace if kind=='cuawright' else BrowserCodeWorkspace
 return cls(output_dir=workspace,start_url='',browser_mode='cdp',command_timeout_seconds=120,output_truncation_chars=24000,shell=str(ROOT/'sandbox.py'),env={'CUA_CHROME_EXECUTABLE':CONFIG['chromium'],'WEBWRIGHT_BROWSER_CDP_URL':ws,'WEBWRIGHT_BROWSER_BACKEND':'cdp'})

def setup_native(local,kind,ws,item):
 local.prepare(task=item['intent'],task_id=item['task_id'],start_url=item['start_url'])
 if kind=='cuawright':
  command=[sys.executable,'-m','cuawright.webwright.tools.browser_session','create','--workspace-dir',str(local._workspace_dir()),'--backend','cdp','--start-url',item['start_url']]
  result=local.execute({'command':shlex.join(command)})
  write_json(local._workspace_dir().parent/'setup-native.json',result)
  if not result['observation']['success']:raise RuntimeError('Initial browser session/navigation failed')
  live_page(ws)
 else:
  target=live_page(ws,navigate=item['start_url']);local.start(ws,target)
  viewport={'code':'await session.Emulation.setDeviceMetricsOverride({width:1280,height:1800,deviceScaleFactor:1,mobile:false}); console.log("viewport-fixed")','description':'Pin the common 1280x1800 viewport used by the matched harness','timeout':30000}
  checked=local.execute({'command':json.dumps(viewport)})
  if not checked['observation']['success'] or 'viewport-fixed' not in checked['output']:raise RuntimeError('Could not pin BrowserCode viewport')
  write_json(local._workspace_dir().parent/'browsercode-viewport.json',checked)
  result={'observation':{'success':True},'native_browsercode_bound':True,'viewport':'1280x1800'}
 return result

def preflight(root,browser_budget):
 directory=root/'startup-preflights'/os.environ.get('SLURM_JOB_ID','offline');directory.mkdir(parents=True,exist_ok=False);browser=OwnedBrowser(directory);locals_=[];row={'model_requests':0,'browser_sessions':1,'complete':False,'started_epoch':time.time()}
 try:
  browser_budget.update(reserve=1);ws=browser.start();item={'task_id':'startup-diagnostic','intent':'Mechanical browser/CDP diagnostic only','start_url':'data:text/html,<html><body><h1>native-startup-ready</h1></body></html>'}
  for kind in ['browsercode']:
   local=environment(directory/kind,ws,kind);locals_.append(local);setup=setup_native(local,kind,ws,item);write_json(directory/(kind+'-setup.json'),setup)
   if kind=='cuawright':
    command=shlex.join([sys.executable,'-m','cuawright.webwright.tools.browser_session','step','--workspace-dir',str(local._workspace_dir()),'--action','Verify native local startup','--code-file','-'])+" <<'PY'\nprint(await page.title())\nprint(await page.locator('h1').inner_text())\nPY"
    result=local.execute({'command':command})
    if not result['observation']['success'] or 'native-startup-ready' not in result['output']:raise RuntimeError('Native CUAWright step diagnostic failed')
    image=directory/'cuawright-final.png';capture_cdp(ws,image)
   else:
    code='const r=await session.Runtime.evaluate({expression:'+json.dumps("document.querySelector('h1').textContent")+',returnByValue:true});console.log(r.result.value);'
    result=local.execute({'command':json.dumps({'code':code,'description':'Verify native BrowserCode startup','timeout':30000})})
    if not result['observation']['success'] or 'native-startup-ready' not in result['output']:raise RuntimeError('Native BrowserCode step diagnostic failed')
    image=directory/'browsercode-final.png';local.capture(image)
   verify_png(image);write_json(directory/(kind+'-step.json'),result)
  row['complete']=True
 finally:
  cleanup_errors=[]
  for local in reversed(locals_):
   try:local.close()
   except BaseException as exc:cleanup_errors.append({'component':'workspace','error_type':type(exc).__name__})
  try:row['browser_cleanup']=browser.close()
  except BaseException as exc:cleanup_errors.append({'component':'browser','error_type':type(exc).__name__})
  row.update(ended_epoch=time.time(),cleanup_errors=cleanup_errors);row['complete']=row['complete']and not cleanup_errors;write_json(directory/'receipt.json',row)
 if not row['complete']:raise RuntimeError('No paid requests before native startup preflight resolves')

def episode(root,item,cell,client,judge_client,budgets,slot):
 kind=cell['harness'];mode=cell['policy'];directory=root/'episodes'/(item['task_id']+'-'+kind+'-'+mode)
 if directory.exists():raise ValueError('Prior episode evidence exists; explicit reviewed recovery required')
 directory.mkdir(parents=True);workspace=directory/'workspace';browser=OwnedBrowser(directory);local=None;sampler=None;timed=None
 row={'task_id':item['task_id'],'harness':kind,'mode':mode,'worker_id':slot,'job_id':os.environ.get('SLURM_JOB_ID'),'execution_source_manifest_sha256':hashlib.sha256((ROOT/'source-manifest.json').read_bytes()).hexdigest(),'plan_sha256':hashlib.sha256((ROOT/'campaign-plan.json').read_bytes()).hexdigest(),'valid':False,'reward':None,'physical_browser_attempt':False,'started_epoch':time.time(),'step_cap':30,'task_latency_excludes_judge':True,'judge_protocol_version':CONFIG['judge_protocol']['version']}
 write_json(directory/'identity.json',row)
 try:
  if any((b.root/'halt.json').exists() for b in budgets.values()):row['not_dispatched_reason']='shared_budget_or_protocol_halt';return row
  budgets['browser'].update(reserve=1);row['physical_browser_attempt']=True;setup_start=time.monotonic();ws=browser.start();local=environment(workspace,ws,kind)
  try:setup=setup_native(local,kind,ws,item);write_json(directory/'setup.json',setup)
  except BaseException as exc:
   write_json(directory/'setup-error.json',{'error_type':type(exc).__name__,'error':str(exc)});raise
  row['setup_seconds']=time.monotonic()-setup_start
  model_name=next(x['model']for x in PLAN['arms']if x['arm']==mode);model=PairedModel(mode,directory,client,budgets['actor'],CONFIG['model_pricing'][model_name],harness=kind,model_name=model_name,reasoning_effort='high',max_output_tokens=12288,response_mode='run_command_tool',action_field='bash_command',openai_api_key='metered-external-client',attach_observation_screenshot=False,observation_template=CONFIG['observation_template'])
  agent=DefaultAgent(model,local,system_template=CONFIG['system_templates'][kind],instance_template='Task: {{ task }}\nStart URL: {{ start_url }}\nWorkspace: {{ workspace_dir }}\nThe controller already initialized the live browser. Use it.',step_limit=30,summary_every_n_steps=0,require_self_reflection_success=False,output_path=directory/'trajectory.json')
  sampler=ResourceSampler();sampler.start();timed=time.monotonic()
  def deadline(*unused):raise TimeoutError('Prespecified900-second episode deadline')
  signal.signal(signal.SIGALRM,deadline);signal.alarm(900)
  try:result=agent.run(task=item['intent'],**dict(local.get_template_vars(),start_url=item['start_url']))
  finally:signal.alarm(0)
  row.update(task_latency_seconds=time.monotonic()-timed,task_latency_excludes_judge=True,agent_result=result,decisions=agent.n_calls)
  if result.get('exit_status') not in ['Submitted','LimitsExceeded']:raise RuntimeError('Unexpected native agent endpoint')
  capture_start=time.monotonic();image=directory/'final.png'
  if kind=='browsercode':local.capture(image)
  else:capture_cdp(ws,image)
  verify_png(image);row['final_capture_seconds']=time.monotonic()-capture_start;row['final_png_sha256']=hashlib.sha256(image.read_bytes()).hexdigest()
  row['process_resources']=sampler.close();sampler=None;local.close();local=None;row['browser_cleanup_before_judge']=browser.close()
  decisions=[json.loads(p.read_text()) for p in sorted((directory/'decisions').glob('*.json'))];row['selected_code_batches']=sum(bool(d['selected_action']['command']) for d in decisions)
  cli=workspace/'browser-steps.jsonl';cdp=workspace/'cdp-calls.jsonl';row['native_cua_cli_batches']=len(cli.read_text().splitlines()) if cli.exists() else None;row['native_browsercode_cdp_attempts']=len(cdp.read_text().splitlines()) if cdp.exists() else None
  judge_start=time.monotonic();row['reward']=judge(judge_client,budgets['judge'],item['intent'],[json.dumps(d['selected_action']) for d in decisions],image,directory);row['judge_seconds']=time.monotonic()-judge_start;row['valid']=True
 except BaseException as exc:
  row.update(error_type=type(exc).__name__,error=str(exc));signal.alarm(0)
  if timed is not None:row.setdefault('task_latency_seconds',time.monotonic()-timed)
  if isinstance(exc,(SystemExit,KeyboardInterrupt)):raise
 finally:
  signal.alarm(0)
  cleanup_errors=[]
  if sampler is not None:
   try:row['process_resources']=sampler.close()
   except BaseException as exc:cleanup_errors.append({'component':'sampler','error_type':type(exc).__name__})
  if local is not None:
   try:local.close()
   except BaseException as exc:cleanup_errors.append({'component':'workspace','error_type':type(exc).__name__})
  try:row['browser_cleanup']=browser.close()
  except BaseException as exc:cleanup_errors.append({'component':'browser','error_type':type(exc).__name__})
  row.update(ended_epoch=time.time(),cleanup_errors=cleanup_errors)
  if cleanup_errors:
   row.update(valid=False,reward=None,cleanup_unverified=True);budgets['actor'].halt('Owned episode cleanup failed; no further episodes before diagnosis')
  decisions=[json.loads(p.read_text()) for p in sorted((directory/'decisions').glob('*.json'))]
  row.setdefault('decisions',len(decisions));row['selected_code_batches']=sum(bool(d['selected_action']['command']) for d in decisions)
  steps=list((workspace/'steps').glob('step_*.sh'));row['browser_code_batches_attempted']=max(0,len(steps)-1)
  cdp=workspace/'cdp-calls.jsonl';row['native_browsercode_cdp_attempts']=len(cdp.read_text().splitlines()) if cdp.exists() else 0
  row['usage_accounting']=receipt_metrics(directory,CONFIG['model_pricing'][next(x['model']for x in PLAN['arms']if x['arm']==mode)])
  if row['usage_accounting']['metric_errors']:
   row.update(valid=False,reward=None,accounting_unverified=True);budgets['actor'].halt('Receipt accounting failed; no further episodes before diagnosis')
  write_json(directory/'result.json',row)
 return row

def prior_episode_disposition(logical,tid,arm):
 paths=list(logical.glob('attempts/attempt-*/episodes/'+tid+'-browsercode-'+arm))
 for p in paths:
  from terminal_sidecar import exact_scope,qualify
  if exact_scope(logical,tid,arm,p):qualify(logical,tid,arm,p)
  elif not(p/'result.json').is_file():raise RuntimeError('Prior incomplete episode preserved; explicit targeted recovery required before replay')
 return bool(paths)

def preserve_or_recover_prior(prior,logical,task,shard_config,judge_client,judge_budget):
 """Archived migration rows are strict; current dispositions are never rerun."""
 prior=Path(prior).resolve();logical=Path(logical).resolve();arm=shard_config['arm'];tid=task['task_id']
 if not prior.is_relative_to(logical):raise ValueError('Prior episode escapes owned logical shard')
 from terminal_sidecar import exact_scope,qualify
 if exact_scope(logical,tid,arm,prior):return qualify(logical,tid,arm,prior)
 if not(prior/'result.json').is_file():raise RuntimeError('Foreign incomplete prior disposition')
 row=json.loads((prior/'result.json').read_text())
 if row.get('task_id')!=tid or row.get('mode')!=arm or row.get('harness')!='browsercode':raise ValueError('Prior disposition task/arm/harness identity mismatch')
 original=logical/'attempts/attempt-000/episodes'/(tid+'-browsercode-'+arm)
 if tid in shard_config.get('original_canary_task_ids',[])and prior==original:
  # Never let a forged version marker bypass the pinned original evidence.
  from judge_recovery import recover_prior_judge
  return recover_prior_judge(prior,task,judge_client,judge_budget)
 if row.get('judge_protocol_version')!=CONFIG['judge_protocol']['version']:
  raise ValueError('Unknown prior judge protocol outside designated original migration evidence')
 if type(row.get('valid'))is not bool or(row['valid']and(type(row.get('reward'))is not int or row['reward']not in(0,1)))or(not row['valid']and row.get('reward')is not None):
  raise ValueError('Malformed prior current-protocol disposition; preserve and inspect')
 return {'preserved':True,'current_protocol_preserved':True,'recovered':False,'valid':row['valid'],'task_id':tid,'actor_rerun':False,'judge_rerun':False}

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--arm',required=True);parser.add_argument('--shard',type=int,required=True);parser.add_argument('--slot',type=int,required=True);parser.add_argument('--attempt-dir',required=True);a=parser.parse_args()
 from controller import verify,validate_sources
 validate_sources();entry=verify(a.arm,a.shard,os.environ['SLURM_JOB_ID']);root=Path(a.attempt_dir).resolve();assert root==Path(entry['attempt_dir']).resolve()
 from financial_plan import verify_enrollment
 sp=verify_enrollment(entry,ROOT);logical=root.parent.parent
 budgets={'actor':ReservationBudget(logical/'budget/model',sp['model_usd_all_attempts'],sp['model_calls_all_roles_and_counters_all_attempts']),'judge':ReservationBudget(logical/'budget/judge',sp['judge_usd_all_attempts'],sp['judge_calls_all_attempts']),'browser':Budget(logical/'budget/browser',sp['browser_starts_all_attempts'],sp['browser_starts_all_attempts'])}
 from dotenv import dotenv_values
 from anthropic import Anthropic
 from openai import OpenAI
 keys=dotenv_values('/gpfs/projects/krishna/zixianma/OpenWebRL/.env');client=Anthropic(api_key=keys['ANTHROPIC_API_KEY'],max_retries=0,timeout=180);jc=OpenAI(api_key=keys.get('JUDGE_API_KEY')or keys['OPENAI_API_KEY'],max_retries=0,timeout=180)
 import wandb
 os.environ['WANDB_API_KEY']=keys['WANDB_API_KEY'];tracking=wandb.init(entity='zixianma',project='openwebrl-evals',id=f'claude-bc-20261028-{a.arm}-s{a.shard:02d}-a{entry["attempt"]:03d}-w{a.slot}',dir=str(root),job_type='browsercode_claude_evaluation',config={'arm':a.arm,'shard':a.shard,'reasoning_effort':'high','max_output_tokens':12288,'max_decisions':30,'judge_protocol':CONFIG['judge_protocol'],'plan_sha256':hashlib.sha256((ROOT/'campaign-plan.json').read_bytes()).hexdigest(),'financial_approval_sha256':entry.get('financial_approval_sha256')},settings=wandb.Settings(disable_git=True),save_code=False)
 tasks={x['task_id']:x for x in json.loads((ROOT/'tasks.json').read_text())};signal.signal(signal.SIGTERM,lambda *unused:(_ for _ in ()).throw(SystemExit(143)))
 try:
  from judge_recovery import qualify_saved_judge
  qualify_saved_judge(logical,sp,tasks,jc,budgets['judge'])
  for i,tid in enumerate(sp['task_ids']):
   if i%2!=a.slot:continue
   # Preserve actor dispositions. Only complete saved terminal evidence can receive an amended judge verdict.
   if prior_episode_disposition(logical,tid,a.arm):
    prior=sorted(logical.glob('attempts/attempt-*/episodes/'+tid+'-browsercode-'+a.arm))[-1]
    recovered=preserve_or_recover_prior(prior,logical,tasks[tid],sp,jc,budgets['judge'])
    tracking.log({'judge_recovery':int(recovered.get('recovered',False)),'judge_recovery_valid':int(recovered.get('valid',False))})
    continue
   if a.shard==0 and tid not in sp['canary_task_ids']:
    from controller import wait_canary_gate
    wait_canary_gate(a.arm,root)
   write_json(root/f'worker{a.slot}-heartbeat.json',{'epoch':time.time(),'task_id':tid,'arm':a.arm,'stage':'episode'})
   row=episode(root,tasks[tid],{'harness':'browsercode','policy':a.arm},client,jc,budgets,a.slot)
   tracking.log({'valid':int(row['valid']),'success':int(row.get('reward')==1),'task_latency_seconds':row.get('task_latency_seconds'),'browser_commands':row.get('selected_code_batches'),'model_cost_conservative':row['usage_accounting'].get('model_excluding_judge',{}).get('charged_or_reserved_usd')})
   if any((b.root/'halt.json').exists()for b in budgets.values()):raise RuntimeError('Pool halted; diagnose before further work')
  write_json(root/f'worker{a.slot}-complete.json',{'epoch':time.time(),'worker_complete':True,'job_id':os.environ['SLURM_JOB_ID']})
 finally:tracking.finish();client.close();jc.close()
if __name__=='__main__':main()
