"""Fresh four-cell episodes, shared native actor/selector/judge accounting."""
from bootstrap import ROOT,CONFIG
import argparse,json,os,signal,sys,time,shlex,hashlib
from pathlib import Path
from preparation_gate import validate
from openwebrl.luna_qwen_policy import Budget
from openwebrl.luna_qwen_metrics import write_json
from cuawright.webwright.agents.default import DefaultAgent
from environments import StableWorkspace,BrowserCodeWorkspace,OwnedBrowser,live_page,capture_cdp,verify_png,ResourceSampler
from paired_model import PairedModel
from common_judge import judge
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
  for local in reversed(locals_):local.close()
  row.update(ended_epoch=time.time(),browser_cleanup=browser.close());write_json(directory/'receipt.json',row)
 if not row['complete']:raise RuntimeError('No paid requests before native startup preflight resolves')

def episode(root,item,cell,client,judge_client,budgets,slot):
 kind=cell['harness'];mode=cell['policy'];directory=root/'episodes'/(item['task_id']+'-'+kind+'-'+mode)
 if directory.exists():raise ValueError('Prior episode evidence exists; explicit reviewed recovery required')
 directory.mkdir(parents=True);workspace=directory/'workspace';browser=OwnedBrowser(directory);local=None;sampler=None;timed=None
 row={'task_id':item['task_id'],'harness':kind,'mode':mode,'worker_id':slot,'job_id':os.environ.get('SLURM_JOB_ID'),'execution_source_manifest_sha256':hashlib.sha256((ROOT/'execution-source-manifest.json').read_bytes()).hexdigest(),'plan_sha256':PLAN['plan_sha256'],'valid':False,'reward':None,'physical_browser_attempt':False,'started_epoch':time.time(),'step_cap':30}
 write_json(directory/'identity.json',row)
 try:
  if any((b.root/'halt.json').exists() for b in budgets.values()):row['not_dispatched_reason']='shared_budget_or_protocol_halt';return row
  budgets['browser'].update(reserve=1);row['physical_browser_attempt']=True;setup_start=time.monotonic();ws=browser.start();local=environment(workspace,ws,kind)
  try:setup=setup_native(local,kind,ws,item);write_json(directory/'setup.json',setup)
  except BaseException as exc:
   write_json(directory/'setup-error.json',{'error_type':type(exc).__name__,'error':str(exc)});raise
  row['setup_seconds']=time.monotonic()-setup_start
  model=PairedModel(mode,directory,client,budgets['actor'],CONFIG['pricing'],harness=kind,model_name='gpt-6.1-sol',reasoning_effort='high' if mode=='serial_high' else 'low',max_output_tokens=12288 if mode=='serial_high' else 4096,response_mode='run_command_tool',action_field='bash_command',openai_api_key='metered-external-client',attach_observation_screenshot=False,observation_template=CONFIG['observation_template'])
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
  if sampler is not None:row['process_resources']=sampler.close()
  if local is not None:local.close()
  row.update(ended_epoch=time.time(),browser_cleanup=browser.close());write_json(directory/'result.json',row)
 return row

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--shard',type=int,required=True);parser.add_argument('--slot',type=int,required=True);args=parser.parse_args();validate(require_approval=True)
 from controller import verify
 verify(os.environ.get('SLURM_JOB_ID'))
 root=ROOT/f'shard{args.shard}';sp=json.loads((root/'plan.json').read_text());limits=PLAN['limits'];budgets={'actor':Budget(root/'budget/actor',limits['sol_usd_all_attempts'],limits['sol_calls_all_attempts']),'judge':Budget(root/'budget/judge',limits['judge_usd_all_attempts'],limits['judge_calls_all_attempts']),'browser':Budget(root/'budget/browser',limits['browser_sessions_all_attempts'],limits['browser_sessions_all_attempts'])}
 from dotenv import dotenv_values
 from openai import OpenAI
 keys=dotenv_values('/gpfs/projects/krishna/zixianma/OpenWebRL/.env');client=OpenAI(api_key=keys['OPENAI_API_KEY'],max_retries=0,timeout=180);judge_client=OpenAI(api_key=keys.get('JUDGE_API_KEY') or keys['OPENAI_API_KEY'],max_retries=0,timeout=180)
 import wandb
 if keys.get('WANDB_API_KEY'):os.environ['WANDB_API_KEY']=keys['WANDB_API_KEY']
 tracking=wandb.init(entity='zixianma',project='openwebrl-evals',id=f'browsercode-onecall3-20261017-s{args.shard}-w{args.slot}',resume='allow',dir=str(root),job_type='paired_harness_evaluation',config=PLAN['protocol'],settings=wandb.Settings(disable_git=True),save_code=False)
 signal.signal(signal.SIGTERM,lambda *unused:(_ for _ in ()).throw(SystemExit(143)));tasks={x['task_id']:x for x in json.loads((ROOT/'tasks.json').read_text())}
 try:
  for index,task_id in enumerate(sp['task_ids']):
   if index%2!=args.slot:continue
   for cell in sp['task_orders'][task_id]:
    write_json(root/f'worker{args.slot}-heartbeat.json',{'epoch':time.time(),'task_id':task_id,**cell,'job_id':os.environ['SLURM_JOB_ID']})
    row=episode(root,tasks[task_id],cell,client,judge_client,budgets,args.slot)
    if row.get('error') and row.get('task_latency_seconds') is None and row.get('physical_browser_attempt'):
     directory=root/'episodes'/(task_id+'-'+cell['harness']+'-'+cell['policy']);setup=directory/'setup-native.json';text=setup.read_text() if setup.exists() else row['error']
     if 'Page.goto:' not in text and 'net::ERR_' not in text:
      write_json(budgets['actor'].root/'halt.json',{'reason':'systemic_native_startup_failure','error_type':row.get('error_type'),'task_id':task_id})
      raise RuntimeError('Stop before more physical attempts; native startup requires diagnosis')
    tracking.log({'harness':cell['harness'],'policy':cell['policy'],'valid':int(row['valid']),'success':int(row.get('reward')==1),'task_latency_seconds':row.get('task_latency_seconds',0)})
  write_json(root/f'worker{args.slot}-complete.json',{'epoch':time.time(),'job_id':os.environ['SLURM_JOB_ID'],'worker_collection_complete':True,'independent_audit_required':True})
 finally:tracking.finish();client.close();judge_client.close()
if __name__=='__main__':main()
