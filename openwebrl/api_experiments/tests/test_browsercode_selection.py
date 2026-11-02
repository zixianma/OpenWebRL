"""Zero-provider tests of actual prospective payloads, concurrency and budget code."""
from __future__ import annotations
import ast
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import threading
import types
import unittest
from unittest.mock import patch

API=Path(__file__).resolve().parents[1]
REPRO=API/'reproduce'
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
p=load('selection_preparer',REPRO/'prospective_selection.py')
c=load('selection_common',REPRO/'common.py')
b=load('selection_frontdoor',REPRO/'browsercode.py')

class SelectionTests(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory(prefix='selection-offline-',dir='/tmp');self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
  self.paths=patch.object(sys,'path',[str(REPRO),*sys.path]);self.paths.start();self.addCleanup(self.paths.stop)
 def runtime(self,name):
  r=self.root/name;r.mkdir();spec=API/'portable_assets/campaign_specs/browsercode_gpt_high';config=p.read(spec/'runtime-config.json');config.pop('support_source_hashes',None)
  for k in('cuawright_source','frozen_support_source','chromium','python'):config[k]=str(r/'dependency')
  config['portable_campaign']=p.BASE_CAMPAIGN;p.write(r/'runtime-config.json',config)
  plan=p.read(spec/'campaign-plan.json');plan['protocol']['max_decisions']=30;plan['arms']=[dict(arm=arm,model=plan['arm_protocols'][arm]['actor_model'],proposal_count=plan['arm_protocols'][arm]['candidates'],reasoning_effort='high',selector_reasoning_effort='high',output_tokens=12288,selector_output_tokens=12288)for arm in plan['arms']]
  from tasks import load_tasks
  p.write(r/'tasks.json',load_tasks(p.BASE_CAMPAIGN));plan['task_manifest_sha256']=p.digest(r/'tasks.json');p.write(r/'campaign-plan.json',plan);p.write(r/'portable-build.json',{'adaptations':[]})
  (r/'paired_model.py').write_bytes((API/'browsercode/models/paired_model.py').read_bytes());(r/'worker.py').write_text(p.portable_worker_source());(r/'common_judge.py').write_text('# archived base judge\n');(r/'episode_metrics.py').write_bytes((API/'browsercode/runtime/gpt_high_episode_metrics.py').read_bytes())
  for n in('browser_bridge.ts','browsercode','node_modules','javascript-runtime/bun','sandbox.py','dependency'):
   path=r/n;path.parent.mkdir(parents=True,exist_ok=True);path.touch()
  inventory={'plan_sha256':p.digest(r/'campaign-plan.json'),'source_hashes':{str(f):p.digest(f)for f in r.rglob('*')if f.is_file()}};p.write(r/'source-manifest.json',inventory);p.write(r/'execution-source-manifest.json',inventory);return r
 def args(self,r,arm,extra=()):return c.parser('gpt').parse_args(['--runtime',str(r),'--arm',arm,'--output',str(self.root/'output'),'--model-usd','2','--judge-usd','1',*extra])
 def test_both_protocols_same_roster_and_exact_terra(self):
  ids=[]
  for count,dec in((2,30),(5,60)):
   r=self.runtime(str(count));before={f:f.read_bytes()for f in (API/'browsercode/models/paired_model.py',API/'browsercode/workers/gpt_high_worker.py')};receipt=p.configure(r,count,dec);self.assertEqual(p.validate(r),receipt);plan,tasks=c.prepare(self.args(r,receipt['arm'],('--limit','1045')),'gpt');ids.append([t['task_id']for t in tasks]);self.assertEqual((plan['candidate_count'],plan['max_decisions'],plan['max_output_tokens'],plan['selector_max_output_tokens']),(count,dec,12288,12288));self.assertEqual(plan['selector_effort'],'high');self.assertEqual(plan['benchmark_roster_counts'],{'online-mind2web':300,'webvoyager':595,'deepshop':150});self.assertEqual(p.read(r/'runtime-config.json')['judge_protocol'],receipt['judge_protocol']);self.assertEqual(receipt['judge_protocol']['seed'],42);self.assertFalse((self.root/'output').exists());self.assertFalse(receipt['paid_authorization'])
   for f,raw in before.items():self.assertEqual(f.read_bytes(),raw)
  self.assertEqual(ids[0],ids[1]);self.assertEqual(len(set(ids[0])),1045)
 def test_invalid_variants_home_and_used_runtime_reject(self):
  for count,dec in((True,30),(2,True),(3,30),(2,60),(5,30),('2',30)):
   with self.subTest(count=count,dec=dec),self.assertRaises(ValueError):p.configure(self.root/'absent',count,dec)
  with self.assertRaisesRegex(ValueError,'home is read-only'):p.configure('/gpfs/home/zixianma/forbidden',2,30)
  r=self.runtime('used');(r/'budget').mkdir()
  with self.assertRaisesRegex(ValueError,'unused'):p.configure(r,2,30)
  self.assertFalse((r/p.ORIGINALS).exists())
 def test_rehashed_scientific_drift_and_missing_marker_reject_before_execute(self):
  for mutation in('paired_model.py','worker.py','runtime-config.json','tasks.json','common_judge.py',p.RECEIPT):
   r=self.runtime(mutation.replace('.','_'));receipt=p.configure(r,2,30)
   if mutation==p.RECEIPT:(r/mutation).rename(r/'preserved-marker.json')
   else:
    path=r/mutation;path.write_text(path.read_text()+'\n');m=p.read(r/'source-manifest.json');m['source_hashes'][str(path)]=p.digest(path);p.write(r/'source-manifest.json',m);p.write(r/'execution-source-manifest.json',m)
   with self.subTest(mutation=mutation),self.assertRaises(ValueError):c.prepare(self.args(r,receipt['arm']),'gpt')
 def test_wrong_provider_arm_executor_and_unmarked_n2_reject(self):
  r=self.runtime('scope');receipt=p.configure(r,2,30)
  for provider,arm,harness in(('luna',receipt['arm'],'browsercode'),('gpt','parallel5_select_high','browsercode'),('gpt',receipt['arm'],'cuawright')):
   a=self.args(r,receipt['arm']);a.arm=arm;a.harness=harness
   with self.assertRaises(ValueError):c.prepare(a,provider)
  r=self.runtime('unmarked')
  with self.assertRaises(ValueError):c.prepare(self.args(r,'parallel2_select_high'),'gpt')
 def test_frontdoor_paired_flags_reject_early_and_defaults_unchanged(self):
  base=['setup','--campaign','gpt-crossbench','--workspace',str(self.root/'fresh')]
  for extra in (['--luna-selection-candidates','2'],['--luna-selection-max-decisions','60'],['--luna-selection-candidates','2','--luna-selection-max-decisions','60'],['--luna-selection-candidates','2','--luna-selection-max-decisions','30','--luna-single-output-tokens','12288']):
   args=b.parser().parse_args(base+extra)
   with patch.object(b,'call',side_effect=AssertionError('No setup')),self.assertRaises(ValueError):b.setup(args)
  args=b.parser().parse_args(base);self.assertIsNone(args.luna_selection_candidates);self.assertIsNone(args.luna_selection_max_decisions)
 def test_ordered_task_selection_keeps_marker_metadata(self):
  r=self.runtime('taskorder');receipt=p.configure(r,2,30);tasks=p.read(r/'tasks.json');ids=[tasks[301]['task_id'],tasks[300]['task_id']];f=self.root/'ids.json';p.write(f,ids);args=self.args(r,receipt['arm'],('--task-ids',str(f),'--benchmark','webvoyager','--limit','1'));plan,chosen=c.prepare(args,'gpt');self.assertEqual([x['task_id']for x in chosen],ids[:1]);self.assertEqual(plan['task_selection_count'],2);self.assertEqual(plan['prospective_selection_protocol'],p.SCHEMA)
 def test_execution_constructs_native_terra_budget_before_any_sdk_request(self):
  r=self.runtime('native-judge-budget');receipt=p.configure(r,2,30);args=self.args(r,receipt['arm']);plan,tasks=c.prepare(args,'gpt');calls=[]
  class ActorBudget:
   def __init__(self,*args):calls.append(('actor',args))
  class JudgeBudget:
   def __init__(self,*args):calls.append(('Terra',args))
  class BrowserBudget:
   def __init__(self,*args):calls.append(('browser',args))
  worker=types.SimpleNamespace(ReservationBudget=ActorBudget,Budget=BrowserBudget,JudgeBudget=JudgeBudget)
  def no_sdk(**kwargs):raise RuntimeError('blocked before SDK')
  with patch.dict(sys.modules,{'worker':worker,'environments':types.SimpleNamespace(),'openai':types.SimpleNamespace(OpenAI=no_sdk)}),patch.dict('os.environ',{'OPENAI_API_KEY':'offline-fixture'}),self.assertRaisesRegex(RuntimeError,'blocked before SDK'):
   c.execute(args,plan,tasks)
  self.assertEqual([x[0]for x in calls],['actor','Terra','browser']);self.assertEqual(str(calls[1][1][1]),'1')
 def actor(self,r):
  class OpenAIModel:
   def __init__(self,**kwargs):self.constructor=kwargs
   def _build_payload(self,messages):return {'input':deepcopy(messages),'include':['reasoning.encrypted_content']}
   def _response_extra(self,native):return {'selected_native_response_id':native['id']}
   def format_message(self,**kwargs):return kwargs
  support=API/'portable_assets/support/openwebrl';metrics=load('real_selection_metrics',support/'luna_qwen_metrics.py');tree=ast.parse((support/'luna_qwen_policy.py').read_text());budgetclass=next(n for n in tree.body if isinstance(n,ast.ClassDef)and n.name=='Budget');ns={'Path':Path};exec(compile(ast.Module(body=[budgetclass],type_ignores=[]),'<original Budget>','exec'),ns)
  fake={'openwebrl.luna_qwen_policy':types.SimpleNamespace(Budget=ns['Budget']),'openwebrl.luna_qwen_metrics':metrics,'cuawright.webwright.models.openai_model':types.SimpleNamespace(OpenAIModel=OpenAIModel),'openwebrl.responses_actor_history':types.SimpleNamespace(response_input_bound=lambda *unused:1000),'openwebrl.sol_serial_parallel_policy':types.SimpleNamespace(selector_input=lambda history,proposals:[{'role':'system','content':'choose'},{'role':'user','content':deepcopy(proposals)}])}
  m=types.ModuleType('actual_adapted_selection_actor');m.__file__=str(r/'paired_model.py')
  with patch.dict(sys.modules,fake):exec(compile((r/'paired_model.py').read_text(),m.__file__,'exec'),m.__dict__)
  return m
 def exercise(self,count,dec,bad_selector=False):
  r=self.runtime('actor'+str(count)+str(bad_selector));receipt=p.configure(r,count,dec);m=self.actor(r);barrier=threading.Barrier(count);lock=threading.Lock();records=[];counter=0
  def create(**payload):
   nonlocal counter
   with lock:counter+=1;ident=counter;records.append(deepcopy(payload))
   if 'text'in payload:
    value=count if bad_selector else 1;output=[{'type':'message','role':'assistant','content':[{'type':'output_text','text':json.dumps({'selected_index':value})}]}]
   else:
    barrier.wait(timeout=4);output=[{'type':'function_call','name':'browser_execute','arguments':json.dumps({'code':f'console.log({ident})','timeout':1000,'description':'inspect'})}]
   return types.SimpleNamespace(model_dump=lambda:{'id':'native-'+str(ident),'model':'gpt-6-luna','status':'completed','reasoning':{'effort':'high'},'output':output,'usage':{'input_tokens':100,'output_tokens':20,'input_tokens_details':{'cached_tokens':0,'cache_write_tokens':0},'total_tokens':120}})
  client=types.SimpleNamespace(responses=types.SimpleNamespace(create=create));pool=m.ReservationBudget(self.root/('budget'+str(count)+str(bad_selector)),1.,100)
  ep=self.root/('episode'+str(count)+str(bad_selector));ep.mkdir();model=m.PairedModel(receipt['arm'],ep,client,pool,p.read(r/'runtime-config.json')['model_pricing']['gpt-6-luna'],max_output_tokens=12288,reasoning_effort='high');model.turn=dec-1;history=[{'role':'user','content':'observed only'}];before=deepcopy(history)
  if bad_selector:
   with self.assertRaises(ValueError):model.query(history)
  else:
   result=model.query(history);self.assertEqual(model.turn,dec);self.assertEqual(history,before);decision=p.read(ep/'decisions'/f'{dec-1:03d}.json');self.assertEqual(result['extra']['selected_native_response_id'],decision['candidate_responses'][1]['id']);self.assertEqual(len(result['extra']['actions']),1);self.assertEqual(result['extra']['actions'][0]['command'],decision['selected_action']['command']);self.assertNotIn('candidate_responses',result['extra']);self.assertNotIn('selector_response',result['extra'])
   with self.assertRaisesRegex(ValueError,'decision cap'):model.query(history)
  self.assertEqual(len(records),count+1);self.assertEqual(p.read(pool.root/'ledger.json')['calls'],count+1);self.assertEqual(p.read(pool.root/'outstanding-reservations.json'),[])
  for q in records:self.assertEqual(q['model'],'gpt-6-luna');self.assertEqual(q['reasoning'],{'effort':'high'});self.assertEqual(q['max_output_tokens'],12288);self.assertFalse(q['store']);self.assertEqual(q['service_tier'],'default')
  selector=next(q for q in records if'text'in q);self.assertEqual(selector['text']['format']['schema']['properties']['selected_index']['enum'],list(range(count)));self.assertEqual(sum('text'not in q for q in records),count)
  calls=p.read(pool.root/'ledger.json')['calls']
  with self.assertRaises(ValueError):model.physical(dict(records[0],max_output_tokens=24576),'candidate_0')
  self.assertEqual(p.read(pool.root/'ledger.json')['calls'],calls)
 def test_n2_real_parallel_http_then_one_high_selector_selected_only(self):self.exercise(2,30)
 def test_n5_turn59_allowed_turn60_blocked_parallel_and_selected_only(self):self.exercise(5,60)
 def test_n2_selector_rejects_out_of_range_without_extra_calls(self):self.exercise(2,30,True)
 def test_actor_scientific_functions_unchanged_except_declared_variant(self):
  source=(API/'browsercode/models/paired_model.py').read_text();base=ast.parse(source)
  for count,dec in((2,30),(5,60)):
   tree=ast.parse(p.adapt_model(source,count,dec))
   for name in('parse_actor','response_output_text','ReservationBudget'):
    self.assertEqual(ast.dump(next(x for x in base.body if getattr(x,'name',None)==name)),ast.dump(next(x for x in tree.body if getattr(x,'name',None)==name)))
 def test_worker_keeps_900second_alarm_and_does_not_add_summary_or_capture(self):
  for dec in(30,60):
   text=p.adapt_worker(p.portable_worker_source(),dec);self.assertIn('signal.alarm(900)',text);self.assertIn(f'step_limit={dec}',text);self.assertIn('summary_every_n_steps=0',text);self.assertIn('from request_budget import ReservationBudget as JudgeBudget',text);self.assertIn("[json.dumps(d['selected_action']) for d in decisions]",text)

if __name__=='__main__':unittest.main()
