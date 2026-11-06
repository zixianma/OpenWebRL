"""Offline Sol30 protocol, strict benchmark routing and native accounting."""
from collections import Counter
from copy import deepcopy
from decimal import Decimal
import ast,importlib.util,json,os,subprocess,sys,types,unittest
from pathlib import Path
from unittest.mock import patch
API=Path(__file__).resolve().parents[1];REPRO=API/'reproduce'
def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value
fixture=module('sol30_sweep_fixture',API/'tests/test_browsercode_output_sweep.py')
s=module('sol30_preparation',REPRO/'sol_single30.py');c=module('sol30_common',REPRO/'common.py');b=module('sol30_frontdoor',REPRO/'browsercode.py')

class Sol30(unittest.TestCase):
 def setUp(self):
  self.base=fixture.SweepTests();self.base.setUp();self.addCleanup(self.base.doCleanups);self.root=self.base.root
 def runtime(self,name='runtime'):
  runtime=self.base.runtime(name);judge=API/'browsercode/judges/terra.py'
  (runtime/'common_judge.py').write_bytes(judge.read_bytes())
  manifest=s.read(runtime/'source-manifest.json');manifest['source_hashes'][str(runtime/'common_judge.py')]=s.digest(runtime/'common_judge.py')
  s.write(runtime/'source-manifest.json',manifest);s.write(runtime/'execution-source-manifest.json',manifest)
  s.configure(runtime);return runtime
 def args(self,runtime,*extra):return c.parser('gpt').parse_args(['--runtime',str(runtime),'--arm',s.ARM,'--output',str(self.root/'output'),'--model-usd','5','--judge-usd','1',*extra])
 def test_exact_protocol_actual_guard_prompt_and_canonical_rosters(self):
  runtime=self.runtime();receipt=s.validate(runtime);config=s.read(runtime/'runtime-config.json');plan=s.read(runtime/'campaign-plan.json');rows=s.read(runtime/'tasks.json')
  self.assertEqual(dict(Counter(r['benchmark']for r in rows)),s.COUNTS);self.assertEqual(len({(r['benchmark'],r['task_id'])for r in rows}),1045);self.assertEqual(len({r['task_id']for r in rows}),1045)
  self.assertEqual((receipt['model'],receipt['reasoning_effort'],receipt['output_tokens'],receipt['max_decisions'],receipt['actor_episode_seconds']),('gpt-6-sol','xhigh',12288,30,900))
  self.assertIn('Only30 model decisions are available.',config['system_templates']['browsercode']);self.assertNotIn('Only60 model decisions are available.',config['system_templates']['browsercode'])
  self.assertEqual((config['judge_protocol']['model'],config['judge_protocol']['reasoning_effort'],config['judge_protocol']['max_completion_tokens'],config['judge_protocol']['seed']),('gpt-5.6-terra','medium',4096,42))
  self.assertIn('signal.alarm(900)',(runtime/'worker.py').read_text());self.assertIn('step_limit=30',(runtime/'worker.py').read_text());self.assertEqual(plan['arms'][0]['proposal_count'],1);self.assertEqual(plan['campaign'],'browsercode_sol_xhigh30');self.assertNotIn('shards',plan);self.assertEqual(config['portable_campaign'],'browsercode_sol_xhigh30')
 def test_full_dry_plans_route_all_three_benchmarks_before_uniqueness_check(self):
  runtime=self.runtime()
  for benchmark,count in s.COUNTS.items():
   plan,rows=c.prepare(self.args(runtime,'--benchmark',benchmark,'--limit',str(count)),'gpt')
   self.assertEqual(len(rows),count);self.assertEqual(len({r['task_id']for r in rows}),count);self.assertEqual({r['benchmark']for r in rows},{benchmark})
   self.assertEqual((plan['candidate_count'],plan['candidate_effort'],plan['max_output_tokens'],plan['max_decisions']),(1,'xhigh',12288,30));self.assertFalse(plan['execute'])
  self.assertFalse((self.root/'output').exists())
  with self.assertRaisesRegex(ValueError,'requires --benchmark'):c.prepare(self.args(runtime),'gpt')
 def test_native_id_selection_is_strict_within_selected_benchmark(self):
  runtime=self.runtime();rows=[r for r in s.read(runtime/'tasks.json')if r['benchmark']=='webvoyager'];selection=self.root/'selected.json';s.write(selection,[rows[1]['task_id'],rows[0]['task_id']])
  plan,tasks=c.prepare(self.args(runtime,'--benchmark','webvoyager','--task-ids',str(selection),'--limit','1'),'gpt');self.assertEqual(tasks,[rows[1]])
  s.write(selection,['FOREIGN_NONCANONICAL_TASK'])
  with self.assertRaisesRegex(ValueError,'absent'):c.prepare(self.args(runtime,'--benchmark','webvoyager','--task-ids',str(selection)),'gpt')
 def test_frontdoor_alias_conflicts_and_provider(self):
  self.assertEqual(b.campaign('sol-xhigh30'),'browsercode_sol_xhigh30');runtime=self.runtime();self.assertEqual(b.provider(runtime,s.ARM),'gpt')
  args=b.parser().parse_args(['setup','--campaign','sol-xhigh30','--workspace',str(self.root/'forbidden'),'--luna-single-output-tokens','12288','--luna-single-reasoning-effort','high'])
  with self.assertRaisesRegex(ValueError,'requires --campaign luna'):b.setup(args)
  self.assertFalse((self.root/'forbidden').exists())
 def test_rehashed_prompt_mutation_still_rejected(self):
  runtime=self.runtime();config=s.read(runtime/'runtime-config.json');config['system_templates']['browsercode']=config['system_templates']['browsercode'].replace('Only30','Only60');s.write(runtime/'runtime-config.json',config)
  receipt=s.read(runtime/s.RECEIPT);receipt['generated_sha256']['runtime-config.json']=s.digest(runtime/'runtime-config.json');s.write(runtime/s.RECEIPT,receipt)
  for name in('source-manifest.json','execution-source-manifest.json'):
   manifest=s.read(runtime/name)
   for changed in('runtime-config.json',s.RECEIPT):manifest['source_hashes'][str(runtime/changed)]=s.digest(runtime/changed)
   s.write(runtime/name,manifest)
  with self.assertRaisesRegex(ValueError,'configuration differs'):s.validate(runtime)
 def load_actor(self,runtime):
  class OpenAIModel:
   def __init__(self,**kwargs):self.constructor=kwargs
   def _build_payload(self,messages):return {'input':deepcopy(messages),'include':['reasoning.encrypted_content']}
   def _response_extra(self,native):return {'response_items':native['output']}
   def format_message(self,**kwargs):return kwargs
  fake={'request_budget':fixture.budget_module,'cuawright.webwright.models.openai_model':types.SimpleNamespace(OpenAIModel=OpenAIModel),'openwebrl.responses_actor_history':types.SimpleNamespace(response_input_bound=lambda *unused:1000),'openwebrl.sol_serial_parallel_policy':types.SimpleNamespace(selector_input=lambda *unused:[])}
  actor=types.ModuleType('sol30_test_actor');actor.__file__=str(runtime/'sol_model.py')
  with patch.dict(sys.modules,fake):exec(compile((runtime/'sol_model.py').read_text(),actor.__file__,'exec'),actor.__dict__)
  return actor
 def test_actual_model_thirty_calls_price_protocol_and_no_thirtyfirst_dispatch(self):
  runtime=self.runtime();actor=self.load_actor(runtime);calls=[]
  def client(payload,role):
   calls.append((deepcopy(payload),role));return {'model':'gpt-6-sol','status':'completed','reasoning':{'effort':'xhigh'},'output':[{'type':'message','role':'assistant','content':[{'type':'output_text','text':'done'}]}],'usage':{'input_tokens':100,'output_tokens':20,'input_tokens_details':{'cached_tokens':0,'cache_write_tokens':0}}}
  pool=fixture.budget_module.ReservationBudget(self.root/'pool','10',1000);gate=fixture.budget_module.SharedHTTPGate(self.root/'gate',1);actor.REQUIRED_HTTP_GATE=gate
  model=actor.SolModel(s.ARM,self.root/'episode',client,pool,actor.PRICING,http_gate=gate,max_output_tokens=12288,reasoning_effort='xhigh')
  for turn in range(30):model.query([{'role':'user','content':'offline observation'}])
  with self.assertRaisesRegex(ValueError,'cap'):model.query([{'role':'user','content':'offline observation'}])
  self.assertEqual(len(calls),30);self.assertEqual(pool.snapshot()['calls'],30)
  for payload,role in calls:self.assertEqual((payload['model'],payload['reasoning'],payload['max_output_tokens'],role),('gpt-6-sol',{'effort':'xhigh'},12288,'candidate_0'))
  self.assertEqual(Decimal(str(pool.snapshot()['charged_or_reserved_usd'])),Decimal('0.012'))
  self.assertTrue(all(r['status']=='known'for r in pool.snapshot()['requests'].values()))
 def test_preparation_writes_only_declared_runtime_in_collaborator_home(self):
  unrelated=self.root/'unrelated';unrelated.mkdir();sentinel=unrelated/'keep.txt';sentinel.write_text('preserve unrelated data')
  before={p.name:p.read_bytes()for p in unrelated.iterdir()}
  with patch.object(Path,'home',return_value=self.root):runtime=self.runtime('declared-runtime')
  self.assertEqual(runtime.parent,self.root);self.assertEqual({p.name:p.read_bytes()for p in unrelated.iterdir()},before)
  self.assertEqual({p.name for p in self.root.iterdir()},{'unrelated','declared-runtime'})
 def test_missing_root_gate_rejects_before_any_dispatch(self):
  runtime=self.runtime();actor=self.load_actor(runtime)
  with self.assertRaisesRegex(ValueError,'gate'):actor.SolModel(s.ARM,self.root/'ungated',None,None)


 def test_real_prefixed_terra_request_uses_shared_gate_and_settles_ledger(self):
  runtime=self.runtime();actor=self.load_actor(runtime)
  with patch.dict(sys.modules,{'claude_budget':fixture.budget_module}):
   judge=module('sol_prefixed_terra_test',runtime/'common_judge.py')
  budgets=types.SimpleNamespace(__file__=str(runtime/'request_budget.py'),SharedHTTPGate=fixture.budget_module.SharedHTTPGate,ReservationBudget=fixture.budget_module.ReservationBudget)
  worker=types.SimpleNamespace(__file__=str(runtime/'worker.py'),PairedModel=actor.SolModel)
  with patch.dict(sys.modules,{'sol_model':actor,'common_judge':judge,'request_budget':budgets}):
   gate=s.install_runtime_hooks(worker,self.root/'execution')
  self.assertIs(actor.REQUIRED_HTTP_GATE,judge.RUNTIME_HTTP_GATE)
  pool=fixture.budget_module.ReservationBudget(self.root/'judge-pool','1',1);calls=[]
  def create(**payload):
   calls.append(payload)
   return {'model':'gpt-5.6-terra','choices':[{'finish_reason':'stop','message':{'content':'Status: "success"'}}],'usage':{'prompt_tokens':100,'completion_tokens':10,'total_tokens':110,'prompt_tokens_details':{'cached_tokens':0,'cache_write_tokens':0}}}
  client=types.SimpleNamespace(max_retries=0,chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create)))
  image=self.root/'terminal.png';image.write_bytes(b'synthetic screenshot fixture')
  prompts=({'SYSTEM_PROMPT':'Judge fixture','USER_PROMPT':'{task} {thoughts_and_actions}'},'f'*64)
  with patch.object(judge,'frozen_prompts',return_value=prompts):
   self.assertEqual(judge.judge(client,pool,'fixture task',['fixture action'],image,self.root/'episode'),1)
  self.assertEqual(len(calls),1);self.assertEqual((calls[0]['model'],calls[0]['reasoning_effort'],calls[0]['max_completion_tokens'],calls[0]['seed']),('gpt-5.6-terra','medium',4096,42))
  snap=pool.snapshot();self.assertEqual(snap['calls'],1);self.assertFalse(pool.halted())
  ident=next(iter(snap['requests']));self.assertTrue(ident.startswith('judge-'));self.assertEqual(snap['requests'][ident]['status'],'known')
  root=self.root/'execution/http-gate'
  self.assertTrue((root/'physical-identities'/(ident[6:]+'.json')).is_file())
  self.assertEqual(len(list((root/'leases').glob('*/released.json'))),1)
  for bad in ['judge-'+('a'*31),'judge-'+('G'*32),'judge-other-'+('a'*32),'other-'+('a'*32),None]:
   with self.subTest(bad=bad),self.assertRaises(ValueError):
    with gate.slot(bad):self.fail('Malformed ID acquired a lease')

 def test_actual_preparation_cli_and_task_local_source_receipt(self):
  runtime=self.base.runtime('cli-runtime');judge=API/'browsercode/judges/terra.py';(runtime/'common_judge.py').write_bytes(judge.read_bytes());manifest=s.read(runtime/'source-manifest.json');manifest['source_hashes'][str(runtime/'common_judge.py')]=s.digest(runtime/'common_judge.py');s.write(runtime/'source-manifest.json',manifest);s.write(runtime/'execution-source-manifest.json',manifest)
  result=subprocess.run([sys.executable,'-B',str(REPRO/'sol_single30.py'),'--runtime',str(runtime)],capture_output=True,text=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',TMPDIR='/tmp'),timeout=30)
  self.assertEqual(result.returncode,0,result.stderr);receipt=s.validate(runtime);self.assertEqual(receipt['known_incomplete_policy']['source_sha256'],s.digest(s.POLICY_SOURCE));self.assertTrue(receipt['known_incomplete_policy']['task_invalid_still_raises']);self.assertTrue(receipt['known_incomplete_policy']['no_retry'])
 def test_wrong_effort_generated_source_rehashed_is_still_rejected(self):
  runtime=self.runtime();path=runtime/'sol_model.py';path.write_text(path.read_text().replace("(1, 'xhigh', None)","(1, 'high', None)"));receipt=s.read(runtime/s.RECEIPT);receipt['generated_sha256']['sol_model.py']=s.digest(path);s.write(runtime/s.RECEIPT,receipt)
  for name in('source-manifest.json','execution-source-manifest.json'):
   manifest=s.read(runtime/name)
   for changed in('sol_model.py',s.RECEIPT):manifest['source_hashes'][str(runtime/changed)]=s.digest(runtime/changed)
   s.write(runtime/name,manifest)
  with self.assertRaisesRegex(ValueError,'transport overlay differs'):s.validate(runtime)
 def test_wrong_arm_and_mixed_markers_rejected_before_output(self):
  runtime=self.runtime();args=self.args(runtime,'--benchmark','online-mind2web');args.arm='sol_single_high60'
  with self.assertRaisesRegex(ValueError,'declared GPT BrowserCode arm'):c.prepare(args,'gpt')
  config=s.read(runtime/'runtime-config.json');config['sol_single120_protocol']='foreign';s.write(runtime/'runtime-config.json',config)
  with self.assertRaisesRegex(ValueError,'cannot be combined'):c.prepare(self.args(runtime,'--benchmark','online-mind2web'),'gpt')
  self.assertFalse((self.root/'output').exists())
 def test_missing_shared_policy_pin_rejected(self):
  runtime=self.runtime();manifest=s.read(runtime/'source-manifest.json');manifest['source_hashes'].pop(str(s.POLICY_SOURCE));s.write(runtime/'source-manifest.json',manifest);s.write(runtime/'execution-source-manifest.json',manifest)
  with self.assertRaisesRegex(ValueError,'source pin missing'):s.validate(runtime)
if __name__=='__main__':unittest.main()
