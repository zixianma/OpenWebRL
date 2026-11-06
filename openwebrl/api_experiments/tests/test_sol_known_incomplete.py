"""Generated Sol SDK boundary and real reservation ledger, with no paid calls."""
from copy import deepcopy
from decimal import Decimal
import ast,importlib.util,json,sys,types,unittest
from pathlib import Path
from unittest.mock import patch

API=Path(__file__).resolve().parents[1];REPRO=API/'reproduce'
def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value
fixture=module('sol_policy_fixture',API/'tests/test_browsercode_output_sweep.py')
sol60tests=module('sol_policy_actor_loader',API/'tests/test_browsercode_sol60.py')
fixture=sol60tests.fixture
policy=module('sol_policy_under_test',REPRO/'sol_known_incomplete.py')

class KnownIncomplete(unittest.TestCase):
 def setUp(self):
  self.base=fixture.SweepTests();self.base.setUp();self.addCleanup(self.base.doCleanups);self.root=self.base.root;self.serial=0
 def runtime(self,horizon,install_hooks=True):
  self.serial+=1;runtime=self.base.runtime('runtime-'+str(self.serial));judge=API/'browsercode/judges/terra.py';(runtime/'common_judge.py').write_bytes(judge.read_bytes())
  s=module('sol_policy_prepare_'+str(horizon),REPRO/('sol_single%d.py'%horizon));manifest=s.read(runtime/'source-manifest.json');manifest['source_hashes'][str(runtime/'common_judge.py')]=s.digest(runtime/'common_judge.py');s.write(runtime/'source-manifest.json',manifest);s.write(runtime/'execution-source-manifest.json',manifest);s.configure(runtime)
  actor=sol60tests.Sol60.load_actor(self,runtime);budgets=types.SimpleNamespace(__file__=str(runtime/'request_budget.py'),SharedHTTPGate=fixture.budget_module.SharedHTTPGate,ReservationBudget=fixture.budget_module.ReservationBudget);judge_module=types.SimpleNamespace(__file__=str(runtime/'common_judge.py'),RUNTIME_HTTP_GATE=None);worker=types.SimpleNamespace(__file__=str(runtime/'worker.py'),PairedModel=actor.SolModel)
  if not install_hooks:return s,runtime,actor,worker,None
  with patch.dict(sys.modules,{'sol_model':actor,'common_judge':judge_module,'request_budget':budgets}):gate=s.install_runtime_hooks(worker,self.root/('gate-'+str(self.serial)))
  return s,runtime,actor,worker,gate
 def request(self,horizon=120,change=None,exception=None,count=False,finish_error=False):
  s,runtime,actor,worker,gate=self.runtime(horizon);effort='xhigh'if horizon==30 else'high';calls=[]
  native={'model':'gpt-6-sol','status':'incomplete','incomplete_details':{'reason':'max_output_tokens'},'reasoning':{'effort':effort},'output':[],'usage':{'input_tokens':100,'output_tokens':12288,'total_tokens':12388,'input_tokens_details':{'cached_tokens':0,'cache_write_tokens':0}}}
  if change:change(native)
  def create(**payload):
   calls.append(deepcopy(payload))
   if exception:raise exception
   return types.SimpleNamespace(model_dump=lambda mode:deepcopy(native))
  client=types.SimpleNamespace(max_retries=0,responses=types.SimpleNamespace(create=create,input_tokens=types.SimpleNamespace(count=create)));pool=fixture.budget_module.ReservationBudget(self.root/('pool-'+str(self.serial)),'5',10)
  model=worker.PairedModel(s.ARM,self.root/('task-'+str(self.serial)),client,pool,actor.PRICING,http_gate=gate,max_output_tokens=12288,reasoning_effort=effort)
  payload={'model':'gpt-6-sol','reasoning':{'effort':effort},'max_output_tokens':12288,'store':False,'service_tier':'default','input':[]}
  error=None
  try:
   if finish_error:
    with patch.object(pool,'finish',side_effect=ValueError('fixture accounting refusal')):model._request(payload,'candidate_0',1000,count=count)
   else:model._request(payload,'candidate_0',1000,count=count)
  except BaseException as exc:error=exc
  rows=[json.loads(p.read_text())for p in(model.directory/'requests').glob('*.json')];self.assertEqual(len(rows),1);self.assertEqual(len(calls),1)
  return pool,rows[0],error,calls,model,worker,actor,s,gate,client
 def test_known_max_output_invalid_once_then_new_task_can_dispatch_all_protocols(self):
  for horizon in(30,60,120):
   with self.subTest(horizon=horizon):
    pool,row,error,calls,model,worker,actor,s,gate,client=self.request(horizon)
    self.assertIs(type(error),ValueError);self.assertEqual(str(error),policy.ERROR);self.assertFalse(pool.halted());self.assertEqual(row['status'],'received');self.assertTrue(row['task_local_known_incomplete_policy']['no_retry']);self.assertEqual(pool.snapshot()['calls'],1);self.assertEqual(pool.snapshot()['requests'][row['physical_id']]['status'],'known')
    # A different task gets its own model/directory and physical ID; the failed
    # task is never retried and its usage row is retained exactly.
    before=deepcopy(pool.snapshot()['requests'][row['physical_id']]);client.responses.create=lambda **payload:types.SimpleNamespace(model_dump=lambda mode:{'model':'gpt-6-sol','status':'completed','reasoning':payload['reasoning'],'output':[{'type':'message','role':'assistant','content':[{'type':'output_text','text':'done'}]}],'usage':{'input_tokens':100,'output_tokens':20,'input_tokens_details':{'cached_tokens':0,'cache_write_tokens':0}}})
    next_model=worker.PairedModel(s.ARM,self.root/('next-task-'+str(self.serial)),client,pool,actor.PRICING,http_gate=gate,max_output_tokens=12288,reasoning_effort='xhigh'if horizon==30 else'high');next_model.query([{'role':'user','content':'next untouched task'}])
    self.assertEqual(pool.snapshot()['calls'],2);self.assertEqual(pool.snapshot()['requests'][row['physical_id']],before);self.assertFalse(pool.halted());self.assertEqual(len(calls),1)
 def test_transport_unknown_still_halts_and_retains_entire_reserve(self):
  pool,row,error,*_=self.request(exception=RuntimeError('fixture transport502'));self.assertIs(type(error),RuntimeError);self.assertTrue(pool.halted());self.assertEqual(row['status'],'usage_unknown');self.assertEqual(pool.snapshot()['requests'][row['physical_id']]['charged_or_reserved_usd'],row['reserved_usd_decimal'])
 def test_accounting_refusal_still_halts_full_reserve(self):
  pool,row,error,*_=self.request(finish_error=True);self.assertTrue(pool.halted());self.assertEqual(row['status'],'usage_unknown');self.assertEqual(pool.snapshot()['requests'][row['physical_id']]['charged_or_reserved_usd'],row['reserved_usd_decimal'])
 def test_other_incomplete_and_failure_conditions_still_halt(self):
  changes=[lambda n:n.update(status='failed'),lambda n:n.update(incomplete_details={'reason':'content_filter'}),lambda n:n.update(model='other-model'),lambda n:n['reasoning'].update(effort='xhigh'),lambda n:n['usage'].update(output_tokens=12287),lambda n:n['usage'].update(output_tokens='malformed')]
  for change in changes:
   with self.subTest(change=change):
    pool,row,error,*_=self.request(change=change);self.assertIsNotNone(error);self.assertTrue(pool.halted());self.assertNotIn('task_local_known_incomplete_policy',row)
 def test_counter_failure_is_not_task_local_even_at_zero_price(self):
  pool,row,error,*_=self.request(count=True);self.assertTrue(pool.halted());self.assertEqual(row['reserved_usd_decimal'],'0');self.assertEqual(pool.snapshot()['calls'],1);self.assertNotIn('task_local_known_incomplete_policy',row)
 def test_over_reservation_usage_still_halts_without_releasing_known_charge(self):
  pool,row,error,*_=self.request(change=lambda n:n['usage'].update(output_tokens=999999));self.assertTrue(pool.halted());self.assertEqual(row['status'],'received_over_reservation');self.assertEqual(pool.snapshot()['requests'][row['physical_id']]['status'],'known');self.assertGreater(Decimal(pool.snapshot()['requests'][row['physical_id']]['charged_or_reserved_usd']),Decimal(row['reserved_usd_decimal']))
 def test_preexisting_halt_and_unsettled_or_mismatched_ledger_never_qualify(self):
  pool,row,error,*_=self.request();payload=row['request'];ident=row['physical_id'];native=row['response'];known=deepcopy(pool.snapshot())
  class FakeBudget:
   def __init__(self,snapshot,halt=False):self.state=snapshot;self.halt=halt
   def snapshot(self):return self.state
   def halted(self):return self.halt
  variants=[]
  for mutate in[lambda s:s['requests'][ident].update(status='reserved'),lambda s:s['requests'][ident].update(actual_usd='0'),lambda s:s['requests'][ident]['metadata'].update(request_sha256='wrong')]:
   state=deepcopy(known);mutate(state);variants.append(FakeBudget(state))
  variants.append(FakeBudget(known,True))
  for budget in variants:self.assertFalse(policy.known_task_local(error,native,row,True,True,False,budget,ident,payload,effort='high'))
 def test_changed_policy_or_generated_source_is_rejected(self):
  s,runtime,actor,worker,gate=self.runtime(30);receipt=s.read(runtime/s.RECEIPT);receipt['known_incomplete_policy']['no_retry']=False;s.write(runtime/s.RECEIPT,receipt)
  with self.assertRaises(ValueError):s.validate(runtime)
  with self.assertRaisesRegex(ValueError,'validated generated Sol source'):policy.derive(actor.SolModel._request,runtime/'sol_model.py','wrong','xhigh')
 def test_derived_function_preserves_every_statement_except_one_halt(self):
  import builtins
  s,runtime,actor,worker,gate=self.runtime(120);original=ast.parse((runtime/'sol_model.py').read_text());method=next(n for n in ast.walk(original)if isinstance(n,ast.FunctionDef)and n.name=='_request');captured=[]
  def compile_ast(tree,*args,**kwargs):captured.append(deepcopy(tree));return builtins.compile(tree,*args,**kwargs)
  with patch.object(policy,'compile',side_effect=compile_ast,create=True):policy.derive(actor.SolModel._request,runtime/'sol_model.py',s.digest(runtime/'sol_model.py'),'high')
  derived=captured[0].body[0];changed=0
  for handler in ast.walk(derived):
   if isinstance(handler,ast.ExceptHandler):
    for i,statement in enumerate(handler.body):
     if isinstance(statement,ast.If)and isinstance(statement.test,ast.Call)and isinstance(statement.test.func,ast.Name)and statement.test.func.id=='_known_task_local':
      self.assertEqual(len(statement.orelse),1);handler.body[i]=statement.orelse[0];changed+=1
  self.assertEqual(changed,1);self.assertEqual(ast.dump(derived,include_attributes=False),ast.dump(method,include_attributes=False))

 def test_mismatched_policy_rejected_before_helper_module_loading(self):
  for horizon in(30,60,120):
   with self.subTest(horizon=horizon):
    s,runtime,actor,worker,_=self.runtime(horizon,install_hooks=False);receipt=s.read(runtime/s.RECEIPT);receipt['known_incomplete_policy']['source_sha256']='mismatched';s.write(runtime/s.RECEIPT,receipt)
    budgets=types.SimpleNamespace(__file__=str(runtime/'request_budget.py'),SharedHTTPGate=fixture.budget_module.SharedHTTPGate,ReservationBudget=fixture.budget_module.ReservationBudget);judge=types.SimpleNamespace(__file__=str(runtime/'common_judge.py'),RUNTIME_HTTP_GATE=None)
    with patch.dict(sys.modules,{'sol_model':actor,'common_judge':judge,'request_budget':budgets}),patch.object(s.importlib.util,'spec_from_file_location')as spec_loader,patch.object(s.importlib.util,'module_from_spec')as module_loader:
     with self.assertRaisesRegex(ValueError,'known-incomplete source policy differs'):s.install_runtime_hooks(worker,self.root/('untrusted-gate-'+str(self.serial)))
     spec_loader.assert_not_called();module_loader.assert_not_called()
if __name__=='__main__':unittest.main()
