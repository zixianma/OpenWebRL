"""Offline real-adapter tests; fake Responses boundary only, no credentials/API/browser."""
import bootstrap
from pathlib import Path
import ast,copy,json,tempfile,threading,unittest
from unittest.mock import patch
from luna_model import LunaModel,MODES,MODEL,MAX_OUTPUT,PRICING,usage_cost,validate_history
from request_budget import ReservationBudget,SharedHTTPGate
from episode_metrics import receipt_metrics

def native(role,payload,final=False):
 item={'type':'reasoning','id':'reason-'+role,'encrypted_content':'opaque-'+role,'summary':[]}
 if role=='selector':output=[{'type':'message','role':'assistant','content':[{'type':'output_text','text':'{"selected_index":4}'}]}]
 elif final:output=[item,{'type':'message','role':'assistant','content':[{'type':'output_text','text':'completed'}]}]
 else:output=[item,{'type':'function_call','id':'item-'+role,'call_id':'call-'+role,'name':'browser_execute','arguments':json.dumps({'code':'console.log('+json.dumps(role)+')','timeout':30000,'description':'fixed candidate'})}]
 return {'id':'response-'+role,'model':MODEL,'status':'completed','reasoning':copy.deepcopy(payload['reasoning']),'output':output,'usage':{'input_tokens':100,'output_tokens':20,'input_tokens_details':{'cached_tokens':20,'cache_write_tokens':10},'output_tokens_details':{'reasoning_tokens':5}}}
class Tests(unittest.TestCase):
 def setUp(self):self.tmp=tempfile.TemporaryDirectory(prefix='luna-effort-offline-',dir='/tmp');self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
 def model(self,mode,client,max_usd=5):
  pool=ReservationBudget(self.root/mode/'budget',max_usd,10000);return LunaModel(mode,self.root/mode/'episode',client,pool,PRICING,openai_api_key='offline-no-provider',response_mode='run_command_tool',action_field='bash_command',attach_observation_screenshot=False,http_gate=SharedHTTPGate(self.root/'campaign-shared-http',8))
 def test_all5_exact_effort_payloads_and_counts(self):
  for mode,(count,actor,selector)in MODES.items():
   calls=[];lock=threading.Lock()
   def client(payload,role):
    with lock:calls.append((copy.deepcopy(payload),role))
    return native(role,payload)
   m=self.model(mode,client);out=m.query([{'role':'system','content':'frozen task instructions'},{'role':'user','content':'actual browser observation'}]);self.assertEqual(len(calls),count+(count>1));self.assertEqual(m.budget.snapshot()['calls'],len(calls))
   for q,role in calls:
    self.assertEqual(q['model'],'gpt-6-luna');self.assertEqual(q['max_output_tokens'],12288);self.assertFalse(q['store']);self.assertEqual(q['service_tier'],'default');self.assertEqual(q['reasoning'],{'effort':selector if role=='selector'else actor})
    if role!='selector':self.assertEqual(q['include'],['reasoning.encrypted_content']);self.assertFalse(q['parallel_tool_calls']);self.assertEqual(q['tools'][0]['name'],'browser_execute')
   decision=json.loads((m.directory/'decisions/000.json').read_text());self.assertEqual(decision['selected_index'],4 if count==5 else 0);self.assertEqual(out['extra']['actions'][0]['command'],decision['selected_action']['command']);self.assertIn('candidate_4'if count==5 else'candidate_0',out['extra']['actions'][0]['command']);self.assertEqual(decision['selection_effort'],selector)
 def test_n5_selected_encrypted_history_only_and_real_result_link(self):
  calls=[]
  def client(payload,role):calls.append((copy.deepcopy(payload),role));return native(role,payload)
  m=self.model('luna_n5_low',client);messages=[{'role':'system','content':'frozen'},{'role':'user','content':'real observation'}];out=m.query(messages);messages+=[out,{'role':'user','content':'actual selected command output','extra':{'tool_call_id':out['extra']['tool_call_id']}}];m.query(messages)
  q=next(q for q,role in calls[6:]if role=='candidate_0');replayed=[x for x in q['input']if x.get('type')=='reasoning'];self.assertEqual([x['encrypted_content']for x in replayed],['opaque-candidate_4']);feedback=[x for x in q['input']if x.get('type')=='function_call_output'];self.assertEqual(feedback,[{'type':'function_call_output','call_id':'call-candidate_4','output':'actual selected command output'}]);selector=next(q for q,role in calls[6:]if role=='selector');self.assertNotIn('encrypted_content',json.dumps(selector));self.assertNotIn('opaque-candidate_',json.dumps(selector))
 def test_unanswered_or_foreign_history_no_synthetic_execution(self):
  calls=[];m=self.model('luna_single_medium',lambda q,r:(calls.append(1),native(r,q))[1]);out=m.query([{'role':'user','content':'observation'}]);before=len(calls)
  with self.assertRaisesRegex(ValueError,'absent'):m.query([{'role':'user','content':'observation'},out])
  with self.assertRaisesRegex(ValueError,'bind'):m.query([{'role':'user','content':'observation'},out,{'role':'user','content':'fake','extra':{'tool_call_id':'foreign'}}])
  self.assertEqual(len(calls),before)
 def test_role_effort_and_output_mismatch_before_any_admission(self):
  m=self.model('luna_n5_medium',lambda*q:None)
  q={'model':MODEL,'reasoning':{'effort':'low'},'max_output_tokens':MAX_OUTPUT,'store':False,'service_tier':'default','input':[]}
  with self.assertRaises(ValueError):m.physical(q,'candidate_0')
  self.assertEqual(m.budget.snapshot()['calls'],0)
  with self.assertRaises(ValueError):LunaModel('luna_single_xhigh',self.root/'bad',lambda*q:None,m.budget,PRICING,max_output_tokens=20000)
 def test_unknown_usage_full_reserve_retained_no_automatic_retry(self):
  calls=[]
  def fail(q,role):calls.append(role);raise RuntimeError('offline provider rejection')
  m=self.model('luna_single_low',fail)
  with self.assertRaises(RuntimeError):m.query([{'role':'user','content':'observation'}])
  state=m.budget.snapshot();self.assertEqual(calls,['candidate_0']);self.assertEqual(state['unknown_calls'],1);self.assertEqual(state['reserved_calls'],0);r=next(iter(state['requests'].values()));self.assertEqual(r['charged_or_reserved_usd'],r['reserved_usd']);self.assertTrue(m.budget.halted());metrics=receipt_metrics(m.directory,{});self.assertEqual(metrics['model_excluding_judge']['unknown_usage_calls'],1);self.assertEqual(metrics['metric_errors'],[])
 def test_known_incomplete_usage_stays_charged_and_invalid(self):
  def client(q,role):d=native(role,q);d['status']='incomplete';return d
  m=self.model('luna_single_xhigh',client)
  with self.assertRaisesRegex(ValueError,'incomplete'):m.query([{'role':'user','content':'observation'}])
  s=m.budget.snapshot();self.assertEqual(s['calls'],1);self.assertEqual(s['unknown_calls'],0);self.assertEqual(s['reserved_calls'],0);self.assertGreater(s['charged_or_reserved_usd'],0);self.assertFalse((m.directory/'decisions/000.json').exists())
 def test_campaign_shared8_across_two_episode_models(self):
  import concurrent.futures,time
  active=0;peak=0;lock=threading.Lock()
  def client(q,role):
   nonlocal active,peak
   with lock:active+=1;peak=max(peak,active)
   time.sleep(.04)
   with lock:active-=1
   return native(role,q)
  a=self.model('luna_n5_medium',client);b=self.model('luna_n5_low',client)
  self.assertEqual(a.http_gate.root,b.http_gate.root)
  with concurrent.futures.ThreadPoolExecutor(max_workers=2)as pool:
   fs=[pool.submit(m.query,[{'role':'user','content':'real observation'}])for m in(a,b)]
   for f in fs:f.result()
  gate=json.loads((a.http_gate.root/'state.json').read_text());self.assertLessEqual(peak,8);self.assertLessEqual(gate['peak'],8);self.assertEqual(gate['active'],{});self.assertEqual(a.budget.snapshot()['calls']+b.budget.snapshot()['calls'],12)
 def test_native_count_stable_id_cached_and_allcall_usage(self):
  calls=[]
  def client(q,role):
   calls.append(role)
   if role=='input_token_count':return {'object':'response.input_tokens','input_tokens':123}
   return native(role,q,final=True)
  m=self.model('luna_single_medium',client)
  with patch('luna_model.response_input_bound',return_value=200001):
   m.query([{'role':'user','content':'same fixed observation'}]);m.query([{'role':'user','content':'same fixed observation'}])
  self.assertEqual(calls,['input_token_count','candidate_0','candidate_0']);self.assertEqual(m.budget.snapshot()['calls'],3);metrics=receipt_metrics(m.directory,{});self.assertEqual(metrics['input_token_count_requests'],1);self.assertEqual(metrics['model_budget_calls_including_token_counts'],3);self.assertEqual(metrics['model_excluding_judge']['calls'],2);self.assertEqual(metrics['metric_errors'],[])
 def test_pricing_unknown_cache_is_upper_bound_and_bad_usage_rejected(self):
  self.assertEqual(usage_cost({'input_tokens':100,'output_tokens':20})['upper_usd_decimal'],'0.0000225');known=usage_cost({'input_tokens':100,'output_tokens':20,'input_tokens_details':{'cached_tokens':20,'cache_write_tokens':10}});self.assertEqual(known['upper_usd_decimal'],'0.00001845')
  for u in [None,{}, {'input_tokens':True,'output_tokens':1},{'input_tokens':100,'output_tokens':2,'input_tokens_details':{'cached_tokens':90,'cache_write_tokens':20}}]:
   with self.assertRaises(ValueError):usage_cost(u)
 def test_external_judge_excluded_from_model_cost_and_time(self):
  m=self.model('luna_single_low',lambda q,r:native(r,q,True));m.query([{'role':'user','content':'observation'}]);(m.directory/'requests/judge-fixture.json').write_text(json.dumps({'role':'external_judge','status':'received','seconds':1000,'reserved_usd':2,'usage':{'prompt_tokens':100,'completion_tokens':10},'cost':{'upper_usd':1}}));d=receipt_metrics(m.directory,{});self.assertEqual(d['by_role']['judge']['calls'],1);self.assertLess(d['model_excluding_judge']['charged_or_reserved_usd'],.01);self.assertLess(d['model_excluding_judge']['summed_request_seconds'],1000)
 def test_worker_current_prior_valid_and_invalid_preserved(self):
  import worker
  logical=self.root/'logical';logical.mkdir();cfg={'arm':'luna_single_low'};task={'task_id':'task'}
  for valid in(True,False):
   p=logical/str(valid);p.mkdir();path=p/'result.json';path.write_text(json.dumps({'task_id':'task','mode':cfg['arm'],'harness':'browsercode','judge_protocol_version':worker.CONFIG['judge_protocol']['version'],'valid':valid,'reward':1 if valid else None}));before=path.read_bytes();r=worker.preserve_or_recover_prior(p,logical,task,cfg);self.assertFalse(r['actor_rerun']);self.assertFalse(r['judge_rerun']);self.assertEqual(path.read_bytes(),before)
  tree=ast.parse(Path(worker.__file__).read_text());self.assertNotIn('judge_recovery',Path(worker.__file__).read_text());self.assertTrue(any(isinstance(n,ast.Call)and ast.unparse(n.func)=='verify_enrollment'for n in ast.walk(tree)))
if __name__=='__main__':unittest.main()
