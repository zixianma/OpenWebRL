import concurrent.futures,copy,hashlib,importlib.util,json,tempfile,threading,time,unittest,urllib.request,urllib.error
from pathlib import Path
from unittest.mock import patch
D=Path(__file__).resolve().parent;s=importlib.util.spec_from_file_location('new_gateway',D/'budget_gateway.py');G=importlib.util.module_from_spec(s);s.loader.exec_module(G)
class Tests(unittest.TestCase):
 def service(self,model='gpt-6-luna',calls=18000,usd=10):
  t=tempfile.TemporaryDirectory(prefix='native-new-gateway-',dir='/tmp');self.addCleanup(t.cleanup);r=Path(t.name);rates={'input':.1,'cached_input':.01,'cache_write_input':.125,'output':.5};rates={k:v*(20 if model=='gpt-6-sol'else 1)for k,v in rates.items()};cap={'shard':0,'task_ids':['case'],'model_usd_all_attempts':usd,'model_calls_all_attempts':calls,'external_judge_usd_all_attempts':5,'external_judge_calls_all_attempts':1100,'browser_starts_all_attempts':60,'model_max_inflight':8};(r/'plan.json').write_text(json.dumps({'model':model,'candidate_count':3,'shards':[cap]}));(r/'pricing-snapshot.json').write_text(json.dumps({'model':model,'usd_per_million_tokens':rates}));return G.ShardGateway(r,0,time.time()+60,'unused-key','secret')
 def payload(self,model='gpt-6-luna',text='x'):
  return {'model':model,'input':[{'role':'user','content':[{'type':'input_text','text':text}]}],'reasoning':{'effort':'high'},'max_output_tokens':20000,'tools':[{'type':'function','name':'run_command','parameters':{'type':'object'}}]}
 def provider(self,endpoint,raw,key,timeout):
  p=json.loads(raw)
  if endpoint=='responses/input_tokens':return 200,G._bytes({'object':'response.input_tokens','input_tokens':100})
  return 200,G._bytes({'model':p['model'],'output':[],'usage':{'input_tokens':100,'input_tokens_details':{'cached_tokens':0,'cache_write_tokens':0},'output_tokens':10}})
 def test_frozen_sol_rates_twentyfold_and_exact_model_no_substitution(self):
  l=self.service();s=self.service('gpt-6-sol',usd=100);p=self.payload();q=self.payload('gpt-6-sol');self.assertAlmostEqual(s._bound(q,'model',{},10000)[0],20*l._bound(p,'model',{},10000)[0]);b=json.loads(self.provider('responses',G._bytes(p),'',1)[1]);self.assertAlmostEqual(G._charge(b,'model',s.rates),20*G._charge(b,'model',l.rates))
  with self.assertRaisesRegex(ValueError,'Exact approved'):s.create('case',p)
 def test_whole_group_denial_dispatches_zero_calls(self):
  for kw in [{'calls':7},{'usd':.1}]:
   s=self.service(**kw)
   with patch.object(G,'_forward',side_effect=AssertionError('paid call'))as f,self.assertRaisesRegex(ValueError,'Whole proposal'):s.group_open({'case':'case','decision':1,'candidate_payload':self.payload()})
   self.assertEqual(f.call_count,0);self.assertEqual(G.totals(s.root,0)['model']['calls'],0)
 def test_group_candidates_selector_then_release_and_private_immutable_records(self):
  s=self.service();p=self.payload();g=s.group_open({'case':'case','decision':1,'candidate_payload':p})['group_id']
  with patch.object(G,'_forward',side_effect=self.provider):
   with self.assertRaisesRegex(ValueError,'three settled'):s.group_create({'group_id':g,'role':'selector','payload':dict(p,tools=[])})
   for i in range(3):self.assertTrue(s.group_create({'group_id':g,'role':'candidate','candidate_index':i,'payload':p})['usage_known'])
   s.group_create({'group_id':g,'role':'selector','payload':dict(p,tools=[],tool_choice='none')})
  finish=s.group_finish({'group_id':g,'selected_candidate_index':1});self.assertEqual(finish['released_unused_calls'],4);self.assertEqual(G._group_holds(s._events()),{});self.assertEqual(G.totals(s.root,0)['model']['calls'],4)
  data={'group_id':g,'kind':'decisions/one','record':{'selected_index':1}};receipt=s.group_record(data);self.assertEqual(receipt['record_sha256'],hashlib.sha256(G._canonical(data['record'])).hexdigest())
  with self.assertRaises(FileExistsError):s.group_record(data)
 def test_group_holds_protect_helper_budget_and_candidate_hash(self):
  s=self.service(calls=8);p=self.payload();g=s.group_open({'case':'case','decision':1,'candidate_payload':p})['group_id']
  with self.assertRaisesRegex(ValueError,'including atomic'):s.create('case',dict(p,tools=[]),'helper')
  with self.assertRaisesRegex(ValueError,'differs'):s.group_create({'group_id':g,'role':'candidate_0','payload':dict(p,max_output_tokens=1)})
  s.group_finish({'group_id':g,'selected_candidate_index':None,'error_type':'BeforeDispatchFailure'});self.assertEqual(G._group_holds(s._events()),{})
 def test_real_counter_each_physical_slot_same_group_and_actual_selector_context(self):
  s=self.service();p=self.payload(text='x'*280000);g=s.group_open({'case':'case','decision':1,'candidate_payload':p})['group_id']
  with patch.object(G,'_forward',side_effect=self.provider):
   for i in range(3):s.group_create({'group_id':g,'role':'candidate_'+str(i),'payload':p})
   sel=dict(self.payload(text='y'*290000),tools=[],tool_choice='none');v=s.group_create({'group_id':g,'role':'selector','payload':sel});self.assertTrue(v['input_counter_id'])
  f=s.group_finish({'group_id':g,'selected_candidate_index':0});self.assertEqual(f['released_unused_calls'],0);self.assertEqual(G.totals(s.root,0)['model']['calls'],8)
  es=s._events();self.assertEqual(sum(e.get('stage')=='input_token_count'and e['event']=='reserve'for e in es),4)
  self.assertTrue(all(e['charge_usd']==0 for e in es if e['event']=='finish'and e.get('stage')=='input_token_count'))
 def test_unknown_keeps_reservation_and_halts_no_automatic_retry(self):
  s=self.service();p=self.payload();g=s.group_open({'case':'case','decision':1,'candidate_payload':p})['group_id']
  with patch.object(G,'_forward',side_effect=TimeoutError):v=s.group_create({'group_id':g,'role':'candidate_0','payload':p})
  self.assertFalse(v['usage_known']);s.group_finish({'group_id':g,'selected_candidate_index':None,'error_type':'TransportFailure'});u=G.totals(s.root,0);self.assertEqual(u['model']['unknown'],1);self.assertGreater(u['model']['usd'],0)
  with self.assertRaisesRegex(ValueError,'halted'):s.create('case',p)
 def test_unknown_counter_zero_usd_physical_attempt_retained_and_halt(self):
  s=self.service();p=self.payload(text='x'*280000)
  with patch.object(G,'_forward',return_value=(502,G._bytes({'error':'failed'}))),self.assertRaisesRegex(ValueError,'count unknown'):s.create('case',p)
  u=G.totals(s.root,0);self.assertEqual(u['model']['calls'],1);self.assertEqual(u['model']['unknown'],1);self.assertEqual(u['model']['usd'],0)
 def test_shared_model_eight_inflight_includes_helpers_and_counters(self):
  s=self.service();lock=threading.Lock();active=peak=0
  def forward(*args):
   nonlocal active,peak
   with lock:active+=1;peak=max(peak,active)
   time.sleep(.03)
   try:return self.provider(*args)
   finally:
    with lock:active-=1
  with patch.object(G,'_forward',side_effect=forward),concurrent.futures.ThreadPoolExecutor(max_workers=12)as ex:
   futures=[ex.submit(s.create,'case',dict(self.payload(text='x'*280000 if i%2 else 'x'),tools=[]),'helper')for i in range(12)]
   for f in futures:self.assertTrue(f.result()['usage_known'])
  self.assertLessEqual(peak,8);self.assertGreater(peak,1);self.assertEqual(G.totals(s.root,0)['model']['calls'],18)
 def test_group_secret_rejects_actor_before_any_admission(self):
  s=self.service();server,base=G.start_gateway(s.root,0,time.time()+60,'unused-key',group_secret='host-only');self.addCleanup(server.server_close);self.addCleanup(server.shutdown)
  req=urllib.request.Request(base+'/group/open',G._bytes({'case':'case','decision':1,'candidate_payload':self.payload()}),{'Content-Type':'application/json'},method='POST')
  with self.assertRaises(urllib.error.HTTPError)as e:urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req)
  self.assertEqual(e.exception.code,403);e.exception.close();self.assertEqual(sum(e['event']=='group_open'for e in s._events()),0)
 def test_group_settlement_rejects_unfinished_physical_receipt(self):
  s=self.service();p=self.payload();g=s.group_open({'case':'case','decision':1,'candidate_payload':p})['group_id']
  with G._lock(s.folder):
   G._append(s.folder,{'event':'reserve','id':'orphan','case':'case','stage':'model','pool':'model','reserve_usd':.01,'group_id':g,'role':'candidate_0','time':time.time()})
  with self.assertRaisesRegex(ValueError,'receipt settlement pending'):s.group_finish({'group_id':g,'selected_candidate_index':None,'error_type':'Interrupted'})
  self.assertIn(g,G._group_holds(s._events()))
 def test_provider_true_input_over_tier_no_generation_dispatch(self):
  s=self.service();calls=[]
  def forward(endpoint,*args):
   calls.append(endpoint);return 200,G._bytes({'object':'response.input_tokens','input_tokens':272000})
  with patch.object(G,'_forward',side_effect=forward),self.assertRaisesRegex(ValueError,'price tier'):s.create('case',self.payload(text='x'*280000))
  self.assertEqual(calls,['responses/input_tokens']);self.assertEqual(G.totals(s.root,0)['model']['calls'],1)
 def test_deadline_denial_zero_physical_calls_and_single_arm_group_rejected(self):
  s=self.service();s.deadline=time.time()-1
  with self.assertRaisesRegex(ValueError,'deadline'):s.group_open({'case':'case','decision':1,'candidate_payload':self.payload()})
  self.assertEqual(G.totals(s.root,0)['model']['calls'],0)
  s=self.service();s.plan['arm_id']='sol_high_single'
  with self.assertRaisesRegex(ValueError,'Single arm'):s.group_open({'case':'case','decision':1,'candidate_payload':self.payload()})
 def test_helper_specific_effort_and_outputs_preserved_exact_arm(self):
  s=self.service('gpt-6-sol',usd=100);p=dict(self.payload('gpt-6-sol'),tools=[],reasoning={'effort':'low'},max_output_tokens=1024)
  with patch.object(G,'_forward',side_effect=self.provider):self.assertTrue(s.create('case',p,'helper')['usage_known'])
  self.assertEqual(p['reasoning']['effort'],'low')
if __name__=='__main__':unittest.main(verbosity=2)
