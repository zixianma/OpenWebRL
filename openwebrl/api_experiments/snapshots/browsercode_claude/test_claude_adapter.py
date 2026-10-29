"""Offline author tests: actual native parser/history/budget/process gate paths.
No provider, browser, scheduler, or credential operations.
"""
from pathlib import Path
import sys,json,tempfile,unittest,copy,time,multiprocessing,threading,os,math
sys.path.insert(0,str(Path(__file__).parent));sys.path.insert(0,'/gpfs/scrubbed/zixianma/openwebrl-runtime/evaluations/serial-vs-parallel-reasoning-20261013/cuawright30-preparation-v1/upstream/src')
from claude_budget import ReservationBudget,SharedHTTPGate,atomic_json
from claude_model import ClaudeModel,MODES,PRICING,parse_response,native_messages,claude_usage_cost,strict_json,input_upper_bound,text_content,selector_observable_content,TOOL

def response(model='claude-sonnet-5-5',ident='tool-1',text=None):
 content=[{'type':'thinking','thinking':'private thought '+ident,'signature':'signed-'+ident}]
 content.append({'type':'text','text':text}if text is not None else{'type':'tool_use','id':ident,'name':'browser_execute','input':{'code':'console.log('+json.dumps(ident)+')','timeout':1000,'description':'inspect'}})
 return {'id':'msg-'+ident,'type':'message','role':'assistant','model':model,'stop_reason':'end_turn'if text is not None else'tool_use','content':content,'usage':{'input_tokens':100,'output_tokens':20,'cache_creation_input_tokens':0,'cache_read_input_tokens':0}}

def gate_process(root,ident,ready):
 gate=SharedHTTPGate(root,limit=3)
 with gate.slot(ident):ready.put(('entered',ident,time.time()));time.sleep(.06)
 ready.put(('exited',ident,time.time()))

class AdapterTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory(prefix='claude-offline-');self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
 def budget(self,usd=10,calls=100):return ReservationBudget(self.root/'budget',usd,calls)
 def model(self,client,mode='sonnet_single',budget=None):
  return ClaudeModel(mode,self.root/'episode',client,budget or self.budget(),model_name=MODES[mode][0],reasoning_effort='high',max_output_tokens=12288,response_mode='run_command_tool',action_field='bash_command',openai_api_key='offline')
 def test_native_tool_and_text_final_parser(self):
  self.assertFalse(parse_response(response(),'claude-sonnet-5-5')['done']);self.assertTrue(parse_response(response(text='done'),'claude-sonnet-5-5')['done'])
 def test_tool_plus_explanation_preserved_not_final(self):
  r=response();r['content'].insert(1,{'type':'text','text':'inspect now'});p=parse_response(r,'claude-sonnet-5-5');self.assertFalse(p['done']);self.assertEqual(p['display_text'],'inspect now')
 def test_native_stop_reasons_and_model_fail_closed(self):
  for field,value in [('stop_reason','max_tokens'),('stop_reason','refusal'),('stop_reason','pause_turn'),('model','foreign')]:
   r=response();r[field]=value
   with self.subTest(value=value),self.assertRaises(ValueError):parse_response(r,'claude-sonnet-5-5')
 def test_unsigned_thinking_and_thinking_only_rejected(self):
  for r in [dict(response(),content=[{'type':'thinking','thinking':'x'}]),dict(response(),content=[{'type':'thinking','thinking':'x','signature':'s'}],stop_reason='end_turn')]:
   with self.assertRaises(ValueError):parse_response(r,'claude-sonnet-5-5')
 def test_bad_tools_and_timeout_bool_rejected(self):
  for kind in ['two','foreign','bool','extra']:
   r=response()
   if kind=='two':r['content'].append(copy.deepcopy(r['content'][-1]))
   if kind=='foreign':r['content'][-1]['name']='bash'
   if kind=='bool':r['content'][-1]['input']['timeout']=True
   if kind=='extra':r['content'][-1]['input']['extra']='x'
   with self.subTest(kind=kind),self.assertRaises(ValueError):parse_response(r,'claude-sonnet-5-5')
 def test_selector_exact_index_and_duplicate_json(self):
  for text in ['{"selected_index":true}','{"selected_index":3}','{"selected_index":0,"selected_index":1}','{"selected_index":1,"extra":0}','notjson']:
   with self.subTest(text=text),self.assertRaises(ValueError):parse_response(response(text=text),'claude-sonnet-5-5',3)
  self.assertEqual(parse_response(response(text='{"selected_index":2}'),'claude-sonnet-5-5',3),2)
 def test_signed_history_native_toolresult_replayed_verbatim(self):
  r=response();m=[{'role':'system','content':'system'},{'role':'user','content':'task'},{'role':'assistant','content':'','extra':{'claude_content':r['content']}},{'role':'user','content':'actual browser output','extra':{'tool_call_id':'tool-1'}}];s,h=native_messages(m);self.assertEqual(h[1]['content'],r['content']);self.assertEqual(h[2]['content'][0]['type'],'tool_result');self.assertEqual(h[2]['content'][0]['tool_use_id'],'tool-1');self.assertEqual(m[2]['extra']['claude_content'],r['content'])
 def test_missing_or_foreign_toolresult_no_synthesis(self):
  r=response();base=[{'role':'user','content':'task'},{'role':'assistant','extra':{'claude_content':r['content']}}]
  for m in [base,base+[{'role':'user','content':'result','extra':{'tool_call_id':'foreign'}}]]:
   with self.assertRaises(ValueError):native_messages(m)
 def test_unknown_assistant_history_and_image_rejected(self):
  for m in [[{'role':'user','content':'task'},{'role':'assistant','content':'fake action'},{'role':'user','content':'observation'}],[{'role':'user','content':[{'type':'input_image','image_url':'x'}]}]]:
   with self.assertRaises(ValueError):native_messages(m)
 def test_usage_cache_all_categories_and_thinking(self):
  u={'input_tokens':10,'output_tokens':20,'cache_read_input_tokens':30,'cache_creation_input_tokens':90,'cache_creation':{'ephemeral_5m_input_tokens':40,'ephemeral_1h_input_tokens':50}};v=claude_usage_cost(u,PRICING['claude-sonnet-5-5']);self.assertAlmostEqual(v['upper_usd'],(10*2+20*10+30*.1+40*2.5+50*4)/1e6);self.assertTrue(v['thinking_tokens_included_in_output']);self.assertFalse(v['thinking_token_count_observed'])
 def test_cache_creation_without_split_conservative_1hour(self):
  v=claude_usage_cost({'input_tokens':0,'output_tokens':0,'cache_creation_input_tokens':100},PRICING['claude-sonnet-5-5']);self.assertEqual(v['upper_usd'],.0004)
 def test_missing_bad_usage_not_zero(self):
  for u in [None,{},dict(input_tokens=True,output_tokens=0),dict(input_tokens=0,output_tokens=-1),dict(input_tokens=0,output_tokens=0,cache_creation_input_tokens=2,cache_creation={'ephemeral_5m_input_tokens':1})]:
   with self.assertRaises(ValueError):claude_usage_cost(u,PRICING['claude-sonnet-5-5'])
 def test_duplicate_id_settlement_matches_id_not_amount(self):
  b=self.budget();b.reserve('a',1);b.reserve('b',1);b.finish('b',.25,{});s=b.snapshot();self.assertEqual(s['requests']['a']['status'],'reserved');self.assertEqual(s['requests']['b']['status'],'known');self.assertEqual(s['charged_or_reserved_usd'],1.25)
  with self.assertRaises(ValueError):b.reserve('a',1)
  with self.assertRaises(ValueError):b.finish('b',0,{})
 def test_unknown_fullreserve_retained_and_halts(self):
  b=self.budget();b.reserve('a',1);b.unknown('a','HTTP500');self.assertEqual(b.snapshot()['charged_or_reserved_usd'],1);self.assertEqual(b.snapshot()['unknown_calls'],1)
  with self.assertRaises(ValueError):b.reserve('b',1)
  with self.assertRaises(ValueError):b.finish('a',0,{})
 def test_caps_and_cap_change_fail_closed(self):
  b=self.budget(1,1);b.reserve('a',1)
  with self.assertRaises(ValueError):b.reserve('b',0)
  with self.assertRaises(ValueError):ReservationBudget(b.root,2,1).snapshot()
 def test_over_reserve_actual_charge_retained(self):
  b=self.budget();b.reserve('a',.1)
  with self.assertRaises(ValueError):b.finish('a',.2,{})
  self.assertEqual(b.snapshot()['charged_or_reserved_usd'],.2);self.assertTrue(b.halted())
 def test_shared_gate_process_capacity_and_finally_release(self):
  ctx=multiprocessing.get_context('fork');q=ctx.Queue();jobs=[ctx.Process(target=gate_process,args=(str(self.root/'gate'),str(i),q))for i in range(9)]
  for p in jobs:p.start()
  for p in jobs:p.join(10);self.assertEqual(p.exitcode,0)
  d=json.loads((self.root/'gate/state.json').read_text());self.assertEqual(d['peak'],3);self.assertEqual(d['active'],{})
 def test_gate_exception_releases_but_orphan_never_autoclears(self):
  g=SharedHTTPGate(self.root/'gate',1,0)
  with self.assertRaises(RuntimeError):
   with g.slot('a'):raise RuntimeError('error')
  d=json.loads((self.root/'gate/state.json').read_text());d['active']['dead']={'pid':99999999};atomic_json(self.root/'gate/state.json',d)
  with self.assertRaises(TimeoutError):
   with g.slot('b'):self.fail('must not reclaim orphan')
 def test_actual_query_single_payload_format_and_signed_return(self):
  calls=[]
  def client(p):calls.append(copy.deepcopy(p));return response()
  m=self.model(client);r=m.query([{'role':'system','content':'policy'},{'role':'user','content':'task'}]);self.assertEqual(len(calls),1);p=calls[0];self.assertEqual(p['thinking'],{'type':'adaptive'});self.assertEqual(p['output_config'],{'effort':'high'});self.assertEqual(p['max_tokens'],12288);self.assertEqual(p['tool_choice'],{'type':'auto','disable_parallel_tool_use':True});self.assertEqual(r['extra']['claude_content'],response()['content']);self.assertEqual(r['extra']['tool_call_id'],'tool-1');self.assertEqual(m.budget.snapshot()['calls'],1)
 def test_n3_selection_never_executes_unselected_or_replays_private_thought(self):
  seen=[];lock=threading.Lock()
  def client(p):
   with lock:i=len(seen);seen.append(copy.deepcopy(p))
   return response(ident=str(i),text='{"selected_index":1}')if'format'in p['output_config']else response(ident=str(i))
  m=self.model(client,'sonnet_n3');r=m.query([{'role':'user','content':'task'}]);self.assertEqual(m.budget.snapshot()['calls'],4);self.assertEqual(r['extra']['selected_candidate_index'],1);record=json.loads((m.directory/'decisions/000.json').read_text());selected_id=record['candidate_responses'][1]['content'][-1]['id'];self.assertEqual(r['extra']['tool_call_id'],selected_id);selector=seen[-1];self.assertNotIn('private thought',json.dumps(selector));self.assertNotIn('signed-',json.dumps(selector));self.assertEqual(len(r['extra']['actions']),1)
  # Next candidate turn replays only selected content/signature and actual result.
  system,history=native_messages([{'role':'user','content':'task'},r,{'role':'user','content':'real result','extra':{'tool_call_id':selected_id}}]);s=json.dumps(history);self.assertIn('signed-'+selected_id,s)
  for candidate in record['candidate_responses']:
   ident=candidate['content'][-1]['id']
   if ident!=selected_id:self.assertNotIn('signed-'+ident,s)
 def test_physical_failure_no_retry_and_fullunknown(self):
  count=[]
  def fail(p):count.append(1);raise RuntimeError('fake500')
  m=self.model(fail)
  with self.assertRaises(RuntimeError):m.query([{'role':'user','content':'task'}])
  self.assertEqual(len(count),1);self.assertEqual(m.budget.snapshot()['unknown_calls'],1);self.assertGreater(m.budget.snapshot()['charged_or_reserved_usd'],0);self.assertEqual(m.turn,0)
 def test_known_malformed_charge_retained_no_decision(self):
  r=response();r['stop_reason']='max_tokens';m=self.model(lambda p:r)
  with self.assertRaises(ValueError):m.query([{'role':'user','content':'task'}])
  self.assertEqual(m.budget.snapshot()['requests'].popitem()[1]['status'],'known');self.assertFalse((m.directory/'decisions/000.json').exists());self.assertTrue(m.budget.halted())
 def test_explicit_protocol_and_sdk_retry_guards(self):
  for kw in [{'model_name':'foreign'},{'reasoning_effort':'medium'},{'max_output_tokens':100}]:
   with self.assertRaises(ValueError):ClaudeModel('sonnet_single',self.root/'x',lambda p:response(),self.budget(),**kw)
  class Retry: max_retries=2
  with self.assertRaises(ValueError):self.model(Retry())
 def test_30decision_cap_before_any_http(self):
  m=self.model(lambda p:self.fail('HTTPmustnotrun'));m.turn=30
  with self.assertRaises(ValueError):m.query([{'role':'user','content':'task'}])
 def test_real_basemodel_image_attachment_native_actor_selector_preserved(self):
  from PIL import Image
  image=self.root/'capture.png';Image.new('RGB',(1280,1800),(20,40,80)).save(image)
  m=self.model(lambda p:response())
  selected=m.query([{'role':'user','content':'task'}])
  # Execute the REAL inherited observation formatter, which attaches explicit
  # images even though automatic screenshots are disabled.
  observation=m.format_observation_messages(selected,[{'observation':{'success':True,'url':'https://example.invalid','title':'fixture','exception':None,'console_output':'','aria_snapshot':'','screenshot_path':None,'image_attachments':[{'path':str(image),'media_type':'image/png'}]},'output':'output'}])
  messages=[{'role':'user','content':'task'},selected,*observation];_,native=native_messages(messages)
  blocks=native[-1]['content'][0]['content'];self.assertEqual(blocks[-1]['type'],'image');source=blocks[-1]['source'];self.assertEqual(source['media_type'],'image/png')
  payload=m.payload([],native);bound,images=input_upper_bound(payload);self.assertEqual(images,[{'width':1280,'height':1800,'token_bound':32768}]);self.assertGreaterEqual(bound,32768)
  observable=selector_observable_content(messages,[{'command':'one'},{'command':'two'}]);self.assertEqual([b for b in observable if b['type']=='image'],[blocks[-1]]);text=' '.join(b['text']for b in observable if b['type']=='text');self.assertIn('actual_selected_assistant_actions',text);self.assertIn('console.log',text);self.assertNotIn(source['data'],text);self.assertNotIn('private thought',text);self.assertNotIn('signed-',text)
 def test_base64_images_not_treated_as_text_token_bound(self):
  from PIL import Image
  from cuawright.webwright.models.base import image_part_from_path
  image=self.root/'random.png';raw=os.urandom(1280*1800*3);Image.frombytes('RGB',(1280,1800),raw).save(image)
  # Lossless random PNG >5MiB intentionally rejects before any HTTP rather
  # than silently resizing/dropping the screenshot.
  with self.assertRaises(ValueError):text_content([image_part_from_path(image)])
  image=self.root/'image.jpg';Image.frombytes('RGB',(1280,1800),raw).save(image,quality=85)
  block=text_content([image_part_from_path(image)])[0];bound,images=input_upper_bound({'messages':[{'role':'user','content':[block]}]});self.assertEqual(images[0]['token_bound'],32768);self.assertLess(bound,len(block['source']['data']))
 def test_image_url_and_malformed_unknown_remote_rejected(self):
  for value in ['https://remote.invalid/image.png','data:image/png;base64,not-valid','data:video/mp4;base64,YQ==']:
   with self.assertRaises(ValueError):text_content([{'type':'input_image','image_url':value}])
 def test_tool_contract_exact_preserved_browsercode_schema(self):
  prior=Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/evaluations/serial-vs-parallel-reasoning-20261013/sol6-single-high-vs-luna-n3-n5-browsercode300-20261018-v1/runtime-config.json');tool=json.loads(prior.read_text())['native_tool_contract'];self.assertEqual(TOOL,{'name':tool['name'],'description':tool['description'],'strict':tool['strict'],'input_schema':tool['parameters']})
 def test_two_real_n5_models_share_default_eight_http_slots(self):
  from concurrent.futures import ThreadPoolExecutor
  import uuid
  b=self.budget(100,100)
  def client(payload):
   time.sleep(.05)
   return response(ident=uuid.uuid4().hex,text='{"selected_index":0}')if'format'in payload['output_config']else response(ident=uuid.uuid4().hex)
  models=[ClaudeModel('sonnet_n5',self.root/('episode'+str(i)),client,b,model_name='claude-sonnet-5-5',reasoning_effort='high',max_output_tokens=12288,response_mode='run_command_tool',action_field='bash_command',openai_api_key='offline')for i in range(2)]
  with ThreadPoolExecutor(max_workers=2)as pool:answers=list(pool.map(lambda m:m.query([{'role':'user','content':'task'}]),models))
  self.assertEqual(b.snapshot()['calls'],12);self.assertEqual(len(answers),2);gate=json.loads((b.root/'http-gate/state.json').read_text());self.assertEqual(gate['peak'],8);self.assertEqual(gate['active'],{});self.assertTrue(all(len(a['extra']['actions'])==1 for a in answers))
 def test_failed_n3_group_drains_known_siblings_before_return(self):
  lock=threading.Lock();count=[];barrier=threading.Barrier(3)
  def client(payload):
   with lock:i=len(count);count.append(i)
   barrier.wait(timeout=5)
   if i==0:raise RuntimeError('unknown500')
   time.sleep(.02);return response(ident=str(i))
  m=self.model(client,'sonnet_n3')
  with self.assertRaises(RuntimeError):m.query([{'role':'user','content':'task'}])
  s=m.budget.snapshot();self.assertEqual(len(count),3);self.assertEqual(s['calls'],3);self.assertEqual(s['unknown_calls'],1);self.assertEqual(s['reserved_calls'],0);self.assertEqual(sum(x['status']=='known'for x in s['requests'].values()),2);self.assertFalse((m.directory/'decisions/000.json').exists())
if __name__=='__main__':unittest.main()
