"""Author offline migration tests. No config/archived-test mutation or API calls."""
from bootstrap import ROOT,CONFIG
from pathlib import Path
from copy import deepcopy
from decimal import Decimal
import unittest,tempfile,types,json,base64,ast,threading
from unittest.mock import patch
import common_judge as J
from claude_budget import ReservationBudget

class TerraJudgeMigration(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory(prefix='terra-judge-offline-');self.addCleanup(self.tmp.cleanup);self.r=Path(self.tmp.name);self.image=self.r/'final.png';self.image.write_bytes(b'original-offline-image-bytes');self.config=deepcopy(J.campaign);self.config['judge_protocol']=deepcopy(J.TERRA_PROTOCOL);self.patch=patch.object(J,'campaign',self.config);self.patch.start();self.addCleanup(self.patch.stop)
 def response(self,text='Thoughts: evaluated\nStatus: "failure"',usage=None,finish='stop',model='gpt-5.6-terra'):
  return {'model':model,'choices':[{'index':0,'finish_reason':finish,'message':{'role':'assistant','content':text,'refusal':None}}],'usage':usage if usage is not None else{'prompt_tokens':100,'completion_tokens':20,'prompt_tokens_details':{'cached_tokens':0}}}
 def client(self,response=None,error=None):
  self.calls=[]
  def create(**request):
   self.calls.append(deepcopy(request))
   if error:raise error
   return types.SimpleNamespace(model_dump=lambda:deepcopy(response if response is not None else self.response()))
  return types.SimpleNamespace(max_retries=0,chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create)))
 def budget(self,root=None,usd=2.5,calls=1100):return ReservationBudget(root or self.r/'budget',usd,calls)
 def receipt(self):return json.loads(next((self.r/'requests').glob('judge-*.json')).read_text())
 def test_protocol_exact_user_model_and_explicit_prior_default(self):
  p=J.protocol();self.assertEqual(p['model'],'gpt-5.6-terra');self.assertEqual(p['reasoning_effort'],'medium');self.assertEqual(p['max_completion_tokens'],4096);self.assertEqual(p['seed'],42)
 def test_config_model_effort_pricing_type_substitution_rejected(self):
  for field,value in [('model','gpt-6.1-terra'),('reasoning_effort','high'),('seed',True),('max_completion_tokens',4096.0)]:
   bad=deepcopy(self.config);bad['judge_protocol'][field]=value
   with self.subTest(field=field),self.assertRaises(ValueError):J.protocol(bad)
  bad=deepcopy(self.config);bad['judge_protocol']['pricing']['input']=1
  with self.assertRaises(ValueError):J.protocol(bad)
  with self.assertRaises(ValueError):J.protocol({'pricing':self.config['pricing']})
 def test_request_frozen_prompts_actions_seed_image_unchanged(self):
  actions=['{"command":"inspect"}','{"done":true}'];request=J.build_judge_request('task',actions,self.image);prompts,_=J.frozen_prompts();trajectory='\n\n'.join(f'Thought {i+1}: \nAction {i+1}: {text}'for i,text in enumerate(actions));old={'model':'o4-mini-2025-04-16','seed':42,'max_completion_tokens':4096,'store':False,'messages':[{'role':'system','content':prompts['SYSTEM_PROMPT']},{'role':'user','content':[{'type':'text','text':prompts['USER_PROMPT'].format(task='task',thoughts_and_actions=trajectory)},{'type':'image_url','image_url':{'url':'data:image/png;base64,'+base64.b64encode(self.image.read_bytes()).decode(),'detail':'high'}}]}]};expected={**old,'model':'gpt-5.6-terra','reasoning_effort':'medium'};self.assertEqual(request,expected);self.assertNotIn('temperature',request);self.assertNotIn('tools',request)
 def test_short_usage_no_cachewrite_telemetry_upper_bound(self):
  cost=J.judge_usage_cost({'prompt_tokens':100,'completion_tokens':20,'prompt_tokens_details':{'cached_tokens':0}});self.assertAlmostEqual(cost['upper_usd'],(100*2.5+20*12)/1e6);self.assertAlmostEqual(cost['lower_usd'],(100*2+20*12)/1e6);self.assertTrue(cost['cache_write_missing_telemetry_conservative_upper_bound']);self.assertIsNone(cost['cache_write_tokens_observed'])
 def test_known_read_and_write_telemetry_no_doublecharge(self):
  usage={'prompt_tokens':100,'completion_tokens':20,'prompt_tokens_details':{'cached_tokens':40,'cache_write_tokens':20},'completion_tokens_details':{'reasoning_tokens':15}};cost=J.judge_usage_cost(usage);expected=(40*2+40*.2+20*2.5+20*12)/1e6;self.assertEqual(cost['upper_usd'],cost['lower_usd']);self.assertAlmostEqual(cost['upper_usd'],expected);self.assertEqual(cost['reasoning_output_tokens_observed'],15);self.assertTrue(cost['reasoning_included_in_output_not_double_charged'])
 def test_missing_read_discount_never_imputed(self):
  cost=J.judge_usage_cost({'prompt_tokens':100,'completion_tokens':0});self.assertEqual(cost['upper_usd'],.00025);self.assertEqual(cost['lower_usd'],.00002);self.assertTrue(cost['cache_read_discount_missing_telemetry_not_imputed'])
 def test_exact_threshold_short_then_fullrequest_long_multipliers(self):
  for tokens,long in [(272000,False),(272001,True)]:
   cost=J.judge_usage_cost({'prompt_tokens':tokens,'completion_tokens':10,'prompt_tokens_details':{'cached_tokens':100,'cache_write_tokens':0}});self.assertIs(cost['long_context_full_request_pricing_applied'],long);im=2 if long else 1;om=1.5 if long else 1;self.assertAlmostEqual(cost['upper_usd'],((tokens-100)*2*im+100*.2*im+10*12*om)/1e6)
 def test_bad_usage_typed_total_subset_and_aliases_rejected(self):
  bad=[None,{}, {'prompt_tokens':True,'completion_tokens':1},{'prompt_tokens':-1,'completion_tokens':0},{'prompt_tokens':1,'completion_tokens':1,'input_tokens':2},{'prompt_tokens':10,'completion_tokens':10,'prompt_tokens_details':{'cached_tokens':True}},{'prompt_tokens':10,'completion_tokens':10,'prompt_tokens_details':{'cached_tokens':8,'cache_write_tokens':3}},{'prompt_tokens':10,'completion_tokens':10,'completion_tokens_details':{'reasoning_tokens':11}}]
  for usage in bad:
   with self.subTest(usage=usage),self.assertRaises(ValueError):J.judge_usage_cost(usage)
 def test_reservation_conservative_cache_and_output_highimage_no_base64tokenization(self):
  request=J.build_judge_request('task',[],self.image);reserve,bound,long=J.judge_reservation(request);self.assertFalse(long);self.assertGreaterEqual(bound,32768);self.assertEqual(reserve,(Decimal(bound)*Decimal('2.5')+Decimal(4096)*Decimal(12))/Decimal(1000000));request['messages'][1]['content'][1]['image_url']['url']='data:image/png;base64,'+'x'*1000000;self.assertEqual(J.judge_input_bound(request),bound)
 def test_long_reservation_at_bound_applies_full_request_multipliers(self):
  request=J.build_judge_request('x'*280000,[],self.image);reserve,bound,long=J.judge_reservation(request);self.assertTrue(long);self.assertEqual(reserve,(Decimal(bound)*5+Decimal(4096)*18)/1000000)
 def test_strict_frozen_status_success_failure_quoted_unquoted(self):
  for status,value in [('"success"',1),("'failure'",0),('success',1),('FAILURE',0)]:self.assertEqual(J.parse_verdict(self.response('Thoughts: x\nStatus: '+status))[0],value)
 def test_ambiguous_substring_refusal_truncation_foreign_model_invalid(self):
  for text in ['Status: unsuccessful','Status: success\nStatus: failure','Status: success maybe', 'No status', 'Status: "success\'']:
   with self.subTest(text=text),self.assertRaises(ValueError):J.parse_verdict(self.response(text))
  for finish in ['length','content_filter','tool_calls',None]:
   with self.subTest(finish=finish),self.assertRaises(ValueError):J.parse_verdict(self.response(finish=finish))
  with self.assertRaises(ValueError):J.parse_verdict(self.response(model='gpt-5.6-sol'))
  r=self.response();r['choices'][0]['message']['refusal']='refused'
  with self.assertRaises(ValueError):J.parse_verdict(r)
 def test_actual_judge_known_receipt_charge_physical_id_no_retry(self):
  client=self.client(self.response());b=self.budget();self.assertEqual(J.judge(client,b,'task',[],self.image,self.r),0);self.assertEqual(len(self.calls),1);self.assertEqual(self.calls[0]['reasoning_effort'],'medium');row=self.receipt();self.assertEqual(row['physical_id'],next(iter(b.snapshot()['requests'])));self.assertEqual(row['status'],'received');self.assertEqual(row['judge_protocol_version'],'gpt56-terra-terminal-success-v1');self.assertEqual(row['usage'],self.response()['usage']);self.assertEqual(Decimal(row['accounting_charge_usd_decimal']),Decimal(row['cost']['upper_usd_decimal']));self.assertEqual(b.snapshot()['reserved_calls'],0)
 def test_http_failure_retains_fullreservation_unknown_no_retry(self):
  client=self.client(error=RuntimeError('offlineHTTP500'));b=self.budget()
  with self.assertRaises(RuntimeError):J.judge(client,b,'task',[],self.image,self.r)
  row=self.receipt();self.assertEqual(row['status'],'usage_unknown');self.assertEqual(len(self.calls),1);self.assertEqual(b.snapshot()['unknown_calls'],1);self.assertEqual(b.snapshot()['charged_or_reserved_usd_decimal'],row['reserved_usd_decimal']);self.assertTrue(b.halted())
 def test_malformed_present_usage_retains_reserve(self):
  native=self.response();native['usage']=None;client=self.client(native);b=self.budget()
  with self.assertRaises(ValueError):J.judge(client,b,'task',[],self.image,self.r)
  self.assertEqual(b.snapshot()['unknown_calls'],1);row=self.receipt();self.assertIsNone(row['usage']);self.assertEqual(row['accounting_charge_usd_decimal'],row['reserved_usd_decimal'])
 def test_known_truncated_parseinvalid_charge_not_refunded(self):
  client=self.client(self.response(finish='length'));b=self.budget()
  with self.assertRaises(ValueError):J.judge(client,b,'task',[],self.image,self.r)
  row=self.receipt();self.assertEqual(row['status'],'received_invalid');self.assertEqual(b.snapshot()['unknown_calls'],0);self.assertGreater(b.snapshot()['charged_or_reserved_usd'],0);self.assertTrue(b.halted());self.assertEqual(len(self.calls),1)
 def test_prior_unknown_charges_preserved_and_old_receipts_never_overwritten(self):
  b=self.budget();b.reserve('old-unknown',.4);b.unknown('old-unknown','oldHTTP');before=(b.root/'ledger.json').read_bytes();(self.r/'requests').mkdir();old=self.r/'requests/judge-old.json';old.write_text('{"historic":true}')
  # Existing halt requires separately reviewed root qualification; the judge
  # cannot clear it or refund historical unknowns on its own.
  client=self.client()
  with self.assertRaises(ValueError):J.judge(client,b,'task',[],self.image,self.r)
  self.assertEqual(len(self.calls),0);self.assertEqual(b.snapshot()['requests']['old-unknown']['charged_or_reserved_usd'],'0.4');self.assertEqual(old.read_text(),'{"historic":true}');self.assertEqual(b.snapshot()['calls'],1)
 def test_call_and_dollar_caps_block_before_provider(self):
  b=self.budget(usd=.01);client=self.client()
  with self.assertRaises(ValueError):J.judge(client,b,'task',[],self.image,self.r)
  self.assertEqual(len(self.calls),0);self.assertEqual(b.snapshot()['calls'],0);self.assertEqual(self.receipt()['status'],'prepared')
 def test_sdk_retry_config_rejected_before_provider_or_reserve(self):
  client=self.client();client.max_retries=2;b=self.budget()
  with self.assertRaises(ValueError):J.judge(client,b,'task',[],self.image,self.r)
  self.assertEqual(len(self.calls),0);self.assertEqual(b.snapshot()['calls'],0)
 def test_contradictory_totals_and_output_details_unknown(self):
  for usage in [{'prompt_tokens':10,'completion_tokens':2,'total_tokens':13},{'prompt_tokens':10,'completion_tokens':2,'total_tokens':True},{'prompt_tokens':10,'completion_tokens':2,'completion_tokens_details':{'reasoning_tokens':1},'output_tokens_details':{'reasoning_tokens':2}}]:
   with self.subTest(usage=usage),self.assertRaises(ValueError):J.judge_usage_cost(usage)
  self.assertGreater(J.judge_usage_cost({'prompt_tokens':10,'completion_tokens':2,'total_tokens':12})['upper_usd'],0)
if __name__=='__main__':unittest.main()
