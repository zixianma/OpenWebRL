"""Frozen BrowserCode terminal-success prompt with the approved Terra judge.
Only model/explicit default effort/pricing change. No automatic provider retry,
model fallback, missing-usage zero fill, historical receipt rewrite, or refunds.
"""
from pathlib import Path
from decimal import Decimal
from copy import deepcopy
import ast,base64,hashlib,json,re,time,uuid
from claude_budget import need,money,atomic_json,exclusive_json

campaign=json.loads(Path(__file__).with_name('runtime-config.json').read_text())
TERRA_PROTOCOL={'model':'gpt-5.6-terra','reasoning_effort':'medium','max_completion_tokens':4096,'seed':42,'pricing':{'input':2,'cached_input':.2,'output':12,'cache_write_multiplier':1.25,'long_context_threshold':272000,'long_input_multiplier':2,'long_output_multiplier':1.5},'version':'gpt56-terra-terminal-success-v1'}
DOC_URL='https://developers.openai.com/api/docs/models/gpt-5.6-terra'

def protocol(config=None):
 p=(campaign if config is None else config).get('judge_protocol')
 need(type(p)is dict,'Explicit approved Terra judge protocol required')
 # Recursive type equality rejects True-as-one/count aliases and any implicit
 # model/effort/pricing substitution. Preserve old CONFIG.pricing as archive.
 def exact(a,b):
  return type(a)is type(b)and(all(k in b and exact(v,b[k])for k,v in a.items())and set(a)==set(b)if type(a)is dict else a==b)
 need(exact(TERRA_PROTOCOL,p),'Terra judge protocol differs from approval');return deepcopy(p)

def frozen_prompts():
 source=Path(campaign['frozen_support_source'])/'openwebrl/eval/reward_online_mind2web.py';tree=ast.parse(source.read_text());prompts={}
 for node in tree.body:
  if isinstance(node,ast.Assign)and any(isinstance(t,ast.Name)and t.id in('SYSTEM_PROMPT','USER_PROMPT')for t in node.targets):prompts[node.targets[0].id]=ast.literal_eval(node.value)
 need(set(prompts)=={'SYSTEM_PROMPT','USER_PROMPT'}and all(type(v)is str for v in prompts.values()),'Frozen terminal prompts missing');return prompts,hashlib.sha256(source.read_bytes()).hexdigest()

def build_judge_request(task,actions,screenshot,judge_protocol=None):
 p=protocol()if judge_protocol is None else protocol({'judge_protocol':judge_protocol});need(type(task)is str and type(actions)is list and all(type(x)is str for x in actions),'Exact task/action strings required');prompts,_=frozen_prompts();trajectory='\n\n'.join(f'Thought {i+1}: \nAction {i+1}: {text}'for i,text in enumerate(actions));image=Path(screenshot).read_bytes();need(bool(image),'Saved final screenshot missing/empty')
 return dict(model=p['model'],reasoning_effort=p['reasoning_effort'],seed=p['seed'],max_completion_tokens=p['max_completion_tokens'],store=False,messages=[{'role':'system','content':prompts['SYSTEM_PROMPT']},{'role':'user','content':[{'type':'text','text':prompts['USER_PROMPT'].format(task=task,thoughts_and_actions=trajectory)},{'type':'image_url','image_url':{'url':'data:image/png;base64,'+base64.b64encode(image).decode(),'detail':'high'}}]}])

def judge_input_bound(request):
 need(request['model']=='gpt-5.6-terra'and request['reasoning_effort']=='medium'and request['max_completion_tokens']==4096,'Unexpected judge request protocol');text=[];images=0
 for message in request['messages']:
  content=message['content']
  if type(content)is str:text.append(content);continue
  need(type(content)is list,'Judge content malformed')
  for block in content:
   if block.get('type')=='text':need(type(block.get('text'))is str,'Judge text malformed');text.append(block['text'])
   elif block.get('type')=='image_url':need(block['image_url']['detail']=='high'and block['image_url']['url'].startswith('data:image/png;base64,'),'Original high-detail PNG required');images+=1
   else:raise ValueError('Unexpected judge multimodal block')
 need(images==1,'Exact one frozen final screenshot required')
 # The caller verifies original1280x1800 final capture before judging. Base64
 # image bytes are NOT counted as text tokens. Keep the prior32768 image bound.
 bound=sum(len(x.encode('utf-8'))for x in text)+4096+images*32768
 need(bound+4096<=1050000,'Conservative judge context bound exceeds model context');return bound

def judge_usage_cost(usage,judge_protocol=None):
 p=protocol()if judge_protocol is None else protocol({'judge_protocol':judge_protocol});need(type(usage)is dict,'Missing native judge usage; full reserve retained')
 def count(primary,alias=None,required=True):
  values=[usage[k]for k in(primary,alias)if k is not None and k in usage];need(not values or all(v==values[0]and type(v)is type(values[0])for v in values),'Contradictory native token aliases');value=values[0]if values else None
  need((value is None and not required)or(type(value)is int and value>=0),'Missing/malformed native usage '+primary);return value
 inp=count('prompt_tokens','input_tokens');out=count('completion_tokens','output_tokens');total=count('total_tokens',required=False);need(total is None or total==inp+out,'Native total_tokens differs from input plus output');details=usage.get('prompt_tokens_details',usage.get('input_tokens_details'))
 if 'prompt_tokens_details'in usage and 'input_tokens_details'in usage:need(usage['prompt_tokens_details']==usage['input_tokens_details'],'Contradictory native detail aliases')
 need(details is None or type(details)is dict,'Malformed input usage details');details=details or{}
 cached=details.get('cached_tokens');writes=details.get('cache_write_tokens')
 for name,value in [('cached_input',cached),('cache_write_input',writes)]:need(value is None or(type(value)is int and 0<=value<=inp),'Malformed/excess '+name)
 need(cached is None or writes is None or cached+writes<=inp,'Cache reads plus writes exceed total input')
 need('completion_tokens_details'not in usage or 'output_tokens_details'not in usage or usage['completion_tokens_details']==usage['output_tokens_details'],'Contradictory output token detail aliases')
 output_details=usage.get('completion_tokens_details',usage.get('output_tokens_details'))
 need(output_details is None or type(output_details)is dict,'Malformed output usage details');reasoning=(output_details or{}).get('reasoning_tokens');need(reasoning is None or(type(reasoning)is int and 0<=reasoning<=out),'Malformed/excess reasoning output')
 price=p['pricing'];long=inp>price['long_context_threshold'];input_multiplier=money(price['long_input_multiplier'])if long else Decimal(1);output_multiplier=money(price['long_output_multiplier'])if long else Decimal(1);ir=money(price['input'])*input_multiplier;cr=money(price['cached_input'])*input_multiplier;wr=ir*money(price['cache_write_multiplier']);orr=money(price['output'])*output_multiplier
 # Native total input includes cached reads and writes. No double charge.
 # Omitted cache-write telemetry => every non-read input could be a write;
 # omitted cache-read telemetry => no discount in the upper bound.
 read_low=0 if cached is None else cached;read_high=inp-(writes or 0)if cached is None else cached
 write_high=inp-read_low if writes is None else writes;write_low=0 if writes is None else writes
 upper=(Decimal(inp-read_low-write_high)*ir+Decimal(read_low)*cr+Decimal(write_high)*wr+Decimal(out)*orr)/Decimal(1000000)
 lower=(Decimal(inp-read_high-write_low)*ir+Decimal(read_high)*cr+Decimal(write_low)*wr+Decimal(out)*orr)/Decimal(1000000)
 return {'upper_usd':float(upper),'upper_usd_decimal':str(upper),'lower_usd':float(lower),'lower_usd_decimal':str(lower),'input_tokens':inp,'output_tokens':out,'cached_input_tokens_observed':cached,'cache_write_tokens_observed':writes,'reasoning_output_tokens_observed':reasoning,'reasoning_included_in_output_not_double_charged':True,'cache_read_discount_missing_telemetry_not_imputed':cached is None,'cache_write_missing_telemetry_conservative_upper_bound':writes is None,'long_context_full_request_pricing_applied':long,'pricing_source':DOC_URL,'pricing':deepcopy(price)}

def judge_reservation(request,judge_protocol=None):
 p=protocol()if judge_protocol is None else protocol({'judge_protocol':judge_protocol});inp=judge_input_bound(request);price=p['pricing'];long=inp>price['long_context_threshold'];ir=money(price['input'])*(money(price['long_input_multiplier'])if long else Decimal(1));orr=money(price['output'])*(money(price['long_output_multiplier'])if long else Decimal(1));amount=(Decimal(inp)*ir*money(price['cache_write_multiplier'])+Decimal(p['max_completion_tokens'])*orr)/Decimal(1000000);return amount,inp,long

def parse_verdict(native):
 need(type(native)is dict and native.get('model')=='gpt-5.6-terra','Judge returned unapproved model');choices=native.get('choices');need(type(choices)is list and len(choices)==1 and type(choices[0])is dict,'Exactly one native judge choice required');choice=choices[0];need(choice.get('finish_reason')=='stop','Judge truncation/refusal/nonterminal finish is invalid');message=choice.get('message');need(type(message)is dict and message.get('refusal')in(None,'')and not message.get('tool_calls'),'Judge refusal/tool output invalid');text=message.get('content');need(type(text)is str and bool(text.strip()),'Empty judge verdict')
 # Frozen prompt asks Thoughts followed by Status: "success" or "failure".
 # Require one exact status field; substrings such as unsuccessful cannot pass.
 matches=re.findall(r'''^\s*status\s*:\s*(?:"(success|failure)"|'(success|failure)'|(success|failure))\s*$''',text,re.I|re.M);status=[next(x for x in match if x)for match in matches];need(len(status)==1 and len(re.findall(r'\bstatus\s*:',text,re.I))==1,'Unparseable/ambiguous frozen status verdict');return int(status[0].lower()=='success'),text

def judge(client,budget,task,actions,screenshot,directory):
 need(getattr(client,'max_retries',0)==0,'Judge SDK retries must be disabled');p=protocol();request=build_judge_request(task,actions,screenshot,p);reserved,input_bound,long=judge_reservation(request,p);directory=Path(directory);ident='judge-'+uuid.uuid4().hex;path=directory/'requests'/(ident+'.json');_,prompt_sha=frozen_prompts();row={'physical_id':ident,'provider':'openai','role':'external_judge','judge_protocol':p,'judge_protocol_version':p['version'],'frozen_prompt_source_sha256':prompt_sha,'request':request,'request_sha256':hashlib.sha256(json.dumps(request,sort_keys=True,ensure_ascii=False).encode()).hexdigest(),'status':'prepared','reserved_usd':float(reserved),'reserved_usd_decimal':str(reserved),'input_token_upper_bound':input_bound,'reservation_long_context_tier':long,'created_epoch':time.time()};exclusive_json(path,row);admitted=False
 try:
  budget.reserve(ident,reserved,{'role':'external_judge','model':p['model'],'protocol_version':p['version'],'directory':str(directory),'request_path':str(path),'request_sha256':row['request_sha256']});admitted=True;row.update(status='reserved',physical_reservation_admitted=True,started_epoch=time.time());atomic_json(path,row);need(not budget.halted(),'Judge pool halted before HTTP');row['physical_http_dispatch_epoch']=time.time();atomic_json(path,row);started=time.monotonic()
  try:reply=client.chat.completions.create(**request)
  finally:row['seconds']=time.monotonic()-started
  native=reply.model_dump()if hasattr(reply,'model_dump')else reply;row.update(response=native,response_received_epoch=time.time());atomic_json(path,row);need(type(native)is dict,'Malformed raw judge response');row['usage']=native.get('usage');cost=judge_usage_cost(row['usage'],p);row['cost']=cost;atomic_json(path,row);budget.finish(ident,cost['upper_usd_decimal'],row['usage']);row.update(status='received',accounting_charge_usd=cost['upper_usd'],accounting_charge_usd_decimal=cost['upper_usd_decimal']);atomic_json(path,row);value,text=parse_verdict(native);row.update(verdict_text=text,verdict=value,verdict_valid=True);return value
 except BaseException as exc:
  row.update(error_type=type(exc).__name__,failed_epoch=time.time(),verdict_valid=False)
  if admitted:
   state=budget.snapshot()['requests'][ident]
   if state['status']=='reserved':budget.unknown(ident,type(exc).__name__);row['status']='usage_unknown'
   elif state['status']=='known':row['status']='received_invalid'
   row['accounting_charge_usd_decimal']=budget.snapshot()['requests'][ident]['charged_or_reserved_usd'];row['accounting_charge_usd']=float(money(row['accounting_charge_usd_decimal']))
  budget.halt('Judge failure; no retries, preserve original usage/reserves and inspect before more paid work');raise
 finally:atomic_json(path,row)
