"""Metered high or three-low-plus-low wrapper around upstream CUAWright model."""
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json,time
from cuawright.webwright.models.openai_model import OpenAIModel
from cuawright.webwright.models.tool_calls import parse_tool_call_output
from openwebrl.luna_qwen_metrics import write_json,api_cost
from openwebrl.responses_actor_history import response_input_bound

def parse_actor(payload,harness):
 calls=[x for x in payload.get('output',[]) if x.get('type')=='function_call']
 text='\n'.join(p.get('text','') for x in payload.get('output',[]) if x.get('type')=='message' for p in x.get('content',[]) if p.get('type')=='output_text').strip()
 if len(calls)>1 or (calls and text):raise ValueError('Each turn must be one action or one final answer')
 if harness=='cuawright':return parse_tool_call_output(payload)
 if harness!='browsercode':raise ValueError('Unknown harness')
 if not calls:
  if not text:raise ValueError('Empty model response')
  return {'done':True,'bash_command':'','final_response':text}
 if calls[0]['name']!='browser_execute':raise ValueError('Wrong BrowserCode tool')
 args=json.loads(calls[0]['arguments'])
 if set(args)!={'code','timeout','description'} or not isinstance(args['code'],str) or not args['code'].strip() or not isinstance(args['description'],str):raise ValueError('Invalid BrowserCode arguments')
 if args['timeout'] is not None and (type(args['timeout']) is not int or not 0<args['timeout']<=120000):raise ValueError('BrowserCode timeout exceeds common command bound')
 return {'done':False,'bash_command':json.dumps(args,sort_keys=True),'final_response':''}

class PairedModel(OpenAIModel):
 def __init__(self,mode,directory,client,budget,pricing,harness="cuawright",**kwargs):
  super().__init__(**kwargs)
  if mode not in ['single_low','single_medium']:raise ValueError('Unknown effort condition')
  self.harness=harness
  self.mode,self.directory,self.client,self.budget,self.pricing=mode,Path(directory),client,budget,pricing
  self.turn=0
 def physical(self,payload,role):
  bound=response_input_bound(payload['input'],payload.get('tools',[]))
  if bound>272000:raise ValueError('Input bound exceeds pinned pricing tier')
  reserve=(bound*self.pricing['luna_cache_write']+payload['max_output_tokens']*self.pricing['luna_output'])/1e6
  ident=self.budget.update(reserve=reserve);path=self.directory/'requests'/f'sol-{ident:05d}.json'
  row=dict(role=role,request=payload,reserved_usd=reserve,started_epoch=time.time(),status='reserved');write_json(path,row);start=time.monotonic()
  try:
   response=self.client.responses.create(**payload);native=response.model_dump();row.update(response=native,usage=native['usage'],status='received')
   write_json(path,row)
   self.budget.update(settle=(reserve,api_cost(native['usage'],'selector' if role=='selector' else 'actor',self.pricing)['upper_usd']))
   if native.get('status')!='completed' or not(native['model']=='gpt-6.1-sol' or native['model'].startswith('gpt-6.1-sol-')):raise ValueError('Invalid native model response')
   return native
  except BaseException as e:
   row['error_type']=type(e).__name__
   if 'usage' not in row:row['status']='usage_unknown'
   write_json(self.budget.root/'halt.json',{'reason':'Physical call or response failure; diagnose before any new attempt','error_type':type(e).__name__});raise
  finally:row['seconds']=time.monotonic()-start;write_json(path,row)
 def query(self,messages,**unused):
  if self.turn>=30:raise ValueError('30 decision cap')
  native=self._build_payload(deepcopy(messages));native.update(model='gpt-6.1-sol',store=False,service_tier='default')
  effort='low' if self.mode=='single_low' else 'medium'
  if self.harness=='browsercode':native['tools']=[json.loads((Path(__file__).with_name('runtime-config.json')).read_text())['native_tool_contract']]
  native['reasoning']={'effort':effort};native['max_output_tokens']=4096
  response=self.physical(native,'actor_'+effort);parsed=parse_actor(response,self.harness)
  command=parsed.get('bash_command','');extra=self._response_extra(response)
  decision={'turn':self.turn,'mode':self.mode,'harness':self.harness,'effort':effort,'response':response,'selected_action':{'command':command,'done':parsed['done'],'final_response':parsed.get('final_response','')}}
  write_json(self.directory/'decisions'/f'{self.turn:03d}.json',decision);self.turn+=1
  actions=[{'command':command,'bash_command':command}] if command else []
  return self.format_message(role='assistant',content=parsed.get('final_response',''),extra=dict(actions=actions,done=parsed['done'],final_response=parsed.get('final_response',''),raw_response=parsed,**extra))
