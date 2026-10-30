"""Metered benchmark-native WV/DeepShop judges with immutable image bindings."""
from bootstrap import ROOT,CONFIG
from pathlib import Path
import ast,base64,hashlib,json,time
from typing import Literal
from pydantic import BaseModel,Field
from openwebrl.luna_qwen_metrics import write_json,api_cost
class DeepShopVerdict(BaseModel):
 thought:str=Field(description='Reasoning behind the verdict')
 verdict:Literal['SUCCESS','NOT SUCCESS']=Field(description='Final verdict for whether the web assistant completed the task or not.')
def native_prompts(benchmark):
 p=ROOT/'native_judge_sources'/('reward_webvoyager.py' if benchmark=='webvoyager' else 'reward_deepshop.py');out={}
 for node in ast.parse(p.read_text()).body:
  if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id in ['SYSTEM_PROMPT','USER_PROMPT']:out[node.targets[0].id]=ast.literal_eval(node.value)
 assert 'SYSTEM_PROMPT' in out
 return out
def parse_verdict(benchmark,response):
 choice=response['choices'][0]
 if choice.get('finish_reason')!='stop':raise ValueError('Incomplete native judge response')
 message=choice['message']
 if message.get('refusal'):raise ValueError('Native judge refusal')
 text=message.get('content') or ''
 if benchmark=='deepshop':return int(DeepShopVerdict.model_validate_json(text).verdict=='SUCCESS')
 if 'NOT SUCCESS' in text:return 0
 if 'SUCCESS' in text:return 1
 raise ValueError('Unparseable native WebVoyager verdict')
def build_request(item,answer,images):
 bench=item['benchmark'];assert bench in ['webvoyager','deepshop'];images=list(images)[-30:];assert images
 prompts=native_prompts(bench);task=item['intent'];num=len(images)
 text=prompts['USER_PROMPT'].format(task=task,answer=answer,num=num) if bench=='webvoyager' else f'TASK:\n{task}\nResult Response:\n{answer}\n{num} screenshots'
 content=[{'type':'text','text':text}]
 for i,p in enumerate(images):
  if bench=='deepshop':content.append({'type':'text','text':f'Step {i+1}'})
  content.append({'type':'image_url','image_url':{'url':'data:image/png;base64,'+base64.b64encode(Path(p).read_bytes()).decode()}})
 messages=[{'role':'system','content':prompts['SYSTEM_PROMPT']},{'role':'user','content':content}]
 if bench=='webvoyager':messages.append({'role':'user','content':'Your verdict:\n'})
 request={'model':'gpt-4o','messages':messages,'seed':42,'max_completion_tokens':4096,'store':False}
 if bench=='deepshop':
  request['temperature']=0
  schema=DeepShopVerdict.model_json_schema();schema['additionalProperties']=False
  request['response_format']={'type':'json_schema','json_schema':{'name':'DeepShopVerdict','strict':True,'schema':schema}}
 return request,images

def judge(client,budget,item,result,images,directory):
 bench=item['benchmark'];answer=result.get('final_response') or result.get('submission') or ''
 if result['exit_status']!='Submitted' or not answer:
  write_json(directory/'native-unjudged-zero.json',{'benchmark':bench,'exit_status':result['exit_status'],'reason':'native_noncompleted_or_missing_answer_zero','reward':0,'model_requests':0});return 0
 request,images=build_request(item,answer,images)
 # Text UTF8 bytes are an upper bound;4096 image tokens per fixed1280x1800
 # image deliberately exceeds native GPT4o tile usage. Unknown calls retain it.
 text_bound=sum(len(x['text'].encode()) for m in request['messages'] for x in (m['content'] if isinstance(m['content'],list) else [{'text':m['content']}]) if 'text' in x)+4096*len(images)+1024
 reserve=(text_bound*CONFIG['pricing']['judge_input']+4096*CONFIG['pricing']['judge_output'])/1e6;ident=budget.update(reserve=reserve);p=directory/'requests'/f'judge-{ident:05d}.json'
 row={'benchmark':bench,'request':request,'status':'reserved','reserved_usd':reserve,'started_epoch':time.time(),'image_sha256':[hashlib.sha256(Path(x).read_bytes()).hexdigest() for x in images],'native_judge_source_sha256':hashlib.sha256((ROOT/'native_judge_sources'/('reward_webvoyager.py' if bench=='webvoyager' else 'reward_deepshop.py')).read_bytes()).hexdigest()};write_json(p,row)
 try:
  response=client.chat.completions.create(**request);native=response.model_dump();row.update(response=native,usage=native['usage'],status='received');write_json(p,row)
  budget.update(settle=(reserve,api_cost(native['usage'],'judge',CONFIG['pricing'])['upper_usd']))
  assert native['model']=='gpt-4o' or native['model'].startswith('gpt-4o-'),'Unexpected native judge model'
  score=parse_verdict(bench,native);row.update(reward=score,verdict_text=native['choices'][0]['message'].get('content'));return score
 except BaseException as exc:
  row['error_type']=type(exc).__name__
  if 'usage' not in row:row['status']='usage_unknown'
  write_json(budget.root/'halt.json',{'reason':'Native judge failure; retain charge and diagnose before any further call','error_type':type(exc).__name__});raise
 finally:row['seconds']=time.time()-row['started_epoch'];write_json(p,row)
