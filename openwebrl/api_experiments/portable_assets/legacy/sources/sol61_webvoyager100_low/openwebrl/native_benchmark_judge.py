"""Single physical native benchmark judge call, bounded before dispatch."""
from pathlib import Path
from types import SimpleNamespace
import time
from openwebrl.luna_qwen_metrics import api_cost,write_json
class NativeBenchmarkJudge:
 def __init__(self,client,budget,pricing):
  self.client,self.budget,self.pricing=client,budget,pricing;self.directory=None
  self.chat=SimpleNamespace(completions=SimpleNamespace(create=self.create))
  self.beta=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(parse=self.parse)))
 async def create(self,**kwargs):return await self._call(False,kwargs)
 async def parse(self,**kwargs):return await self._call(True,kwargs)
 async def _call(self,structured,kwargs):
  if kwargs.get('model')!='gpt-4o' or kwargs.get('seed')!=42:raise ValueError('Native judge model/seed changed')
  if structured and kwargs.get('temperature')!=0:raise ValueError('Structured native judge temperature changed')
  if list((Path(self.directory)/'requests').glob('judge-*.json')):raise RuntimeError('One physical judge call per episode; no retries')
  request=dict(kwargs,max_completion_tokens=4096,store=False)
  # Full128K input bound at uncached rate plus maximum output; image/history
  # content cannot be billed above model context. This is a reservation, not a bill.
  reserve=(128000*self.pricing['judge_input']+4096*self.pricing['judge_output'])/1e6
  ident=self.budget.update(reserve=reserve)
  saved=dict(request)
  if structured:
   from openai.lib._parsing._completions import type_to_response_format_param
   saved['response_format']=type_to_response_format_param(request['response_format'])
  path=Path(self.directory)/'requests'/f'judge-{ident:05d}.json'
  row=dict(provider='api',role='judge',status='reserved',request=saved,reserved_usd=reserve,started_epoch=time.time(),structured_parse=structured);write_json(path,row);start=time.monotonic()
  try:
   fn=self.client.beta.chat.completions.parse if structured else self.client.chat.completions.create
   response=await fn(**request);row.update(status='received',response=response.model_dump(),usage=response.usage.model_dump())
   if row['usage'].get('prompt_tokens',row['usage'].get('input_tokens',0))>128000:raise ValueError('Native judge exceeded conservative context bound')
   actual=api_cost(row['usage'],'judge',self.pricing)['upper_usd'];self.budget.update(settle=(reserve,actual))
   if self.budget.root.joinpath('halt.json').exists():raise RuntimeError('Judge budget halt')
   if not(response.model=='gpt-4o' or response.model.startswith('gpt-4o-')):raise ValueError('Unexpected native judge model')
   return response
  except BaseException as exc:
   row.update(error_type=type(exc).__name__,http_status=getattr(exc,'status_code',None))
   if 'usage' not in row:row['status']='usage_unknown'
   write_json(self.budget.root/'halt.json',dict(reason='Native judge failure; review before any new physical calls',error_type=type(exc).__name__))
   raise
  finally:row['seconds']=time.monotonic()-start;write_json(path,row)
