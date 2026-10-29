"""All admitted raw Responses generations/counters; external judge isolated."""
from pathlib import Path
import json

def receipt_metrics(directory,pricing):
 roles={};errors=[];counts=0;counter_unknown=0;counter_received=0
 fields=['calls','received_usage_calls','unknown_usage_calls','input_tokens','cached_input_tokens','cache_write_tokens','output_tokens','received_cost_usd','charged_or_reserved_usd','summed_request_seconds']
 for p in sorted((Path(directory)/'requests').glob('*.json')):
  try:
   d=json.loads(p.read_text())
   if d.get('status')=='prepared':continue
   role='judge'if p.name.startswith('judge-')else d['role']
   if role=='input_token_count':
    counts+=1
    if d.get('status')=='usage_unknown':counter_unknown+=1
    else:counter_received+=1
    if d['reserved_usd']!=0 or(d.get('cost')or{}).get('upper_usd',0)!=0:raise ValueError('Native input counter unexpectedly has generation fee')
    continue
   v=roles.setdefault(role,{k:0 for k in fields});v['calls']+=1;v['summed_request_seconds']+=d.get('seconds',0);u=None if d.get('status')=='usage_unknown'else d.get('usage')
   if u is None:v['unknown_usage_calls']+=1;v['charged_or_reserved_usd']+=d['reserved_usd'];continue
   v['received_usage_calls']+=1;cost=d['cost']['upper_usd'];v['received_cost_usd']+=cost;v['charged_or_reserved_usd']+=cost
   if role=='judge':v['input_tokens']+=u['prompt_tokens'];v['output_tokens']+=u['completion_tokens'];details=u.get('prompt_tokens_details')or{}
   else:v['input_tokens']+=u['input_tokens'];v['output_tokens']+=u['output_tokens'];details=u.get('input_tokens_details')or{}
   v['cached_input_tokens']+=details.get('cached_tokens')or 0;v['cache_write_tokens']+=details.get('cache_write_tokens')or 0
  except Exception as exc:errors.append({'receipt':p.name,'error_type':type(exc).__name__,'error':str(exc)})
 model={k:sum(v[k]for role,v in roles.items()if role!='judge')for k in fields}
 return {'by_role':roles,'model_excluding_judge':model,'input_token_count_requests':counts,'input_token_count_received':counter_received,'input_token_count_unknown':counter_unknown,'model_budget_calls_including_token_counts':model['calls']+counts,'metric_errors':errors,'thinking_token_count_observed':False,'thinking_cost_included_in_output':True,'cache_read_or_write_missing_does_not_mean_zero':True,'latency_note':'Parallel summed request seconds are not native task latency; external judge excluded from task latency and model cost.'}
