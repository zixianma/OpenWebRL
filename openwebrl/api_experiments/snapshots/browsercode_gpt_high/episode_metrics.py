"""Receipt-derived, all-call accounting; generates no API or browser requests."""
import json
from pathlib import Path
from openwebrl.luna_qwen_metrics import api_cost

def receipt_metrics(directory,pricing):
 roles={};count_requests=0;errors=[]
 for p in sorted((Path(directory)/'requests').glob('*.json')):
  try:
   d=json.loads(p.read_text())
   if d.get('role')=='input_token_count':count_requests+=1;continue
   role='judge' if p.name.startswith('judge-') else 'selector' if d.get('role')=='selector' else 'proposal';v=roles.setdefault(role,{'calls':0,'received_usage_calls':0,'unknown_usage_calls':0,'input_tokens':0,'cached_input_tokens':0,'cache_write_tokens':0,'output_tokens':0,'reported_reasoning_tokens':0,'received_cost_usd':0.,'charged_or_reserved_usd':0.,'summed_request_seconds':0.})
   v['calls']+=1;v['summed_request_seconds']+=d.get('seconds',0);u=d.get('usage')
   if u:
    v['received_usage_calls']+=1;inp=u.get('input_tokens',u.get('prompt_tokens',0));out=u.get('output_tokens',u.get('completion_tokens',0));details=u.get('input_tokens_details') or u.get('prompt_tokens_details') or {};od=u.get('output_tokens_details') or u.get('completion_tokens_details') or {};v['input_tokens']+=inp;v['output_tokens']+=out;v['cached_input_tokens']+=details.get('cached_tokens',0);v['cache_write_tokens']+=details.get('cache_write_tokens',0);v['reported_reasoning_tokens']+=od.get('reasoning_tokens',0);cost=api_cost(u,'judge' if role=='judge' else 'actor',pricing)['upper_usd'];v['received_cost_usd']+=cost;v['charged_or_reserved_usd']+=cost
   else:v['unknown_usage_calls']+=1;v['charged_or_reserved_usd']+=d['reserved_usd']
  except Exception as exc:errors.append({'receipt':p.name,'error_type':type(exc).__name__,'error':str(exc)})
 model={k:sum(v[k] for r,v in roles.items() if r!='judge') for k in ['calls','received_usage_calls','unknown_usage_calls','input_tokens','cached_input_tokens','cache_write_tokens','output_tokens','reported_reasoning_tokens','received_cost_usd','charged_or_reserved_usd','summed_request_seconds']};return {'by_role':roles,'model_excluding_judge':model,'input_token_count_requests':count_requests,'model_budget_calls_including_token_counts':model['calls']+count_requests,'metric_errors':errors,'latency_note':'Summed request seconds add parallel proposal durations and are not episode wall latency.'}
