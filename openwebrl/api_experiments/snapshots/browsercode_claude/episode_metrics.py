"""Saved raw physical receipt accounting; never synthesizes missing telemetry."""
from pathlib import Path
import json

def receipt_metrics(directory,pricing):
 roles={};errors=[]
 for p in sorted((Path(directory)/'requests').glob('*.json')):
  try:
   d=json.loads(p.read_text());role='judge'if p.name.startswith('judge-')else d['role'];v=roles.setdefault(role,{'calls':0,'received_usage_calls':0,'unknown_usage_calls':0,'input_tokens':0,'cached_input_tokens':0,'cache_write_tokens':0,'output_tokens':0,'received_cost_usd':0.,'charged_or_reserved_usd':0.,'summed_request_seconds':0.})
   if d.get('status')=='prepared':continue
   v['calls']+=1;v['summed_request_seconds']+=d.get('seconds',0);u=None if d.get('status')=='usage_unknown'else d.get('usage')
   if u is not None:
    v['received_usage_calls']+=1
    if role=='judge':
     from openwebrl.luna_qwen_metrics import api_cost
     c=json.loads(Path(__file__).with_name('runtime-config.json').read_text());cost=d['cost']['upper_usd'] if 'cost'in d else api_cost(u,'judge',c['pricing'])['upper_usd'];v['input_tokens']+=u.get('prompt_tokens',0);v['output_tokens']+=u.get('completion_tokens',0);v['cached_input_tokens']+=(u.get('prompt_tokens_details')or{}).get('cached_tokens',0)
    else:
     cost=d['cost']['upper_usd'];v['input_tokens']+=u['input_tokens'];v['output_tokens']+=u['output_tokens'];v['cached_input_tokens']+=u.get('cache_read_input_tokens',0);v['cache_write_tokens']+=u.get('cache_creation_input_tokens',0)
    v['received_cost_usd']+=cost;v['charged_or_reserved_usd']+=cost
   else:v['unknown_usage_calls']+=1;v['charged_or_reserved_usd']+=d['reserved_usd']
  except Exception as e:errors.append({'receipt':p.name,'error_type':type(e).__name__,'error':str(e)})
 fields=['calls','received_usage_calls','unknown_usage_calls','input_tokens','cached_input_tokens','cache_write_tokens','output_tokens','received_cost_usd','charged_or_reserved_usd','summed_request_seconds'];model={k:sum(v[k]for r,v in roles.items()if r!='judge')for k in fields}
 return {'by_role':roles,'model_excluding_judge':model,'input_token_count_requests':0,'model_budget_calls_including_token_counts':model['calls'],'metric_errors':errors,'thinking_token_count_observed':False,'thinking_cost_included_in_output':True,'latency_note':'Parallel request durations are not native episode wall time.'}
