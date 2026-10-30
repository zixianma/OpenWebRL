"""Fail closed on task/source/model identity, explicit approvals and all-attempt caps."""
from pathlib import Path
import hashlib,json,collections
ROOT=Path(__file__).resolve().parent
read=lambda p:json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(1024*1024),b''):h.update(x)
 return h.hexdigest()
def validate(require_approval=False):
 p=read(ROOT/'campaign-plan.json');assert hashlib.sha256(json.dumps({k:v for k,v in p.items() if k!='plan_sha256'},sort_keys=True).encode()).hexdigest()==p['plan_sha256'];assert sha(ROOT/'tasks.json')==p['task_manifest_sha256'];assert p['planned_dispositions']==2980 and len(p['shards'])==40
 protocols={'serial_high':{'actor_model':'gpt-6-sol','selector_model':None,'candidates':1},'parallel3_select_high':{'actor_model':'gpt-6-luna','selector_model':'gpt-6-luna','candidates':3},'parallel5_select_high':{'actor_model':'gpt-6-luna','selector_model':'gpt-6-luna','candidates':5},'luna_single_high':{'actor_model':'gpt-6-luna','selector_model':None,'candidates':1}};assert p['arm_protocols']==protocols
 assert p['protocol']['actor_max_output_tokens']==p['protocol']['selector_max_output_tokens']==12288 and p['protocol']['actor_effort']==p['protocol']['selector_effort']=='high' and p['protocol']['max_decisions']==30
 tasks=read(ROOT/'tasks.json');assert len(tasks)==745;byid={x['task_id']:x for x in tasks};assert len(byid)==745;expected={};actual=collections.defaultdict(list)
 for bench,b in p['benchmarks'].items():
  assert sha(b['source_file'])==b['source_sha256'] and sha(b['native_judge_source'])==b['native_judge_source_sha256'];raw=[json.loads(x) for x in Path(b['source_file']).read_text().splitlines()];assert len(raw)==b['tasks']==({'webvoyager':595,'deepshop':150}[bench]);mapping={(x['id'] if bench=='webvoyager' else x['task_id']):x for x in raw};expected[bench]={t['task_id'] for t in tasks if t['benchmark']==bench}
  for t in tasks:
   if t['benchmark']!=bench:continue
   x=mapping[t['native_task_id']];assert t['native_metadata']==x and t['source_row_sha256']==hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest();assert t['intent']==(x['ques'] if bench=='webvoyager' else x['task_name']) and t['start_url']==(x['web'] if bench=='webvoyager' else x['website'])
 totals=collections.Counter()
 for i,s in enumerate(p['shards']):
  assert i==s['shard'] and s['arm'] in protocols and read(ROOT/f'shard{i}/plan.json')=={**s,'plan_sha256':p['plan_sha256']};wv=s['benchmark']=='webvoyager';resources={'billing_account':'vlm','gpus':0,'cpus':4,'memory_gib':16,'total_seconds_all_attempts':21600 if wv else 10800};assert s['resources']==resources
  assert s['episode_count']==len(s['task_ids']) and set(s['task_ids'])<=expected[s['benchmark']];actual[(s['benchmark'],s['arm'])]+=s['task_ids'];a=read(ROOT/f'shard{i}/approval.json');assert a['resources']==resources and a['limits']==s['limits'] and a['plan_sha256']==p['plan_sha256']
  ai=list(protocols).index(s['arm']);assert s['limits']=={'model_usd_all_attempts':([35,15,20,10] if wv else [10,5,7,3])[ai],'model_calls_all_attempts':([9000,24000,32000,6000] if wv else [2400,6000,8000,2400])[ai],'judge_usd_all_attempts':12 if wv else 3,'judge_calls_all_attempts':130 if wv else 35,'browser_sessions_all_attempts':135 if wv else 35}
  assert s['episode_count']+1<=s['limits']['browser_sessions_all_attempts']
  for k,v in s['limits'].items():totals[k]+=v
  if require_approval:assert a['approved'] and a['execution_source_manifest_sha256']==sha(ROOT/'execution-source-manifest.json')
 for bench in expected:
  for arm in protocols:assert len(actual[(bench,arm)])==len(expected[bench]) and set(actual[(bench,arm)])==expected[bench]
 assert totals=={'model_usd_all_attempts':525,'model_calls_all_attempts':449000,'judge_usd_all_attempts':300,'judge_calls_all_attempts':3300,'browser_sessions_all_attempts':3400}
 m=read(ROOT/'execution-source-manifest.json')
 for name,digest in m['files'].items():assert sha(ROOT/name)==digest,name
 assert sha(ROOT/'javascript-runtime/bun')==m['runtime_dependency']['sha256'];cfg=read(ROOT/'runtime-config.json');vendor=read(ROOT/'browsercode/source-manifest.json');assert vendor['revision']==p['protocol']['browsercode_revision']
 for name,digest in vendor['files'].items():assert sha(ROOT/'browsercode'/name)==digest
 for name,digest in cfg['support_source_hashes'].items():assert sha(Path(cfg['frozen_support_source'])/name)==digest
 if require_approval:
  a=read(ROOT/'approval.json');assert a['approved'] and a['plan_sha256']==p['plan_sha256'] and a['campaign_caps']==p['campaign_caps'] and a['execution_source_manifest_sha256']==sha(ROOT/'execution-source-manifest.json')
 return {'valid':True,'tasks':745,'dispositions':2980,'arms':4,'jobs':40,'approved':read(ROOT/'approval.json')['approved']}
if __name__=='__main__':print(validate(True))
