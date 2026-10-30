"""New Luna single arm, frozen300 tasks, no original evidence writes; exact caps fail closed."""
from pathlib import Path
import json,hashlib
ROOT=Path(__file__).resolve().parent
read=lambda p:json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def validate(require_approval=False):
 p=read(ROOT/'campaign-plan.json');assert hashlib.sha256(json.dumps({k:v for k,v in p.items() if k!='plan_sha256'},sort_keys=True).encode()).hexdigest()==p['plan_sha256']
 ref=Path(p['reference_campaign']);assert sha(ROOT/'tasks.json')==p['task_manifest_sha256']==p['original_task_manifest_sha256']==sha(ref/'tasks.json');assert read(ref/'completion.json')['verified_complete'] and sha(ref/'completion.json')==p['reference_completion_sha256']
 ids={t['task_id'] for t in read(ROOT/'tasks.json')};assert len(ids)==300;src=read(ROOT/'task-source-audit.json');assert sha(src['source'])==src['source_sha256']
 assert p['arm_protocols']=={'luna_single_high':{'actor_model':'gpt-6-luna','selector_model':None,'candidates':1}} and p['protocol']==read(ref/'campaign-plan.json')['protocol']
 assert len(p['shards'])==5 and p['campaign_caps']==dict(jobs=5,gpus=0,concurrent_cpus=20,concurrent_memory_gib=80,allocation_job_hours=15,model_usd=15,judge_usd=15,total_api_usd=30,model_calls_including_token_counts=25000,judge_calls=350,browsers=350)
 seen=set()
 for i,s in enumerate(p['shards']):
  assert s['shard']==i and s['arm']=='luna_single_high' and s['episode_count']==60 and len(s['task_ids'])==60 and not seen&set(s['task_ids']);seen.update(s['task_ids']);assert all(order==[{'harness':'browsercode','policy':'luna_single_high'}] for order in s['task_orders'].values())
  assert s['limits']==dict(model_usd_all_attempts=3,model_calls_all_attempts=5000,judge_usd_all_attempts=3,judge_calls_all_attempts=70,browser_sessions_all_attempts=70)
  assert read(ROOT/f'shard{i}/plan.json')=={**s,'plan_sha256':p['plan_sha256']};a=read(ROOT/f'shard{i}/approval.json');assert a['plan_sha256']==p['plan_sha256'] and a['resources']==p['resources_per_shard'] and a['limits']==s['limits']
  if require_approval:assert a['approved'] and a['execution_source_manifest_sha256']==sha(ROOT/'execution-source-manifest.json')
 assert seen==ids and p['resources_per_shard']==dict(billing_account='vlm',cpus=4,gpus=0,memory_gib=16,total_seconds_all_attempts=10800)
 m=read(ROOT/'execution-source-manifest.json')
 for name,digest in m['files'].items():assert sha(ROOT/name)==digest,name
 assert sha(ROOT/'javascript-runtime/bun')==m['runtime_dependency']['sha256'];cfg=read(ROOT/'runtime-config.json');vendor=read(ROOT/'browsercode/source-manifest.json');assert vendor['revision']==p['protocol']['browsercode_revision']
 for name,digest in vendor['files'].items():assert sha(ROOT/'browsercode'/name)==digest
 for name,digest in cfg['support_source_hashes'].items():assert sha(Path(cfg['frozen_support_source'])/name)==digest
 if require_approval:
  a=read(ROOT/'approval.json');assert a['approved'] and a['plan_sha256']==p['plan_sha256'] and a['campaign_caps']==p['campaign_caps'] and a['execution_source_manifest_sha256']==sha(ROOT/'execution-source-manifest.json') and a.get('approved_epoch') and a.get('user_approval_quote')
 return {'valid':True,'tasks':300,'new_episodes':300,'shards':5,'model':'gpt-6-luna','candidates':1,'step_limit':30,'approved':read(ROOT/'approval.json')['approved']}
if __name__=='__main__':print(validate())
