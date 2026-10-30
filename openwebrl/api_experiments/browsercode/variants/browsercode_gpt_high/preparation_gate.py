"""Fail closed on exact model, source, task, preserved evidence and approval bindings."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def validate(require_approval=False):
 p=read(ROOT/'campaign-plan.json');assert hashlib.sha256(json.dumps({k:v for k,v in p.items() if k!='plan_sha256'},sort_keys=True).encode()).hexdigest()==p['plan_sha256'];assert sha(ROOT/'tasks.json')==p['task_manifest_sha256'];assert sha(ROOT/'preserved-evidence.json')==p['preserved_evidence_sha256'];ids={t['task_id'] for t in read(ROOT/'tasks.json')};assert len(ids)==300
 a=read(ROOT/'task-source-audit.json');assert sha(Path(a['source']))==a['source_sha256'];saved=read(ROOT/'preserved-evidence.json');assert sha(Path(saved['completion']))==saved['completion_sha256'] and read(Path(saved['completion']))['verified_complete'];assert sha(Path(saved['audit']))==saved['audit_sha256'] and read(Path(saved['audit']))['verified_phase_complete'];kept={arm:set(v) for arm,v in saved['records'].items()}
 for arm,rows in saved['records'].items():
  for tid,ref in rows.items():
   f=Path(ref['path']);d=read(f);assert tid in ids and sha(f)==ref['sha256'] and d['task_id']==tid and d['mode']==arm and d['valid']==ref['valid'];assert (d['reward'] in [0,1]) if d['valid'] else d['reward'] is None
 assert sum(len(v) for v in kept.values())==200
 assert p['arm_protocols']=={'serial_high':{'actor_model':'gpt-6-sol','selector_model':None,'candidates':1},'parallel3_select_high':{'actor_model':'gpt-6-luna','selector_model':'gpt-6-luna','candidates':3},'parallel5_select_high':{'actor_model':'gpt-6-luna','selector_model':'gpt-6-luna','candidates':5}}
 assert p['protocol']['actor_effort']==p['protocol']['selector_effort']=='high' and p['protocol']['actor_max_output_tokens']==p['protocol']['selector_max_output_tokens']==12288 and p['protocol']['max_steps']==30
 seen={a:set() for a in p['arms']}
 for i,s in enumerate(p['shards']):
  assert read(ROOT/f'shard{i}/plan.json')=={**s,'plan_sha256':p['plan_sha256']};arm=s['arm'];assert not seen[arm]&set(s['task_ids']);seen[arm].update(s['task_ids']);assert len(s['task_ids'])==s['episode_count'] and s['episode_count']+1<=s['limits']['browser_sessions_all_attempts']
  ap=read(ROOT/f'shard{i}/approval.json');assert ap['plan_sha256']==p['plan_sha256'] and ap['resources']==p['resources_per_shard'] and ap['limits']==s['limits']
  if require_approval:assert ap['approved'] is True and ap['execution_source_manifest_sha256']==sha(ROOT/'execution-source-manifest.json'),'Exact shard resource approval is required before launch'
 assert seen['serial_high']==ids
 for arm,k in kept.items():assert not seen[arm]&k and seen[arm]|k==ids
 assert [len(seen[a]) for a in p['arms']]==[300,200,200]
 caps=p['campaign_caps'];assert len(p['shards'])==20 and sum(s['limits']['model_usd_all_attempts'] for s in p['shards'])==caps['model_usd']==200;assert sum(s['limits']['judge_usd_all_attempts'] for s in p['shards'])==caps['judge_usd']==20;assert sum(s['limits']['model_calls_all_attempts'] for s in p['shards'])==caps['model_calls_including_token_counts']==124000
 m=read(ROOT/'execution-source-manifest.json')
 for name,digest in m['files'].items():assert sha(ROOT/name)==digest,name
 assert sha(ROOT/'javascript-runtime/bun')==m['runtime_dependency']['sha256'];cfg=read(ROOT/'runtime-config.json');vendor=read(ROOT/'browsercode/source-manifest.json');assert vendor['revision']==p['protocol']['browsercode_revision']
 for name,digest in vendor['files'].items():assert sha(ROOT/'browsercode'/name)==digest
 for name,digest in cfg['support_source_hashes'].items():assert sha(Path(cfg['frozen_support_source'])/name)==digest
 if require_approval:
  ap=read(ROOT/'approval.json');assert ap['approved'] is True and ap['plan_sha256']==p['plan_sha256'] and ap['campaign_caps']==caps,'Exact campaign resource/API approval is required before launch'
 return {'valid':True,'tasks':300,'new_episodes':700,'preserved_records':200,'approved':read(ROOT/'approval.json')['approved'],'models':['gpt-6-sol','gpt-6-luna']}
if __name__=='__main__':print(validate())
