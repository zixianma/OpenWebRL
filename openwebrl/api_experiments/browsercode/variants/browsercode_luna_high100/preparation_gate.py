from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def validate(require_approval=False):
 p=read(ROOT/'campaign-plan.json');assert hashlib.sha256(json.dumps({k:v for k,v in p.items() if k!='plan_sha256'},sort_keys=True).encode()).hexdigest()==p['plan_sha256']
 assert sha(ROOT/'tasks.json')==p['task_manifest_sha256'] and sha(ROOT/'study-tasks.json')==p['task_selection']['source_manifest_sha256']
 assert sha(ROOT/'target-manifest.json')==p['recovery']['target_manifest_sha256'] and sha(ROOT/'preserved-evidence.json')==p['recovery']['preserved_evidence_sha256']
 ids={x['task_id'] for x in read(ROOT/'tasks.json')};assert len(ids)==100
 saved=read(ROOT/'preserved-evidence.json');seen={a:set() for a in p['arms']};kept={a:set() for a in p['arms']}
 assert sha(Path(saved['independent_audit']))==saved['independent_audit_sha256'] and not read(Path(saved['independent_audit']))['errors']
 for a,rows in saved['valid'].items():
  for tid,ref in rows.items():
   f=Path(ref['path']);d=read(f);assert sha(f)==ref['sha256'] and d['valid'] is True and d['reward'] in [0,1] and tid in ids;kept[a].add(tid)
 m=read(ROOT/'execution-source-manifest.json')
 for name,digest in m['files'].items():assert sha(ROOT/name)==digest,name
 assert sha(ROOT/'javascript-runtime/bun')==m['runtime_dependency']['sha256']
 cfg=read(ROOT/'runtime-config.json');vendor=read(ROOT/'browsercode/source-manifest.json');assert vendor['revision']==p['protocol']['browsercode_revision']
 for name,digest in vendor['files'].items():assert sha(ROOT/'browsercode'/name)==digest
 for name,digest in cfg['support_source_hashes'].items():assert sha(Path(cfg['frozen_support_source'])/name)==digest
 assert p['protocol']['actor_model']==p['protocol']['selector_model']=='gpt-6-luna';assert p['protocol']['actor_effort']==p['protocol']['selector_effort']=='high';assert p['protocol']['actor_max_output_tokens']==p['protocol']['selector_max_output_tokens']==12288 and p['protocol']['max_steps']==30
 for i in range(10):
  s=read(ROOT/f'shard{i}/plan.json');assert s=={**p['shards'][i],'plan_sha256':p['plan_sha256']};a=s['arm'];assert len(s['task_ids'])==s['episode_count'];assert not seen[a]&set(s['task_ids']);seen[a].update(s['task_ids']);assert len(s['task_ids'])+1<=25
  if require_approval:
   ap=read(ROOT/f'shard{i}/approval.json');assert ap['approved'] and ap['plan_sha256']==p['plan_sha256'] and ap['execution_source_manifest_sha256']==sha(ROOT/'execution-source-manifest.json') and ap['limits']==s['limits'] and ap['resources']==p['resources_per_shard']
 assert [len(seen[a]) for a in p['arms']]==[86,95] and [len(kept[a]) for a in p['arms']]==[14,5]
 for a in seen:assert not seen[a]&kept[a] and seen[a]|kept[a]==ids
 assert sum(s['limits']['luna_usd_all_attempts'] for s in p['shards'])==40 and sum(s['limits']['judge_usd_all_attempts'] for s in p['shards'])==5
 return dict(valid=True,new_episodes=181,preserved_valid=19,paired_tasks=100,old_budget_transferred=False)
if __name__=='__main__':print(validate())
