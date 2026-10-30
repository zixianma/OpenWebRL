from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parent
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def validate(require_approval=False):
 p=json.loads((ROOT/'campaign-plan.json').read_text());payload={k:v for k,v in p.items() if k!='plan_sha256'}
 assert hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()==p['plan_sha256'],'plan hash mismatch'
 assert sha(ROOT/'tasks.json')==p['task_manifest_sha256'],'task manifest changed'
 assert sha(ROOT/'study-tasks.json')==p['task_selection']['source_manifest_sha256'],'sampling frame changed'
 cfg=json.loads((ROOT/'runtime-config.json').read_text()); vendor=json.loads((ROOT/'browsercode/source-manifest.json').read_text())
 assert vendor['revision']==p['protocol']['browsercode_revision'],'BrowserCode revision changed'
 assert (ROOT/'node_modules').resolve()==(ROOT.parent/'harness-paired20-preparation-v1/node_modules').resolve(),'pinned JavaScript dependencies changed'
 for n in ['package.json','bun.lock']: assert (ROOT/n).is_file(),f'missing dependency lock {n}'
 for name,want in vendor['files'].items(): assert sha(ROOT/'browsercode'/name)==want,f'BrowserCode source changed: {name}'
 for name,want in cfg['support_source_hashes'].items(): assert sha(Path(cfg['frozen_support_source'])/name)==want,f'frozen support source changed: {name}'
 tasks=json.loads((ROOT/'tasks.json').read_text()); study=json.loads((ROOT/'study-tasks.json').read_text())
 assert len(tasks)==80 and len({x['task_id'] for x in tasks})==80
 assert len(study)==100 and len({x['task_id'] for x in study})==100
 shards=p['shards']; assert len(shards)==4
 all_ids=[]
 for s,sp in enumerate(shards):
  q=json.loads((ROOT/f'shard{s}/plan.json').read_text()); assert q=={**sp,'plan_sha256':p['plan_sha256']}
  assert q['episode_count']==20 and len(q['task_ids'])==20 and set(q['task_ids'])==set(q['task_orders'])
  all_ids.extend(q['task_ids'])
 assert len(set(all_ids))==80 and set(all_ids)=={x['task_id'] for x in tasks}
 m=json.loads((ROOT/'execution-source-manifest.json').read_text())
 for name,want in m['files'].items(): assert sha(ROOT/name)==want,f'execution source changed: {name}'
 campaign=json.loads((ROOT/'approval.json').read_text()); assert campaign['approved'] and campaign['plan_sha256']==p['plan_sha256']
 for s in range(4):
  a=json.loads((ROOT/f'shard{s}/approval.json').read_text())
  if require_approval:
   assert a.get('approved') is True and a.get('plan_sha256')==p['plan_sha256'] and a.get('resources')==p['resources_per_shard'] and a.get('limits')==p['limits_per_shard'],'exact shard approval missing/mismatched'
   assert a.get('execution_source_manifest_sha256')==sha(ROOT/'execution-source-manifest.json'),'execution source is not approval-bound'
 return {'valid':True,'approved':all(json.loads((ROOT/f'shard{s}/approval.json').read_text()).get('approved') for s in range(4)),'tasks':80,'episodes':80,'shards':4}
if __name__=='__main__':
 import argparse
 q=argparse.ArgumentParser();q.add_argument('--require-approval',action='store_true');x=q.parse_args();print(json.dumps(validate(x.require_approval)))
