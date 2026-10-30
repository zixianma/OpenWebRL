from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parent
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def validate(require_approval=False):
 p=json.loads((ROOT/'campaign-plan.json').read_text());payload={k:v for k,v in p.items() if k!='plan_sha256'}
 assert hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()==p['plan_sha256'],'plan hash mismatch'
 assert sha(ROOT/'tasks.json')==p['task_manifest_sha256'] and sha(ROOT/'study-tasks.json')==p['task_selection']['source_manifest_sha256']
 cfg=json.loads((ROOT/'runtime-config.json').read_text());vendor=json.loads((ROOT/'browsercode/source-manifest.json').read_text())
 assert vendor['revision']==p['protocol']['browsercode_revision']
 assert (ROOT/'node_modules').resolve()==(ROOT.parent/'harness-paired100-extension-20261016-v1/node_modules').resolve()
 for name,want in vendor['files'].items(): assert sha(ROOT/'browsercode'/name)==want,f'BrowserCode source changed: {name}'
 for name,want in cfg['support_source_hashes'].items(): assert sha(Path(cfg['frozen_support_source'])/name)==want,f'support source changed: {name}'
 tasks=json.loads((ROOT/'tasks.json').read_text());assert len(tasks)==100 and len({x['task_id'] for x in tasks})==100
 pids={x['task_id'] for x in tasks};shards=p['shards'];assert len(shards)==10
 by_arm={m:set() for m in p['arms']}
 for s,sp in enumerate(shards):
  q=json.loads((ROOT/f'shard{s}/plan.json').read_text());assert q=={**sp,'plan_sha256':p['plan_sha256']}
  assert q['episode_count']==20 and len(q['task_ids'])==20 and set(q['task_ids'])==set(q['task_orders'])
  assert q['arm'] in by_arm and all(v==[{'harness':'browsercode','policy':q['arm']}] for v in q['task_orders'].values())
  assert not by_arm[q['arm']] & set(q['task_ids']);by_arm[q['arm']].update(q['task_ids'])
 assert all(v==pids for v in by_arm.values()),'Each condition must cover exactly the common 100 tasks'
 m=json.loads((ROOT/'execution-source-manifest.json').read_text())
 for name,want in m['files'].items(): assert sha(ROOT/name)==want,f'execution source changed: {name}'
 campaign=json.loads((ROOT/'approval.json').read_text())
 assert campaign.get('approved') is True and campaign.get('plan_sha256')==p['plan_sha256'] and campaign.get('execution_source_manifest_sha256')==sha(ROOT/'execution-source-manifest.json')
 for s in range(10):
  a=json.loads((ROOT/f'shard{s}/approval.json').read_text())
  if require_approval: assert a.get('approved') is True and a.get('plan_sha256')==p['plan_sha256'] and a.get('execution_source_manifest_sha256')==sha(ROOT/'execution-source-manifest.json') and a.get('resources')==p['resources_per_shard'] and a.get('limits')==p['shards'][s]['limits']
 return {'valid':True,'tasks_per_arm':100,'episodes':200,'shards':10}
if __name__=='__main__':
 import argparse;q=argparse.ArgumentParser();q.add_argument('--require-approval',action='store_true');a=q.parse_args();print(json.dumps(validate(a.require_approval)))
