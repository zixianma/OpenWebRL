from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def validate(require_approval=False):
 p=json.loads((ROOT/'campaign-plan.json').read_text()); payload={k:v for k,v in p.items() if k!='plan_sha256'}
 assert hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()==p['plan_sha256'],'plan hash mismatch'
 assert sha(ROOT/'tasks.json')==p['task_manifest_sha256'] and sha(ROOT/'study-tasks.json')==p['task_selection']['source_manifest_sha256']
 cfg=json.loads((ROOT/'runtime-config.json').read_text()); vendor=json.loads((ROOT/'browsercode/source-manifest.json').read_text())
 assert vendor['revision']==p['protocol']['browsercode_revision']
 assert (ROOT/'node_modules').resolve()==(ROOT.parent/'harness-paired100-extension-20261016-v1/node_modules').resolve()
 for name,want in vendor['files'].items():assert sha(ROOT/'browsercode'/name)==want, f'BrowserCode source changed: {name}'
 for name,want in cfg['support_source_hashes'].items():assert sha(Path(cfg['frozen_support_source'])/name)==want, f'support source changed: {name}'
 all_tasks=json.loads((ROOT/'tasks.json').read_text()); ids={x['task_id'] for x in all_tasks}; assert len(ids)==100
 audit=json.loads((ROOT/'predecessor-attempt-audit.json').read_text()); assert sha(ROOT/'predecessor-attempt-audit.json')==p['recovery']['predecessor_audit_sha256']
 old_approval=json.loads((ROOT.parent/'luna-high-selector-n3-n5-browsercode100-20261017-v1/approval.json').read_text())
 assert audit['old_approved_source_manifest_sha256']==old_approval['execution_source_manifest_sha256']
 m=json.loads((ROOT/'execution-source-manifest.json').read_text())
 for name,want in m['files'].items():assert sha(ROOT/name)==want, f'execution source changed: {name}'
 campaign=json.loads((ROOT/'approval.json').read_text()); assert campaign.get('approved') is True and campaign.get('plan_sha256')==p['plan_sha256'] and campaign.get('execution_source_manifest_sha256')==sha(ROOT/'execution-source-manifest.json')
 perarm={a:set() for a in p['arms']}
 for s in range(10):
  q=json.loads((ROOT/f'shard{s}/plan.json').read_text()); assert q=={**p['shards'][s],'plan_sha256':p['plan_sha256']}
  assert q['episode_count']==len(q['task_ids'])==len(q['task_orders']) and set(q['task_ids'])==set(q['task_orders'])
  assert q['arm'] in perarm and all(v==[{'harness':'browsercode','policy':q['arm']}] for v in q['task_orders'].values())
  perarm[q['arm']].update(q['task_ids'])
  a=json.loads((ROOT/f'shard{s}/approval.json').read_text())
  if require_approval: assert a.get('approved') is True and a.get('plan_sha256')==p['plan_sha256'] and a.get('execution_source_manifest_sha256')==sha(ROOT/'execution-source-manifest.json') and a.get('resources')==p['resources_per_shard'] and a.get('limits')==p['shards'][s]['limits']
 assert perarm[p['arms'][0]]==perarm[p['arms'][1]]
 assert len(perarm[p['arms'][0]])==p['recovery']['paired_task_count']
 assert p['planned_episodes']==2*p['recovery']['paired_task_count']
 assert len(audit['jobs'])==10 and all(x['luna_usage_unknown_calls']>=0 for x in audit['jobs'])
 return {'valid':True,'paired_tasks':len(perarm[p['arms'][0]]),'episodes':p['planned_episodes'],'shards':10,'partial':True}
if __name__=='__main__':
 import argparse;q=argparse.ArgumentParser();q.add_argument('--require-approval',action='store_true');a=q.parse_args();print(json.dumps(validate(a.require_approval)))
