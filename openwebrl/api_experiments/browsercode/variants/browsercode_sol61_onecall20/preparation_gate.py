"""Fail-closed checks for the one-call BrowserCode pilot."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def validate(require_approval=False):
 p=json.loads((ROOT/'campaign-plan.json').read_text());payload={k:v for k,v in p.items() if k!='plan_sha256'}
 assert hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()==p['plan_sha256'],'plan hash mismatch'
 assert sha(ROOT/'tasks.json')==p['task_manifest_sha256'],'task manifest changed'
 assert sha(ROOT/'study-tasks.json')==p['task_selection']['source_manifest_sha256'],'sampling frame changed'
 cfg=json.loads((ROOT/'runtime-config.json').read_text()); vendor=json.loads((ROOT/'browsercode/source-manifest.json').read_text())
 assert vendor['revision']==p['protocol']['browsercode_revision'],'BrowserCode revision changed'
 assert (ROOT/'node_modules').resolve()==(Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/evaluations/serial-vs-parallel-reasoning-20261013/harness-paired20-preparation-v1/node_modules')).resolve(),'pinned JavaScript dependencies changed'
 for n in ['package.json','bun.lock']: assert (ROOT/n).is_file(),f'missing dependency lock {n}'
 for name,want in vendor['files'].items(): assert sha(ROOT/'browsercode'/name)==want, f'BrowserCode source changed: {name}'
 for name,want in cfg['support_source_hashes'].items(): assert sha(Path(cfg['frozen_support_source'])/name)==want, f'frozen support source changed: {name}'
 tasks=json.loads((ROOT/'tasks.json').read_text());assert len(tasks)==20 and len({t['task_id'] for t in tasks})==20
 sp=json.loads((ROOT/'shard0/plan.json').read_text());assert sp['plan_sha256']==p['plan_sha256'] and sp['episode_count']==20 and len(sp['task_orders'])==20
 assert set(sp['task_ids'])=={t['task_id'] for t in tasks}
 m=json.loads((ROOT/'execution-source-manifest.json').read_text())
 for name,want in m['files'].items(): assert sha(ROOT/name)==want, f'execution source changed: {name}'
 a=json.loads((ROOT/'approval.json').read_text())
 if a.get('execution_source_manifest_sha256') != sha(ROOT/'execution-source-manifest.json'): raise ValueError('execution source manifest is not approval-bound')
 if require_approval:
  assert a.get('approved') is True and a.get('plan_sha256')==p['plan_sha256'] and a.get('resources')==p['resources'] and a.get('limits')==p['limits'],'exact approval missing/mismatched'
 return {'valid':True,'approved':a.get('approved'),'tasks':20,'episodes':20}
if __name__=='__main__':
 import argparse
 q=argparse.ArgumentParser();q.add_argument('--require-approval',action='store_true');x=q.parse_args();print(json.dumps(validate(x.require_approval)))
