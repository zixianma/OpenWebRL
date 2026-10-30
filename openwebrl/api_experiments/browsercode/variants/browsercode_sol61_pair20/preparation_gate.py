"""Validate the matched experiment and refuse allocation/API work before approval."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()

def validate(require_approval=False):
    plan=json.loads((ROOT/'campaign-plan.json').read_text())
    assert digest({k:v for k,v in plan.items() if k!='plan_sha256'})==plan['plan_sha256'],'Plan changed'
    tasks=json.loads((ROOT/'tasks.json').read_text())
    assert len(tasks)==20 and len({t['task_id'] for t in tasks})==20
    assert hashlib.sha256((ROOT/'tasks.json').read_bytes()).hexdigest()==plan['task_manifest_sha256']
    grid=[(s['shard'],t,cell['harness'],cell['policy']) for s in plan['shards'] for t in s['task_ids'] for cell in plan['cells']]
    assert len(grid)==80 and len({(t,h,m) for _,t,h,m in grid})==80
    assert {t for _,t,_,_ in grid}=={t['task_id'] for t in tasks}
    assert all(len(s['task_ids'])==5 for s in plan['shards'])
    for n,want in plan['preparation_source_hashes'].items():
        assert hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==want,n
    vendor=json.loads((ROOT/'browsercode/source-manifest.json').read_text())
    assert vendor['revision']==plan['browsercode_revision']
    for n,want in vendor['files'].items():
        assert hashlib.sha256((ROOT/'browsercode'/n).read_bytes()).hexdigest()==want,n
    preflight=json.loads((ROOT/'offline-native-preflight.json').read_text())
    assert preflight['results']['execution'] and preflight['results']['model_requests']==0 and preflight['results']['browser_sessions']==0
    approval=json.loads((ROOT/'approval.json').read_text())
    if require_approval and (not approval['approved'] or approval['plan_sha256']!=plan['plan_sha256']):
        raise PermissionError('Exact four-shard compute/browser/API approval required before sbatch or physical requests')
    return {'valid':True,'approved':approval['approved'],'fixed_tasks':20,'cells':4,'episodes':80,'shards':4}

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--require-approval',action='store_true');args=parser.parse_args()
    print(json.dumps(validate(args.require_approval)))
