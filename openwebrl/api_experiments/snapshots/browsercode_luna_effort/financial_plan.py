from pathlib import Path
import json,hashlib
ROOT=Path(__file__).resolve().parent
def effective_plan(root=None):
 r=Path(root) if root is not None else ROOT
 return json.loads((r/'campaign-plan.json').read_text())
def enrollment_binding(root=None):return {'financial_approval_sha256':None}
NEW_CAPS={}
def verify_enrollment(entry,root=None):
 p=effective_plan(root);cfg=next(s for s in p['shards'] if s['arm']==entry['arm'] and s['shard']==entry['shard'])
 assert entry['shard_caps']==cfg,'Frozen enrolled shard config changed'
 return cfg
