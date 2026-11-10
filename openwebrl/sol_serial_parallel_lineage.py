"""Allow hashed mechanical-recovery ancestors without rewriting saved evidence."""
from pathlib import Path
import hashlib
import json
from openwebrl.luna_qwen_metrics import digest
from openwebrl.luna_qwen_eval import verified_record

# Only implementation identity and preparation status can change mechanically.
IMPLEMENTATION_FIELDS = {'source', 'source_hashes', 'source_manifest_sha256', 'prior_plans', 'status', 'approved'}

def accepted_plan_hashes(plan):
    hashes = {digest(plan)}
    scientific = {k:v for k,v in plan.items() if k not in IMPLEMENTATION_FIELDS}
    for ref in plan.get('prior_plans', []):
        path = Path(ref['path'])
        if hashlib.sha256(path.read_bytes()).hexdigest() != ref['sha256']:
            raise ValueError('Mechanical recovery ancestor bytes changed')
        old = json.loads(path.read_text())
        if digest(old) != ref['plan_sha256']:
            raise ValueError('Mechanical recovery ancestor identity changed')
        if {k:v for k,v in old.items() if k not in IMPLEMENTATION_FIELDS} != scientific:
            raise ValueError('Recovery changed scientific protocol, cohort, outputs or caps')
        hashes.add(ref['plan_sha256'])
    return hashes

def verify_existing_records(plan, plan_sha256):
    if plan_sha256 != digest(plan):
        raise ValueError('Current plan identity mismatch')
    allowed = accepted_plan_hashes(plan)
    expected = {digest([item['task_id'], mode])[:24]:(item['task_id'],mode)
                for item in plan['schedule'] for mode in item['modes']}
    for path in (Path(plan['output'])/'records').glob('*.json'):
        if path.stem not in expected:
            raise ValueError('Unexpected saved paired episode')
        row = verified_record(path, digest(plan['protocol']))
        if (row.get('task_id'),row.get('mode')) != expected[path.stem]:
            raise ValueError('Saved episode filename/identity differs')
        if (row.get('experiment_id'),row.get('family')) != (plan['experiment_id'],plan['family']):
            raise ValueError('Saved episode belongs to another study')
        if row.get('plan_sha256') not in allowed:
            raise ValueError('Saved episode belongs to an unregistered plan ancestor')
