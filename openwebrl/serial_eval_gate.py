"""Durable serial-evaluation promotion gates, independent of model imports."""
import hashlib
import json
from pathlib import Path

RUNTIME_FILES = ['openwebrl/serial_actions.py', 'openwebrl/serial_eval_gate.py',
    'openwebrl/generate_browser.py', 'openwebrl/arm_eval.py',
    'openwebrl/run_evaluate.py', 'openwebrl/reward_browser.py',
    'openwebrl/base/utils.py', 'openwebrl/base/adapter.py',
    'openwebrl/adapters/browser_adapter.py']


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def identity(model,variant,task_file,repo):
    model=Path(model).resolve();repo=Path(repo)
    manifest=model/'merge-manifest.json';merge=json.loads(manifest.read_text())
    checkpoint=Path(merge['checkpoint']).resolve()
    checksum=digest(checkpoint/'adapter_model.safetensors')
    if checksum!=merge['adapter_sha256']:raise ValueError('Gate: merged model adapter provenance differs')
    return dict(variant=variant,model=str(model),checkpoint=str(checkpoint),adapter_sha256=checksum,
        merge_manifest_sha256=digest(manifest),task_file_sha256=digest(task_file),
        runtime_hashes={name:digest(repo/name) for name in RUNTIME_FILES},
        sampling=dict(temperature=.7,top_p=.9,max_new_tokens=6144,seed=42,max_steps=30),
        judge='o4-mini',judge_protocol='online_mind2web/AgentTrek',
        history='selected alternative reasoning/action',
        model_metadata_sha256={p.name:digest(p) for p in model.glob('*.json') if p.name!='merge-manifest.json'})


def validate_gate(report,current,*,pilot=False,task_ids=()):
    if report.get('schema_version')!=1 or report.get('passed') is not True:
        raise ValueError('Gate: missing or failed validation')
    if report.get('identity')!=current:raise ValueError('Gate: model/runtime/settings identity changed')
    stage=report.get('stage')
    if pilot:
        if stage!='free_generation' or not 1<=len(task_ids)<=5:
            raise ValueError('Gate: pilot requires free-generation proof and at most five tasks')
        if list(task_ids)!=report.get('pilot_task_ids'):
            raise ValueError('Gate: pilot cohort differs from frozen cohort')
    elif stage!='browser_pilot':
        raise ValueError('Gate: broad evaluation requires a passed browser pilot')
    if report.get('checks',{}).get('adapter_merge_parity') is not True:
        raise ValueError('Gate: adapter/merge parity missing')
    if report.get('checks',{}).get('free_generation') is not True:
        raise ValueError('Gate: free generation missing')
    if not pilot and report.get('checks',{}).get('browser_pilot') is not True:
        raise ValueError('Gate: browser execution missing')
    if not report.get('artifacts'):raise ValueError('Gate: missing evidence artifacts')
    for name,checksum in report['artifacts'].items():
        if digest(name)!=checksum:raise ValueError('Gate: validation artifact changed')
    return digest_json(report)


def digest_json(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def load_gate(path,current,*,pilot=False,task_ids=()):
    if not path:raise ValueError('Serial evaluation requires --serial-gate; broad evaluation is blocked until the pilot passes')
    report=json.loads(Path(path).read_text())
    return validate_gate(report,current,pilot=pilot,task_ids=task_ids)


def generation_pass(records,expected_ids):
    expected={(i,t) for i in expected_ids for t in (0.,.7)}
    keys=[(r['id'],r['temperature']) for r in records]
    return (len(keys)==len(expected) and set(keys)==expected and
        all(r.get('valid') is True and r.get('finish')=='stop' and r.get('prompt_tokens_match') is True for r in records))


def browser_pass(records,expected_ids):
    """Mechanics gate, not a minimum task-success threshold."""
    if len(records)!=len(expected_ids) or {r['task_id'] for r in records}!=set(expected_ids):return False
    if any('serial_protocol_error' in str(r.get('terminate_reason','')) or
           r.get('terminate_reason')=='generation_length_limit' for r in records):return False
    valid=[r for r in records if r.get('valid')]
    if len(valid)<3:return False
    for r in valid:
        meta=r.get('metadata',{})
        if not meta.get('serial_action_executed') or not meta.get('serial_history_projected'):return False
        messages=meta.get('messages',[])
        history=[m for m in messages if m.get('role')=='assistant']
        if not history or any('<alternative' in str(m.get('content','')) for m in history):return False
    # At least two trajectories must exercise the next-turn projected history.
    return sum(len([m for m in r['metadata'].get('messages',[]) if m.get('role')=='assistant'])>=2 for r in valid)>=2
