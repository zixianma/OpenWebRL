"""Freeze the fresh OM2W-300 rerun packages J1/J2/J3 (actor gpt-4o-2024-08-06, judge gpt-5.6-terra).

User approvals 2026-10-23: rerun all 300 tasks x 5 methods fresh ("Approve, account vlm"), judge
gpt-5.6-terra ("ok yes use gpt5.6 terra"). OpenAI shut down gpt-4o-2024-05-13 and o4-mini on
2026-10-23. The only scientific change from the earlier campaigns is that model substitution (plus its
token prices); prompts, critics, routes, selectors, judge prompt/parser and worker code are unchanged.
Parent: full300-four-arms-vlm. Never reserves or submits.

Usage: build_rerun.py J1|J2|J3
"""
import collections, hashlib, json, os, shutil, sys
from pathlib import Path

O = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-om2w-table5-20261012')
HERE = Path(__file__).resolve().parent
PARENT = O/'full300-four-arms-vlm'
RECEIPT = HERE/'user-approval-20261023.json'
JUDGE_RECEIPT = HERE/'judge-choice-20261023.json'
OLD_ACTOR, NEW_ACTOR = 'gpt-4o-2024-05-13', 'gpt-4o-2024-08-06'
OLD_JUDGE, NEW_JUDGE = 'o4-mini-2025-04-16', 'gpt-5.6-terra'
PRICES = {NEW_ACTOR: [2.5e-06, 1e-05], NEW_JUDGE: [2e-06, 1.2e-05]}
STAGE = sys.argv[1]
SPEC = dict(J1=dict(selectors=['no_search', 'random5'], local=12),
            J2=dict(selectors=['arbiter7b', 'piotr_selectionarm'], local=120),
            J3=dict(selectors=['shepherd8b'], local=181))[STAGE]
NEW = O/f'rerun300-{STAGE.lower()}-vlm'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(Path(p).read_text())
sub = lambda s: s.replace(OLD_ACTOR, NEW_ACTOR).replace(OLD_JUDGE, NEW_JUDGE)
assert not NEW.exists()
keep_root = {'campaign.sbatch', 'proposal.json', 'resource-request.json', 'worker-template.json', 'source'}
shutil.copytree(PARENT, NEW, symlinks=True, ignore=lambda d, names: [n for n in names if Path(d) == PARENT and n not in keep_root])
os.chmod(NEW, 0o755)

# 1. Source: model constants only.
edits = {
    'openwebrl/webprm_om2w_table5.py': [("JUDGE = 'o4-mini-2025-04-16'", "JUDGE = 'gpt-5.6-terra'  # o4-mini shut down 2026-10-23; user-chosen replacement"),
                                        ("JUDGE_PRICES = (1.10e-6, 4.40e-6)", "JUDGE_PRICES = (2.00e-6, 12.00e-6)")],
    'openwebrl/webprm_webarena_worker.py': [("ACTOR_SNAPSHOTS = (ACTOR, 'gpt-4o-2024-05-13')",
                                             "ACTOR_SNAPSHOTS = (ACTOR, 'gpt-4o-2024-08-06')  # 05-13 shut down 2026-10-23")],
}
launch_pairs = [
    ("    if isinstance(extension, dict) and extension.get('identity') == FULL300_ID:\n",
     "    if isinstance(extension, dict) and extension.get('identity') == RERUN300_ID:\n"
     "        return rerun300(proposal, template, extension)\n"
     "    if isinstance(extension, dict) and extension.get('identity') == FULL300_ID:\n"),
]
rerun_helper = '''RERUN300_ID = 'om2w_rerun300_0806_terra_v1'
RERUN_SUBSTITUTION = {'gpt-4o-2024-05-13': 'gpt-4o-2024-08-06', 'o4-mini-2025-04-16': 'gpt-5.6-terra'}


def _substituted(value):
    text = json.dumps(value, sort_keys=True)
    for old, new in RERUN_SUBSTITUTION.items():
        text = text.replace(old, new)
    return json.loads(text)


def rerun300(proposal, template, extension):
    """Fresh 300-task cells; identical science to the parent except the approved actor/judge substitution."""
    keys = {'identity', 'paid_dispatch_interval_seconds', 'requires_closed_job_id', 'selectors', 'approval_stage',
            'parent_proposal', 'parent_template', 'user_approval', 'judge_choice', 'substitution'}
    if (set(extension) != keys or extension['paid_dispatch_interval_seconds'] != 3.0
            or extension['requires_closed_job_id'] != '373588' or extension['substitution'] != RERUN_SUBSTITUTION
            or any(k in proposal for k in ('candidate_controls', 'repair_of', 'predecessor', 'preproduction_ledger'))):
        raise ValueError('Exact rerun300 extension required')
    source = proposal['source_sha256']; values = {}
    for key in ('parent_proposal', 'parent_template', 'user_approval', 'judge_choice'):
        item = extension[key]
        if (not isinstance(item, dict) or set(item) != {'path', 'sha256'} or sha(checked(Path(item['path']))) != item['sha256']
                or source.get(item['path']) != item['sha256']):
            raise ValueError('Rerun300 frozen reference changed: '+key)
        values[key] = read(item['path'])
    parent, old_template, receipt = values['parent_proposal'], values['parent_template'], values['user_approval']
    stage = receipt['stages'][extension['approval_stage']]
    if (receipt.get('account') != 'vlm' or receipt.get('user_quote') != 'Approve, account vlm'
            or receipt.get('actor') != 'gpt-4o-2024-08-06' or values['judge_choice'].get('judge') != JUDGE
            or JUDGE != 'gpt-5.6-terra' or extension['selectors'] != stage['selectors']
            or proposal['limits']['api_usd'] != stage['api_usd_cap']
            or proposal['resources'] != dict(gpus=1, gpu_type='H200', cpus=8, memory_gib=240, total_seconds=stage['total_gpu_seconds_cap'])):
        raise ValueError('Rerun300 resources/API cap/models differ from the exact user approvals')
    selectors = extension['selectors']
    if template['actors'] != ['gpt-4o-2024-08-06'] or template['selectors'] != selectors:
        raise ValueError('Rerun300 actor/selector scope differs')
    ignored = {'code_sha256', 'selectors', 'actors', 'prices_usd_per_token', 'price_provenance', 'protocol_contract', 'judge_model'}
    if ({k: v for k, v in template.items() if k not in ignored}
            != _substituted({k: v for k, v in old_template.items() if k not in ignored})):
        raise ValueError('Rerun300 changed the worker scientific template beyond the model substitution')
    if (template['protocol_contract'] != _substituted(old_template['protocol_contract'])
            or template['judge_model'] != JUDGE):
        raise ValueError('Rerun300 protocol differs beyond the model substitution')
    for key in ('task_file', 'cohorts_file'):
        if sha(proposal[key]) != sha(parent[key]):
            raise ValueError('Rerun300 frozen input differs: '+key)
    if proposal['env_file'] != parent['env_file'] or proposal['models'] != parent['models']:
        raise ValueError('Rerun300 critics/provider environment differ')
    if read(proposal['reference_proposal'])['broker_routes'] != _substituted(read(parent['reference_proposal'])['broker_routes']):
        raise ValueError('Rerun300 routes differ beyond the model substitution')
    old_root = Path(extension['parent_template']['path']).parent/'source'
    old_code = {str(Path(p).relative_to(old_root)): v for p, v in old_template['code_sha256'].items()}
    new_code = {str(Path(p).relative_to(REPO)): v for p, v in template['code_sha256'].items()}
    allowed = {'scripts/launch_webprm_om2w_table5.py', 'openwebrl/webprm_om2w_table5.py', 'openwebrl/webprm_webarena_worker.py'}
    if old_code.keys() != new_code.keys() or any(old_code[p] != new_code[p] for p in old_code if p not in allowed):
        raise ValueError('Rerun300 changed frozen worker implementation')
    if proposal['campaign_root'] != str(REPO.parent/'campaign') or proposal['stage'] != 'full300':
        raise ValueError('Rerun300 requires its own claim/budget store and stage')
    expected = om2w_cohort(proposal['task_file'], proposal['cohorts_file'], template['actors'], selectors)
    wanted = [c['cell_id'] for c in expected['cells']]
    if (proposal.get('cell_ids') != wanted or proposal['stage_cells'] != len(wanted) or len(wanted) != 300*len(selectors)
            or proposal['stage_tasks'] != 300):
        raise ValueError('Rerun300 must hold every cell of the 300 tasks for its selectors')
    own = checked(Path(proposal['campaign_root']))
    for path in OM2W_RUNTIME.glob('*/campaign/queue/claims/*.json'):
        if own not in path.parents and read(path).get('cell_id') in set(wanted):
            raise ValueError('Rerun300 cell already claimed by another campaign: '+str(path))
    return expected


'''
for rel, pairs in edits.items():
    p = NEW/'source'/rel; t = p.read_text()
    for a, b in pairs:
        assert t.count(a) == 1, a
        t = t.replace(a, b)
    p.write_text(t)
launch = NEW/'source/scripts/launch_webprm_om2w_table5.py'; t = launch.read_text()
for a, b in launch_pairs:
    assert t.count(a) == 1, a
    t = t.replace(a, b)
t = t.replace("def full300_completion(proposal, template, extension):", rerun_helper+"def full300_completion(proposal, template, extension):", 1)
launch.write_text(t)

for name in ('proposal.json', 'worker-template.json', 'resource-request.json', 'campaign.sbatch'):
    p = NEW/name; p.write_text(p.read_text().replace(str(PARENT)+'/', str(NEW)+'/').replace(str(PARENT)+'"', str(NEW)+'"'))
batch = NEW/'campaign.sbatch'; bt = batch.read_text()
assert bt.count('--job-name=om2w-full300-A') == 1 and bt.count('--time=24:00:00') == 1 and str(PARENT) not in bt
batch.write_text(bt.replace('--job-name=om2w-full300-A', f'--job-name=om2w-rerun-{STAGE.lower()}'))

# 2. Reference routes with the substitution (frozen inside the package).
parent = read(PARENT/'proposal.json')
reference = read(parent['reference_proposal'])
ref = dict(reference, broker_routes=json.loads(sub(json.dumps(reference['broker_routes']))),
           cell_routes=json.loads(sub(json.dumps(reference['cell_routes']))),
           rerun_substitution_note='broker/cell routes of '+parent['reference_proposal']+' with '+OLD_ACTOR+'->'+NEW_ACTOR+', '+OLD_JUDGE+'->'+NEW_JUDGE)
ref_path = NEW/'reference-proposal-rerun.json'; ref_path.write_text(json.dumps(ref, indent=2, sort_keys=True)+'\n')

sys.path.insert(0, str(NEW/'source'))
from scripts import launch_webprm_om2w_table5 as L
from openwebrl import webprm_om2w_table5 as om
from openwebrl.webprm_webarena import digest
assert Path(L.__file__).resolve() == launch.resolve() and om.JUDGE == NEW_JUDGE

old_template = read(PARENT/'worker-template.json')
template = json.loads(sub(json.dumps({k: v for k, v in old_template.items() if k not in ('code_sha256', 'prices_usd_per_token', 'price_provenance')})))
template['code_sha256'] = {p: sha(p) for p in read(NEW/'worker-template.json')['code_sha256']}
template['selectors'] = SPEC['selectors']; template['actors'] = [NEW_ACTOR]
template['prices_usd_per_token'] = {k: v for k, v in old_template['prices_usd_per_token'].items() if k not in (OLD_ACTOR, OLD_JUDGE)}
template['prices_usd_per_token'].update(PRICES)
template['price_provenance'] = {k: v for k, v in old_template['price_provenance'].items() if k not in (OLD_ACTOR, OLD_JUDGE)}
for m in PRICES:
    template['price_provenance'][m] = dict(source='https://developers.openai.com/api/docs/pricing (checked 2026-10-23, standard tier)', usd_per_token=PRICES[m])
assert template['protocol_contract'] == om.protocol_contract(template['arbiter_source']), 'protocol contract recompute differs'
(NEW/'worker-template.json').write_text(json.dumps(template, indent=2, sort_keys=True)+'\n')

expected = om.cohort(parent['task_file'].replace(str(PARENT)+'/', str(NEW)+'/'), parent['cohorts_file'].replace(str(PARENT)+'/', str(NEW)+'/'),
                     template['actors'], SPEC['selectors'])
(NEW/'cohort.json').write_text(json.dumps(expected, indent=2, sort_keys=True)+'\n')
broker_routes, cell_routes = om.routes(ref['broker_routes'], ref['cell_routes'], template['actors'], candidate_controls=True)
stage = read(RECEIPT)['stages'][STAGE]
n = len(expected['cells'])
proposal = read(NEW/'proposal.json')
proposal.update(stage='full300', stage_cells=n, stage_tasks=300, cell_ids=[c['cell_id'] for c in expected['cells']],
    cohort_path=str(NEW/'cohort.json'), cohort_sha256=digest(expected), reference_proposal=str(ref_path),
    broker_routes=broker_routes, cell_routes=cell_routes, protocol_sha256=digest(template['protocol_contract']),
    resources=dict(gpus=1, gpu_type='H200', cpus=8, memory_gib=240, total_seconds=stage['total_gpu_seconds_cap']),
    limits=dict(proposal['limits'], api_usd=stage['api_usd_cap'], paid_requests=91*n, local_requests=SPEC['local']*n+11,
                total_requests=91*n+SPEC['local']*n+11, maximum_attempts=2),
    status='prepared_rerun300_'+STAGE+'_pending_reservation',
    matched_shepherd_extension=dict(identity='om2w_rerun300_0806_terra_v1', paid_dispatch_interval_seconds=3.0,
        requires_closed_job_id='373588', selectors=SPEC['selectors'], approval_stage=STAGE,
        substitution={OLD_ACTOR: NEW_ACTOR, OLD_JUDGE: NEW_JUDGE},
        parent_proposal=dict(path=str(PARENT/'proposal.json'), sha256=sha(PARENT/'proposal.json')),
        parent_template=dict(path=str(PARENT/'worker-template.json'), sha256=sha(PARENT/'worker-template.json')),
        user_approval=dict(path=str(RECEIPT), sha256=sha(RECEIPT)),
        judge_choice=dict(path=str(JUDGE_RECEIPT), sha256=sha(JUDGE_RECEIPT))),
    scientific_scope=dict(benchmark='Online-Mind2Web live, pinned 300-task file', identity='om2w_rerun300_0806_terra_v1_'+STAGE,
        scope=f'Fresh cells of all 300 tasks for {SPEC["selectors"]} with actor {NEW_ACTOR} and judge {NEW_JUDGE}.',
        comparability='Same prompts/critics/selectors/judge prompt+parser/worker code as earlier campaigns; models substituted after the '
                      '2026-10-23 OpenAI shutdown. Not pooled with the gpt-4o-2024-05-13/o4-mini cohort.',
        invalids='Invalid outcomes stay invalid; claimed cells are never rerun.'),
    cost_estimate=dict(basis='earlier campaigns ~USD1.3-1.4/cell at 05-13 prices; 08-06 is ~half', hard_USD_cap=stage['api_usd_cap'], cells=n))
proposal.pop('cohort_sha256_parent', None)
src = {p: sha(p) for p in proposal['source_sha256'] if Path(p).exists()}
for p in (PARENT/'proposal.json', PARENT/'worker-template.json', RECEIPT, JUDGE_RECEIPT, Path(__file__).resolve(),
          ref_path, NEW/'cohort.json', NEW/'campaign.sbatch', NEW/'worker-template.json', proposal['task_file'], proposal['cohorts_file']):
    src[str(p)] = sha(p)
for k in [k for k in src if 'claimed-exclusion' in k or k.endswith('full300-20261023/user-approval-20261023.json')]:
    del src[k]
proposal['source_sha256'] = src
proposal['worker_template_sha256'] = sha(NEW/'worker-template.json')
(NEW/'proposal.json').write_text(json.dumps(proposal, indent=2, sort_keys=True)+'\n')
print(json.dumps(L.preview(proposal), indent=1)[:600])

request = read(NEW/'resource-request.json')
t = request['approval_template']
t.update(api_usd_cap=stage['api_usd_cap'], limits=proposal['limits'], resources=proposal['resources'], stage='full300', proposal_sha256=digest(proposal))
request = dict(approval_template=t, account='vlm', actor=NEW_ACTOR, judge=NEW_JUDGE,
    proposal=dict(path=str(NEW/'proposal.json'), sha256=sha(NEW/'proposal.json')), proposal_digest=digest(proposal),
    status='approved_rerun300_'+STAGE+'_pending_submission', first_allocation_minutes=1440, maximum_allocations=2,
    user_approval=dict(path=str(RECEIPT), sha256=sha(RECEIPT)))
(NEW/'resource-request.json').write_text(json.dumps(request, indent=2, sort_keys=True)+'\n')
for p in (NEW/'proposal.json', NEW/'resource-request.json', NEW/'worker-template.json'):
    os.chmod(p, 0o600)
changed = sorted(str(Path(p).relative_to(NEW)) for p in proposal['source_sha256']
                 if p.startswith(str(NEW)+'/source/') and Path(str(PARENT)+p[len(str(NEW)):]).exists() and sha(p) != sha(str(PARENT)+p[len(str(NEW)):]))
print(json.dumps(dict(package=str(NEW), cells=n, by_selector=collections.Counter(c['selector'] for c in expected['cells']),
                      digest=digest(proposal), changed_source=changed), indent=1))
