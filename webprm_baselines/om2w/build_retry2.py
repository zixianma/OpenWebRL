"""(retry2, 2026-10-25 evening: retry1 jobs ended at 13:52 PDT when OpenAI returned 429 organization_spend_limit_exceeded; user: "yes ok to rereun om2w cells do it now". Same science; cells whose every attempt is invalid.)"""
"""(retry2, 2026-10-25: user "can we rerun all invalid tasks" / "Approve 3 retry jobs": one labelled second attempt for cells whose every recorded attempt is invalid.)
(derived from completion3: completion2 J1/J3 386864/386866 failed at startup on a 15 s scontrol timeout; two attempts allowed. cells left after attempt 2 / the first J3 completion.)
Freeze an OM2W rerun300 completion package: the never-claimed cells of one closed rerun300 stage.

Why: rerun300 J3 attempt 1 (383631) ended with its broker killed before writing epoch_closed, so its
attempt 2 (385775) refused to start and the stage's 2 allocation attempts are used up with 87 cells
never claimed. J1/J2 need the same route for the user-approved cap raise ("Raise as proposed",
user-cap-raise-20261024.json), because the API cap is bound in each frozen journal header.

A completion package copies its parent unchanged except for the launcher, which gains the
om2w_rerun300_completion_v1 extension. It holds only cells no OM2W campaign has claimed, uses its own
claim store and budget journal, and takes its API cap and GPU time from completion-approvals.json,
which must stay within the approved totals minus everything the parent already spent.
Never reserves or submits.

Usage: build_completion.py J1|J2|J3
"""
import collections, glob, hashlib, json, os, shutil, sys
from pathlib import Path

O = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-om2w-table5-20261012')
HERE = Path(__file__).resolve().parent
STAGE = sys.argv[1]
PARENT = O/f'rerun300-{STAGE.lower()}-vlm'
NEW = O/f'rerun300-{STAGE.lower()}-retry2-vlm'
APPROVALS = HERE/'retry2-approvals-20261025.json'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(Path(p).read_text())
assert not NEW.exists()
spec = read(APPROVALS)['stages'][STAGE]

# Retry set: cells of this stage whose every recorded rerun300 attempt is invalid (claimed and closed, no valid outcome).
attempts = collections.defaultdict(list)
for f in glob.glob(str(O/'rerun300-*/campaign/queue/outcomes/*.json')):
    o = read(f); attempts[o.get('cell_id') or o['result']['cell_id']].append(dict(path=f, valid=bool(o['result'].get('valid'))))
claimed_any = {read(p)['cell_id'] for p in glob.glob(str(O/'*/campaign/queue/claims/*.json'))}
parent_ids = set(read(PARENT/'proposal.json')['cell_ids'])
retry = sorted(c for c, rows in attempts.items() if c in parent_ids and rows and not any(r['valid'] for r in rows))
assert all(c in claimed_any for c in retry)
never = sorted(c for c in parent_ids if c not in claimed_any)  # e.g. cells a time-capped completion never reached
retry = sorted(set(retry) | set(never))
assert retry
exclusion = HERE/f'retry-set-{STAGE.lower()}-retry2.json'
assert not exclusion.exists()
exclusion.write_text(json.dumps(dict(cell_ids=retry, prior_attempts={c: attempts[c] for c in retry}), indent=1)+'\n')

keep_root = {'campaign.sbatch', 'cohort.json', 'proposal.json', 'resource-request.json', 'worker-template.json', 'source',
             'reference-proposal-rerun.json'}
shutil.copytree(PARENT, NEW, symlinks=True, ignore=lambda d, names: [n for n in names if Path(d) == PARENT and n not in keep_root])
os.chmod(NEW, 0o755)

launch = NEW/'source/scripts/launch_webprm_om2w_table5.py'; t = launch.read_text()
helper = '''RETRY_ID = 'om2w_rerun300_retry_invalid_v1'


def rerun300_retry(proposal, template, extension):
    """One labelled second attempt for rerun300 cells whose every recorded attempt is invalid; parent science, own store."""
    keys = {'identity', 'paid_dispatch_interval_seconds', 'requires_closed_job_id', 'approval_stage',
            'parent_proposal', 'parent_template', 'claimed_exclusion', 'completion_approvals'}
    if (set(extension) != keys or extension['paid_dispatch_interval_seconds'] != 3.0
            or extension['requires_closed_job_id'] != '373588'
            or any(k in proposal for k in ('candidate_controls', 'repair_of', 'predecessor', 'preproduction_ledger'))):
        raise ValueError('Exact rerun300 completion extension required')
    source = proposal['source_sha256']; values = {}
    for key in ('parent_proposal', 'parent_template', 'claimed_exclusion', 'completion_approvals'):
        item = extension[key]
        if (not isinstance(item, dict) or set(item) != {'path', 'sha256'} or sha(checked(Path(item['path']))) != item['sha256']
                or source.get(item['path']) != item['sha256']):
            raise ValueError('Completion frozen reference changed: '+key)
        values[key] = read(item['path'])
    parent, old_template = values['parent_proposal'], values['parent_template']
    stage = values['completion_approvals']['stages'][extension['approval_stage']]
    if (values['completion_approvals'].get('account') != 'vlm' or proposal['limits']['api_usd'] != stage['api_usd_cap']
            or stage['api_usd_cap'] > stage['api_usd_available']
            or proposal['resources'] != dict(gpus=1, gpu_type='H200', cpus=8, memory_gib=240, total_seconds=stage['total_seconds'])
            or stage['total_seconds'] > stage['gpu_seconds_available']):
        raise ValueError('Completion resources/API cap exceed the remaining approved amounts')
    if parent.get('matched_shepherd_extension', {}).get('identity') != RERUN300_ID:
        raise ValueError('Completion parent must be a rerun300 stage')
    ignored = {'code_sha256'}
    if {k: v for k, v in template.items() if k not in ignored} != {k: v for k, v in old_template.items() if k not in ignored}:
        raise ValueError('Completion changed the worker scientific template')
    for key in ('task_file', 'cohorts_file', 'reference_proposal'):
        if sha(proposal[key]) != sha(parent[key]):
            raise ValueError('Completion frozen input differs: '+key)
    for key in ('protocol_sha256', 'env_file', 'models', 'broker_routes', 'cell_routes'):
        if proposal[key] != parent[key]:
            raise ValueError('Completion science differs: '+key)
    old_root = Path(extension['parent_template']['path']).parent/'source'
    old_code = {str(Path(p).relative_to(old_root)): v for p, v in old_template['code_sha256'].items()}
    new_code = {str(Path(p).relative_to(REPO)): v for p, v in template['code_sha256'].items()}
    if old_code.keys() != new_code.keys() or any(old_code[p] != new_code[p] for p in old_code
                                                  if p != 'scripts/launch_webprm_om2w_table5.py'):
        raise ValueError('Completion changed frozen worker implementation')
    if proposal['campaign_root'] != str(REPO.parent/'campaign') or proposal['stage'] != 'full300':
        raise ValueError('Completion requires its own claim/budget store and stage')
    expected = om2w_cohort(proposal['task_file'], proposal['cohorts_file'], template['actors'], template['selectors'])
    retry = set(values['claimed_exclusion']['cell_ids'])
    wanted = [c['cell_id'] for c in expected['cells'] if c['cell_id'] in retry]
    if (proposal.get('cell_ids') != wanted or proposal['stage_cells'] != len(wanted) or set(wanted) - set(parent['cell_ids'])
            or proposal['stage_tasks'] != len({c['task_id'] for c in expected['cells'] if c['cell_id'] in set(wanted)})):
        raise ValueError('Retry cell membership differs from the frozen invalid-only retry set')
    own = checked(Path(proposal['campaign_root']))
    for path in OM2W_RUNTIME.glob('*/campaign/queue/claims/*.json'):
        if own not in path.parents and read(path).get('cell_id') in set(wanted):
            outcome = path.parent.parent/'outcomes'/path.name
            if not outcome.exists() or read(outcome)['result'].get('valid'):
                raise ValueError('Retry cell has an open or valid attempt elsewhere: '+str(path))
    return expected


'''
a = "    if isinstance(extension, dict) and extension.get('identity') == RERUN300_ID:\n"
assert t.count(a) == 1
t = t.replace(a, "    if isinstance(extension, dict) and extension.get('identity') == RETRY_ID:\n"
                 "        return rerun300_retry(proposal, template, extension)\n"+a)
b = "def rerun300(proposal, template, extension):"
assert t.count(b) == 1
t = t.replace(b, helper+b)
launch.write_text(t)

for name in ('proposal.json', 'worker-template.json', 'resource-request.json', 'campaign.sbatch', 'cohort.json', 'reference-proposal-rerun.json'):
    p = NEW/name; p.write_text(p.read_text().replace(str(PARENT)+'/', str(NEW)+'/').replace(str(PARENT)+'"', str(NEW)+'"'))
batch = NEW/'campaign.sbatch'; bt = batch.read_text()
assert bt.count(f'--job-name=om2w-rerun-{STAGE.lower()}\n') == 1 and str(PARENT) not in bt
batch.write_text(bt.replace(f'--job-name=om2w-rerun-{STAGE.lower()}\n', f'--job-name=om2w-rerun-{STAGE.lower()}-retry2\n'))

sys.path.insert(0, str(NEW/'source'))
from scripts import launch_webprm_om2w_table5 as L
from openwebrl.webprm_webarena import digest
assert Path(L.__file__).resolve() == launch.resolve()

template = read(NEW/'worker-template.json')
template['code_sha256'] = {p: sha(p) for p in template['code_sha256']}
(NEW/'worker-template.json').write_text(json.dumps(template, indent=2, sort_keys=True)+'\n')
parent = read(PARENT/'proposal.json')
full = L.om2w_cohort(parent['task_file'].replace(str(PARENT)+'/', str(NEW)+'/'),
                     parent['cohorts_file'].replace(str(PARENT)+'/', str(NEW)+'/'), template['actors'], template['selectors'])
cells = [c for c in full['cells'] if c['cell_id'] in set(retry)]
cohort = full  # stage_cells() selects cell_ids from the full fixed cohort
(NEW/'cohort.json').write_text(json.dumps(cohort, indent=2, sort_keys=True)+'\n')
n = len(cells)
proposal = read(NEW/'proposal.json')
local_per_cell = (parent['limits']['local_requests']-11)//parent['stage_cells']
proposal.update(stage_cells=n, stage_tasks=len({c['task_id'] for c in cells}), cell_ids=[c['cell_id'] for c in cells],
    cohort_path=str(NEW/'cohort.json'), cohort_sha256=digest(cohort),
    resources=dict(gpus=1, gpu_type='H200', cpus=8, memory_gib=240, total_seconds=spec['total_seconds']),
    limits=dict(proposal['limits'], api_usd=spec['api_usd_cap'], paid_requests=91*n, local_requests=local_per_cell*n+11,
                total_requests=91*n+local_per_cell*n+11, maximum_attempts=2),
    status=f'prepared_rerun300_{STAGE}_retry2_pending_approval',
    matched_shepherd_extension=dict(identity='om2w_rerun300_retry_invalid_v1', paid_dispatch_interval_seconds=3.0,
        requires_closed_job_id='373588', approval_stage=STAGE,
        parent_proposal=dict(path=str(PARENT/'proposal.json'), sha256=sha(PARENT/'proposal.json')),
        parent_template=dict(path=str(PARENT/'worker-template.json'), sha256=sha(PARENT/'worker-template.json')),
        claimed_exclusion=dict(path=str(exclusion), sha256=sha(exclusion)),
        completion_approvals=dict(path=str(APPROVALS), sha256=sha(APPROVALS))),
    scientific_scope=dict(parent['scientific_scope'], identity=parent['scientific_scope']['identity']+'_retry2',
        scope=f'Labelled second attempt for rerun300 {STAGE} cells whose every attempt was invalid ({n} cells); same science as the parent.',
        pooling='Labelled retry; first attempts preserved and reported. Identical worker, prompts, models, judge and routes.'),
    cost_estimate=dict(basis=spec['cost_basis'], hard_USD_cap=spec['api_usd_cap'], cells=n))
src = {p: sha(p) for p in proposal['source_sha256'] if Path(p).exists()}
for p in (PARENT/'proposal.json', PARENT/'worker-template.json', exclusion, APPROVALS, Path(__file__).resolve(),
          NEW/'cohort.json', NEW/'campaign.sbatch', NEW/'worker-template.json', NEW/'reference-proposal-rerun.json'):
    src[str(p)] = sha(p)
proposal['source_sha256'] = src
proposal['worker_template_sha256'] = sha(NEW/'worker-template.json')
(NEW/'proposal.json').write_text(json.dumps(proposal, indent=2, sort_keys=True)+'\n')
L.validate_proposal(proposal)

request = read(NEW/'resource-request.json')
t = request['approval_template']
t.update(api_usd_cap=spec['api_usd_cap'], limits=proposal['limits'], resources=proposal['resources'], stage='full300',
         proposal_sha256=digest(proposal), status='approved', user_approval_quote='<exact user approval required>',
         shared_gpu_reservation_id='<reserve after exact approval>')
request = dict(request, approval_template=t, proposal=dict(path=str(NEW/'proposal.json'), sha256=sha(NEW/'proposal.json')),
               proposal_digest=digest(proposal), status=f'rerun300_{STAGE}_completion_pending_approval',
               first_allocation_minutes=spec['total_seconds']//60, maximum_allocations=1)
(NEW/'resource-request.json').write_text(json.dumps(request, indent=2, sort_keys=True)+'\n')
for p in (NEW/'proposal.json', NEW/'resource-request.json', NEW/'worker-template.json'):
    os.chmod(p, 0o600)
print(json.dumps(dict(package=str(NEW), cells=n, tasks=proposal['stage_tasks'],
                      by_selector=collections.Counter(c['selector'] for c in cells), digest=digest(proposal),
                      resources=proposal['resources'], api_usd=spec['api_usd_cap']), indent=1))
