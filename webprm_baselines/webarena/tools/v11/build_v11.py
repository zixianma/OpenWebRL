"""Build one v11 WebArena-Lite Table 5 package: gpt-4o-2024-08-06 no_search + Piotr SelectionARM on one site, 1280x720.

v11 (2026-10-27): copy of v10-tools/build_v10.py (v10 packages and tools stay untouched). User chose the 1280x720 setup:
the actor viewport is 1280x720 for BOTH methods, no_search (rerun) and piotr_selectionarm (Online-Mind2Web protocol:
released WebArbiter 5-BLOCK prompt in one call, released filter/dedup, Piotr SelectionARM selects), and the ARM sees
exactly the actor's 1280x720 screenshot. Same 128 tasks; each package interleaves the two methods per task.
Built from the frozen v8 package launch-preparation-v8-j2-arb-a exactly as the v9 builders do (v9b-tools exec'd unchanged),
with every v9 fix (v9 cell/launcher, v9b broker validator, v9c CMS login, v9e broker concurrency, v9f GitLab monitoring),
then v11: patch_v11_cell.py (incl. 1280x720 viewport), patch_v11_launcher.py, openwebrl/webprm_native_piotr_v11.py,
native-audit-v8. Never reserves or submits.
Usage (env): V11_NAME V11_SITE V11_TASKS (comma task ids) V11_API_USD V11_SECONDS V11_MAX_WORKERS
  stage 1 (copy + patches); once v11-tools/integration-review-v11-<name>.json exists, stage 2 (proposal, routes, request).
"""
import hashlib, json, os, re, shutil, sys
from pathlib import Path

D = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-webarena-20261011/next-stage-preparation/webarena-subset-review-20261021')
N = D.parent
TOOLS = D/'v9b-tools'  # v9 patches (exec'd unchanged; their free names: NEW, V9, ct)
V11T = Path(__file__).resolve().parent
OLD = D/'launch-preparation-v8-j2-arb-a'
NAME = os.environ['V11_NAME']; NEW = D/('launch-preparation-v11-'+NAME)
SITE = os.environ['V11_SITE']
MAGENTO_SOURCE = N/'shopping-table5-450-multitab-recovery-v10/source'
MAGENTO_PROPOSALS = {'shopping': N/'shopping-table5-450-multitab-recovery-v10/proposal.json',
                     'shopping_admin': N/'cms-table5-350-bounded-reap-recovery-v10/proposal.json'}
PIOTR_REFERENCE = N/'cms-table5-350-bounded-reap-recovery-v10'  # the earlier CMS Piotr campaign (proposal + worker template)
MAGENTO_FILES = ('openwebrl/webprm_cms_hosting.py', 'openwebrl/webprm_cms_materializer.py', 'openwebrl/webprm_cms_native.py',
                 'openwebrl/webprm_cms_site.py', 'openwebrl/webprm_shopping_native.py', 'openwebrl/webprm_shopping_site.py',
                 'scripts/run_webprm_table5_cms.py', 'scripts/audit_webprm_table5_cms_cell.py',
                 'scripts/validate_webprm_cms_hosting.py', 'scripts/validate_webprm_shopping_hosting.py',
                 'tests/test_audit_webprm_table5_cms.py', 'tests/test_run_webprm_table5_cms.py', 'tests/test_run_webprm_table5_shopping.py',
                 'tests/test_validate_webprm_cms_hosting.py', 'tests/test_validate_webprm_shopping_hosting.py',
                 'tests/test_webprm_cms_hosting.py', 'tests/test_webprm_cms_materializer.py', 'tests/test_webprm_cms_native.py',
                 'tests/test_webprm_cms_site.py')
GITLAB_ENTRY = dict(endpoint='http://127.0.0.1:8023', state_apparent_bytes=40*2**30,  # job 389698 base: 34,179,881,242 B
    hosting_proposal=dict(path=str(N/'gitlab-hosting-validation-v8/proposal.json'), sha256='7b48bf4a359ff406b361eb2938ae0298052c210c4cb0af747f13364721115b6f'),
    hosting_validation=dict(path=str(N/'gitlab-hosting-validation-v8/campaign/attempt-389698/validation-result.json'), sha256='2b1af66e6a3608033d25a9ef9ad3c309b6a3e5d7ccb0f60ff67214a84b4ea178'),
    base_content_sha256='3c68a967a1aa285c8cc6c466e34ee0614959821d12ac90a13d46209eea8279f4')
BROWSER_V3 = D/'native-browser-v3/native_episode.py'
AUD = D/'native-audit-v8'; AUD_FILES = ('native_cell_audit.py', 'native_lowlevel_replay.py')
HELPER = 'openwebrl/webprm_native_piotr_v11.py'
REVIEW = V11T/f'integration-review-v11-{NAME}.json'
DECISION = V11T/'v11-decision-20261027.json'
OLD_NAME = OLD.name
ACTOR = 'gpt-4o-2024-08-06'
SELECTOR = 'piotr_selectionarm'
SELECTORS = ['no_search', 'piotr_selectionarm']  # v11: interleaved per task
TASKS = json.loads((Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-baselines-20261009/sources/arbiter')
                    / 'viewer/config_files/wa/test_webarena_lite.raw.json').read_text())
V9_SITES = ('shopping', 'shopping_admin', 'reddit', 'gitlab')
V9_TASK_IDS = sorted(t['task_id'] for t in TASKS if len(t['sites']) == 1 and t['sites'][0] in V9_SITES and t['task_id'] != 5)
assert len(V9_TASK_IDS) == 128, len(V9_TASK_IDS)
assert SITE in V9_SITES, SITE
V9 = dict(site=SITE, task_ids=V9_TASK_IDS, api_usd=float(os.environ['V11_API_USD']), total_seconds=int(os.environ['V11_SECONDS']),
          max_workers=int(os.environ['V11_MAX_WORKERS']), paid_requests=12000, local_requests=24000)
BLOCK_TASKS = [int(x) for x in os.environ['V11_TASKS'].split(',')]
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(Path(p).read_text())


def retarget(text):
    return text.replace(str(OLD)+'/', str(NEW)+'/').replace(str(OLD)+'"', str(NEW)+'"')


if not NEW.exists():
    keep = {'campaign.sbatch', 'cohort.json', 'proposal.json', 'resource-request.json', 'worker-template.json', 'source',
            'browser-identity-v1.json', 'offline-canary-evidence-v1', 'offline-canary-proof.json'}
    shutil.copytree(OLD, NEW, symlinks=True,
                    ignore=lambda d, names: [n for n in names if (Path(d) == OLD and n not in keep) or n == '__pycache__'])
    os.chmod(NEW, 0o755)
    for rel in MAGENTO_FILES:
        assert not (NEW/'source'/rel).exists(), rel
        shutil.copy2(MAGENTO_SOURCE/rel, NEW/'source'/rel)
    if SITE in MAGENTO_PROPOSALS:  # v9c: CMS submit_admin_login repair (Magento packages, as build_v9e.py)
        CMS_SOURCE = N/'cms-table5-350-bounded-reap-recovery-v10/source'
        for rel in ('openwebrl/webprm_cms_site.py', 'tests/test_webprm_cms_site.py'):
            shutil.copy2(CMS_SOURCE/rel, NEW/'source'/rel)
        assert 'def submit_admin_login' in (NEW/'source/openwebrl/webprm_cms_site.py').read_text()
    exec(compile((TOOLS/'patch_v9_cell.py').read_text(), 'patch_v9_cell', 'exec'))
    exec(compile((TOOLS/'patch_v9_broker.py').read_text(), 'patch_v9_broker', 'exec'))
    exec(compile((TOOLS/'patch_v9e_broker.py').read_text(), 'patch_v9e_broker', 'exec'))
    if SITE == 'gitlab':  # as build_v9f_gitlab.py
        G = TOOLS/'gitlab'
        exec(compile((G/'patch_v9_gitlab.py').read_text(), 'patch_v9_gitlab', 'exec'))
        exec(compile((TOOLS/'patch_v9f_gitlab.py').read_text(), 'patch_v9f_gitlab', 'exec'))
    cell = NEW/'source/scripts/run_native_webarena_cell.py'
    ct = cell.read_text()
    # v9: CMS endpoints carry a path (http://127.0.0.1:7780/admin); the browser allowlist is the origin.
    a = "allowed_origins=[instance['endpoint'].rstrip('/')]"
    assert ct.count(a) == 1
    ct = ct.replace(a, "allowed_origins=['{0.scheme}://{0.netloc}'.format(__import__('urllib.parse').parse.urlsplit(instance['endpoint']))]")
    cell.write_text(ct)
    assert not (NEW/'source'/HELPER).exists()
    shutil.copy2(V11T/'webprm_native_piotr_v11.py', NEW/'source'/HELPER)
    exec(compile((V11T/'patch_v11_cell.py').read_text(), 'patch_v11_cell', 'exec'))
    launch = NEW/'source/scripts/launch_webprm_table5_campaign.py'
    t = launch.read_text()
    exec(compile((TOOLS/'patch_v9_launcher.py').read_text(), 'patch_v9_launcher', 'exec'))
    exec(compile((V11T/'patch_v11_launcher.py').read_text(), 'patch_v11_launcher', 'exec'))
    launch.write_text(t)
    for name in ('proposal.json', 'worker-template.json', 'resource-request.json', 'campaign.sbatch', 'cohort.json'):
        p = NEW/name; p.write_text(retarget(p.read_text()))
    b = NEW/'campaign.sbatch'; bt = b.read_text()
    h, m = divmod(V9['total_seconds']//60, 60)
    bt = re.sub(r'--time=\d\d:\d\d:00', f'--time={h:02d}:{m:02d}:00', bt)
    bt = re.sub(r'--job-name=\S+\n', '--job-name=wa-v11-'+NAME+'\n', bt, count=1)
    assert '--account=vlm\n' in bt
    b.write_text(bt)

if not REVIEW.exists():
    comps = {p: sha(p) for p in (NEW/'source/scripts/run_native_webarena_cell.py', NEW/'source'/HELPER, BROWSER_V3,
                                 D/'native-broker-v2/native_broker_bridge.py', *[AUD/n for n in AUD_FILES]) if Path(p).exists()}
    print(json.dumps(dict(stage='patched; write integration-review then rerun', package=str(NEW),
                          components={str(k): v for k, v in comps.items()}), indent=1))
    sys.exit(0)

cell = NEW/'source/scripts/run_native_webarena_cell.py'
launch = NEW/'source/scripts/launch_webprm_table5_campaign.py'
sys.path.insert(0, str(NEW/'source'))
from scripts import launch_webprm_table5_campaign as L
assert Path(L.__file__).resolve() == launch.resolve()
from scripts.prepare_webprm_table5_campaign import digest as cohort_task_digest
from openwebrl.webprm_webarena_piotr import validate_piotr_selectionarm
from openwebrl.webprm_native_piotr_v11 import RECEIPT as V11_RECEIPT

# Cohort: the site's released single-site tasks (Reddit: minus DROPPED_TASK_IDS) x (no_search, Piotr), task-major.
old_cohort = read(OLD/'cohort.json')
ids = sorted(t['task_id'] for t in TASKS if t['sites'] == [SITE] and t['task_id'] not in L.DROPPED_TASK_IDS)
assert len(ids) == L.V9_SITE_TASKS[SITE]
selectors = list(SELECTORS)
cells = [dict(actor=ACTOR, cell_id=f'native_table5_v11/{SITE}/task_{tid:03d}/{ACTOR}__{sel}', domain=SITE,
              original_task_id=TASKS[tid]['old_task_id'], selector=sel, task_config_sha256=cohort_task_digest(TASKS[tid]),
              task_id=tid) for tid in ids for sel in selectors]
assert all(TASKS[tid]['task_id'] == tid for tid in ids)
cohort = dict(old_cohort, cells=cells, domain_counts={SITE: len(ids)}, episode_count=len(cells), task_count=len(ids),
              protocol=L.V9_COHORT_PROTOCOL, selectors=selectors, actors=[ACTOR],
              excluded_tasks=[e for e in old_cohort.get('excluded_tasks', []) if SITE == 'reddit'])

piotr_template = read(PIOTR_REFERENCE/'worker-template.json')
template = read(NEW/'worker-template.json')
template['protocol'] = L.V9_COHORT_PROTOCOL
template['selectors'] = selectors
for key in ('piotr_checkpoint', 'piotr_model', 'piotr_representation', 'piotr_revision', 'piotr_source'):
    template[key] = piotr_template[key]
template['piotr_endpoint'] = None
template['piotr_manifest'] = validate_piotr_selectionarm(template)
assert template['piotr_manifest']['representation'] == V11_RECEIPT['representation']
v9_changes = dict(
    user_decision='ah yes can we follow that config for webarbiter/webshepherd (read the papers again to confirm). also let\'s do B (2026-10-26)',
    candidate_protocol=dict(methods=['arbiter7b', 'shepherd8b'], samples=20, temperature=1.0, top_p=1.0, top_n=5,
        prompt='released single-action envelope (disclosed_single_action_envelope_v2)', ranking='frequency desc, then first sampled',
        validity='released filter_candidates on each sample', confidence='sample share among valid samples',
        sources=['WebShepherd (arXiv 2505.15277v2) App. F.2: 20 samples, nucleus sampling, temperature 1.0, top-n most frequent',
                 'WebArbiter (arXiv 2601.21872v1) Sec. 5.2.1: following WebShepherd, Best-of-N, N=5, knockout tournament']),
    no_search='unchanged v8 single-action control (v11: rerun at 1280x720)', webshepherd_ties='more frequent (lower index) wins, as the paper',
    new_tab='native-browser-v3 lazily creates the CDP session (device metrics + Accessibility.enable) for action-opened tabs',
    site=SITE, sites_in_campaign=['shopping', 'shopping_admin', 'reddit', 'gitlab'], map_excluded='WebArbiter Table 5 excludes Map',
    fuzzy_judge='retired gpt-4-1106-preview unavailable: cell invalid; offline gpt-4.1 re-score is a labelled sensitivity analysis',
    auditor='native-audit-v8')
if SITE == 'gitlab':
    v9_changes['gitlab'] = 'gitlab-hosting-validation-v8 job 389698; v9-tools/gitlab/patch_v9_gitlab.py; v9f monitoring disabled'
v11_changes = dict(
    user_decisions=['ah I see. yes do A. so then we can use gpt4o + our ARM right? no need for openwebrl-sft (2026-10-27)',
                    'chose the 1280x720 setup: actor viewport 1280x720 for both no_search (rerun) and piotr_selectionarm (2026-10-27)'],
    methods=SELECTORS, interleaving='task-major cohort order: each task\'s no_search and ARM cells run back to back, each on a fresh copy of the same site base',
    viewport=dict(width=1280, height=720, sources=['WebArena browser_env default viewport_size 1280x720 (arbiter viewer/src/browser_env/envs.py:73)',
        'replaces the released WebArbiter service default 1024x2490 (viewer/src/server/service/run_management.py:172) used in v8-v10'],
        current_viewport_only=True, note='the AX tree lists only on-screen elements, as before; a smaller viewport shows fewer elements per observation'),
    piotr=V11_RECEIPT, om2w_protocol_source='openwebrl/webprm_om2w_table5.py + webprm_webarena_worker.py (5 raw slots, released policy prompt, '
                                            'T1.0/top_p 0.9, representation v2, 1280x720)',
    deviations_from_om2w=['1..5 deduplicated candidates instead of exactly five admissible raw slots (OM2W invalidated the cell otherwise)',
                          'ungroundable actions shown as a canonical descriptor instead of invalidating the cell',
                          'native WebArena harness (released coordinator, WebArena AX tree/evaluator) instead of the OM2W browsergym viewer harness'],
    auditor='native-audit-v8')
for key in ('native_contract', 'protocol_contract'):
    template[key] = dict(template[key], protocol=L.V9_COHORT_PROTOCOL, v9_operational_changes=v9_changes,
                         v11_operational_changes=v11_changes)
template['endpoints'] = {SITE: (read(MAGENTO_PROPOSALS[SITE])['domains'][SITE]['endpoint'] if SITE in MAGENTO_PROPOSALS
                                else GITLAB_ENTRY['endpoint'] if SITE == 'gitlab' else template['endpoints']['reddit'])}
template['native_components']['episode'] = dict(path=str(BROWSER_V3), sha256=sha(BROWSER_V3))
template['native_episode'] = dict(path=str(BROWSER_V3), sha256=sha(BROWSER_V3))
template['native_auditor'] = dict(path=str(AUD/AUD_FILES[0]), sha256=sha(AUD/AUD_FILES[0]))
swap = {str(D/'native-browser-v2/native_episode.py'): str(BROWSER_V3),
        **{str(D/'native-audit-v4'/n): str(AUD/n) for n in AUD_FILES}}
template['code_sha256'] = {swap.get(p, p): None for p in template['code_sha256']}
for rel in MAGENTO_FILES + (HELPER, 'openwebrl/webprm_webarena_piotr.py', 'openwebrl/webprm_arbiter_harness.py'):
    if not rel.startswith('tests/'):
        template['code_sha256'][str(NEW/'source'/rel)] = None
if SITE == 'gitlab':
    for rel in ('openwebrl/webprm_gitlab_site.py', 'scripts/run_webprm_table5_gitlab.py', 'scripts/validate_webprm_gitlab_hosting.py',
                'openwebrl/webprm_gitlab_native.py', 'openwebrl/webprm_gitlab_recovery.py'):
        if (NEW/'source'/rel).exists():
            template['code_sha256'][str(NEW/'source'/rel)] = None
template['code_sha256'] = {p: sha(p) for p in template['code_sha256']}
(NEW/'worker-template.json').write_text(json.dumps(template, sort_keys=True, indent=2)+'\n')
cohort['protocol_sha256'] = L.digest(template['native_contract'])
(NEW/'cohort.json').write_text(json.dumps(cohort, sort_keys=True, indent=2)+'\n')

proposal = read(NEW/'proposal.json')
if SITE in MAGENTO_PROPOSALS:
    proposal['domains'] = {SITE: read(MAGENTO_PROPOSALS[SITE])['domains'][SITE]}
elif SITE == 'gitlab':
    proposal['domains'] = {SITE: GITLAB_ENTRY}
proposal['models'] = {SELECTOR: read(PIOTR_REFERENCE/'proposal.json')['models'][SELECTOR]}
proposal['resources'] = dict(proposal['resources'], total_seconds=V9['total_seconds'])
proposal['limits'] = dict(proposal['limits'], api_usd=V9['api_usd'], paid_requests=V9['paid_requests'],
                          local_requests=V9['local_requests'], total_requests=V9['paid_requests']+V9['local_requests'],
                          max_workers=V9['max_workers'])
assert len(set(BLOCK_TASKS)) == len(BLOCK_TASKS) and set(BLOCK_TASKS) <= set(ids), 'v10 task block outside the site cohort'
proposal['v8_cell_blocks'] = [dict(selector=sel, task_ids=BLOCK_TASKS) for sel in selectors]
proposal['broker_routes'] = L.native_routes(proposal['models'])
proposal['cell_routes'] = {ACTOR: {sel: L.native_cell_routes(sel, proposal['models']) for sel in selectors}}
proposal['native_contract'] = template['native_contract']
proposal['protocol_sha256'] = L.digest(template['native_contract'])
proposal['cohort_sha256'] = L.digest(cohort); proposal['cohort_file_sha256'] = sha(NEW/'cohort.json')
proposal['native_episode'] = dict(path=str(BROWSER_V3), sha256=sha(BROWSER_V3))
proposal['native_worker'] = dict(path=str(cell), sha256=sha(cell))
proposal['native_auditor'] = template['native_auditor']
proposal['reviewed_worker_bytes_reused_at'] = dict(path=str(cell), sha256=sha(cell))
proposal['integration_review'] = dict(path=str(REVIEW), sha256=sha(REVIEW))
proposal['v7_fresh_experiment'] = dict(identity='native_table5_v11', actor=ACTOR, site=SITE,
    parent=dict(path=str(OLD/'proposal.json'), sha256=sha(OLD/'proposal.json')),
    user_decisions=[str(D/'v7-decision-20261023.json'), str(D/'v8-tools/v8-decision-20261025.json'), str(TOOLS/'v9-decision-20261026.json'),
                    str(DECISION)])
proposal['worker_template_sha256'] = sha(NEW/'worker-template.json')
proposal['status'] = 'prepared_v11_pending_approval'
proposal['operational_change'] = 'V11 fresh experiment (no_search + Piotr SelectionARM, 1280x720): see v11_operational_changes in native_contract.'
src = {p: sha(p) for p in proposal['source_sha256'] if Path(p).exists() and 'native-browser-v2' not in p and 'native-audit-v4' not in p}
for p in sorted((NEW/'source').rglob('*')):
    if p.is_file() and '__pycache__' not in p.parts:
        src[str(p)] = sha(p)
gitlab_tools = [TOOLS/'gitlab'/n for n in ('patch_v9_gitlab.py', 'gen_gitlab_stage.py', 'webprm_gitlab_site.py')] if SITE == 'gitlab' else []
for p in (BROWSER_V3, D/'native-browser-v3/test_newtab.py', REVIEW, Path(__file__).resolve(), TOOLS/'v9-decision-20261026.json', DECISION,
          *[TOOLS/n for n in ('patch_v9_cell.py', 'patch_v9_launcher.py', 'patch_v9_broker.py', 'patch_v9e_broker.py', 'patch_v9f_gitlab.py')],
          *[V11T/n for n in ('patch_v11_cell.py', 'patch_v11_launcher.py', 'webprm_native_piotr_v11.py', 'build_audit_v8.py')],
          *gitlab_tools, NEW/'campaign.sbatch', NEW/'cohort.json', NEW/'worker-template.json', OLD/'proposal.json',
          *[AUD/n for n in AUD_FILES]):
    src[str(p)] = sha(p)
piotr_source = Path(template['piotr_source'])
for p in (piotr_source/'manifest.json', piotr_source/'inference/selection_prompt.py', piotr_source/'inference/selection_infer.py',
          Path(proposal['models'][SELECTOR]['processor_path'])/'serving-view-manifest.json',
          Path(proposal['models'][SELECTOR]['processor_path'])/'processor-verification.json',
          Path(proposal['models'][SELECTOR]['weights_content_audit'])):
    src[str(p)] = sha(p)
if SITE in MAGENTO_PROPOSALS:
    for key in ('hosting_proposal', 'hosting_validation'):
        ref = proposal['domains'][SITE][key]; src[ref['path']] = sha(ref['path']); assert src[ref['path']] == ref['sha256']
    msrc = read(MAGENTO_PROPOSALS[SITE])['source_sha256']
    for path, value in msrc.items():  # validated hosting receipts / OCI proofs the site module reads
        if 'shopping-table5-450' not in path and 'cms-table5-350' not in path and Path(path).exists() and sha(path) == value:
            src.setdefault(path, value)
elif SITE == 'gitlab':
    for key in ('hosting_proposal', 'hosting_validation'):
        ref = proposal['domains'][SITE][key]; src[ref['path']] = sha(ref['path']); assert src[ref['path']] == ref['sha256']
    gv = read(GITLAB_ENTRY['hosting_validation']['path']); gp = read(GITLAB_ENTRY['hosting_proposal']['path'])
    for ref in (gv['first'], gv['second'], gp['service_plan'], gp['account_binding'], gp['manifest'], gp['layer_proof']):
        src[ref['path']] = sha(ref['path']); assert src[ref['path']] == ref['sha256'], ref['path']
    plan = read(gp['service_plan']['path'])
    for ref in plan['planned_configs'].values():
        src[ref['path']] = sha(ref['path']); assert src[ref['path']] == ref['sha256']
for path, value in read(REVIEW)['component_sha256'].items():
    src[path] = value
proposal['source_sha256'] = src
for key, comp in (('native_episode', 'episode'), ('native_broker', 'broker')):
    assert template['native_components'][comp] == proposal[key], ('template/proposal component mismatch', key)
for blob in (json.dumps({k: v for k, v in template.items() if k != 'code_sha256'}),
             json.dumps({k: v for k, v in proposal.items() if k not in ('source_sha256', 'v7_fresh_experiment')})):
    blob = blob.replace('"auditor": "native-audit-v4"', '')  # v8_operational_changes keeps its historical label
    assert ('native-browser-v2' not in blob and 'native-audit-v4' not in blob and 'native-audit-v5' not in blob
            and 'native-audit-v6' not in blob and 'native-audit-v7' not in blob and OLD_NAME not in blob), 'stale v8/v9/v10 reference'
L.validate_proposal(proposal)
(NEW/'proposal.json').write_text(json.dumps(proposal, sort_keys=True, indent=2)+'\n')

request = read(NEW/'resource-request.json')
tmpl = request['approval_template']
tmpl.update(proposal_sha256=L.digest(proposal), resources=proposal['resources'], limits=proposal['limits'],
            protocol_sha256=proposal['protocol_sha256'], ephemeral_policy_sha256=L.digest(proposal['ephemeral_policy']),
            api_usd_cap=proposal['limits']['api_usd'], account='vlm', maximum_allocation_attempts=1, attempt_job_ids=[], job_id=None)
request = dict(approval_template=tmpl, proposal=dict(path=str(NEW/'proposal.json'), sha256=sha(NEW/'proposal.json')),
               proposal_digest=L.digest(proposal), status='v11_pending_exact_user_approval', account='vlm')
(NEW/'resource-request.json').write_text(json.dumps(request, sort_keys=True, indent=2)+'\n')
for p in (NEW/'proposal.json', NEW/'resource-request.json', NEW/'worker-template.json', NEW/'cohort.json'):
    os.chmod(p, 0o600)
print(json.dumps(dict(package=str(NEW), digest=L.digest(proposal), site=SITE, cohort_cells=len(cells),
                      admitted_cells=len(L.v8_block_cells(proposal, cells)), total_seconds=V9['total_seconds'], api_usd=V9['api_usd'],
                      routes=[(r['model'], r['purpose'], r.get('max_choices', 1)) for r in proposal['broker_routes']]), indent=1))
