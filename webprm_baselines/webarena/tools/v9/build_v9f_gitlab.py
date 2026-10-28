"""Build one v9 GitLab Table 5 package (copy of build_v9.py; that file and its packages stay untouched).

GitLab additions: v9-tools/gitlab/patch_v9_gitlab.py (site adapter, generated stage script, validated GitLab hosting
primitives from gitlab-hosting-validation-v8 / job 389698, launcher/site-registry/reset-auditor dispatch) and native-audit-v6.
Original build_v9 description follows.

Build one v9 WebArena-Lite Table 5 package: one site, the paper's Best-of-N candidate protocol, native-browser-v3.

v9 (2026-10-26, user: "ah yes can we follow that config for webarbiter/webshepherd (read the papers again to
confirm). also let's do B"). Built on top of the frozen, fully built v8 package launch-preparation-v8-j2-arb-a
(v8-tools/build_v8.py and its packages stay untouched):
  1. arbiter7b / shepherd8b candidates: 20 policy samples (n=20, temperature 1.0, top_p 1.0, released single-action
     prompt), the 5 most frequent distinct valid actions (WebShepherd App. F.2; WebArbiter Sec. 5.2.1 N=5);
     no_search unchanged (patch_v9_cell.py). Host broker admits n=20 only on native_policy (patch_v9_broker.py).
  2. native-browser-v3: lazily creates the CDP session for tabs opened by actions (v8 'Page' object has no
     attribute 'client', 4/54 cells).
  3. Four Table 5 sites; one hosted site per package. Shopping / CMS reuse the validated third-party Magento
     hosting (job 358540 / 357831 recipes) and its stage script / reset auditor from
     shopping-table5-450-multitab-recovery-v10/source, unchanged.
  4. native-audit-v5 mirrors 1 and 3.
Never reserves or submits. Usage (env): V9_NAME V9_SITE V9_BLOCKS V9_API_USD V9_SECONDS V9_MAX_WORKERS
  build_v9.py            stage 1 (copy + patches) and, once integration-review-v9f-<name>.json exists, stage 2.
"""
import hashlib, json, os, re, shutil, sys
from pathlib import Path

D = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-webarena-20261011/next-stage-preparation/webarena-subset-review-20261021')
N = D.parent
TOOLS = Path(__file__).resolve().parent
OLD = D/'launch-preparation-v8-j2-arb-a'
NAME = os.environ['V9_NAME']; NEW = D/('launch-preparation-v9f-'+NAME)
SITE = os.environ['V9_SITE']
MAGENTO_SOURCE = N/'shopping-table5-450-multitab-recovery-v10/source'
MAGENTO_PROPOSALS = {'shopping': N/'shopping-table5-450-multitab-recovery-v10/proposal.json',
                     'shopping_admin': N/'cms-table5-350-bounded-reap-recovery-v10/proposal.json'}
MAGENTO_FILES = ('openwebrl/webprm_cms_hosting.py', 'openwebrl/webprm_cms_materializer.py', 'openwebrl/webprm_cms_native.py',
                 'openwebrl/webprm_cms_site.py', 'openwebrl/webprm_shopping_native.py', 'openwebrl/webprm_shopping_site.py',
                 'scripts/run_webprm_table5_cms.py', 'scripts/audit_webprm_table5_cms_cell.py',
                 'scripts/validate_webprm_cms_hosting.py', 'scripts/validate_webprm_shopping_hosting.py',
                 'tests/test_audit_webprm_table5_cms.py', 'tests/test_run_webprm_table5_cms.py', 'tests/test_run_webprm_table5_shopping.py',
                 'tests/test_validate_webprm_cms_hosting.py', 'tests/test_validate_webprm_shopping_hosting.py',
                 'tests/test_webprm_cms_hosting.py', 'tests/test_webprm_cms_materializer.py', 'tests/test_webprm_cms_native.py',
                 'tests/test_webprm_cms_site.py')
BROWSER_V3 = D/'native-browser-v3/native_episode.py'
AUD = D/'native-audit-v6'; AUD_FILES = ('native_cell_audit.py', 'native_lowlevel_replay.py')
REVIEW = TOOLS/f'integration-review-v9f-{NAME}.json'
OLD_NAME = OLD.name
ACTOR = 'gpt-4o-2024-08-06'
TASKS = json.loads((Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-baselines-20261009/sources/arbiter')
                    / 'viewer/config_files/wa/test_webarena_lite.raw.json').read_text())
V9_SITES = ('shopping', 'shopping_admin', 'reddit', 'gitlab')
GITLAB_ENTRY = dict(endpoint='http://127.0.0.1:8023', state_apparent_bytes=40*2**30,  # job 389698 base: 34,179,881,242 B
    hosting_proposal=dict(path=str(N/'gitlab-hosting-validation-v8/proposal.json'), sha256='7b48bf4a359ff406b361eb2938ae0298052c210c4cb0af747f13364721115b6f'),
    hosting_validation=dict(path=str(N/'gitlab-hosting-validation-v8/campaign/attempt-389698/validation-result.json'), sha256='2b1af66e6a3608033d25a9ef9ad3c309b6a3e5d7ccb0f60ff67214a84b4ea178'),
    base_content_sha256='3c68a967a1aa285c8cc6c466e34ee0614959821d12ac90a13d46209eea8279f4')
V9_TASK_IDS = sorted(t['task_id'] for t in TASKS if len(t['sites']) == 1 and t['sites'][0] in V9_SITES and t['task_id'] != 5)
assert len(V9_TASK_IDS) == 128, len(V9_TASK_IDS)
V9 = dict(site=SITE, task_ids=V9_TASK_IDS, api_usd=float(os.environ['V9_API_USD']), total_seconds=int(os.environ['V9_SECONDS']),
          max_workers=int(os.environ['V9_MAX_WORKERS']), paid_requests=12000, local_requests=24000)
assert SITE == 'gitlab', 'build_v9_gitlab.py builds GitLab packages only (others: build_v9.py)'
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
    exec(compile((TOOLS/'patch_v9_cell.py').read_text(), 'patch_v9_cell', 'exec'))
    exec(compile((TOOLS/'patch_v9_broker.py').read_text(), 'patch_v9_broker', 'exec'))
    exec(compile((TOOLS/'patch_v9e_broker.py').read_text(), 'patch_v9e_broker', 'exec'))
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
    launch = NEW/'source/scripts/launch_webprm_table5_campaign.py'
    t = launch.read_text()
    exec(compile((TOOLS/'patch_v9_launcher.py').read_text(), 'patch_v9_launcher', 'exec'))
    launch.write_text(t)
    for name in ('proposal.json', 'worker-template.json', 'resource-request.json', 'campaign.sbatch', 'cohort.json'):
        p = NEW/name; p.write_text(retarget(p.read_text()))
    b = NEW/'campaign.sbatch'; bt = b.read_text()
    h, m = divmod(V9['total_seconds']//60, 60)
    bt = re.sub(r'--time=\d\d:\d\d:00', f'--time={h:02d}:{m:02d}:00', bt)
    bt = re.sub(r'--job-name=\S+\n', '--job-name=wa-v9f-'+NAME+'\n', bt, count=1)
    assert '--account=vlm\n' in bt
    b.write_text(bt)

if not REVIEW.exists():
    comps = {p: sha(p) for p in (NEW/'source/scripts/run_native_webarena_cell.py', BROWSER_V3,
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

# Cohort: the site's released single-site tasks (Reddit: minus DROPPED_TASK_IDS) x three methods.
old_cohort = read(OLD/'cohort.json')
ids = sorted(t['task_id'] for t in TASKS if t['sites'] == [SITE] and t['task_id'] not in L.DROPPED_TASK_IDS)
assert len(ids) == L.V9_SITE_TASKS[SITE]
selectors = ['no_search', 'arbiter7b', 'shepherd8b']
cells = [dict(actor=ACTOR, cell_id=f'native_table5_v9/{SITE}/task_{tid:03d}/{ACTOR}__{sel}', domain=SITE,
              original_task_id=TASKS[tid]['old_task_id'], selector=sel, task_config_sha256=cohort_task_digest(TASKS[tid]),
              task_id=tid) for sel in selectors for tid in ids]
assert all(TASKS[tid]['task_id'] == tid for tid in ids)
cohort = dict(old_cohort, cells=cells, domain_counts={SITE: len(ids)}, episode_count=len(cells), task_count=len(ids),
              protocol=L.V9_COHORT_PROTOCOL, selectors=selectors, actors=[ACTOR],
              excluded_tasks=[e for e in old_cohort.get('excluded_tasks', []) if SITE == 'reddit'])

template = read(NEW/'worker-template.json')
template['protocol'] = L.V9_COHORT_PROTOCOL
for key in ('native_contract', 'protocol_contract'):
    template[key] = dict(template[key], protocol=L.V9_COHORT_PROTOCOL, v9_operational_changes=dict(
        user_decision='ah yes can we follow that config for webarbiter/webshepherd (read the papers again to confirm). also let\'s do B (2026-10-26)',
        candidate_protocol=dict(methods=['arbiter7b', 'shepherd8b'], samples=20, temperature=1.0, top_p=1.0, top_n=5,
            prompt='released single-action envelope (disclosed_single_action_envelope_v2)', ranking='frequency desc, then first sampled',
            validity='released filter_candidates on each sample', confidence='sample share among valid samples',
            sources=['WebShepherd (arXiv 2505.15277v2) App. F.2: 20 samples, nucleus sampling, temperature 1.0, top-n most frequent',
                     'WebArbiter (arXiv 2601.21872v1) Sec. 5.2.1: following WebShepherd, Best-of-N, N=5, knockout tournament']),
        no_search='unchanged v8 single-action control', webshepherd_ties='more frequent (lower index) wins, as the paper',
        new_tab='native-browser-v3 lazily creates the CDP session (device metrics + Accessibility.enable) for action-opened tabs',
        site=SITE, sites_in_campaign=['shopping', 'shopping_admin', 'reddit', 'gitlab'], map_excluded='WebArbiter Table 5 excludes Map',
        fuzzy_judge='retired gpt-4-1106-preview unavailable: cell invalid; offline gpt-4.1 re-score is a labelled sensitivity analysis',
        auditor='native-audit-v6', gitlab='gitlab-hosting-validation-v8 job 389698; v9-tools/gitlab/patch_v9_gitlab.py'))
template['endpoints'] = {SITE: GITLAB_ENTRY['endpoint']}
template['native_components']['episode'] = dict(path=str(BROWSER_V3), sha256=sha(BROWSER_V3))
template['native_episode'] = dict(path=str(BROWSER_V3), sha256=sha(BROWSER_V3))
template['native_auditor'] = dict(path=str(AUD/AUD_FILES[0]), sha256=sha(AUD/AUD_FILES[0]))
swap = {str(D/'native-browser-v2/native_episode.py'): str(BROWSER_V3),
        **{str(D/'native-audit-v4'/n): str(AUD/n) for n in AUD_FILES}}
template['code_sha256'] = {swap.get(p, p): None for p in template['code_sha256']}
for rel in MAGENTO_FILES:
    if not rel.startswith('tests/'):
        template['code_sha256'][str(NEW/'source'/rel)] = None
template['code_sha256'] = {p: sha(p) for p in template['code_sha256']}
(NEW/'worker-template.json').write_text(json.dumps(template, sort_keys=True, indent=2)+'\n')
cohort['protocol_sha256'] = L.digest(template['native_contract'])
(NEW/'cohort.json').write_text(json.dumps(cohort, sort_keys=True, indent=2)+'\n')

proposal = read(NEW/'proposal.json')
proposal['domains'] = {SITE: GITLAB_ENTRY}
proposal['resources'] = dict(proposal['resources'], total_seconds=V9['total_seconds'])
proposal['limits'] = dict(proposal['limits'], api_usd=V9['api_usd'], paid_requests=V9['paid_requests'],
                          local_requests=V9['local_requests'], total_requests=V9['paid_requests']+V9['local_requests'],
                          max_workers=V9['max_workers'])
proposal['v8_cell_blocks'] = [dict(selector=b.split(':')[0], task_ids=[int(x) for x in b.split(':')[1].split(',')])
                              for b in os.environ['V9_BLOCKS'].split(';')]
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
proposal['v7_fresh_experiment'] = dict(identity='native_table5_v9', actor=ACTOR, site=SITE,
    parent=dict(path=str(OLD/'proposal.json'), sha256=sha(OLD/'proposal.json')),
    user_decisions=[str(D/'v7-decision-20261023.json'), str(D/'v8-tools/v8-decision-20261025.json'), str(TOOLS/'v9-decision-20261026.json')])
proposal['worker_template_sha256'] = sha(NEW/'worker-template.json')
proposal['status'] = 'prepared_v9_pending_approval'
proposal['operational_change'] = 'V9 fresh experiment: see v9_operational_changes in native_contract.'
src = {p: sha(p) for p in proposal['source_sha256'] if Path(p).exists() and 'native-browser-v2' not in p and 'native-audit-v4' not in p}
for p in sorted((NEW/'source').rglob('*')):
    if p.is_file() and '__pycache__' not in p.parts:
        src[str(p)] = sha(p)
for p in (BROWSER_V3, D/'native-browser-v3/test_newtab.py', *[TOOLS/'gitlab'/n for n in ('patch_v9_gitlab.py', 'gen_gitlab_stage.py', 'webprm_gitlab_site.py')], REVIEW, Path(__file__).resolve(), TOOLS/'v9-decision-20261026.json',
          *[TOOLS/n for n in ('patch_v9_cell.py', 'patch_v9_launcher.py', 'patch_v9_broker.py', 'patch_v9e_broker.py', 'patch_v9f_gitlab.py')],
          NEW/'campaign.sbatch', NEW/'cohort.json', NEW/'worker-template.json', OLD/'proposal.json',
          *[AUD/n for n in AUD_FILES], AUD/'test_native_audit_v6.py'):
    src[str(p)] = sha(p)
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
    assert 'native-browser-v2' not in blob and 'native-audit-v4' not in blob and 'native-audit-v5' not in blob and OLD_NAME not in blob, 'stale v8 reference'
L.validate_proposal(proposal)
(NEW/'proposal.json').write_text(json.dumps(proposal, sort_keys=True, indent=2)+'\n')

request = read(NEW/'resource-request.json')
tmpl = request['approval_template']
tmpl.update(proposal_sha256=L.digest(proposal), resources=proposal['resources'], limits=proposal['limits'],
            protocol_sha256=proposal['protocol_sha256'], ephemeral_policy_sha256=L.digest(proposal['ephemeral_policy']),
            api_usd_cap=proposal['limits']['api_usd'], account='vlm', maximum_allocation_attempts=1, attempt_job_ids=[], job_id=None)
request = dict(approval_template=tmpl, proposal=dict(path=str(NEW/'proposal.json'), sha256=sha(NEW/'proposal.json')),
               proposal_digest=L.digest(proposal), status='v9_pending_exact_user_approval', account='vlm')
(NEW/'resource-request.json').write_text(json.dumps(request, sort_keys=True, indent=2)+'\n')
for p in (NEW/'proposal.json', NEW/'resource-request.json', NEW/'worker-template.json', NEW/'cohort.json'):
    os.chmod(p, 0o600)
print(json.dumps(dict(package=str(NEW), digest=L.digest(proposal), site=SITE, cohort_cells=len(cells),
                      admitted_cells=len(L.v8_block_cells(proposal, cells)), total_seconds=V9['total_seconds'], api_usd=V9['api_usd'],
                      routes=[(r['model'], r['purpose'], r.get('max_choices', 1)) for r in proposal['broker_routes'] if not r['local']]), indent=1))
