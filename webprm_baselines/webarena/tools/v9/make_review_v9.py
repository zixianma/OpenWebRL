"""Write integration-review-v9b-<name>.json for a stage-1 v9 package (amends the v8 review; binds the actual v9 components
and the zero-paid test evidence). Usage: make_review_v9.py NAME TESTS_LOG [SMOKE_RECEIPT]"""
import datetime, hashlib, json, re, sys
from pathlib import Path
D = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-webarena-20261011/next-stage-preparation/webarena-subset-review-20261021')
T = D/'v9b-tools'; name = sys.argv[1]; NEW = D/('launch-preparation-v9b-'+name); log = Path(sys.argv[2])
smoke = Path(sys.argv[3]) if len(sys.argv) > 3 else None
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest(); ref = lambda p: dict(path=str(p), sha256=sha(p))
v8 = T.parent/'v8-tools/integration-review-v8-j2-arb-a.json'
r = json.loads(v8.read_text())
text = log.read_text()
assert 'Ran 4 tests' in text and text.count('\nOK') >= 2 and 'FAIL' not in re.split(r'=== (?:\S+ )?package unit suites', text)[0], 'v9 tests did not pass'
comp = {str(NEW/'source/scripts/run_native_webarena_cell.py'): None, str(D/'native-browser-v3/native_episode.py'): None,
        str(D/'native-broker-v2/native_broker_bridge.py'): None, str(D/'native-audit-v5/native_cell_audit.py'): None,
        str(D/'native-audit-v5/native_lowlevel_replay.py'): None, str(NEW/'source/openwebrl/webprm_concurrent_broker.py'): None}
comp = {p: sha(p) for p in comp}
r.update(amends=ref(v8), component_sha256=comp, created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    v9_note=('v9 (user: "follow that config for webarbiter/webshepherd (read the papers again to confirm). also let\'s do B"): '
             'arbiter7b/shepherd8b candidates = 5 most frequent valid actions of 20 samples (n=20, T=1, top_p=1, single-action '
             'envelope), no_search unchanged; native-browser-v3 lazy CDP session for action-opened tabs; one Table 5 site per '
             'package (Magento sites reuse the validated hosting/stage/reset chain); native-audit-v5. Built by v9-tools/build_v9.py with '
             + ', '.join(f'{n}={sha(T/n)}' for n in ('patch_v9_cell.py', 'patch_v9_launcher.py', 'patch_v9_broker.py', 'build_audit_v5.py'))),
    v9_evidence=dict(
        cell_and_replay_tests=dict(test=ref(T/'test_v9.py'), log=ref(log), passed=4, tests=4,
            covers=['20-sample aggregation: invalid/filtered samples dropped, frequency order, first-sampled ties, <=5, empty pool',
                    'top-5 reaches the released tournament (arbiter7b) and the WebShepherd-style pool (frequency order)',
                    'native-audit-v5 whole offline replay of a real v9 fixture episode re-aggregates the saved 20-sample replies',
                    'tampered 20-sample reply rejected', 'static cell/launcher/broker contract incl. n=20 only on native_policy']),
        new_tab=dict(test=ref(D/'native-browser-v3/test_newtab.py'), v2_reproduces_crash=True, v3_passes=True,
                     log=ref(D/'native-browser-v3/tests-389659.log')),
        live_site_smoke=ref(smoke) if smoke else None))
r['limitations'] = r['limitations'] + [
    'v9 n=20 sampling and the WebShepherd-8B ViewerSelector on the top-5 pool are exercised live only in production (fixture tests use a fake broker).',
    'Shopping/CMS use the validated third-party Magento images (original-state equivalence unverified, as in the earlier Magento campaigns).',
    'Cells invalid only because the retired gpt-4-1106-preview fuzzy judge is unavailable stay invalid; their site state may be disposed after reset/closure audit.']
r['paid_calls'] = 0
(T/f'integration-review-v9b-{name}.json').write_text(json.dumps(r, sort_keys=True, indent=1)+'\n')
print(name, 'review written')
