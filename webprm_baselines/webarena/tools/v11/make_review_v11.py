"""Write integration-review-v11-<name>.json for a stage-1 v11 package (copy of v10-tools/make_review_v10.py; amends the v8
review; binds the actual v11 components and the zero-paid test evidence). Usage: make_review_v11.py NAME TESTS_LOG"""
import datetime, hashlib, json, re, sys
from pathlib import Path
D = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-webarena-20261011/next-stage-preparation/webarena-subset-review-20261021')
T = D/'v11-tools'; name = sys.argv[1]; NEW = D/('launch-preparation-v11-'+name); log = Path(sys.argv[2])
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest(); ref = lambda p: dict(path=str(p), sha256=sha(p))
v8 = D/'v8-tools/integration-review-v8-j2-arb-a.json'
r = json.loads(v8.read_text())
text = log.read_text()
head = text.split('=== ' + name + ' package unit suites')[0]
assert head.count('\nOK') >= 3 and 'FAIL' not in head and 'Traceback' not in head, 'v11 tests did not pass'
assert re.search(r'Ran 4 tests', head) and 'piotr v11 replay' in head and 'no_search v11 replay' in head and 'tampered Piotr selection rejected' in head, 'v11 cell/replay evidence missing'
units = text.split('package unit suites')[1]
# Known, pre-existing in every v9e package (tests-v9e-*.log): test_audit_webprm_table5_cms pins the CMS campaign's legacy
# auditor hash (2 failures: CMS/Shopping test_wrapper_runs_the_unchanged_reviewed_auditor_with_only_reset_dispatch).
blocks = [b for b in units.split('\n--- ')[1:]]
# Also pre-existing in every v9f GitLab package (tests-v9f-gitlab-*.log): test_validate_webprm_gitlab_hosting errors=1.
KNOWN = {'test_audit_webprm_table5_cms': 'FAILED (failures=2)', 'test_validate_webprm_gitlab_hosting': 'FAILED (errors=1)'}
bad = [b.split('\n')[0] for b in blocks if ('FAILED' in b or 'Error' in b) and KNOWN.get(b.split('\n')[0].strip(), '\0') not in b]
assert not bad, ('package unit suites failed', bad)
comp = [NEW/'source/scripts/run_native_webarena_cell.py', NEW/'source/openwebrl/webprm_native_piotr_v11.py',
        D/'native-browser-v3/native_episode.py', D/'native-broker-v2/native_broker_bridge.py', D/'native-audit-v8/native_cell_audit.py',
        D/'native-audit-v8/native_lowlevel_replay.py', NEW/'source/openwebrl/webprm_concurrent_broker.py',
        NEW/'source/scripts/launch_webprm_table5_campaign.py']
r.update(amends=ref(v8), component_sha256={str(p): sha(p) for p in comp},
    created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    v11_note=('v11 (user 2026-10-27: gpt-4o + our ARM, option A; then the 1280x720 setup): no_search (rerun) and piotr_selectionarm '
              'interleaved per task on the v9 native harness at a 1280x720 actor viewport (released 5-BLOCK policy prompt, released '
              'filter/dedup/rerank, terminal detector short-circuit, Piotr SelectionARM over 1..5 candidates seeing the actor\'s own '
              '1280x720 screenshot); all v9 operational fixes; native-audit-v8. Built by v11-tools/build_v11.py with '
              + ', '.join(f'{n}={sha(T/n)}' for n in ('patch_v11_cell.py', 'patch_v11_launcher.py', 'webprm_native_piotr_v11.py', 'build_audit_v8.py'))),
    v11_evidence=dict(tests=dict(log=ref(log), cell_test=ref(T/'test_v11.py'), broker_test=ref(T/'test_v11_broker.py'),
            status_test=ref(T/'test_v11_status.py'),
            covers=['real released coordinator + Chromium fixture at 1280x720: released 5-BLOCK prompt sent unchanged (no envelope, no n)',
                    'Piotr request image is byte-identical to the actor observation PNG, 1280x720',
                    'targets below the old 576 px fold get a visible normalized point (in_viewport, y > 800/1000)',
                    'filtered and duplicate slots never reach Piotr; terminal detector short-circuits selection',
                    'native-audit-v8 whole offline replay of piotr_selectionarm (byte-identical Piotr payloads; tampered selection rejected) and of no_search at 1280x720',
                    'audit-v8 bridge-config contract for both methods with negative controls (1024x2490 viewport, crop frame, missing v11 receipt)',
                    'real ConcurrentBrokerProcess: local selector_piotr route; native_policy n=20 rejected',
                    'v9e status read / same-cell parallel calls on the v11 routes', 'package unit suites (known pre-existing failures only)'])))
r['limitations'] = r['limitations'] + [
    'The Piotr SelectionARM server (sglang, processor view v3, compatible RoPE config) and real gpt-4o 5-BLOCK replies are exercised live only in production (tests use a fake broker); the same serving recipe ran in the OM2W and CMS Piotr campaigns.',
    'Shopping/CMS/GitLab use the validated third-party images (original-state equivalence unverified, as in v9).',
    'Cells invalid only because the retired gpt-4-1106-preview fuzzy judge is unavailable stay invalid, as in v9.']
r['paid_calls'] = 0
(T/f'integration-review-v11-{name}.json').write_text(json.dumps(r, sort_keys=True, indent=1)+'\n')
print(name, 'review written')
