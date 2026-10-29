"""Write integration-review-v12-<name>.json for a stage-1 v12 package from its CPU test log (copy of v11b-tools/
make_review_v11c.py with the v12 evidence). Usage: make_review_v12r2.py NAME TESTS_LOG (r2: + disposal regression)"""
import datetime, hashlib, json, re, sys
from pathlib import Path
D = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-webarena-20261011/next-stage-preparation/webarena-subset-review-20261021')
T = D/'v12-tools'; name = sys.argv[1]; NEW = D/('launch-preparation-v12-'+name); log = Path(sys.argv[2])
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest(); ref = lambda p: dict(path=str(p), sha256=sha(p))
v8 = D/'v8-tools/integration-review-v8-j2-arb-a.json'
r = json.loads(v8.read_text())
text = log.read_text()
cellpart = text.split('=== ' + name + ' v12 launcher')[0]
launchpart = text.split('=== ' + name + ' v12 launcher')[1].split('v12 server startup')[0]
startpart = text.split('v12 server startup (r1 proxy regression)')[1].split('v12 r2 disposal regression')[0]
disppart = text.split('v12 r2 disposal regression')[1].split('package unit suites')[0]
assert 'Ran 1 test' in disppart and '\nOK' in disppart and 'disposal regression: r0 lacked' in disppart, 'v12 r2 disposal regression did not pass'
assert 'Ran 2 tests' in startpart and '\nOK' in startpart and 'FAILED' not in startpart and 'patched switch_critic passed under the cluster proxy env' in startpart and 'negative control' in startpart, 'v12 r1 startup regression did not pass'
assert re.search(r'Ran 9 tests', cellpart) and '\nOK' in cellpart and 'FAILED' not in cellpart, 'v12 cell tests did not pass'
for needle in ('import layout: harness from package', 'projection: done/N-A/mixed/negative ok', 'v11 string_match parity:',
               'no_search: ', 'piotr_selectionarm: ', 'screenshots 1280x1000', 'transcript replay verified, audit evidence verified, tampered picks rejected'):
    assert needle in cellpart, ('missing v12 evidence', needle)
assert 'Ran 2 tests' in launchpart and '\nOK' in launchpart and 'FAILED' not in launchpart and 'v12 preflight through relays passed' in launchpart, 'v12 launcher tests did not pass'
units = text.split('package unit suites')[1]
blocks = [b for b in units.split('\n--- ')[1:]]
KNOWN = {'test_audit_webprm_table5_cms': 'FAILED (failures=2)', 'test_validate_webprm_gitlab_hosting': 'FAILED (errors=1)'}
bad = [b.split('\n')[0] for b in blocks if ('FAILED' in b or 'Error' in b) and KNOWN.get(b.split('\n')[0].strip(), '\0') not in b]
assert not bad, ('package unit suites failed', bad)
comp = [NEW/'source/scripts/run_owrl_webarena_cell.py', T/'r2/run_owrl_webarena_cell.py', NEW/'source/openwebrl/webprm_owrl_v12.py',
        D/'native-browser-v3/native_episode.py', D/'native-broker-v2/native_broker_bridge.py', T/'native-audit-v12/native_cell_audit.py',
        NEW/'source/openwebrl/webprm_concurrent_broker.py', NEW/'source/scripts/launch_webprm_table5_campaign.py', T/'owrl_uds_relay.py',
        T/'owrl-snapshot/SNAPSHOT_SHA256SUMS']
evidence = [l for l in cellpart.splitlines() if re.match(r'(import layout|projection|v11 string_match parity|no_search: |piotr_selectionarm: )', l)]
r.update(amends=ref(v8), component_sha256={str(p): sha(p) for p in comp},
    created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    v12_note=('v12 (user 2026-10-28: "ah yes try C"): OpenWebRL-4B-SFT no_search vs + Piotr SelectionARM in the OpenWebRL action space on '
              'the v11 hosted sites/evaluator at 1280x1000; crossbench agent loop/serving from the frozen owrl-snapshot; native-audit-v12. '
              'native_episode is retained only as the Chromium staging/identity component of the controller (the v12 worker does not run it).'),
    v12_r1_fix=dict(note='jobs 395121/395122 failed at server bring-up: the inherited cluster http_proxy (no_proxy lacked 127.0.0.1) carried sglang self-warmup requests to squid, so the servers never became ready and /server_info timed out; r1 adds localhost/127.0.0.1/::1 to NO_PROXY for the launcher and every server (as crossbench) and polls /server_info until ready (600 s)',
        patch=ref(T/'patch_v12_launcher.py'), test=ref(T/'test_v12_server_startup.py'), fake_server=ref(T/'fake_sglang_v12.py'),
        evidence=[l for l in startpart.splitlines() if 'passed' in l or 'negative' in l]),
    v12_r2_fix=dict(note='smoke 395908: 6/6 cells valid and audit-verified but no state disposed: audit_input lacked the required terminal artifact; r2 worker writes native-evaluation/terminal.json + terminal.png and drops proxy env (no external network in the namespace)',
        worker=ref(T/'r2/run_owrl_webarena_cell.py'), test=ref(T/'r2/test_v12_disposal.py'), evidence=[l for l in disppart.splitlines() if 'disposal regression' in l]),
    v12_evidence=dict(tests=dict(log=ref(log), cell_test=ref(T/'test_v12.py'), launcher_test=ref(T/'test_v12_launcher.py'), lines=evidence,
        covers=['import layout (package harness, snapshot agent stack, venv packages ahead of the NLTK runtime)', 'trajectory projection incl. N/A fallback',
                'released-evaluator parity on every saved valid v11 string_match-only final', 'end-to-end local site: real Chromium + snapshot WebEnv at 1280x1000 + '
                'snapshot generate_turn_sample + crossbench MeteredActor/CrossbenchSelector + relays + fake sglang/select; released evaluator '
                '(string/url/program_html) on the live page through SyncPage; transcript replay; native-audit-v12 OpenWebRL evidence + tamper control',
                'launcher preflight through relays with negative controls; crossbench serving recipe and v12 identity in the patched launcher',
                'package unit suites (known pre-existing failures only)'])))
r['limitations'] = r['limitations'] + [
    'The real sglang actor/selector servers, the Piotr /select service on real screenshots and the hosted sites are exercised only by the GPU smoke test.',
    'Shopping/CMS/GitLab use the validated third-party images (original-state equivalence unverified, as in v9-v11).',
    'Cells invalid only because the retired gpt-4-1106-preview fuzzy judge is unavailable stay invalid, as in v9-v11.']
r['paid_calls'] = 0
(T/f'integration-review-v12-{name}.json').write_text(json.dumps(r, sort_keys=True, indent=1)+'\n')
print(name, 'review written')
