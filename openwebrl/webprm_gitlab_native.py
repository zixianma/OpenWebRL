"""Native GitLab hosting controls; inert until called inside an owned private site."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import re
from pathlib import Path
from urllib.parse import urlsplit

from .webprm_cms_hosting import Closed, read, reference, put
from .webprm_webarena import TASK_FILE, verify_source, load_native_evaluator

ORIGIN = 'http://127.0.0.1:8023'
NAME_SELECTOR = '#user_name'
NAME_JS = "document.querySelector('#user_name').value"


def released_account(source, binding_ref):
    binding = read(reference(binding_ref))
    if binding.get('domain') != 'gitlab' or any(
            hashlib.sha256((Path(source)/name).read_bytes()).hexdigest() != value
            for name, value in binding['source']['files'].items()):
        raise Closed('Released GitLab authentication source changed')
    path = reference(binding['native_account'], maximum=16384)
    if path.stat().st_mode & 0o077:
        raise Closed('Private native GitLab account required')
    account = read(path, 16384)
    if set(account) != {'username', 'password'} or not all(isinstance(x, str) and x for x in account.values()):
        raise Closed('Released GitLab account fields differ')
    return account


def user_sql(username):
    if not isinstance(username, str) or not re.fullmatch(r'[A-Za-z0-9_.-]{1,128}', username):
        raise Closed('Bounded released username required')
    return ("SELECT json_build_object('id',id,'name',name) FROM users WHERE username='"+username+"';\n").encode()


def validate_user(row):
    if (not isinstance(row, dict) or set(row) != {'id', 'name'} or type(row['id']) is not int
            or row['id'] <= 0 or not isinstance(row['name'], str) or not 1 <= len(row['name']) <= 512):
        raise Closed('One deterministic native GitLab user required')
    return row


def reset_gate(baseline, mutation, reset, *, repository_marker_absent):
    if (any(x.get('status') != 'passed' for x in (baseline, mutation, reset))
            or baseline['instance_id'] != mutation['instance_id'] or baseline['instance_id'] == reset['instance_id']
            or len({x['base_sha256'] for x in (baseline, mutation, reset)}) != 1
            or baseline['user'] != reset['user'] or baseline['user']['id'] != mutation['user']['id']
            or baseline['user']['name'] == mutation['user']['name']
            or baseline.get('session_sha256') == reset.get('session_sha256')
            or repository_marker_absent is not True
            or not all(x.get('native_positive') == 1 and x.get('native_negative') == 0 for x in (baseline, mutation, reset))):
        raise Closed('Fresh GitLab state did not restore DB, repository marker and session')
    return dict(status='passed', native_authentication_verified=True, fresh_reset_verified=True,
                repository_storage_marker_reset_verified=True, native_git_commit_mutation_tested=False,
                original_state_equivalence_verified=False, benchmark_tasks_executed=0, model_calls=0,
                user_id=baseline['user']['id'], baseline_instance=baseline['instance_id'], reset_instance=reset['instance_id'])


def evaluator_controls(native, page, expected):
    expected = validate_user(expected)
    task = dict(intent='Synthetic GitLab hosting reset validation', eval=dict(eval_types=['program_html'],
        program_html=[dict(url='last', locator=NAME_JS, required_contents=dict(exact_match=expected['name']))]))
    trajectory = [dict(observation=dict(text='Synthetic GitLab hosting control'), info={})]
    positive = native.evaluate(task, trajectory, page, termination='stop')
    negative = deepcopy(task); negative['eval']['program_html'][0]['required_contents']['exact_match'] += ' __not_expected__'
    return dict(positive=positive, negative=native.evaluate(negative, trajectory, page, termination='stop'))


def commit_profile_change(page, marker, action, proof):
    """Await the native remote-form response before reloading persisted state.

    The pinned profile view uses remote:true. Waiting for load alone can return
    on the old document, then goto cancels the in-flight profile save.
    """
    def response_matches(response):
        url=urlsplit(response.url)
        return (url.scheme=='http' and url.netloc=='127.0.0.1:8023' and url.path=='/-/profile'
                and response.request.method in ('POST','PUT','PATCH'))
    action(page.locator(NAME_SELECTOR).fill, marker)
    with page.expect_response(response_matches,timeout=60000) as pending:
        action(page.get_by_role('button',name='Update profile settings',exact=True).click)
    response=pending.value
    proof.update(method=response.request.method,path=urlsplit(response.url).path,status=response.status)
    if not 200<=response.status<400 or response.finished() is not None:
        raise Closed('Native GitLab profile update response did not complete successfully')
    page.wait_for_load_state('load')
    action(page.goto,ORIGIN+'/-/profile',wait_until='load')
    observed=page.locator(NAME_SELECTOR).input_value()
    proof['observed_name_after_reload']=observed
    if observed!=marker:raise Closed('GitLab profile save did not persist the requested native mutation')


def browser_control(options):
    from playwright.sync_api import sync_playwright
    out = Path(options['output']); out.mkdir(mode=0o700, exist_ok=False)
    source = Path(options['native_source']); manifest = verify_source(source)
    account = released_account(source, options['account_binding']); native = load_native_evaluator(source)
    expected = validate_user(options['user']); phase = options['phase']; marker = options['mutation_name']
    if phase not in ('baseline', 'mutate', 'reset') or not re.fullmatch(r'OpenWebRL GitLab reset [a-f0-9]{16}', marker):
        raise Closed('Bounded GitLab phase/marker required')
    report = dict(status='failed', phase=phase, instance_id=options['instance_id'], base_sha256=options['base_sha256'],
                  source=manifest, model_calls=0, benchmark_tasks_executed=0, third_party_candidate=True,
                  original_state_equivalence_verified=False, allowed_requests=0, blocked_requests=0, browser_actions=0)
    def action(fn, *args, **kwargs):
        report['browser_actions'] += 1
        if report['browser_actions'] > 100: raise Closed('GitLab browser action cap')
        return fn(*args, **kwargs)
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(executable_path=options['chromium'], headless=True)
            context = browser.new_context(viewport=dict(width=1280, height=720))
            try:
                def restrict(route):
                    url = urlsplit(route.request.url)
                    if url.scheme in ('data', 'blob') or (url.scheme == 'http' and url.netloc == '127.0.0.1:8023'):
                        report['allowed_requests'] += 1
                        if report['allowed_requests'] > 10000: route.abort(); return
                        route.continue_()
                    else:
                        report['blocked_requests'] += 1; route.abort()
                context.route('**/*', restrict)
                page = context.new_page(); page.set_default_timeout(60000)
                action(page.goto, ORIGIN+'/users/sign_in', wait_until='load')
                action(page.get_by_test_id('username-field').fill, account['username'])
                action(page.get_by_test_id('password-field').fill, account['password'])
                action(page.get_by_test_id('sign-in-button').click)
                page.wait_for_load_state('load')
                action(page.goto, ORIGIN+'/-/profile', wait_until='load')
                if page.url != ORIGIN+'/-/profile': raise Closed('Released GitLab login did not reach profile')
                if phase == 'mutate':
                    report['profile_update']={}
                    commit_profile_change(page,marker,action,report['profile_update'])
                    expected = dict(expected, name=marker)
                if page.locator(NAME_SELECTOR).input_value() != expected['name']:
                    raise Closed('GitLab profile does not match native database fixture')
                controls = evaluator_controls(native, page, expected)
                if not (controls['positive'].get('valid') and controls['positive'].get('score') == 1
                        and controls['negative'].get('valid') and controls['negative'].get('score') == 0):
                    raise Closed('GitLab native positive/negative evaluator controls failed')
                put(out/'evaluator-controls.json', controls)
                if phase == 'baseline':
                    tasks = [t for t in json.loads((source/TASK_FILE).read_text()) if t['sites'] == ['gitlab']]
                    urls = sorted({t['start_url'].replace('__GITLAB__', ORIGIN) for t in tasks})
                    if len(urls) > 32: raise Closed('Released GitLab route bound changed')
                    for url in urls:
                        response = action(page.goto, url, wait_until='load')
                        if not response or response.status >= 400 or urlsplit(page.url).netloc != '127.0.0.1:8023':
                            raise Closed('Released GitLab start route failed')
                    report.update(released_task_count=len(tasks), released_start_urls_checked=len(urls))
                state = context.storage_state(); private = out/'storage-state-private.json'
                private.write_text(json.dumps(state)); private.chmod(0o600)
                report.update(status='passed', native_positive=1, native_negative=0, user=expected,
                              session_sha256=hashlib.sha256(private.read_bytes()).hexdigest())
                if report['allowed_requests'] > 10000: raise Closed('GitLab request cap')
            finally:
                context.close(); browser.close()
    except Exception as error:
        report['status'] = 'failed'
        report['error'] = (type(error).__name__+': '+str(error)).replace(account['password'], '[REDACTED]')[:1000]
        put(out/'result.json', report)
        raise Closed(report['error']) from None
    put(out/'result.json', report)
    return report
