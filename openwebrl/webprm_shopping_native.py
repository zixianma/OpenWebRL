"""Zero-model native Shopping authentication/customer mutation/reset controls.

Mirrors webprm_cms_native for the customer-facing Magento store (port 7770). The
caller must already own the approved private service namespace. Functions are
inert until invoked; credentials and browser storage remain private.

The mutation is the released customer account form: one first-name save, a plain
form POST that Magento confirms with a success message. It (and the wishlist add that
probe 358020 saw silently fail) needs the storefront JS to have synced the form_key
input with its cookie first; probe 358448 verified both persist once synced.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import re
from pathlib import Path
from urllib.parse import urlsplit

from .webprm_cms_hosting import Closed, put, read, reference
from .webprm_cms_native import validate_product as validate_sentinel
from .webprm_webarena import TASK_FILE, load_native_evaluator, verify_source

ORIGIN = 'http://127.0.0.1:7770'
LOGIN_PATH = '/customer/account/login/'
# Released auto_login expiry check for shopping: exact final URL {SHOPPING}/wishlist/.
CHECK_PATH = '/wishlist/'
EDIT_PATH = '/customer/account/edit/'
FIRSTNAME_SELECTOR = '#firstname'
FIRSTNAME_JS = 'document.querySelector(' + json.dumps(FIRSTNAME_SELECTOR) + ').value'
MARKER_PREFIX = 'OpenWebRL Reset '
FORM_KEY_SYNCED_JS = ('() => { const input = document.querySelector(\'#form-validate input[name="form_key"]\');'
                      ' const match = document.cookie.match(/(?:^|; )form_key=([^;]*)/);'
                      ' return !!(input && match && input.value === decodeURIComponent(match[1])); }')
EMAIL = re.compile(r'[A-Za-z0-9._%+-]{1,64}@[A-Za-z0-9.-]{1,190}')


def start_urls(tasks, origin=ORIGIN):
    """Released shopping start URLs; multi-tab '|AND|' entries open one tab per part."""
    urls = set()
    for task in tasks:
        for part in task['start_url'].split(' |AND| '):
            url = part.replace('__SHOPPING__', origin)
            if not url.startswith(origin + '/') and url != origin or any(c.isspace() for c in url):
                raise Closed('Released Shopping start URL differs from owned endpoint')
            urls.add(url)
    return sorted(urls)


def customer_sql(email):
    """Deterministic DB sentinel for the released customer; the email never reaches logs."""
    if not isinstance(email, str) or not EMAIL.fullmatch(email):
        raise Closed('Released customer email has an unexpected form')
    return ("SELECT JSON_OBJECT('id',entity_id,'name',firstname) FROM customer_entity WHERE email='"
            + email + "';\n").encode()


def marker_ok(marker):
    return (isinstance(marker, str) and marker.startswith(MARKER_PREFIX) and len(marker) <= 64
            and re.fullmatch(r'[A-Za-z ]+', marker) is not None)


def reset_gate(baseline, mutation, reset, *, media_absent):
    if (any(row.get('status') != 'passed' for row in (baseline, mutation, reset)) or
            baseline['instance_id'] != mutation['instance_id'] or baseline['instance_id'] == reset['instance_id'] or
            baseline['base_sha256'] != reset['base_sha256'] or baseline['base_sha256'] != mutation['base_sha256'] or
            baseline['customer'] != reset['customer'] or mutation['customer']['id'] != baseline['customer']['id'] or
            mutation['customer']['name'] == baseline['customer']['name'] or
            reset.get('session_sha256') == baseline.get('session_sha256') or media_absent is not True or
            not all(row.get('native_positive') == 1 and row.get('native_negative') == 0 for row in (baseline, mutation, reset)) or
            not all(row.get('helper', {}).get('status') == 'passed' for row in (baseline, reset))):
        raise Closed('Independent fresh Shopping state did not restore the native fixture')
    return dict(status='passed', native_authentication_verified=True, fresh_reset_verified=True,
                original_state_equivalence_verified=False, benchmark_tasks_executed=0, model_calls=0,
                customer_id=baseline['customer']['id'], baseline_instance=baseline['instance_id'],
                reset_instance=reset['instance_id'])


def evaluator_controls(native, page, expected):
    """Exact live-control task/trajectory on the account form, positive and negative."""
    expected = validate_sentinel(expected)
    task = dict(intent='Synthetic Shopping hosting reset validation', eval=dict(eval_types=['program_html'],
        program_html=[dict(url='last', locator=FIRSTNAME_JS, required_contents=dict(exact_match=expected['name']))]))
    trajectory = [dict(observation=dict(text='Synthetic Shopping native control'), info={})]
    positive = native.evaluate(task, trajectory, page, termination='stop')
    negative = deepcopy(task); negative['eval']['program_html'][0]['required_contents']['exact_match'] += ' __not_expected__'
    rejected = native.evaluate(negative, trajectory, page, termination='stop')
    return dict(positive=positive, negative=rejected)


def released_accounts(source, binding_ref):
    """Customer login and helper REST admin accounts via the reviewed private binding."""
    binding = read(reference(binding_ref))
    if binding.get('domain') != 'shopping' or any(
            hashlib.sha256((Path(source)/name).read_bytes()).hexdigest() != value
            for name, value in binding['source']['files'].items()):
        raise Closed('Released Shopping authentication source/binding changed')
    accounts = []
    for key in ('native_account', 'native_helper_accounts'):
        path = reference(binding[key], maximum=16384)
        if path.stat().st_mode & 0o077:
            raise Closed('Private native account permissions required')
        accounts.append(read(path, 16384))
    account, helpers = accounts
    if set(account) != {'username', 'password'} or not all(isinstance(v, str) and v for v in account.values()):
        raise Closed('Released account fields differ')
    if (set(helpers) != {'shopping_site_admin'} or set(helpers['shopping_site_admin']) != {'username', 'password'}
            or not all(isinstance(v, str) and v for v in helpers['shopping_site_admin'].values())):
        raise Closed('Released helper account fields differ')
    return account, helpers


def helper_control(source, helpers):
    """Released shopping_get_latest_order_url through the owned two-call REST boundary."""
    from .webprm_webarena_helpers import NativeHelpers, SHOPPING
    native = NativeHelpers(source, shopping_origin=ORIGIN, shopping_accounts=helpers)
    with native.requests.session():
        url = native.functions[SHOPPING]()
    match = re.fullmatch(re.escape(ORIGIN) + r'/sales/order/view/order_id/([0-9]+)/', str(url))
    if not match:
        raise Closed('Native Shopping helper returned an unexpected order URL')
    return dict(status='passed', latest_order_id=int(match.group(1)))


def form_key_evidence(context, page):
    """Whether the storefront form_key cookie exists and matches the form input (booleans only)."""
    cookies = {c['name']: c['value'] for c in context.cookies()}
    inputs = page.locator('#form-validate input[name="form_key"]')
    count = inputs.count()
    return dict(inputs=count, cookie_present='form_key' in cookies,
                input_equals_cookie=count == 1 and cookies.get('form_key') == inputs.input_value())


def mage_messages(context):
    """Server flash messages the storefront renders client-side from the mage-messages cookie."""
    from urllib.parse import unquote
    raw = {c['name']: c['value'] for c in context.cookies()}.get('mage-messages', '')
    try:
        rows = json.loads(unquote(raw) or '[]')
    except ValueError:
        return [dict(type='unparsed', text=unquote(raw)[:300])]
    return [dict(type=row.get('type'), text=str(row.get('text'))[:300]) for row in rows if isinstance(row, dict)][:5]


def diagnose(page, out, report):
    """Private, bounded failure evidence: screenshot, path, title and visible form errors."""
    try:
        page.screenshot(path=str(out/'failure.png'))
        put(out/'failure-diagnostics.json', dict(path=urlsplit(page.url).path, title=page.title()[:200],
            form_errors=page.locator('.mage-error:visible, div.field-error:visible').all_inner_texts()[:10],
            messages=page.locator('.message:visible').all_inner_texts()[:10],
            save_landing_path=report.get('save_landing_path'), save_posts=report.get('save_posts'),
            mage_messages=report.get('mage_messages'), form_key=report.get('form_key')))
    except Exception:
        pass


def browser_control(options):
    """Released customer login + native evaluator/helper path; no live use in tests."""
    from playwright.sync_api import sync_playwright
    out = Path(options['output']); out.mkdir(mode=0o700, exist_ok=False)
    source = Path(options['native_source']); manifest = verify_source(source)
    account, helpers = released_accounts(source, options['account_binding'])
    expected = validate_sentinel(options['customer']); phase = options['phase']
    if phase not in ('baseline', 'mutate', 'reset'):
        raise Closed('Unknown Shopping native fixture phase')
    marker = options['mutation_name']
    if not marker_ok(marker):
        raise Closed('Bounded fixed mutation marker required')
    native = load_native_evaluator(source)
    report = dict(status='failed', phase=phase, instance_id=options['instance_id'], base_sha256=options['base_sha256'],
                  source=manifest, model_calls=0, benchmark_tasks_executed=0, third_party_candidate=True,
                  original_state_equivalence_verified=False, allowed_requests=0, blocked_requests=0)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(executable_path=options['chromium'], headless=True)
        context = browser.new_context(viewport=dict(width=1280, height=720))
        def restrict(route):
            parsed = urlsplit(route.request.url)
            if (parsed.scheme, parsed.netloc) == ('http', '127.0.0.1:7770'):
                report['allowed_requests'] += 1; route.continue_()
            else:
                report['blocked_requests'] += 1; route.abort()
        context.route('**/*', restrict)
        posts = []
        def on_response(response):
            # Bounded POST evidence: path, status and redirect path only (no bodies or cookies).
            if response.request.method == 'POST':
                posts.append(dict(path=urlsplit(response.url).path, status=response.status,
                                  location=urlsplit(response.headers.get('location', '')).path))
        context.on('response', on_response)
        page = context.new_page(); page.set_default_timeout(30000)
        def visit(url):
            response = page.goto(url if url.startswith('http') else ORIGIN + url, wait_until='domcontentloaded', timeout=60000)
            if (response is None or response.status != 200 or
                    (urlsplit(page.url).scheme, urlsplit(page.url).netloc) != ('http', '127.0.0.1:7770')):
                raise Closed('Shopping route failed or escaped its owned origin')
        try:
            # Exact released shopping login locators; no alternate account.
            visit(LOGIN_PATH)
            page.get_by_label('Email', exact=True).fill(account['username'])
            page.get_by_label('Password', exact=True).fill(account['password'])
            page.get_by_role('button', name='Sign In').click()
            page.wait_for_load_state('domcontentloaded')
            visit(CHECK_PATH)
            if page.url != ORIGIN + CHECK_PATH:
                raise Closed('Native Shopping customer authentication failed')
            state = context.storage_state(); put(out/'storage-state-private.json', state)
            report['session_sha256'] = hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest()
            if phase == 'baseline':
                tasks = [t for t in json.loads((source/TASK_FILE).read_text()) if t['sites'] == ['shopping']]
                urls = start_urls(tasks)
                for url in urls:
                    visit(url)
                report['released_task_count'] = len(tasks); report['released_start_urls_checked'] = len(urls)
            visit(EDIT_PATH)
            field = page.locator(FIRSTNAME_SELECTOR)
            if field.count() != 1 or field.input_value() != expected['name']:
                raise Closed('Native account form disagrees with deterministic DB sentinel')
            if phase == 'mutate':
                # Exclusive durable intent before the sole Save click. No retry.
                put(out/'mutation-intent.json', dict(customer_id=expected['id'], original_name=expected['name'],
                    replacement_name=marker, instance_id=options['instance_id'], save_attempts=1))
                # Probe 358448: Save persists once the storefront JS has synced the form_key input
                # with its cookie (PageCache reads the session key from that cookie); validations
                # 358132/358247 clicked right after domcontentloaded and Magento silently redirected
                # back to the edit form. Wait for that sync; never click with a stale key.
                page.wait_for_load_state('load')
                try:
                    page.wait_for_function(FORM_KEY_SYNCED_JS, timeout=60000)
                except Exception:
                    report['form_key'] = form_key_evidence(context, page)
                    raise Closed('Storefront form_key input never matched its cookie: ' + json.dumps(report['form_key']))
                report['form_key'] = form_key_evidence(context, page)
                field.fill(marker)
                save = page.locator('#form-validate button.action.save')
                if save.count() != 1:
                    raise Closed('Native account Save button is not unique')
                # The storefront renders flash messages client-side from customer-data (unlike
                # the admin), so no message appeared within 60 s in job 358132. Wait for the
                # form POST's own navigation instead; persistence below (form + DB) decides.
                posts.clear()
                with page.expect_navigation(wait_until='domcontentloaded', timeout=60000):
                    save.click()
                report['save_landing_path'] = urlsplit(page.url).path
                report['save_posts'] = list(posts)[:5]
                report['mage_messages'] = mage_messages(context)
                try:
                    page.wait_for_selector('.message-success, .message-error, .message-warning', timeout=20000)
                except Exception:
                    pass
                report['save_messages'] = dict(
                    success=page.locator('.message-success').all_inner_texts()[:3],
                    error=page.locator('.message-error, .message-warning').all_inner_texts()[:3])
                report['save_messages']['error'] = report['save_messages']['error'] or [
                    m['text'] for m in report['mage_messages'] if m['type'] == 'error']
                if report['save_messages']['error']:
                    raise Closed('Native account save did not succeed: ' + ' | '.join(report['save_messages']['error'])[:500])
                visit(EDIT_PATH)
                expected = dict(expected, name=marker)
                if page.locator(FIRSTNAME_SELECTOR).input_value() != marker:
                    raise Closed('Saved Shopping first name did not persist: ' + json.dumps(dict(
                        landing=report['save_landing_path'], posts=report['save_posts'], form_key=report['form_key'],
                        messages=report['mage_messages']))[:1500])
            controls = evaluator_controls(native, page, expected)
            put(out/'native-verdicts.json', controls)
            positive, rejected = controls['positive'], controls['negative']
            if not positive['valid'] or positive['score'] != 1 or not rejected['valid'] or rejected['score'] != 0:
                raise Closed('Native positive/negative evaluator controls failed')
            if phase != 'mutate':
                report['helper'] = helper_control(source, helpers)
            page.screenshot(path=str(out/'terminal.png'))
            report.update(status='passed', customer=expected, native_positive=1, native_negative=0)
            put(out/'outcome-before-browser-close.json', report)
        except Exception as error:
            message = str(error)
            # Longest first: a short username can be a substring of a password.
            for value in sorted((*account.values(), *helpers['shopping_site_admin'].values()), key=len, reverse=True):
                message = message.replace(value, '[REDACTED]')
            diagnose(page, out, report)
            raise Closed('Shopping native control failed: ' + message[:2000]) from None
        finally:
            context.close(); browser.close()
    put(out/'result.json', report)
    return report
