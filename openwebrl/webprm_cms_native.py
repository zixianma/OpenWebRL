"""Zero-model native CMS authentication/product mutation/reset controls.

The caller must already own the approved private service namespace. Functions
are inert until invoked; credentials and browser storage remain private.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
from urllib.parse import urlsplit

from .webprm_cms_hosting import Closed, put, read, reference
from .webprm_webarena import TASK_FILE, load_native_evaluator, verify_source

ORIGIN = 'http://127.0.0.1:7780'
# Magento's admin area frontName is 'admin' and the dashboard route is admin/dashboard, so the
# canonical URL is /admin/admin/dashboard/; '/admin/dashboard' is a frontend 404 (job 357728).
DASHBOARD_PATH = '/admin/admin/dashboard/'
NAME_SELECTOR = 'input[name="product[name]"]'
NAME_JS = 'document.querySelector(' + json.dumps(NAME_SELECTOR) + ').value'


def product_name_field(page, timeout=60000):
    """Magento renders the product form with UI components after domcontentloaded (probe
    357743): wait for one visible, populated name input before reading or editing it."""
    page.wait_for_function('(s)=>{const e=document.querySelectorAll(s);return e.length===1&&e[0].offsetParent!==null&&e[0].value!==""}',
                           arg=NAME_SELECTOR,timeout=timeout)
    field=page.locator(NAME_SELECTOR)
    if field.count()!=1:
        raise Closed('Native product name field is not unique')
    return field


def validate_product(row):
    if (not isinstance(row,dict) or set(row)!={'id','name'} or type(row['id']) is not int or
            row['id']<=0 or not isinstance(row['name'],str) or not 1<=len(row['name'])<=512):
        raise Closed('One bounded deterministic native product required')
    return row


def reset_gate(baseline, mutation, reset, *, media_absent):
    if (any(row.get('status')!='passed' for row in (baseline,mutation,reset)) or
            baseline['instance_id']!=mutation['instance_id'] or baseline['instance_id']==reset['instance_id'] or
            baseline['base_sha256']!=reset['base_sha256'] or baseline['base_sha256']!=mutation['base_sha256'] or
            baseline['product']!=reset['product'] or mutation['product']['id']!=baseline['product']['id'] or
            mutation['product']['name']==baseline['product']['name'] or
            reset.get('session_sha256')==baseline.get('session_sha256') or media_absent is not True or
            not all(row.get('native_positive')==1 and row.get('native_negative')==0 for row in (baseline,mutation,reset))):
        raise Closed('Independent fresh CMS state did not restore the native fixture')
    return dict(status='passed',native_authentication_verified=True,fresh_reset_verified=True,
                original_state_equivalence_verified=False,benchmark_tasks_executed=0,model_calls=0,
                product_id=baseline['product']['id'],baseline_instance=baseline['instance_id'],
                reset_instance=reset['instance_id'])


def evaluator_controls(native, page, expected):
    """Exact live-control task/trajectory, also exercised on an offline page."""
    expected=validate_product(expected)
    task=dict(intent='Synthetic CMS hosting reset validation',eval=dict(eval_types=['program_html'],
        program_html=[dict(url='last',locator=NAME_JS,required_contents=dict(exact_match=expected['name']))]))
    trajectory=[dict(observation=dict(text='Synthetic CMS native control'),info={})]
    positive=native.evaluate(task,trajectory,page,termination='stop')
    negative=deepcopy(task);negative['eval']['program_html'][0]['required_contents']['exact_match']+=' __not_expected__'
    rejected=native.evaluate(negative,trajectory,page,termination='stop')
    return dict(positive=positive,negative=rejected)


def browser_control(options):
    """Concrete released-login + native evaluator path; no live use in tests."""
    from playwright.sync_api import sync_playwright
    out=Path(options['output']);out.mkdir(mode=0o700,exist_ok=False)
    source=Path(options['native_source']);manifest=verify_source(source)
    binding=read(reference(options['account_binding']))
    if binding.get('domain')!='shopping_admin' or any(
            hashlib.sha256((source/name).read_bytes()).hexdigest()!=value
            for name,value in binding['source']['files'].items()):
        raise Closed('Released CMS authentication source/binding changed')
    account_path=reference(binding['native_account'],maximum=16384)
    if account_path.stat().st_mode&0o077:
        raise Closed('Private native account permissions required')
    account=read(account_path,16384)
    if set(account)!={'username','password'} or not all(isinstance(v,str) and v for v in account.values()):
        raise Closed('Released account fields differ')
    expected=validate_product(options['product']);phase=options['phase']
    if phase not in ('baseline','mutate','reset'):
        raise Closed('Unknown CMS native fixture phase')
    marker=options['mutation_name']
    if not isinstance(marker,str) or not marker.startswith('OpenWebRL CMS reset ') or len(marker)>128:
        raise Closed('Bounded fixed mutation marker required')
    native=load_native_evaluator(source)
    report=dict(status='failed',phase=phase,instance_id=options['instance_id'],base_sha256=options['base_sha256'],
                source=manifest,model_calls=0,benchmark_tasks_executed=0,third_party_candidate=True,
                original_state_equivalence_verified=False,allowed_requests=0,blocked_requests=0)
    with sync_playwright() as playwright:
        browser=playwright.chromium.launch(executable_path=options['chromium'],headless=True)
        context=browser.new_context(viewport=dict(width=1280,height=720))
        def restrict(route):
            parsed=urlsplit(route.request.url)
            if (parsed.scheme,parsed.netloc)==('http','127.0.0.1:7780'):
                report['allowed_requests']+=1;route.continue_()
            else:
                report['blocked_requests']+=1;route.abort()
        context.route('**/*',restrict)
        page=context.new_page();page.set_default_timeout(15000)
        def visit(path):
            response=page.goto(ORIGIN+path,wait_until='domcontentloaded',timeout=30000)
            if (response is None or response.status!=200 or
                    (urlsplit(page.url).scheme,urlsplit(page.url).netloc)!=('http','127.0.0.1:7780')):
                raise Closed('CMS route failed or escaped its owned origin')
        try:
            # Exact released shopping_admin login locators; no alternate account.
            visit('/admin')
            page.get_by_placeholder('user name').fill(account['username'])
            page.get_by_placeholder('password').fill(account['password'])
            page.get_by_role('button',name='Sign in',exact=True).click()
            visit(DASHBOARD_PATH)
            if 'Dashboard' not in page.content() or page.get_by_placeholder('user name').count():
                raise Closed('Native CMS dashboard authentication failed')
            state=context.storage_state();put(out/'storage-state-private.json',state)
            report['session_sha256']=hashlib.sha256(json.dumps(state,sort_keys=True).encode()).hexdigest()
            if phase=='baseline':
                tasks=json.loads((source/TASK_FILE).read_text())
                tasks=[t for t in tasks if t['sites']==['shopping_admin']]
                urls=sorted({t['start_url'].replace('__SHOPPING_ADMIN__',ORIGIN+'/admin') for t in tasks})
                for url in urls:
                    if not url.startswith(ORIGIN+'/admin'):
                        raise Closed('Released CMS start URL differs from owned endpoint')
                    visit(url[len(ORIGIN):])
                report['released_task_count']=len(tasks);report['released_start_urls_checked']=len(urls)
            path=f"/admin/catalog/product/edit/id/{expected['id']}/"
            visit(path)
            field=product_name_field(page)
            if field.input_value()!=expected['name']:
                raise Closed('Native product form disagrees with deterministic DB sentinel')
            if phase=='mutate':
                # Exclusive durable intent before the sole Save click. No retry.
                put(out/'mutation-intent.json',dict(product_id=expected['id'],original_name=expected['name'],
                    replacement_name=marker,instance_id=options['instance_id'],save_attempts=1))
                field.fill(marker)
                save=page.locator('#save-button')
                if save.count()!=1:
                    raise Closed('Native product Save button is not unique')
                save.click()
                # Wait for whichever admin message the save produced; a non-success is reported verbatim
                # (bounded) instead of a bare timeout (probe 357750).
                page.wait_for_selector('.message-success, .message-error, .message-warning',timeout=60000)
                if not page.locator('.message-success').count():
                    text=' | '.join(page.locator('.message-error, .message-warning').all_inner_texts())[:500]
                    raise Closed('Native product save did not succeed: '+text)
                visit(path)
                expected=dict(expected,name=marker)
                if product_name_field(page).input_value()!=marker:
                    raise Closed('Saved CMS product name did not persist')
            controls=evaluator_controls(native,page,expected)
            put(out/'native-verdicts.json',controls)
            positive,rejected=controls['positive'],controls['negative']
            if not positive['valid'] or positive['score']!=1 or not rejected['valid'] or rejected['score']!=0:
                raise Closed('Native positive/negative evaluator controls failed')
            page.screenshot(path=str(out/'terminal.png'))
            report.update(status='passed',product=expected,native_positive=1,native_negative=0)
            put(out/'outcome-before-browser-close.json',report)
        finally:
            context.close();browser.close()
    put(out/'result.json',report)
    return report
