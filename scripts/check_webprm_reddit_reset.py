#!/usr/bin/env python3
"""Zero-model-call original Reddit login, mutation and fresh-instance reset fixture.

Run baseline, mutate, then reset independently owned hosting to a new writable
instance and run after-reset. This script never starts/stops website services,
resets a database, or deletes anything. Every phase is process-supervised.
"""
from __future__ import annotations

import argparse
import ast
from copy import deepcopy
import hashlib
import ipaddress
import json
import multiprocessing
from multiprocessing import resource_tracker
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import tempfile
import time
from urllib.parse import quote, urlsplit
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from openwebrl.webprm_lifecycle import OwnedEpisodeTree, _process, _processes, _subreaper
from openwebrl.webprm_webarena import (ARBITER_REVISION, TASK_FILE, digest,
                                     load_native_evaluator, verify_source)

AUTH_FILES = ('viewer/src/browser_env/env_config.py', 'viewer/src/browser_env/auto_login.py')
REDDIT_IDS = [5, 14, 97, 98, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 151, 152, 153, 154]
FORUM = 'books'  # Chosen before hosting or task outcomes; synthetic fixture only.
TITLE_JS = 'document.querySelector(".submission__title").textContent.trim()'
BODY_JS = 'document.querySelector(".submission__body").textContent.trim()'


def write_new(path, value, private=False):
    path = Path(path)
    if path.is_symlink():
        raise ValueError('Refusing a result symlink')
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600 if private else 0o644)
    with os.fdopen(fd, 'w') as handle:
        json.dump(value, handle, indent=2)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())


def safe_path(value):
    path = Path(value).resolve()
    approved = (Path('/tmp'), Path('/gpfs/scrubbed/zixianma/openwebrl-runtime'))
    if not any(path == base or base in path.parents for base in approved):
        raise ValueError('Use approved runtime or /tmp paths')
    return path


def load_auth(source):
    """Read released fixture credentials without importing settings or logging them."""
    hashes = {}
    for name in AUTH_FILES:
        released = subprocess.check_output(['git', '-C', str(source), 'show', f'{ARBITER_REVISION}:{name}'])
        actual = (source / name).read_bytes()
        if actual != released:
            raise ValueError('Native authentication source changed')
        hashes[name] = hashlib.sha256(actual).hexdigest()
    tree = ast.parse((source / AUTH_FILES[0]).read_text())
    assignment = next(node for node in tree.body if isinstance(node, ast.Assign) and
                      any(isinstance(target, ast.Name) and target.id == 'ACCOUNTS' for target in node.targets))
    account = ast.literal_eval(assignment.value)['reddit']
    if set(account) != {'username', 'password'} or not all(isinstance(v, str) and v for v in account.values()):
        raise ValueError('Unexpected native Reddit credential contract')
    return account, hashes


def validate_instance(receipt, *, enforce_owner=True):
    required = {'image_sha256', 'instance_id', 'state_path', 'endpoint', 'job_id',
                'service_pid', 'service_start_ticks', 'hostname'}
    if not required <= receipt.keys() or not re.fullmatch('[0-9a-f]{64}', receipt['image_sha256']):
        raise ValueError('Incomplete instance identity receipt')
    if not isinstance(receipt['instance_id'], str) or not receipt['instance_id']:
        raise ValueError('Missing instance identity')
    validate_fixture_scope(receipt, 'baseline')
    provenance = None
    if receipt['scope'] == 'original':
        from openwebrl.webprm_reddit_provenance import validate_original_provenance
        provenance = validate_original_provenance(receipt)
    safe_path(receipt['state_path'])
    parsed = urlsplit(receipt['endpoint'])
    if (parsed.scheme != 'http' or not parsed.hostname or parsed.path not in ('', '/') or
            parsed.username or parsed.password or parsed.query or parsed.fragment or
            any(character.isspace() for character in receipt['endpoint'])):
        raise ValueError('Expected an explicit credential-free HTTP origin')
    try:
        loopback = ipaddress.ip_address(parsed.hostname).is_loopback
    except ValueError:
        loopback = False
    if not loopback:
        raise ValueError('Website endpoint must use a literal loopback IP address')
    parsed.port
    if enforce_owner:
        if str(receipt['job_id']) != os.environ.get('SLURM_JOB_ID'):
            raise ValueError('Fixture must execute inside its approved owned Slurm allocation')
        if receipt['hostname'] != socket.gethostname():
            raise ValueError('Fixture and website must run on the recorded compute host')
        identity = _process(int(receipt['service_pid']))
        if identity['start_ticks'] != int(receipt['service_start_ticks']) or identity['state'] == 'Z':
            raise ValueError('Website service PID identity is stale')
        browser_netns = os.readlink('/proc/self/ns/net')
        service_netns = os.readlink(f'/proc/{int(receipt["service_pid"])}/ns/net')
        if browser_netns != service_netns:
            raise ValueError('Browser and website service must share the same network namespace')
        if _process(int(receipt['service_pid']))['start_ticks'] != identity['start_ticks']:
            raise ValueError('Website service PID changed while checking network namespace')
        return {'browser_network_namespace': browser_netns,
                'service_network_namespace': service_netns,
                'original_provenance': provenance}
    return {'original_provenance': provenance}


def validate_reset_identity(baseline, current):
    if baseline['image_sha256'] != current['image_sha256']:
        raise ValueError('Reset must reuse the identical immutable original image')
    if baseline['instance_id'] == current['instance_id']:
        raise ValueError('Reset must create a new instance identity')
    if Path(baseline['state_path']).resolve() == Path(current['state_path']).resolve():
        raise ValueError('Reset must use a fresh writable state path')
    for receipt in [baseline, current]:
        for field in ['build_receipt_sha256', 'state_provenance_sha256']:
            if not isinstance(receipt.get(field), str) or not re.fullmatch('[0-9a-f]{64}', receipt[field]):
                raise ValueError('Reset identities require verified build and state provenance digests')
    if baseline['build_receipt_sha256'] != current['build_receipt_sha256']:
        raise ValueError('Reset instances must share the same verified original-image build receipt')
    if baseline['state_provenance_sha256'] == current['state_provenance_sha256']:
        raise ValueError('Reset instances must have distinct fresh-state provenance receipts')


def validate_fixture_scope(instance, phase, predecessors=()):
    if instance.get('scope') not in {'diagnostic', 'original'}:
        raise ValueError('Instance scope must explicitly be diagnostic or original')
    diagnostic = instance.get('scope') == 'diagnostic'
    if phase != 'baseline' and (diagnostic or any(
            report.get('eligible_as_reset_baseline') is False for report in predecessors)):
        raise ValueError('Partial-image diagnostic evidence cannot authorize mutation or serve as a reset baseline')
    if phase != 'baseline' and any(
            report.get('eligible_as_reset_baseline') is not True or
            report.get('instance', {}).get('scope') != 'original' for report in predecessors):
        raise ValueError('Predecessor evidence must explicitly establish original-image reset eligibility')
    return {'scope': 'diagnostic_partial_image_not_original_image_validation' if diagnostic else 'original_image_validation',
            'eligible_as_reset_baseline': not diagnostic}


def browser_runtime():
    parent = safe_path(os.environ['TMPDIR'])
    if parent == Path('/tmp') or len(str(parent)) > 45:
        raise ValueError('Set TMPDIR to a short, approved job scratch subdirectory')
    root = Path(tempfile.mkdtemp(prefix='b-', dir=parent))
    tempfile.tempdir = str(root)
    os.environ['TMPDIR'] = str(root)
    for name in ['XDG_CONFIG_HOME', 'XDG_CACHE_HOME', 'XDG_DATA_HOME', 'XDG_RUNTIME_DIR']:
        path = root / name.lower()
        path.mkdir(mode=0o700)
        os.environ[name] = str(path)
    return root


def record_response_failure(network, allowed_origin, response):
    """Count same-origin failures without recording query strings or credentials."""
    parsed = urlsplit(response.url)
    if (parsed.scheme, parsed.netloc) != allowed_origin or response.status < 400:
        return
    kind = response.request.resource_type
    key = f'{kind}:{response.status}'
    counts = network['http_failures_by_type_status']
    counts[key] = counts.get(key, 0) + 1
    if len(network['http_failure_paths']) < 20:
        network['http_failure_paths'].append({'resource_type': kind, 'status': response.status,
                                              'path': parsed.path[:256]})


def record_request_failure(network, allowed_origin, request):
    parsed = urlsplit(request.url)
    if (parsed.scheme, parsed.netloc) != allowed_origin:
        return
    kind = request.resource_type
    counts = network['request_failures_by_type']
    counts[kind] = counts.get(kind, 0) + 1
    if len(network['request_failure_paths']) < 20:
        network['request_failure_paths'].append({'resource_type': kind, 'path': parsed.path[:256],
                                                 'failure': str(request.failure)[:100]})


def mutation_claim_path(baseline_path):
    baseline_path = safe_path(baseline_path)
    baseline = json.loads(baseline_path.read_text())
    if safe_path(baseline['baseline_directory']) != baseline_path.parent:
        raise ValueError('Use the original baseline receipt, not a copied mutation lineage')
    return baseline_path.parent / 'mutation-claim.json'


def reject_previous_mutation(baseline_path):
    claim = mutation_claim_path(baseline_path)
    if claim.exists() or claim.is_symlink():
        raise ValueError('A mutation attempt already exists for this baseline; inspect its durable '
                         'outcome/intent and reset to a fresh baseline before another submission')


def reserve_mutation(baseline_path, output, baseline, instance):
    """An atomic claim prevents duplicate clicks across concurrent or later retries."""
    write_new(mutation_claim_path(baseline_path), {
        'status': 'one_submission_reserved_not_yet_clicked',
        'run_id': baseline['run_id'], 'output_directory': str(output),
        'title': baseline['synthetic_title'], 'instance': instance})


def close_browser_phase(report, output, context, browser):
    """A cleanup stall cannot erase an already verified mutation's identity."""
    if report['phase'] == 'mutate':
        if not report.get('post_path'):
            raise ValueError('Verified mutation is missing its durable post identity')
        outcome = deepcopy(report)
        outcome.update(status='passed', cleanup_pending=True,
                       scope='verified_mutation_outcome_before_browser_cleanup')
        write_new(Path(output) / 'mutation-outcome.json', outcome)
    context.close()
    browser.close()


def child_run(options):
    os.setsid()
    out = Path(options['output'])
    account = {}
    stage = 'initializing'
    report = {'status': 'failed', 'phase': options['phase'], 'model_provider_calls': 0,
              'benchmark_tasks_executed': 0, 'checks': []}
    browser = None
    try:
        source = Path(options['source'])
        report['source'] = verify_source(source)
        account, report['auth_source_hashes'] = load_auth(source)
        instance = json.loads(Path(options['instance_receipt']).read_text())
        report['network_namespace'] = validate_instance(instance)
        report['instance'] = instance
        endpoint = instance['endpoint'].rstrip('/')
        tasks = [task for task in json.loads((source / TASK_FILE).read_text()) if task['sites'] == ['reddit']]
        if [task['task_id'] for task in tasks] != REDDIT_IDS:
            raise ValueError('Released 19-task Reddit membership changed')
        report['task_group'] = {'count': 19, 'task_ids': REDDIT_IDS, 'sha256': digest(tasks),
                                'membership': 'all released single-site Reddit tasks; no outcome-based selection'}
        baseline = json.loads(Path(options['baseline']).read_text()) if options.get('baseline') else None
        mutation = json.loads(Path(options['mutation']).read_text()) if options.get('mutation') else None
        for prior in [baseline, mutation]:
            if prior and prior.get('status') != 'passed':
                raise ValueError('A predecessor fixture phase is incomplete')
        report.update(validate_fixture_scope(instance, options['phase'],
                                             [prior for prior in [baseline, mutation] if prior]))
        if baseline and baseline['source'] != report['source']:
            raise ValueError('Source changed across reset phases')
        if options['phase'] == 'mutate' and instance != baseline['instance']:
            raise ValueError('Mutation must use the verified baseline instance')
        if options['phase'] == 'after-reset':
            validate_reset_identity(baseline['instance'], instance)
            if mutation['instance'] != baseline['instance']:
                raise ValueError('Mutation and baseline instances differ')
        native = load_native_evaluator(source)
        report['browser_runtime'] = str(browser_runtime())
        from playwright.sync_api import sync_playwright
        allowed_origin = (urlsplit(endpoint).scheme, urlsplit(endpoint).netloc)
        network = {'allowed_requests': 0, 'blocked_external_requests': 0,
                   'http_failures_by_type_status': {}, 'http_failure_paths': [],
                   'request_failures_by_type': {}, 'request_failure_paths': []}
        report['network'] = network

        def progress(name):
            nonlocal stage
            stage = name
            with (out / 'progress.jsonl').open('a') as handle:
                handle.write(json.dumps({'stage': name, 'unix_time': time.time()}) + '\n')

        def navigate(page, path, expected=200):
            url = endpoint + path if path.startswith('/') else path
            response = page.goto(url, wait_until='domcontentloaded', timeout=15000)
            if response is None or response.status != expected:
                raise ValueError(f'Unexpected HTTP status at fixture stage {stage}: '
                                 f'{None if response is None else response.status}')
            if (urlsplit(page.url).scheme, urlsplit(page.url).netloc) != allowed_origin:
                raise ValueError('Website navigated outside the owned origin')
            return response.status

        def html_check(page, label, locator, expected):
            task = {'intent': 'Synthetic original Reddit deployment validation', 'eval': {
                'eval_types': ['program_html'], 'program_html': [{'url': 'last', 'locator': locator,
                    'required_contents': {'exact_match': expected}}]}}
            state = {'observation': {'text': 'Synthetic deployment fixture'}, 'info': {}}
            positive = native.evaluate(task, [state], page, termination='stop')
            negative = deepcopy(task)
            negative['eval']['program_html'][0]['required_contents']['exact_match'] += ' __not_expected__'
            rejection = native.evaluate(negative, [state], page, termination='stop')
            if not positive['valid'] or not positive['success'] or not rejection['valid'] or rejection['success']:
                raise ValueError('Native positive/negative HTML fixture failed')
            report['checks'].append({'name': label, 'passed': True,
                                      'native_positive_score': positive['score'],
                                      'native_negative_score': rejection['score']})

        with sync_playwright() as pw:
            browser = pw.chromium.launch(executable_path=options['chromium'], headless=True)
            context = browser.new_context()
            def route_request(route):
                parsed = urlsplit(route.request.url)
                if (parsed.scheme, parsed.netloc) == allowed_origin:
                    network['allowed_requests'] += 1
                    route.continue_()
                else:
                    network['blocked_external_requests'] += 1
                    route.abort()
            context.route('**/*', route_request)
            context.on('response', lambda response: record_response_failure(network, allowed_origin, response))
            context.on('requestfailed', lambda request: record_request_failure(network, allowed_origin, request))
            page = context.new_page()
            page.set_default_timeout(10000)
            progress('health_and_native_ui_login')
            navigate(page, '/')
            navigate(page, '/login')
            page.get_by_label('Username').fill(account['username'])
            page.get_by_label('Password').fill(account['password'])
            page.get_by_role('button', name='Log in', exact=True).click()
            page.wait_for_url(lambda url: urlsplit(str(url)).path != '/login', timeout=10000)
            expected_account_path = '/user/' + quote(account['username']) + '/account'
            navigate(page, expected_account_path)
            if urlsplit(page.url).path.rstrip('/') != expected_account_path:
                raise ValueError('Native UI login did not establish account access')
            report['checks'].append({'name': 'native_ui_login_and_account_access', 'passed': True})
            write_new(out / 'storage-state-private.json', context.storage_state(), private=True)

            if options['phase'] == 'baseline':
                progress('released_task_start_urls')
                urls = sorted(set(task['start_url'].replace('__REDDIT__', endpoint) for task in tasks))
                for url in urls:
                    navigate(page, url)
                report['checks'].append({'name': 'released_reddit_start_urls', 'passed': True,
                                        'unique_urls': len(urls), 'task_count': 19})
                progress('baseline_sentinels')
                navigate(page, '/')
                def post_paths():
                    hrefs = page.locator('.submission__title a[href], a.submission__title[href]').evaluate_all(
                        '(elements) => elements.map(e => e.href)')
                    return sorted({urlsplit(href).path for href in hrefs
                                   if (urlsplit(href).scheme, urlsplit(href).netloc) == allowed_origin and
                                   re.match(r'^/f/[^/]+/\d+/', urlsplit(href).path)})
                posts = post_paths()
                report['sentinel_selection'] = {'homepage_post_count': len(posts), 'source_path': '/',
                                                'rule': 'Lexicographically first three internal post-title links'}
                if len(posts) < 3:
                    # The released account has no subscriptions and this image
                    # has no featured forums. Use the already fixed synthetic
                    # fixture forum; this does not select benchmark outcomes.
                    report['sentinel_selection'].update(source_path='/f/' + FORUM,
                        reason='Homepage lacks three post-title links; use the prospectively fixed fixture forum')
                    navigate(page, '/f/' + FORUM)
                    posts = post_paths()
                if len(posts) < 3:
                    raise ValueError('Fewer than three original fixed-forum post sentinels')
                report['sentinels'] = []
                for path in posts[:3]:
                    navigate(page, path)
                    title = page.evaluate('() => ' + TITLE_JS)
                    if not isinstance(title, str) or not title:
                        raise ValueError('Original post lacks a title')
                    html_check(page, 'baseline_post_title', TITLE_JS, title)
                    report['sentinels'].append({'path': path, 'title': title})
                report['run_id'] = uuid.uuid4().hex
                report['baseline_directory'] = str(out)
                report['synthetic_title'] = 'OpenWebRL reset fixture ' + report['run_id']
                report['synthetic_body'] = 'Synthetic deployment validation marker ' + report['run_id'] + '.'
                report['synthetic_forum'] = FORUM

            elif options['phase'] == 'mutate':
                progress('synthetic_submission_form')
                navigate(page, '/f/' + FORUM)
                page.get_by_role('link', name='Submit', exact=True).click()
                page.get_by_label(re.compile(r'^Title\s*\*?$')).fill(baseline['synthetic_title'])
                page.get_by_label('Body', exact=True).fill(baseline['synthetic_body'])
                # Following the forum's Submit link must retain its forum. Do
                # not silently choose another forum or retry a submitted post.
                form = page.get_by_label(re.compile(r'^Forum\s*\*?$'))
                if form.count() != 1:
                    raise ValueError('Submission form has no unique Forum control')
                forum_label = form.evaluate('(e) => e.tagName === "SELECT" ? e.selectedOptions[0]?.textContent : e.value')
                if str(forum_label).strip().removeprefix('/f/').lower() != FORUM:
                    raise ValueError('Submission form did not retain the fixed fixture forum')
                write_new(out / 'mutation-intent.json', {'status': 'submit_not_yet_clicked',
                    'run_id': baseline['run_id'], 'title': baseline['synthetic_title'],
                    'forum': FORUM, 'instance': instance})
                reserve_mutation(options['baseline'], out, baseline, instance)
                progress('synthetic_submission_click_pending')
                page.get_by_role('button', name='Create submission', exact=True).click()
                page.wait_for_url(re.compile(r'/f/' + FORUM + r'/\d+/'), timeout=15000)
                write_new(out / 'submission-response.json', {
                    'status': 'post_navigation_observed_native_checks_pending',
                    'run_id': baseline['run_id'], 'post_path': urlsplit(page.url).path,
                    'instance': instance})
                progress('synthetic_submission_verification')
                html_check(page, 'synthetic_post_title', TITLE_JS, baseline['synthetic_title'])
                html_check(page, 'synthetic_post_body', BODY_JS, baseline['synthetic_body'])
                report.update(run_id=baseline['run_id'], post_path=urlsplit(page.url).path,
                              synthetic_title=baseline['synthetic_title'])

            else:
                progress('reset_sentinel_restoration')
                for sentinel in baseline['sentinels']:
                    navigate(page, sentinel['path'])
                    html_check(page, 'restored_original_post_title', TITLE_JS, sentinel['title'])
                progress('synthetic_post_absent_after_reset')
                navigate(page, mutation['post_path'], expected=404)
                if baseline['synthetic_title'] in page.locator('body').inner_text():
                    raise ValueError('Synthetic post survived the fresh database reset')
                report['checks'].append({'name': 'synthetic_post_absent_after_fresh_instance_reset',
                                        'passed': True, 'http_status': 404})
                report.update(run_id=baseline['run_id'], reset_identity_verified=True,
                              baseline_receipt_sha256=hashlib.sha256(Path(options['baseline']).read_bytes()).hexdigest(),
                              mutation_receipt_sha256=hashlib.sha256(Path(options['mutation']).read_bytes()).hexdigest())
            report['network'] = network
            report['status'] = 'passed'
            progress('browser_cleanup')
            close_browser_phase(report, out, context, browser)
            browser = None
        report['final_stage'] = 'complete'
    except Exception as error:
        message = str(error)
        for value in account.values():
            message = message.replace(value, '[REDACTED]')
        report.update(status='failed', failed_stage=stage, error_type=type(error).__name__, error=message[:3000])
    finally:
        write_new(out / 'report.json', report)
    raise SystemExit(0 if report['status'] == 'passed' else 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['baseline', 'mutate', 'after-reset'], required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--chromium', type=Path, required=True)
    parser.add_argument('--instance-receipt', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--mutation', type=Path)
    parser.add_argument('--seconds', type=int, default=180)
    args = parser.parse_args()
    if not 1 <= args.seconds <= 180:
        parser.error('Phase limit must be 1 through 180 seconds')
    if args.phase != 'baseline' and args.baseline is None:
        parser.error('This phase requires --baseline REPORT_JSON')
    if args.phase == 'after-reset' and args.mutation is None:
        parser.error('Reset validation requires --mutation REPORT_JSON')
    out = safe_path(args.output)
    if out.exists():
        parser.error('Use a fresh phase output directory; preserve every attempt')
    instance = json.loads(args.instance_receipt.read_text())
    validate_instance(instance)
    validate_fixture_scope(instance, args.phase)
    if args.phase == 'mutate':
        reject_previous_mutation(args.baseline)
    out.mkdir(parents=True, mode=0o700)
    options = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    options['output'] = str(out)
    previous_subreaper = _subreaper(True)
    resource_tracker.ensure_running()
    baseline_pids = set(_processes())
    child = multiprocessing.get_context('spawn').Process(target=child_run, args=(options,))
    child.start()
    tree = OwnedEpisodeTree(child, baseline_pids)
    deadline = time.monotonic() + args.seconds
    timed_out = False
    def interrupt(signum, frame):
        raise KeyboardInterrupt(f'Fixture supervisor received signal {signum}')
    old_term = signal.signal(signal.SIGTERM, interrupt)
    try:
        while child.is_alive() and time.monotonic() < deadline:
            tree.refresh()
            child.join(timeout=.25)
        timed_out = child.is_alive()
    finally:
        cleanup = tree.stop(grace=2)
        signal.signal(signal.SIGTERM, old_term)
        _subreaper(previous_subreaper)
        write_new(out / 'supervisor.json', {'status': 'timed_out' if timed_out else 'finished',
            'limit_seconds': args.seconds, 'cleanup': cleanup,
            'verified_mutation_outcome_available': (out / 'mutation-outcome.json').is_file(),
            'submission_response_available': (out / 'submission-response.json').is_file(),
            'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    passed = not timed_out and child.exitcode == 0 and not cleanup['remaining_owned_processes']
    print(json.dumps({'status': 'passed' if passed else 'failed', 'phase': args.phase,
                      'verified_mutation_outcome_available': (out / 'mutation-outcome.json').is_file(),
                      'model_provider_calls': 0, 'output': str(out)}))
    raise SystemExit(0 if passed else 1)


if __name__ == '__main__':
    main()
