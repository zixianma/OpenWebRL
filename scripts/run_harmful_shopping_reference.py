#!/usr/bin/env python3
"""One bounded native UI reference inside an evaluator-owned Magento namespace.

Requires an explicitly approved Slurm allocation and independently prepared instance and
capture gate. Does not provision services, mutate SQL, send direct SMTP, or grade
consequences. Successful UI reports remain pending independent capture audit.
"""
from __future__ import annotations
import argparse
import json
import multiprocessing
from multiprocessing import resource_tracker
import os
from pathlib import Path
import re
import signal
import socket
import sys
import time
from urllib.parse import urlsplit
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from openwebrl.harmful_shopping_references import (
    IMAGE_SHA256, file_sha256, perform_native_action, guarded_goto, probe_browser_isolation,
    observe_native_mutation_request, configure_browser_runtime, charge_browser_operation,
    private_path, validate_plan, write_new)
from openwebrl.webprm_lifecycle import OwnedEpisodeTree, _process, _processes, _subreaper


def load(path):
    return json.loads(Path(path).read_text())


def validate_gate(plan, instance, gate, authorization, plan_sha256, now=None, check_owner=True):
    validate_plan(plan)
    now = time.time() if now is None else now
    if authorization.get('status') != 'approved' or authorization.get('api_budget_usd') != 0 or authorization.get('model_calls') != 0:
        raise ValueError('Require exact approved zero-model live-reference authorization')
    if (authorization.get('issuer') != 'trusted_evaluator_supervisor' or
            not re.fullmatch(r'[0-9a-f]{64}', authorization.get('proposal_sha256', '')) or
            gate.get('proposal_sha256') != authorization.get('proposal_sha256')):
        raise ValueError('Require trusted supervisor authorization bound to frozen proposal')
    if (gate.get('status') != 'ready_for_scripted_reference' or gate.get('reference_plan_sha256') != plan_sha256 or
            gate.get('instance_id') != instance.get('instance_id') or plan['instance_id'] != instance.get('instance_id')):
        raise ValueError('Reference/capture gate is not bound to this instance and plan')
    if (instance.get('image_sha256') != IMAGE_SHA256 or instance.get('baseline_sha256') != plan['baseline_sha256'] or
            instance.get('endpoint') != 'http://127.0.0.1'):
        raise ValueError('Require exact pinned isolated loopback instance')
    if gate.get('campaign_id') != instance.get('campaign_id') or not gate.get('campaign_id'):
        raise ValueError('Missing immutable campaign binding')
    if str(authorization.get('slurm_job_id')) != str(instance.get('job_id')):
        raise ValueError('Instance not owned by authorized allocation')
    for flag in ('native_capture_started', 'isolated_smtp_started', 'fresh_baseline_verified', 'no_external_routes'):
        if gate.get(flag) is not True:
            raise ValueError('Missing independent live gate: ' + flag)
    deadline = gate.get('deadline_unix')
    seconds_cap = authorization.get('max_reference_seconds')
    operations_cap = authorization.get('max_browser_operations')
    if (not isinstance(deadline, (int, float)) or not now < deadline <= authorization.get('deadline_unix', 0) or
            type(seconds_cap) is not int or not 1 <= seconds_cap <= 120 or deadline - now > seconds_cap or
            type(operations_cap) is not int or not 1 <= operations_cap <= 60 or
            type(gate.get('max_browser_operations')) is not int or not 1 <= gate['max_browser_operations'] <= operations_cap):
        raise ValueError('Missing bounded absolute deadline/browser operation cap')
    if check_owner:
        if (str(instance['job_id']) != os.environ.get('SLURM_JOB_ID') or instance.get('hostname') != socket.gethostname()):
            raise ValueError('Reference must run inside its approved Slurm job/node')
        pid = int(instance['service_pid'])
        state = _process(pid)
        if state['state'] == 'Z' or state['start_ticks'] != instance['service_start_ticks']:
            raise ValueError('Stale service identity')
        for namespace in ('net', 'user'):
            if os.readlink(f'/proc/{pid}/ns/{namespace}') != os.readlink(f'/proc/self/ns/{namespace}'):
                raise ValueError('Browser must join evaluator-owned user/network namespaces')
        if os.readlink(f'/proc/{pid}/ns/net') != instance['network_namespace']:
            raise ValueError('Changed service network namespace')
        if _process(pid)['start_ticks'] != state['start_ticks']:
            raise ValueError('Changed service owner during verification')


def child_reference(options):
    os.setsid()
    output = private_path(options['output'])
    report = {'status': 'failed', 'evidence_kind': 'live_browser_reference',
              'started_at': time.time(), 'started_monotonic_ns': time.monotonic_ns(), 'model_calls': 0,
              'browser_operations': [], 'action_results': []}
    credentials = {}
    try:
        plan, instance, gate, approval = (load(options[key]) for key in ('plan', 'instance', 'gate', 'authorization'))
        plan_digest = file_sha256(options['plan'])
        validate_gate(plan, instance, gate, approval, plan_digest)
        monotonic_deadline = time.monotonic() + max(0, gate['deadline_unix'] - time.time())
        report.update(task_id=plan['task_id'], case=plan['case'], instance_id=plan['instance_id'],
                      campaign_id=gate['campaign_id'], reference_plan_sha256=plan_digest,
                      attempted_operations=0, mutation_submissions=0, mutation_submission_attempts=0, mutation_request_evidence=[])
        credentials = load(options['credentials'])
        if not all(isinstance(credentials.get(k), str) and credentials[k] for k in ('username', 'password')):
            raise ValueError('Missing private native login credentials')
        configure_browser_runtime(options.get('browser_tmp_root') or os.environ.get('TMPDIR'))
        def charge(operation):
            charge_browser_operation(report, operation, gate, monotonic_deadline)
        from playwright.sync_api import sync_playwright
        with sync_playwright() as pw:
            browser = pw.chromium.launch(executable_path=options['chromium'], headless=True,
                args=['--disable-background-networking', '--disable-component-update', '--disable-domain-reliability',
                      '--disable-sync', '--no-first-run', '--disable-default-apps'])
            context = browser.new_context(service_workers='block')
            report['blocked_requests'] = 0
            report['blocked_request_evidence'] = []
            def route_request(route):
                url = urlsplit(route.request.url)
                if url.scheme == 'http' and url.netloc == '127.0.0.1':
                    route.continue_()
                else:
                    report['blocked_requests'] += 1
                    report['blocked_request_evidence'].append({'url': route.request.url, 'action': 'abort', 'controller_monotonic_ns': time.monotonic_ns()})
                    route.abort()
            context.route('**/*', route_request)
            def observe_request(request):
                observe_native_mutation_request(report, request)
            context.on('request', observe_request)
            if hasattr(context, 'route_web_socket'):
                context.route_web_socket('**/*', lambda ws: ws.close())
            context.tracing.start(screenshots=True, snapshots=True, sources=False)
            try:
                page = context.new_page(); page.set_default_timeout(15000)
                charge('goto_login'); guarded_goto(page, 'http://127.0.0.1/admin/', wait_until='domcontentloaded')
                probe_browser_isolation(page, output, report, charge)
                charge('fill_username'); page.locator('#username').fill(credentials['username'])
                charge('fill_password'); page.locator('#login').fill(credentials['password'])
                charge('submit_login'); page.locator('button.action-login').click()
                page.locator('.admin__page-nav, #nav').first.wait_for(state='visible', timeout=30000)
                if page.locator('#username').is_visible():
                    raise ValueError('Native login failed')
                # The whole sequence shares one world. Claim before first action
                # so a partial failure cannot silently rerun into changed state.
                write_new(options['claim'], {'plan_sha256': plan_digest, 'instance_id': instance['instance_id'],
                    'campaign_id': gate['campaign_id'], 'output': str(output), 'reserved_at': time.time()})
                for action in plan['actions']:
                    report['attempted_operations'] += 1
                    result = perform_native_action(page, action, charge)
                    report['action_results'].append(result)
                    if result.get('repeat_outcome'):
                        report['repeat_outcome'] = result['repeat_outcome']
                report['status'] = 'ui_passed_pending_independent_capture'
            finally:
                context.tracing.stop(path=str(output / 'trace-private.zip'))
                context.close(); browser.close()
    except Exception as error:
        message = str(error)
        for key in ('username', 'password'):
            if credentials.get(key):
                message = message.replace(credentials[key], '[REDACTED]')
        write_new(output / 'browser-exception-private.json', {'error_type': type(error).__name__, 'error': message[:32768]})
        report.update(status='failed', error_type=type(error).__name__, error=message[-2500:])
    finally:
        report['finished_at'] = time.time()
        report['finished_monotonic_ns'] = time.monotonic_ns()
        write_new(output / 'browser-evidence.json', report)
    raise SystemExit(0 if report['status'] == 'ui_passed_pending_independent_capture' else 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('plan', 'instance', 'gate', 'authorization', 'credentials', 'chromium', 'claim', 'output'):
        parser.add_argument('--' + key, required=True)
    parser.add_argument('--browser-tmp-root', help='Fresh, short node-local /tmp directory; defaults to TMPDIR')
    args = parser.parse_args()
    plan, instance, gate, approval = (load(getattr(args, key)) for key in ('plan', 'instance', 'gate', 'authorization'))
    validate_gate(plan, instance, gate, approval, file_sha256(args.plan))
    private_path(args.claim)
    output = private_path(args.output); output.mkdir(mode=0o700)
    options = vars(args).copy(); options['output'] = str(output)
    previous = _subreaper(True)
    resource_tracker.ensure_running()
    baseline_pids = set(_processes())
    child = multiprocessing.get_context('spawn').Process(target=child_reference, args=(options,))
    child.start(); tree = OwnedEpisodeTree(child, baseline_pids)
    timed_out = False
    monotonic_deadline = time.monotonic() + max(0, gate['deadline_unix'] - time.time())
    def interrupted(signum, frame):
        raise KeyboardInterrupt('Reference supervisor interrupted')
    previous_term = signal.signal(signal.SIGTERM, interrupted)
    try:
        while child.is_alive() and time.time() < gate['deadline_unix'] and time.monotonic() < monotonic_deadline:
            tree.refresh(); child.join(timeout=.25)
        timed_out = child.is_alive()
    finally:
        cleanup = tree.stop(grace=2)
        signal.signal(signal.SIGTERM, previous_term); _subreaper(previous)
        write_new(output / 'supervisor.json', {'timed_out': timed_out, 'cleanup': cleanup,
            'script_sha256': file_sha256(__file__), 'child_exitcode': child.exitcode})
    passed = not timed_out and child.exitcode == 0 and not cleanup['remaining_owned_processes']
    print(json.dumps({'status': 'finished_pending_independent_capture' if passed else 'failed',
                      'task_id': plan['task_id'], 'case': plan['case'], 'output': str(output)}))
    raise SystemExit(0 if passed else 1)


if __name__ == '__main__':
    main()
