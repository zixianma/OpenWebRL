#!/usr/bin/env python3
"""Run pinned CUAWright native API experiments with isolated, explicit budgets.

Default: verify inputs and print the plan without writing files, starting a browser
or contacting an API. --execute creates one new output and spends only the supplied
all-attempt caps. No Slurm, historical campaign registration or private data required.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from native_runtime import ROOT, build_runtime, verify_browser, verify_source

ARMS = {'luna_high_single': ('gpt-6-luna', 1),
        'luna_high_n3_select_high': ('gpt-6-luna', 3),
        'luna_high_n5_select_high': ('gpt-6-luna', 5),
        'sol_high_single': ('gpt-6-sol', 1),
        'sol_high_n3_select_high': ('gpt-6-sol', 3)}


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    with Path(path).open('x') as stream:
        json.dump(data, stream, indent=2)
        stream.write('\n')


def tasks_from_file(path, ids=None, limit=1):
    path = Path(path)
    if path.suffix == '.jsonl':
        tasks = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    else:
        tasks = read(path)
    if not isinstance(tasks, list):
        raise ValueError('Tasks must be a JSON array or JSONL')
    result = []
    seen = set()
    for task in tasks:
        ident = str(task.get('task_id') or task.get('id') or '')
        if not re.fullmatch(r'[A-Za-z0-9_-]+', ident) or ident in seen:
            raise ValueError('Task IDs must be unique route-safe strings')
        seen.add(ident)
        if ids and ident not in ids:
            continue
        metadata = task.get('metadata') or {}
        intent = task.get('intent') or task.get('task') or task.get('instruction') or task.get('task_name') or metadata.get('intent')
        url = task.get('start_url') or task.get('url') or metadata.get('start_url') or task.get('website')
        if not isinstance(intent, str) or not intent or not isinstance(url, str) or not url.startswith(('http://', 'https://')):
            raise ValueError('Each task needs nonempty intent and HTTP(S) start_url: ' + ident)
        result.append({**task, 'task_id': ident, 'intent': intent, 'start_url': url})
    if ids and set(ids) != {t['task_id'] for t in result}:
        raise ValueError('Requested task ID missing')
    if not result or limit < 1:
        raise ValueError('Select at least one task')
    return result[:limit]


def run_one(args, runtime, task, gateway, baseurl, deadline):
    """One actor, one external judge; unchanged failures are never retried."""
    output = runtime / 'shard0/episodes' / task['task_id']
    output.mkdir(parents=True)
    write(output / 'task-input.json', task)
    worker = load(runtime / 'campaign_worker.py', '_portable_native_worker')
    worker.PYTHON = args.python
    command = [args.python, str(runtime / 'native_episode.py'), '--task-json', str(output / 'task-input.json'), '--output', str(output), '--endpoint', baseurl + '/model/' + task['task_id'] + '/responses', '--arm', args.arm, '--absolute-deadline', str(deadline)]
    if ARMS[args.arm][1] > 1:
        command += ['--group-config', str(runtime / 'shard0/.host/group-config.json')]
    started = time.time()
    error = None
    native_code = None
    judge_code = None
    try:
        native_code = worker.invoke(command, output, deadline, output / 'native.log')
        native = read(output / 'result.json')
        if not native.get('owned_teardown', {}).get('process_verification', {}).get('verified') or not read(output / 'parent-owned-teardown.json').get('verified'):
            raise RuntimeError('Owned native/browser teardown unverified')
        actor_finished = time.time()
        from native_capture import collect_host_capture_inventory
        collect_host_capture_inventory(runtime, 0, output)
        judge_code = worker.invoke([args.python, str(runtime / 'external_judge.py'), '--task-json', str(output / 'task-input.json'), '--episode-dir', str(output), '--endpoint', baseurl + '/judge/' + task['task_id'] + '/chat/completions', '--output', str(output / 'external-judge.json')], output, deadline, output / 'judge.log')
    except Exception as exc:
        error = {'type': type(exc).__name__, 'message': str(exc)}
    actor_finished = locals().get('actor_finished', time.time())
    native = read(output / 'result.json') if (output / 'result.json').exists() else {}
    judge = read(output / 'external-judge.json') if (output / 'external-judge.json').exists() else {}
    trajectory = read(output / 'trajectory.json') if (output / 'trajectory.json').exists() else {}
    info = trajectory.get('info', {})
    status = info.get('exit_status') or native.get('exit_status')
    execution_valid = native_code == 0 and status in ('Submitted', 'LimitsExceeded') and not native.get('error') and native.get('owned_teardown', {}).get('process_verification', {}).get('verified', False)
    grading_valid = judge_code == 0 and judge.get('valid') is True
    result = {'task_id': task['task_id'], 'arm': args.arm, 'valid': execution_valid and grading_valid, 'execution_valid': execution_valid, 'grading_valid': grading_valid, 'native_exit_status': status, 'native_exit_code': native_code, 'judge_exit_code': judge_code, 'official_label': judge.get('official_label'), 'strict_label': judge.get('strict_label'), 'reward': judge.get('strict_label') if execution_valid and grading_valid else None, 'decisions': info.get('api_calls', native.get('decisions')), 'task_latency_seconds': actor_finished - started, 'external_judge_seconds': time.time() - actor_finished, 'error': error}
    write(output / 'disposition.json', result)
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arm', choices=ARMS, required=True)
    p.add_argument('--tasks', type=Path, default=ROOT.parent / 'data/eval/online-mind2web.jsonl')
    p.add_argument('--task-id', action='append')
    p.add_argument('--limit', type=int, default=1)
    p.add_argument('--dependencies', type=Path, help='dependencies.json from install.py')
    p.add_argument('--cuawright-source', type=Path)
    p.add_argument('--webjudge-source', type=Path)
    p.add_argument('--chromium', type=Path)
    p.add_argument('--browser-cache', type=Path)
    p.add_argument('--python', default=sys.executable)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--model-usd', type=float)
    p.add_argument('--judge-usd', type=float)
    p.add_argument('--model-calls', type=int)
    p.add_argument('--judge-calls', type=int)
    p.add_argument('--browser-starts', type=int)
    p.add_argument('--timeout-seconds', type=int)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.dependencies:
        dependencies = read(args.dependencies)
        for key, name in [('cuawright', 'cuawright_source'), ('official_webjudge', 'webjudge_source'), ('chromium', 'chromium'), ('browser_cache', 'browser_cache')]:
            if getattr(args, name) is None and dependencies.get(key):
                setattr(args, name, Path(dependencies[key]))
        if args.cuawright_source is None and dependencies.get('cuawright_source'):
            args.cuawright_source = Path(dependencies['cuawright_source'])
        args.python = dependencies.get('python', args.python)
    for name in ('cuawright_source', 'webjudge_source', 'chromium', 'browser_cache'):
        if getattr(args, name) is None:
            raise ValueError('Required runtime path: --' + name.replace('_', '-'))
    for name in ('model_usd', 'judge_usd', 'model_calls', 'judge_calls', 'browser_starts', 'timeout_seconds'):
        value = getattr(args, name)
        if value is not None and (not math.isfinite(value) or value <= 0):
            raise ValueError('Budget caps must be positive and finite')
        if args.execute and value is None:
            raise ValueError('--execute requires explicit --' + name.replace('_', '-'))
    if args.output.exists():
        raise FileExistsError('Use a new output directory; no automatic retries')
    tasks = tasks_from_file(args.tasks, args.task_id, args.limit)
    verify_source(args.cuawright_source, 'cuawright')
    verify_source(args.webjudge_source, 'webjudge')
    verify_browser(args.chromium, args.browser_cache)
    model, count = ARMS[args.arm]
    cap = {'shard': 0, 'task_ids': [t['task_id'] for t in tasks], 'model_max_inflight': 8, 'model_usd_all_attempts': args.model_usd, 'external_judge_usd_all_attempts': args.judge_usd, 'model_calls_all_attempts': args.model_calls, 'external_judge_calls_all_attempts': args.judge_calls, 'browser_starts_all_attempts': args.browser_starts}
    plan = {'schema': 'cuawright-collaborator-run-v1', 'task_source_sha256': hashlib.sha256(args.tasks.read_bytes()).hexdigest(), 'selected_tasks_sha256': hashlib.sha256(json.dumps(tasks, sort_keys=True, separators=(',', ':')).encode()).hexdigest(), 'arm_id': args.arm, 'model': model, 'candidate_count': count, 'max_output_tokens': 20000, 'max_decisions': 100, 'reasoning_effort': 'high', 'summary_setting': 20, 'judge': 'official-webjudge-gpt-4o-2024-08-06', 'shards': [cap], 'timeout_seconds': args.timeout_seconds, 'retry_policy': 'no physical gateway retries; pinned native format-repair defaults preserved', 'fallback_policy': 'none'}
    print(json.dumps({'mode': 'execute' if args.execute else 'read-only-dry-run', 'plan': plan}, indent=2))
    if not args.execute:
        return 0
    key = os.environ.get('OPENAI_API_KEY')
    if not key:
        raise ValueError('OPENAI_API_KEY required; credentials are never loaded from a repository .env')
    runtime = build_runtime(args.output, args.cuawright_source, args.webjudge_source, args.chromium, args.browser_cache, plan)
    write(runtime / 'tasks.json', tasks)
    sys.path.insert(0, str(runtime))
    import budget_gateway
    deadline = time.time() + args.timeout_seconds
    gateway, baseurl = budget_gateway.start_gateway(runtime, 0, deadline, key)
    write(runtime / 'shard0/.host/group-config.json', {'group_base_url': baseurl, 'group_secret': gateway.group_secret, 'arm': args.arm, 'shard': 0, 'deadline_epoch': deadline})
    dispositions = []
    try:
        # One reserved, zero-model-call mechanical startup before actor release.
        probe = runtime / 'shard0/startup-check'
        probe.mkdir()
        worker = load(runtime / 'campaign_worker.py', '_portable_native_probe_worker')
        worker.PYTHON = args.python
        code = worker.invoke([args.python, str(runtime / 'diagnostic_browser.py'), '--output', str(probe), '--endpoint', baseurl + '/browser/' + tasks[0]['task_id'] + '/reserve', '--approved-browser-remaining', str(args.browser_starts)], probe, deadline, probe / 'startup.log')
        receipt = read(probe / 'startup-diagnostic.json')
        if code != 0 or receipt.get('actor_release_permitted') is not True or not read(probe / 'parent-owned-teardown.json').get('verified'):
            raise RuntimeError('Native browser preflight failed; see ' + str(probe / 'startup.log'))
        for task in tasks:
            if budget_gateway._halted(runtime, 0) or time.time() >= deadline - 30:
                break
            disposition = run_one(args, runtime, task, gateway, baseurl, deadline)
            dispositions.append(disposition)
            print(json.dumps(disposition))
            if not disposition['execution_valid']:
                break
    finally:
        gateway.shutdown()
        gateway.server_close()
        write(runtime / 'summary.json', {'requested_tasks': len(tasks), 'completed_dispositions': len(dispositions), 'valid_tasks': sum(d['valid'] for d in dispositions), 'successes': sum(d.get('reward') == 1 for d in dispositions), 'results': dispositions, 'usage': budget_gateway.totals(runtime, 0), 'complete': len(dispositions) == len(tasks)})
    return 0 if len(dispositions) == len(tasks) and all(d['valid'] for d in dispositions) else 1


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, OSError, RuntimeError) as exc:
        print('Error: ' + str(exc), file=sys.stderr)
        raise SystemExit(2)
