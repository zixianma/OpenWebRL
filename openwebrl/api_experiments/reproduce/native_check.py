#!/usr/bin/env python3
"""Verify native installation; optionally run one real browser with zero API calls.

Use --browser-preflight to assemble a new isolated runtime, start Chromium on
about:blank through the original native tools and kernel confinement, verify
cleanup and save receipts. Without it, source/dependency checks are read-only.
Neither path loads credentials or permits a provider request.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

from native import load, read, write
from native_runtime import build_runtime, verify_browser, verify_source


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dependencies', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='New isolated check directory')
    parser.add_argument('--browser-preflight', action='store_true')
    parser.add_argument('--timeout-seconds', type=int, default=300)
    args = parser.parse_args(argv)
    dependencies = read(args.dependencies)
    cuawright = Path(dependencies.get('cuawright_source', dependencies.get('cuawright', '')))
    judge = Path(dependencies['official_webjudge'])
    chromium = Path(dependencies['chromium'])
    cache = Path(dependencies['browser_cache'])
    python = dependencies['python']
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError('Output exists; preserve earlier check evidence')
    if args.output.resolve().is_relative_to(Path('/gpfs/home/zixianma')):
        raise ValueError('Use a project or temporary output directory')
    if args.timeout_seconds < 60:
        raise ValueError('Browser check needs a timeout of at least 60 seconds')
    verify_source(cuawright, 'cuawright')
    verify_source(judge, 'webjudge')
    verify_browser(chromium, cache)
    if not Path(python).is_file():
        raise ValueError('Incomplete installed Python/Chromium dependencies')
    if not args.browser_preflight:
        print(json.dumps({'verified': True, 'mode': 'read-only', 'paid_provider_calls': 0, 'browser_starts': 0}))
        return 0
    # Zero model/judge credits are immutable. A second independent barrier below
    # forbids the gateway's physical HTTP primitive even if admission regresses.
    cap = {'shard': 0, 'task_ids': ['native-startup-check'], 'model_max_inflight': 8,
           'model_usd_all_attempts': 0, 'external_judge_usd_all_attempts': 0,
           'model_calls_all_attempts': 0, 'external_judge_calls_all_attempts': 0,
           'browser_starts_all_attempts': 1}
    plan = {'arm_id': 'luna_high_single', 'model': 'gpt-6-luna', 'candidate_count': 1,
            'purpose': 'zero-provider-call browser installation check', 'shards': [cap]}
    runtime = build_runtime(args.output, cuawright, judge, chromium, cache, plan)
    sys.path.insert(0, str(runtime))
    gateway_module = load(runtime / 'budget_gateway.py', '_zero_provider_native_gateway')
    dispatches = []
    def deny_dispatch(*args, **kwargs):
        dispatches.append(time.time())
        raise AssertionError('Provider dispatch is forbidden in installation checks')
    gateway_module._forward = deny_dispatch
    deadline = time.time() + args.timeout_seconds
    gateway, baseurl = gateway_module.start_gateway(runtime, 0, deadline, 'not-a-provider-key')
    probe = runtime / 'shard0/startup-check'
    probe.mkdir()
    worker = load(runtime / 'campaign_worker.py', '_native_installation_worker')
    worker.PYTHON = python
    result = {'verified': False, 'paid_provider_calls': 0, 'error': None}
    try:
        code = worker.invoke([python, str(runtime / 'diagnostic_browser.py'), '--output', str(probe), '--endpoint', baseurl + '/browser/native-startup-check/reserve', '--approved-browser-remaining', '1'], probe, deadline, probe / 'startup.log')
        receipt = read(probe / 'startup-diagnostic.json')
        cleanup = read(probe / 'parent-owned-teardown.json')
        usage = gateway_module.totals(runtime, 0)
        verified = (code == 0 and receipt.get('actor_release_permitted') is True
                    and receipt.get('offline_mock') is False
                    and receipt.get('paid_model_calls') == receipt.get('model_dispatches') == 0
                    and cleanup.get('verified') is True
                    and usage['model']['calls'] == usage['judge']['calls'] == 0
                    and usage['browser_starts'] == 1 and not dispatches)
        # Import the pinned judge and required dependencies without invoking it.
        judge_module = load(runtime / 'external_judge.py', '_native_installation_judge')
        judge_module.load_official()
        result.update(verified=verified, native_exit_code=code,
                      browser_starts=usage['browser_starts'], usage=usage,
                      kernel_guard=receipt.get('kernel_guard'), landlock_abi=receipt.get('landlock_abi'),
                      judge_source_import_verified=True, owned_cleanup_verified=cleanup.get('verified'),
                      startup_receipt=str(probe / 'startup-diagnostic.json'))
    except Exception as error:
        result['error'] = {'type': type(error).__name__, 'message': str(error)}
    finally:
        gateway.shutdown()
        gateway.server_close()
        result['physical_dispatch_primitive_calls'] = len(dispatches)
        write(runtime / 'native-installation-check.json', result)
    print(json.dumps(result, indent=2))
    return 0 if result['verified'] else 1


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError) as error:
        print('Error: ' + str(error), file=sys.stderr)
        raise SystemExit(2)
