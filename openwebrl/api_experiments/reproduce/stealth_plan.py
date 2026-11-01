"""Prepare a deterministic paired OM2W stealth pilot without API calls.

The output contains task identities and stays local. This planner does not
authorize spending, allocate resources, or promise a hard proxy-dollar cap.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
from urllib.parse import urlsplit


def money(value):
    try:
        amount = Decimal(value)
    except (InvalidOperation, TypeError) as exc:
        raise ValueError('Budget must be a positive finite decimal') from exc
    if not amount.is_finite() or amount <= 0:
        raise ValueError('Budget must be a positive finite decimal')
    return amount


def rank(seed, kind, value):
    return hashlib.sha256(json.dumps([seed, kind, value], separators=(',', ':')).encode()).hexdigest()


def build(tasks, *, count=20, seed=42, single_usd='1', n5_usd='3', judge_usd='2', browser_usd='10'):
    if type(count) is not int or type(seed) is not int or count < 1:
        raise ValueError('Count must be positive and seed must be an integer')
    selected = [t for t in tasks if t.get('benchmark', 'online-mind2web') == 'online-mind2web']
    ids = [t['task_id'] for t in selected]
    if len(selected) != 300 or len(set(ids)) != 300 or any(not isinstance(i, str) or not i for i in ids):
        raise ValueError('Use the complete pinned roster of 300 distinct OM2W tasks')
    if count > 300:
        raise ValueError('Count exceeds the OM2W roster')
    budgets = dict(single=money(single_usd), n5=money(n5_usd), terra=money(judge_usd), browser=money(browser_usd))
    groups = defaultdict(list)
    for task in selected:
        host = urlsplit(task['start_url']).hostname
        if not host:
            raise ValueError('Each task needs an absolute start URL')
        groups[host.lower()].append(task['task_id'])
    hosts = sorted(groups, key=lambda h: rank(seed, 'host', h))
    for host in hosts:
        groups[host].sort(key=lambda i: rank(seed, 'task', i))
    ordered = []
    depth = 0
    while len(ordered) < count:
        for host in hosts:
            if depth < len(groups[host]):
                ordered.append((groups[host][depth], host))
                if len(ordered) == count:
                    break
        depth += 1
    pairs = [dict(pair=i, task_id=task_id, host=host,
                  worker_arms=['luna_single_high', 'parallel5_select_high'] if i % 2 == 0 else ['parallel5_select_high', 'luna_single_high'])
             for i, (task_id, host) in enumerate(ordered)]
    return dict(
        schema='browsercode-om2w-us-stealth-pilot-plan-v1', status='prepared_not_authorized',
        benchmark='online-mind2web', task_pairs=count, actor_sessions=2 * count,
        selection=dict(seed=seed, rule='SHA256-ranked hosts and tasks; round-robin hosts, without outcomes',
                       distinct_hosts=len({h for _, h in ordered}), interpretation='Website-coverage pilot; not a representative full-benchmark estimate'),
        pairs=pairs,
        protocol=dict(harness='browsercode', actor='gpt-6-luna', reasoning_effort='high',
                      max_output_tokens=12288, selector_reasoning_effort='high', selector_max_output_tokens=12288,
                      candidate_calls='five concurrent candidate calls followed by one selector for N5; one call for single',
                      max_decisions=30, actor_loop_wall_seconds=900, sdk_timeout_seconds=180,
                      judge=dict(model='gpt-5.6-terra', reasoning_effort='medium', max_output_tokens=4096, seed=42),
                      browser=dict(provider='Browser Use', api='v4', proxyCountryCode='us',
                                   viewport=[1280, 1800], timeout_minutes=20,
                                   profile='fresh, no shared profile', enableRecording=False),
                      scheduling='One matched task at a time; two fresh browser sessions start together; swap worker arms each pair; await both before next pair'),
        proposed_limits=dict(
            usd={k: str(v) for k, v in budgets.items()}, total_usd=str(sum(budgets.values())),
            browser_allowance_kind='operational target, not a provider-enforced hard cap; explicit overshoot approval required',
            browser_stop_new_admission_usd=str(budgets['browser'] * Decimal('0.8')),
            browser_watchdog_stop_usd=str(budgets['browser']),
            browser_creates_all_attempts=2 * count + 2, mechanical_preflight_sessions=2,
            generation_calls_all_attempts=count * 30 * 7,
            model_http_calls_including_input_counters_all_attempts=count * 30 * 7 * 2,
            terra_calls_all_attempts=2 * count, concurrent_browsers=2, concurrent_generation_http=6,
            resources=dict(nodes=1, cpus=2, memory_gib=16, gpus=0, seconds_all_attempts=14400)),
        accounting=dict(all_attempts=True, unknown_cost='preserve as unknown; block new pairs pending reconciliation',
                        fees=['actor', 'Terra', 'provider hosting', 'provider proxy'],
                        timing=['provider lifetime', 'union of model-call intervals', 'browser commands', 'setup/capture/cleanup', 'unclassified or overlapping intervals'],
                        cleanup='Explicit provider stop and stopped-state verification before Terra; retain raw post-stop usage receipts',
                        creation='No automatic create retries; reconcile ambiguous creations before further admission'),
        launch_requires='Separate explicit approval of exact resources, model caps and browser spending/overshoot policy; this file is not approval')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tasks', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--count', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--expected-roster', type=Path, default=Path(__file__).resolve().parents[1] / 'portable_assets/campaign_specs/browsercode_gpt_high/task-roster.json')
    args = parser.parse_args()
    output = args.output.resolve()
    if output.is_relative_to(Path('/gpfs/home/zixianma')):
        raise ValueError('Choose an output outside the read-only home directory')
    raw = args.tasks.read_bytes()
    tasks = json.loads(raw)
    expected_raw = args.expected_roster.read_bytes()
    expected_ids = [row['task_id'] for row in json.loads(expected_raw)]
    actual_ids = [row['task_id'] for row in tasks if row.get('benchmark', 'online-mind2web') == 'online-mind2web']
    if len(expected_ids) != 300 or len(set(expected_ids)) != 300 or set(actual_ids) != set(expected_ids):
        raise ValueError('OM2W identities do not match the pinned reference roster')
    plan = build(tasks, count=args.count, seed=args.seed)
    plan['task_source_sha256'] = hashlib.sha256(raw).hexdigest()
    plan['reference_roster_sha256'] = hashlib.sha256(expected_raw).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as handle:
        json.dump(plan, handle, indent=2, sort_keys=True)
        handle.write('\n')
    print(json.dumps({k: plan[k] for k in ('status', 'task_pairs', 'actor_sessions', 'proposed_limits')}, indent=2))


if __name__ == '__main__':
    main()
