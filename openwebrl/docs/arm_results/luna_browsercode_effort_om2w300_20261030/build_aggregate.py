"""Export aggregate-only provisional Luna BrowserCode results from local evidence.

This exporter reads dispositions and cumulative per-shard ledgers. It validates
rosters, uniqueness, finite observed metrics and ledger caps, but does not perform
the campaign's complete independent request/scheduler/process audit. It publishes
no task IDs, task text, raw provider payloads or absolute evidence paths.
"""
from __future__ import annotations

import argparse
from collections import Counter
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
from pathlib import Path
import statistics

SCHEMA = 'provisional-luna-browsercode-effort-aggregate-v1'


def read(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('Duplicate JSON key in evidence')
            result[key] = value
        return result
    with Path(path).open() as stream:
        return json.load(stream, object_pairs_hook=pairs,
                         parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite evidence JSON')))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(65536), b''):
            digest.update(block)
    return digest.hexdigest()


def money(value):
    try:
        result = Decimal(str(value))
    except InvalidOperation as error:
        raise ValueError('Invalid ledger amount') from error
    if not result.is_finite() or result < 0:
        raise ValueError('Ledger amount must be finite and nonnegative')
    return result


def observed(value):
    if value is None:
        return None
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError('Observed metric must be finite and nonnegative')
    return value


def metric(values):
    values = list(values)
    present = [value for value in values if value is not None]
    return {'mean': statistics.fmean(present) if present else None,
            'observed_tasks': len(present), 'missing_tasks': len(values) - len(present)}


def input_fingerprint(files, root):
    # Neither individual task names nor the underlying relative paths are exported.
    digest = hashlib.sha256()
    for path in sorted(files, key=lambda value: str(value.relative_to(root))):
        relative = str(path.relative_to(root)).encode()
        digest.update(hashlib.sha256(relative).digest())
        digest.update(bytes.fromhex(sha(path)))
    return digest.hexdigest()


def disposition(path, derived):
    raw = read(path)
    if type(raw.get('valid')) is not bool:
        raise ValueError('Disposition lacks Boolean validity')
    if raw['valid']:
        if type(raw.get('reward')) is not int or raw['reward'] not in (0, 1):
            raise ValueError('Valid disposition lacks a native binary reward')
    elif raw.get('reward') is not None:
        raise ValueError('Invalid disposition has an imputed reward')
    if raw.get('harness') != 'browsercode' or raw.get('step_cap') != 30:
        raise ValueError('Unexpected browser harness or decision limit')
    if raw.get('task_latency_excludes_judge') is not True:
        raise ValueError('Unknown task-latency endpoint')
    if raw.get('usage_accounting', {}).get('metric_errors'):
        raise ValueError('Unresolved saved episode metric error')
    if derived and (raw.get('raw_native_result_absent') is not True or raw['valid']):
        raise ValueError('Derived record must remain an explicit invalid annotation')
    # Keep only scalar metadata; discard task content and provider output fields.
    return {'task_id': raw['task_id'], 'arm': raw['mode'], 'valid': raw['valid'],
            'reward': raw['reward'], 'derived': derived,
            'latency_seconds': None if derived else observed(raw.get('task_latency_seconds')),
            'browser_batches': None if derived else observed(raw.get('selected_code_batches')),
            'ended_epoch': observed(raw.get('ended_epoch')),
            'source_manifest_sha256': raw.get('execution_source_manifest_sha256'),
            'judge_protocol': raw.get('judge_protocol_version')}


def pool_ledger(path, *, cap_usd, cap_calls, arm, shard, task_by_episode, seen_ids, per_task):
    ledger = read(path)
    requests = ledger.get('requests')
    if not isinstance(requests, dict) or type(ledger.get('calls')) is not int:
        raise ValueError('Expected physical-ID ledger')
    if ledger['calls'] != len(requests) or ledger['calls'] > cap_calls or ledger['max_calls'] != cap_calls:
        raise ValueError('Ledger call count or frozen call cap differs')
    if money(ledger['max_usd']) != money(cap_usd):
        raise ValueError('Ledger dollar cap differs from frozen shard plan')
    totals = {'calls': 0, 'known_usage_calls': 0, 'unknown_usage_calls': 0,
              'reserved_inflight_calls': 0, 'known_fee_usd': Decimal(0),
              'unknown_charge_or_reserve_usd': Decimal(0), 'charged_or_reserved_usd': Decimal(0)}
    for ident, entry in requests.items():
        if ident in seen_ids or entry.get('physical_id') != ident:
            raise ValueError('Duplicate or inconsistent physical reservation ID')
        seen_ids.add(ident)
        status = entry.get('status')
        charge = money(entry['charged_or_reserved_usd'])
        reserve = money(entry['reserved_usd'])
        if status == 'known':
            if charge != money(entry['actual_usd']) or charge > reserve or not isinstance(entry.get('usage'), dict):
                raise ValueError('Invalid known-usage settlement')
            totals['known_usage_calls'] += 1
            totals['known_fee_usd'] += charge
        elif status in ('usage_unknown', 'reserved'):
            if charge != reserve:
                raise ValueError('Unknown usage did not retain its full reservation')
            totals['unknown_usage_calls'] += 1
            totals['reserved_inflight_calls'] += int(status == 'reserved')
            totals['unknown_charge_or_reserve_usd'] += charge
        else:
            raise ValueError('Unknown reservation status')
        totals['calls'] += 1
        totals['charged_or_reserved_usd'] += charge
        request = Path(entry['metadata']['request_path'])
        try:
            relative = request.relative_to(shard)
        except ValueError as error:
            raise ValueError('Ledger request belongs to another shard') from error
        parts = relative.parts
        if len(parts) < 6 or parts[0] != 'attempts' or parts[2] != 'episodes' or parts[4] != 'requests':
            raise ValueError('Ledger request lacks an episode identity')
        episode = parts[3]
        if episode not in task_by_episode:
            raise ValueError('Ledger request is outside the fixed task panel')
        per_task[task_by_episode[episode]] += charge
    if totals['charged_or_reserved_usd'] != money(ledger['charged_or_reserved_usd']):
        raise ValueError('Ledger total differs from its physical request charges')
    if totals['charged_or_reserved_usd'] > money(cap_usd):
        raise ValueError('Ledger exceeds the frozen shard dollar cap')
    return totals


def add_pool(total, extra):
    for key, value in extra.items():
        total[key] = total.get(key, 0) + value


def public_pool(pool):
    return {key: str(value) if isinstance(value, Decimal) else value for key, value in pool.items()}


def build(campaign):
    root = Path(campaign).resolve()
    plan_path = root / 'campaign-plan.json'
    config_path = root / 'runtime-config.json'
    source_path = root / 'source-manifest.json'
    plan, config = read(plan_path), read(config_path)
    protocol = plan['protocol']
    if (protocol['max_decisions'] != 30 or protocol['actor_max_output_tokens'] != 12288
            or protocol['selector_max_output_tokens'] != 12288 or protocol['selected_only_execution'] is not True):
        raise ValueError('Unexpected scientific decision protocol')
    arms = plan['arms']
    names = [arm['arm'] for arm in arms]
    if len(names) != len(set(names)) or any(arm['model'] != 'gpt-6-luna' or arm['output_tokens'] != 12288 for arm in arms):
        raise ValueError('Expected distinct Luna arm definitions')
    files = {plan_path, config_path, source_path}
    tasks_path = root / 'tasks.json'
    if sha(tasks_path) != plan['task_manifest_sha256']:
        raise ValueError('Task manifest differs from the frozen plan')
    files.add(tasks_path)
    # Publish basenames and hashes only, never local source-manifest paths.
    source_snapshot = {}
    for raw_path, expected_hash in read(source_path)['source_hashes'].items():
        path = Path(raw_path)
        if path.parent == root and path.suffix in ('.py', '.ts'):
            if sha(path) != expected_hash:
                raise ValueError('Current source snapshot differs from its manifest')
            source_snapshot[path.name] = expected_hash
            files.add(path)
    derived_path = root / 'analysis/derived-disposition-index-v1.json'
    derived = {}
    if derived_path.exists():
        files.add(derived_path)
        for entry in read(derived_path)['entries']:
            path = Path(entry['derived_result_path']).resolve()
            if not path.is_relative_to(root) or path in derived or sha(path) != entry['derived_result_sha256']:
                raise ValueError('Derived-invalid source binding differs')
            derived[path] = entry
    records, accounting, aggregate_rows = {}, {}, []
    current = {key: sha(path) for key, path in [('plan_sha256', plan_path), ('runtime_config_sha256', config_path), ('current_source_manifest_sha256', source_path)]}
    source_counts, judge_protocols = Counter(), set()
    shared_roster = None
    physical_ids = set()
    total_pools = {'model': {}, 'judge': {}}
    global_caps = {'model': Decimal(0), 'judge': Decimal(0)}
    for arm in arms:
        name = arm['arm']
        shards = sorted((row for row in plan['shards'] if row['arm'] == name), key=lambda row: row['shard'])
        planned = [ident for row in shards for ident in row['task_ids']]
        if not planned or len(planned) != len(set(planned)):
            raise ValueError('Duplicate or empty planned arm roster')
        expected = set(planned)
        if shared_roster is None:
            shared_roster = expected
        elif expected != shared_roster:
            raise ValueError('Arms do not use identical task IDs')
        rows, locations = {}, {}
        for path in sorted((root / 'arms' / name).glob('shard-*/attempts/attempt-*/episodes/*/result.json')):
            path = path.resolve()
            row = disposition(path, path in derived)
            ident = row['task_id']
            if row['arm'] != name or ident not in expected or ident in rows:
                raise ValueError('Unexpected or duplicate arm/task disposition')
            row_shard = next(item for item in shards if ident in item['task_ids'])
            if path.parents[4].name != f'shard-{row_shard["shard"]:02d}':
                raise ValueError('Disposition appeared in the wrong shard')
            rows[ident] = row
            locations[ident] = path
            files.add(path)
            source_counts[row['source_manifest_sha256']] += 1
            judge_protocols.add(row['judge_protocol'])
        if set(rows) != expected:
            raise ValueError('Fixed arm panel is incomplete')
        per_task = {'model': {ident: Decimal(0) for ident in expected}, 'judge': {ident: Decimal(0) for ident in expected}}
        pools = {'model': {}, 'judge': {}}
        browser_starts, browser_cap = 0, 0
        for cfg in shards:
            shard = root / 'arms' / name / f'shard-{cfg["shard"]:02d}'
            task_by_episode = {ident + '-browsercode-' + name: ident for ident in cfg['task_ids']}
            for pool in ('model', 'judge'):
                path = shard / 'budget' / pool / 'ledger.json'
                files.add(path)
                calls = cfg['model_calls_all_roles_and_counters_all_attempts' if pool == 'model' else 'judge_calls_all_attempts']
                totals = pool_ledger(path, cap_usd=cfg[pool + '_usd_all_attempts'], cap_calls=calls,
                    arm=name, shard=shard, task_by_episode=task_by_episode, seen_ids=physical_ids, per_task=per_task[pool])
                add_pool(pools[pool], totals)
                global_caps[pool] += money(cfg[pool + '_usd_all_attempts'])
            browser_path = shard / 'budget/browser/ledger.json'
            browser = read(browser_path)
            files.add(browser_path)
            if type(browser.get('calls')) is not int or not 0 <= browser['calls'] <= cfg['browser_starts_all_attempts'] or money(browser['charged_or_reserved_usd']) != browser['calls']:
                raise ValueError('Browser-start ledger exceeds frozen cap or differs from calls')
            browser_starts += browser['calls']
            browser_cap += cfg['browser_starts_all_attempts']
        if any(sum(per_task[pool].values()) != pools[pool]['charged_or_reserved_usd'] for pool in pools):
            raise ValueError('Per-task allocation did not reconcile to cumulative ledgers')
        for ident, row in rows.items():
            # Derived annotations are not native task-metric observations, even
            # when an annotation contains a synthetic zero-valued summary.
            row['model_usd'] = None if row['derived'] else float(per_task['model'][ident])
        native = [row for row in rows.values() if not row['derived']]
        valid = [row for row in rows.values() if row['valid']]
        successes = sum(row['reward'] == 1 for row in valid)
        cohort_metrics = {}
        for label, cohort in [('valid_only', valid), ('all_observed', list(rows.values()))]:
            cohort_metrics[label] = {'cohort_tasks': len(cohort),
                'model_charge_or_reserve_usd': metric(row['model_usd'] for row in cohort),
                'task_seconds': metric(row['latency_seconds'] for row in cohort),
                'selected_browser_batches': metric(row['browser_batches'] for row in cohort)}
        item = {'arm': name, 'recorded': len(rows), 'valid': len(valid), 'invalid': len(rows) - len(valid),
            'successes': successes, 'success_rate': successes / len(valid) if valid else None,
            'overall_success_rate': successes / len(expected),
            'mean_model_usd_per_task': float(pools['model']['charged_or_reserved_usd'] / len(expected)),
            'mean_model_usd_per_task_observed_tasks': len(expected),
            'mean_task_seconds': cohort_metrics['all_observed']['task_seconds']['mean'],
            'mean_task_seconds_observed_tasks': cohort_metrics['all_observed']['task_seconds']['observed_tasks'],
            'mean_browser_steps': cohort_metrics['all_observed']['selected_browser_batches']['mean'],
            'mean_browser_steps_observed_tasks': cohort_metrics['all_observed']['selected_browser_batches']['observed_tasks'],
            'unknown_model_usage_calls': pools['model']['unknown_usage_calls'],
            'derived_invalid_count': len(rows) - len(native), 'native_dispositions': len(native),
            'valid_only_metrics': cohort_metrics['valid_only'], 'all_observed_metrics': cohort_metrics['all_observed'],
            'all_attempt_accounting': {**{pool: public_pool(value) for pool, value in pools.items()},
                'browser_starts': browser_starts, 'browser_start_cap': browser_cap},
            'arm_settings': arm}
        aggregate_rows.append(item)
        records[name] = rows
        accounting[name] = pools
        for pool in pools:
            add_pool(total_pools[pool], pools[pool])
    if len(derived) != sum(row['derived_invalid_count'] for row in aggregate_rows):
        raise ValueError('Derived annotation index contains an unused entry')
    if judge_protocols != {config['judge_protocol']['version']}:
        raise ValueError('Mixed or unknown terminal judge protocol')
    for pool in total_pools:
        if global_caps[pool] != money(plan['api_caps'][pool + '_usd']) or total_pools[pool]['charged_or_reserved_usd'] > global_caps[pool]:
            raise ValueError('Global pool total or frozen cap differs')
    if sum(global_caps.values()) != money(plan['api_caps']['combined_usd']):
        raise ValueError('Combined API cap differs from the frozen pool caps')
    settings = {'benchmark': 'Online-Mind2Web', 'harness': 'BrowserCode', 'tasks_per_arm': len(shared_roster),
        'actor_model': 'gpt-6-luna', 'max_output_tokens_per_candidate_or_selector': 12288,
        'max_decisions': 30, 'viewport': [1280, 1800],
        'terminal_judge': {key: config['judge_protocol'][key] for key in
                           ('model', 'reasoning_effort', 'max_completion_tokens', 'seed', 'version')}}
    fingerprint = input_fingerprint(files, root)
    total_records = sum(row['recorded'] for row in aggregate_rows)
    return {'schema': SCHEMA, 'status': 'PROVISIONAL', 'provisional': True, 'final_audit_complete': False,
        'dispositions_complete': True,
        'qualification': 'Aggregate-only reconstruction from saved dispositions and cumulative ledgers. Full independent request/scheduler/cleanup qualification remains incomplete.',
        'settings': settings,
        'counts': {'arms': len(names), 'tasks_per_arm': len(shared_roster), 'recorded': total_records,
            'native_dispositions': sum(row['native_dispositions'] for row in aggregate_rows),
            'derived_invalid': sum(row['derived_invalid_count'] for row in aggregate_rows)},
        'metric_definitions': {
            'success_rate': 'Successes divided by native valid outcomes; invalid outcomes are excluded and counted separately.',
            'model_cost': 'Model/counter cumulative ledger charges or full retained unknown reservations, excluding external judge; conservative bounds, not invoices.',
            'mean_model_usd_per_task': 'All-attempt model ledger charge/reserve divided by every planned task; includes invalid outcomes and all attempts.',
            'mean_model_usd_per_task_observed_tasks': 'Fixed-task allocation denominator; every task is represented in the cumulative-ledger allocation, not a count of exact observed provider bills.',
            'valid_only_model_cost': 'Model ledger charges/reserves attributed by physical request-path task identity; includes any attempts for each selected valid task.',
            'all_observed_model_cost': 'Same task-attributed model charges/reserves over native dispositions; derived synthetic-zero annotation excluded from metric observations.',
            'task_seconds': 'Native actor-loop latency including online model/browser waits; setup, final judge-only capture, cleanup and external judge excluded. Missing values remain missing.',
            'browser_steps': 'Selected executable BrowserCode batches, not primitive CDP actions. Missing values remain missing.',
            'comparison': 'Five Luna effort arms on the same task roster and terminal judge. Earlier high-effort GPT cohorts use a different judge and collection window; cross-cohort comparisons are descriptive.'},
        'aggregate_rows': aggregate_rows,
        'all_attempt_accounting': {pool: public_pool(value) for pool, value in total_pools.items()},
        'provenance': {**current, 'current_source_snapshot_sha256': dict(sorted(source_snapshot.items())),
            'disposition_execution_source_manifest_sha256_counts': dict(sorted(source_counts.items())),
            'input_snapshot_sha256': fingerprint, 'input_files': len(files), 'builder_sha256': sha(Path(__file__))},
        'public_data_policy': 'Aggregate numeric data and settings only; task identifiers, raw payloads, trajectories, absolute local paths and private audit records are not exported.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.is_relative_to(Path('/gpfs/home/zixianma')):
        parser.error('Choose a writable project or temporary output directory')
    if output.exists() or output.is_symlink():
        parser.error('Output must be a new file; existing aggregates are not overwritten')
    value = build(args.campaign)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'output': str(output), 'status': value['status'], 'recorded': value['counts']['recorded'],
                      'final_audit_complete': value['final_audit_complete'], 'aggregate_sha256': sha(output)}))


if __name__ == '__main__':
    main()
