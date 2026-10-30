"""Aggregate fresh collaborator outputs without exporting prompts or trajectories.

Outcome rates and valid-task efficiency use saved episode dispositions. Fees use
all-attempt ledgers, including retained unknown reservations. Paired comparisons
require matching benchmark, harness, judge, protocol context and task identity.
Bootstrap intervals resample common-valid task pairs (seed 20261029 by default).
"""
from __future__ import annotations

import argparse
from decimal import Decimal, InvalidOperation
import hashlib
import itertools
import json
import math
import platform
from pathlib import Path
import random
import statistics


def read(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('Duplicate JSON key: ' + key)
            result[key] = value
        return result
    return json.loads(Path(path).read_text(), object_pairs_hook=pairs,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite JSON value')))


def hash_value(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def evidence(path, root):
    return {'file': str(Path(path).relative_to(root)), 'sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest()}


def number(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        return None
    return value


def amount(value):
    try:
        result = Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError('Invalid accounting amount') from exc
    if not result.is_finite() or result < 0:
        raise ValueError('Accounting amount must be finite and nonnegative')
    return result


def metric(values):
    observed = [value for value in values if value is not None]
    return {'mean': statistics.fmean(observed) if observed else None,
            'observations': len(observed), 'missing': len(values) - len(observed)}


def accounting(root):
    """Read immutable ledger files directly; never import a mutating gateway."""
    pools, sources = {}, []
    native = root / 'shard0/.host/meter/events.jsonl'
    if native.exists():
        events = [json.loads(line) for line in native.read_text().splitlines() if line.strip()]
        reserves = [event for event in events if event.get('event') == 'reserve']
        if len({event['id'] for event in reserves}) != len(reserves):
            raise ValueError('Duplicate native reservation identity')
        finishes = [event for event in events if event.get('event') == 'finish']
        if len({event['id'] for event in finishes}) != len(finishes):
            raise ValueError('Duplicate native settlement identity')
        if any(event['pool'] not in ('model', 'judge') for event in reserves):
            raise ValueError('Unknown native accounting pool')
        if any(event['id'] not in {row['id'] for row in reserves} for event in finishes):
            raise ValueError('Settlement has no reservation')
        ends = {event['id']: event for event in finishes}
        for pool in ('model', 'judge'):
            selected = [event for event in reserves if event['pool'] == pool]
            known = [event for event in selected if ends.get(event['id'], {}).get('usage_known') is True]
            unresolved = [event for event in selected if event not in known]
            fees = sum((amount(ends[event['id']]['charge_usd']) for event in known), Decimal(0))
            retained = sum((amount(ends.get(event['id'], {}).get('charge_usd', event['reserve_usd'])) for event in unresolved), Decimal(0))
            pools[pool] = {'calls': len(selected), 'known_usage_calls': len(known), 'unresolved_calls': len(unresolved),
                           'known_usage_fee_upper_usd': str(fees), 'unresolved_charge_or_reserve_usd': str(retained),
                           'charged_or_reserved_usd': str(fees + retained)}
        groups = [event for event in events if event.get('event') == 'group_open']
        if len({event['id'] for event in groups}) != len(groups):
            raise ValueError('Duplicate native admission group')
        closed = {event['group_id'] for event in events if event.get('event') == 'group_finish'}
        held_usd, held_calls = Decimal(0), 0
        for group in groups:
            if group['id'] in closed:
                continue
            used = [event for event in reserves if event.get('group_id') == group['id']]
            remaining = amount(group['held_usd']) - sum((amount(event['reserve_usd']) for event in used), Decimal(0))
            calls = group['held_model_calls'] - len(used)
            if remaining < Decimal('-1e-10') or calls < 0:
                raise ValueError('Native admission group exceeds its credits')
            held_usd += max(Decimal(0), remaining)
            held_calls += calls
        pools['model'].update(undispatched_group_hold_usd=str(held_usd), undispatched_group_call_credits=held_calls)
        sources.append(evidence(native, root))
    else:
        candidates = {'model': ('budget/model/ledger.json', 'budgets/actor-budget/ledger.json'),
                      'judge': ('budget/judge/ledger.json', 'budgets/judge/ledger.json')}
        for pool, names in candidates.items():
            paths = [root / name for name in names if (root / name).is_file()]
            if len(paths) > 1:
                raise ValueError('Ambiguous pool ledgers')
            if not paths:
                pools[pool] = None  # Absence does not establish zero spending.
                continue
            path = paths[0]
            ledger = read(path)
            if type(ledger.get('calls')) is not int or ledger['calls'] < 0:
                raise ValueError('Invalid ledger call count')
            total = amount(ledger['charged_or_reserved_usd'])
            requests = ledger.get('requests')
            detail = {'calls': ledger['calls'], 'charged_or_reserved_usd': str(total),
                      'known_usage_calls': None, 'unresolved_calls': None,
                      'known_usage_fee_upper_usd': None, 'unresolved_charge_or_reserve_usd': None}
            if isinstance(requests, dict):
                if ledger['calls'] != len(requests):
                    raise ValueError('Ledger calls differ from physical request IDs')
                charges = sum((amount(row['charged_or_reserved_usd']) for row in requests.values()), Decimal(0))
                if total != charges:
                    raise ValueError('Ledger total differs from request charges')
                known = [row for row in requests.values() if row['status'] == 'known']
                unknown = [row for row in requests.values() if row['status'] in ('reserved', 'usage_unknown')]
                if len(known) + len(unknown) != len(requests):
                    raise ValueError('Unknown request accounting status')
                detail.update(known_usage_calls=len(known), unresolved_calls=len(unknown),
                              known_usage_fee_upper_usd=str(sum((amount(row['charged_or_reserved_usd']) for row in known), Decimal(0))),
                              unresolved_charge_or_reserve_usd=str(sum((amount(row['charged_or_reserved_usd']) for row in unknown), Decimal(0))))
            pools[pool] = detail
            sources.append(evidence(path, root))
    return {'scope': 'entire input run, shared across its arms; never added to valid-task means',
            'pools': pools, 'evidence': sources,
            'fee_definition': 'Accounting upper bounds from native receipts, not provider invoices; unresolved usage retains its charge/reserve. Undispatched group admission holds are reported separately from physical calls and fees.'}


def task_map(rows):
    result = {}
    for task in rows:
        metadata = task.get('metadata', task)
        ident = str(metadata.get('task_id', task.get('task_id', task.get('id'))))
        if ident in result:
            raise ValueError('Duplicate task identity in input manifest: ' + ident)
        intent = metadata.get('intent', metadata.get('confirmed_task'))
        url = metadata.get('start_url', metadata.get('website'))
        result[ident] = {'benchmark': metadata.get('benchmark', task.get('benchmark', 'online-mind2web')),
                         'content_sha256': hash_value([intent, url]) if all(type(value) is str and value for value in (intent, url)) else None}
    return result


def load_run(name, folder):
    root = Path(folder).resolve()
    if not root.is_dir():
        raise ValueError('Run directory missing: ' + str(root))
    sources = []
    def get(path):
        path = root / path
        sources.append(evidence(path, root))
        return read(path)
    task_info, runtime_config, source_identity = {}, {}, {}
    if (root / 'reproduction-plan.json').exists():
        family, plan = 'browsercode', get('reproduction-plan.json')
        selected = get('selected-task-ids.json')
        planned = [(str(ident), plan['arm']) for ident in selected]
        requested = set(planned)
        rows = [(path, read(path)) for path in sorted((root / 'episodes').glob('*/result.json'))]
        runtime = Path(plan['runtime'])
        if (runtime / 'runtime-config.json').is_file():
            runtime_config = read(runtime / 'runtime-config.json')
        if (runtime / 'tasks.json').is_file():
            task_info = task_map(read(runtime / 'tasks.json'))
        judge = runtime_config.get('judge_protocol', {}).get('model')
        if not judge and (runtime / 'campaign-plan.json').is_file():
            judge = read(runtime / 'campaign-plan.json').get('protocol', {}).get('judge_model')
        protocol = {'max_decisions': plan['max_decisions'], 'judge': judge,
                    'judge_protocol': runtime_config.get('judge_protocol', {}),
                    'actor_prompt_sha256': hash_value([runtime_config.get('system_templates'), runtime_config.get('observation_template')]) if runtime_config else None}
        source_identity = {key: plan.get(key) for key in ('source_manifest_sha256', 'task_manifest_sha256')}
        model = plan.get('model')
        harness = plan.get('harness', 'browsercode')
    elif (root / 'portable-execution.json').exists():
        family, execution = 'coordinate', get('portable-execution.json')
        plan = get('plan.json')
        protocol = plan['protocol']
        planned = [(str(row['task_id']), arm) for row in plan['schedule'] for arm in row['modes']]
        requested = set(planned)
        rows = [(path, read(path)) for path in sorted((root / 'records').glob('*.json'))]
        task_info = task_map([json.loads(line) for line in (root / 'tasks.jsonl').read_text().splitlines() if line.strip()])
        sources.append(evidence(root / 'tasks.jsonl', root))
        model, harness = protocol.get('actor'), 'coordinate'
        source_identity = {'task_manifest_sha256': plan.get('task_file_sha256'), 'protocol_sha256': hash_value(protocol)}
    elif (root / 'plan.json').exists() and read(root / 'plan.json').get('schema') == 'cuawright-collaborator-run-v1':
        family, plan = 'native', get('plan.json')
        planned = [(str(ident), plan['arm_id']) for shard in plan['shards'] for ident in shard['task_ids']]
        requested = set(planned)
        rows = [(path, read(path)) for path in sorted((root / 'shard0/episodes').glob('*/disposition.json'))]
        task_info = task_map(get('tasks.json'))
        protocol = {key: plan[key] for key in ('max_decisions', 'summary_setting', 'judge', 'retry_policy', 'fallback_policy')}
        model, harness = plan['model'], 'native-cuawright'
        source_identity = {key: plan.get(key) for key in ('task_source_sha256', 'selected_tasks_sha256')}
        if (root / 'runtime-inventory.json').exists():
            source_identity['runtime_inventory_sha256'] = evidence(root / 'runtime-inventory.json', root)['sha256']
    else:
        raise ValueError('Unrecognized collaborator output: ' + str(root))
    if len(planned) != len(requested):
        raise ValueError('Requested schedule contains duplicate task/arm cells')
    groups, seen = {}, set()
    for path, row in rows:
        sources.append(evidence(path, root))
        arm = row.get('mode', row.get('arm', plan.get('arm')))
        key = (str(row['task_id']), arm)
        if key in seen:
            raise ValueError('Duplicate task/arm disposition; resolve attempts explicitly: ' + str(key))
        seen.add(key)
        if key not in requested:
            raise ValueError('Recorded disposition is outside the requested cohort: ' + str(key))
        if type(row.get('valid')) is not bool:
            raise ValueError('Disposition requires an explicit boolean validity')
        if row['valid'] and (type(row.get('reward')) not in (int, float) or row['reward'] not in (0, 1)):
            raise ValueError('Valid disposition lacks a binary verdict')
        if family == 'coordinate' and row.get('protocol_sha256') != hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest():
            raise ValueError('Disposition protocol hash differs from the frozen protocol')
        if family == 'coordinate' and row.get('artifact_sha256'):
            artifact = Path(row['artifact']).resolve()
            if not artifact.is_relative_to(root) or not artifact.is_file() or hashlib.sha256(artifact.read_bytes()).hexdigest() != row['artifact_sha256']:
                raise ValueError('Legacy result artifact binding changed')
        info = task_info.get(key[0], {})
        benchmark = row.get('benchmark', info.get('benchmark', plan.get('benchmark', 'online-mind2web')))
        current_harness = row.get('harness', harness)
        if current_harness != harness:
            raise ValueError('Disposition harness differs from the requested executor')
        if not isinstance(benchmark, str) or not benchmark:
            raise ValueError('Disposition benchmark identity is missing')
        group_key = (arm, benchmark, current_harness)
        group = groups.setdefault(group_key, [])
        group.append({'task_id': key[0], 'valid': row['valid'], 'success': int(row['reward']) if row['valid'] else None,
                      'task_content_sha256': info.get('content_sha256'),
                      'metrics': row_metrics(row), 'protocol_sha256': row.get('protocol_sha256'),
                      'judge_protocol_version': row.get('judge_protocol_version')})
    # Include completely unstarted arms/groups, not just groups with results.
    for ident, arm in requested:
        benchmark = task_info.get(ident, {}).get('benchmark', plan.get('benchmark') or 'online-mind2web')
        groups.setdefault((arm, benchmark, harness), [])
    summaries = []
    for (arm, benchmark, current_harness), records in sorted(groups.items()):
        ids = {ident for ident, candidate in requested if candidate == arm and task_info.get(ident, {}).get('benchmark', plan.get('benchmark') or 'online-mind2web') == benchmark}
        valid = [row for row in records if row['valid']]
        successes = sum(row['success'] for row in valid)
        context = {'benchmark': benchmark, 'harness': current_harness, 'max_decisions': protocol.get('max_decisions', protocol.get('max_steps')),
                   'judge': protocol.get('judge'), 'judge_protocol': protocol.get('judge_protocol'),
                   'viewport': protocol.get('viewport'), 'actor_prompt_sha256': protocol.get('actor_prompt_sha256', hash_value(protocol['required_browser_prompt_sha256']) if protocol.get('required_browser_prompt_sha256') else None)}
        versions = sorted({row['judge_protocol_version'] for row in records if row['judge_protocol_version']})
        if versions:
            context['judge_protocol_version'] = versions
        arm_configuration = {key: plan.get(key) for key in ('candidate_count', 'candidate_effort', 'selector_effort', 'max_output_tokens', 'selector_max_output_tokens', 'reasoning_effort')}
        if family == 'coordinate':
            parallel = arm.startswith('parallel')
            arm_configuration = {'candidate_count': protocol.get('candidates') if parallel else 1,
                'candidate_effort': protocol.get('parallel_effort' if parallel else 'serial_effort'),
                'selector_effort': protocol.get('selector_efforts', {}).get(arm, protocol.get('selector_effort')) if parallel else None,
                'max_output_tokens': protocol.get('parallel_max_output_tokens' if parallel else 'serial_max_output_tokens')}
        identity = {'family': family, 'model': model, 'protocol_sha256': hash_value(protocol), 'source': source_identity,
                    'arm_configuration': arm_configuration, 'comparison_context': context}
        summaries.append({'id': '/'.join((name, arm, benchmark, current_harness)), 'run': name, 'arm': arm,
            'requested': len(ids), 'recorded': len(records), 'unrecorded': len(ids) - len(records),
            'valid': len(valid), 'invalid': len(records) - len(valid), 'successes': successes,
            'rates': {'requested_success': successes / len(ids) if ids else None,
                      'recorded_success': successes / len(records) if records else None,
                      'valid_only_success': successes / len(valid) if valid else None},
            'valid_task_efficiency': {key: metric([row['metrics'][key] for row in valid]) for key in METRICS},
            'identity': identity, 'task_records': records, 'requested_task_ids': sorted(ids)})
    return {'name': name, 'groups': summaries, 'all_attempt_accounting': accounting(root), 'evidence': sources}


METRICS = ('task_latency_seconds', 'selected_browser_batches', 'model_cost_upper_usd', 'model_input_tokens', 'model_output_tokens', 'decisions')


def row_metrics(row):
    model = row.get('usage_accounting', {}).get('model_excluding_judge', {})
    known = model.get('unknown_usage_calls') == 0 and not row.get('usage_accounting', {}).get('metric_errors')
    return dict(task_latency_seconds=number(row.get('task_latency_seconds')) if row.get('task_latency_excludes_judge') is not False else None,
                selected_browser_batches=number(row.get('selected_code_batches', row.get('browser_steps'))),
                model_cost_upper_usd=number(row.get('model_cost_upper_usd', model.get('received_cost_usd') if known else None)),
                model_input_tokens=number(row.get('model_input_tokens', model.get('input_tokens') if known else None)),
                model_output_tokens=number(row.get('model_output_tokens', model.get('output_tokens') if known else None)),
                decisions=number(row.get('decisions')))


def paired(left, right, *, seed, draws):
    result = {'left': left['id'], 'right': right['id'], 'direction': 'left minus right'}
    a, b = left['identity']['comparison_context'], right['identity']['comparison_context']
    if a != b or any(a.get(key) is None for key in ('benchmark', 'harness', 'judge', 'max_decisions')):
        return dict(result, eligible=False, reason='Different or missing benchmark/harness/judge/protocol context')
    common_requested = set(left['requested_task_ids']) & set(right['requested_task_ids'])
    x = {row['task_id']: row for row in left['task_records']}
    y = {row['task_id']: row for row in right['task_records']}
    shared = sorted(common_requested & x.keys() & y.keys())
    for ident in shared:
        if x[ident]['task_content_sha256'] is None or x[ident]['task_content_sha256'] != y[ident]['task_content_sha256']:
            return dict(result, eligible=False, reason='Missing or different task content identity')
    valid = [ident for ident in shared if x[ident]['valid'] and y[ident]['valid']]
    differences = [x[ident]['success'] - y[ident]['success'] for ident in valid]
    n = len(valid)
    discordant_left = sum(x[ident]['success'] == 1 and y[ident]['success'] == 0 for ident in valid)
    discordant_right = sum(x[ident]['success'] == 0 and y[ident]['success'] == 1 for ident in valid)
    discordant = discordant_left + discordant_right
    p = min(1., 2 * sum(math.comb(discordant, k) for k in range(min(discordant_left, discordant_right) + 1)) / 2 ** discordant) if discordant else 1.
    rng = random.Random(seed)
    samples = sorted(statistics.fmean(rng.choices(differences, k=n)) for _ in range(draws)) if n else []
    interval = [samples[math.floor((draws - 1) * .025)], samples[math.ceil((draws - 1) * .975)]] if samples else None
    metrics = {}
    for key in METRICS:
        usable = [ident for ident in valid if x[ident]['metrics'][key] is not None and y[ident]['metrics'][key] is not None]
        metrics[key] = {'left': metric([x[ident]['metrics'][key] for ident in usable]), 'right': metric([y[ident]['metrics'][key] for ident in usable]),
                        'paired_observations': len(usable), 'missing_pairs': n - len(usable)}
    return dict(result, eligible=True, common_requested=len(common_requested), common_recorded=len(shared), common_valid=n,
                excluded_unrecorded=len(common_requested) - len(shared), excluded_invalid=len(shared) - n,
                common_valid_task_ids=valid, left_successes=sum(x[ident]['success'] for ident in valid), right_successes=sum(y[ident]['success'] for ident in valid),
                success_rate_difference=statistics.fmean(differences) if n else None,
                paired_bootstrap_95_percentile_interval=interval,
                mcnemar={'left_only_successes': discordant_left, 'right_only_successes': discordant_right, 'exact_two_sided_p': p if n else None},
                common_valid_efficiency=metrics)


def build_report(runs, *, seed=20261029, draws=10000):
    if type(draws) is not int or not 1 <= draws <= 100000:
        raise ValueError('Bootstrap draws must be between 1 and 100000')
    names = [name for name, _ in runs]
    if not names or len(set(names)) != len(names) or any(not name for name in names):
        raise ValueError('Run names must be nonempty and unique')
    if len({str(Path(folder).resolve()) for _, folder in runs}) != len(runs):
        raise ValueError('The same physical run directory cannot be counted twice')
    loaded = [load_run(name, folder) for name, folder in runs]
    groups = [group for run in loaded for group in run['groups']]
    comparisons = [paired(left, right, seed=seed, draws=draws) for left, right in itertools.combinations(groups, 2)]
    return {'schema': 'api-collaborator-analysis-v1', 'analysis_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'statistics': {'python_version': platform.python_version(), 'seed': seed, 'bootstrap_draws': draws,
            'interval': 'Paired task bootstrap, percentile95%; binary outcomes on common-valid tasks only; no multiple-comparison correction.',
            'mcnemar': 'Exact two-sided conditional binomial test on discordant common-valid pairs; descriptive, no multiple-comparison correction.',
            'denominators': 'requested includes unrecorded; recorded includes invalid; valid-only excludes invalid. Efficiency means use available valid observations, missing values remain missing. Common-valid comparisons condition on both outcomes being valid; inspect excluded counts alongside each estimate.'},
            'runs': loaded, 'comparisons': comparisons}


def markdown(report):
    lines = ['| Run / arm / benchmark / harness | Requested | Recorded | Valid | Invalid | Successes | Recorded success (%) | Valid-only success (%) |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for run in report['runs']:
        for group in run['groups']:
            percentages = ['—' if group['rates'][key] is None else f"{100 * group['rates'][key]:.2f}" for key in ('recorded_success', 'valid_only_success')]
            lines.append('| ' + ' | '.join([group['id'].replace('|', '\\|'), *map(str, [group[key] for key in ('requested', 'recorded', 'valid', 'invalid', 'successes')]), *percentages]) + ' |')
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='append', required=True, help='NAME=OUTPUT_DIRECTORY; repeat for paired comparisons')
    parser.add_argument('--output', type=Path, required=True, help='New JSON report; existing reports are never overwritten')
    parser.add_argument('--seed', type=int, default=20261029)
    parser.add_argument('--bootstrap-draws', type=int, default=10000)
    parser.add_argument('--markdown', action='store_true', help='Print a compact table; JSON remains the complete analysis')
    args = parser.parse_args()
    try:
        output = args.output.resolve()
        if output.is_relative_to(Path('/gpfs/home/zixianma')):
            raise ValueError('Use a project or temporary report destination')
        runs = []
        for item in args.run:
            if '=' not in item:
                raise ValueError('--run requires NAME=OUTPUT_DIRECTORY')
            name, folder = item.split('=', 1)
            runs.append((name, folder))
        result = build_report(runs, seed=args.seed, draws=args.bootstrap_draws)
        with output.open('x') as stream:
            json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write('\n')
        if args.markdown:
            print(markdown(result))
        else:
            print('Saved ' + str(output))
    except (ValueError, OSError, KeyError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
