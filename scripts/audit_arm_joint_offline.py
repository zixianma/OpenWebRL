#!/usr/bin/env python3
"""Audit joint SFT/DPO objective transfer and endpoint availability on CPU."""
from collections import Counter
import json
import math
from pathlib import Path
import statistics


RUNTIME = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-reproduction')
RUNS = RUNTIME / 'runs'
DATA = RUNTIME / 'joint-data-v2-20260910'
OUT = Path(__file__).resolve().parents[1] / 'openwebrl/docs/arm_results/joint_data_v2'
UPDATES = [0, 44, 87, 131, 174]


def read_jsonl(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def mean(values):
    return sum(values) / len(values) if values else None


def median(values):
    return statistics.median(values) if values else None


def pearson(xs, ys):
    if len(xs) < 2:
        return None
    mx, my = mean(xs), mean(ys)
    xx = sum((x - mx) ** 2 for x in xs)
    yy = sum((y - my) ** 2 for y in ys)
    if not xx or not yy:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(xx * yy)


def load_reference(root):
    result = {}
    for rank in range(2):
        for item in read_jsonl(root / f'reference-rank-{rank}.jsonl'):
            result[item['id']] = item['scores']
    return result


def offline_audit():
    panels = {}
    for source in ['c2', 'piotr']:
        for row in read_jsonl(DATA / f'validation-panel-{source}.jsonl'):
            panels[row['id']] = row
    report = {
        'scope': 'Fixed held-out panels; margins are sequence-summed log probabilities.',
        'panel_examples': len(panels),
        'runs': {},
    }
    for objective in ['sft', 'dpo']:
        root = RUNS / f'joint-v2-{objective}-2gpu-r2'
        reference = load_reference(root)
        run = {}
        for update in UPDATES:
            payload = json.loads((root / f'student/validation-{update:06d}.json').read_text())
            examples = []
            for item in payload['examples']:
                row = panels[item['id']]
                ref = reference[item['id']]
                policy_full = item['chosen']['full'] - item['rejected']['full']
                policy_action = item['chosen']['action'] - item['rejected']['action']
                ref_full = ref['chosen']['full'] - ref['rejected']['full']
                ref_action = ref['chosen']['action'] - ref['rejected']['action']
                rel_full = policy_full - ref_full
                rel_action = policy_action - ref_action
                examples.append({
                    'source': item['source'],
                    'pair_type': row['pair_type'],
                    'full': rel_full,
                    'action': rel_action,
                    'residual': rel_full - rel_action,
                    'response_tokens': (row['chosen']['tokens'] + row['rejected']['tokens']) / 2,
                    'action_tokens': (row['chosen']['action_tokens'] + row['rejected']['action_tokens']) / 2,
                })
            summaries = {}
            for source in ['C2', 'Piotr', 'all']:
                subset = examples if source == 'all' else [x for x in examples if x['source'] == source]
                full = [x['full'] for x in subset]
                action = [x['action'] for x in subset]
                residual = [x['residual'] for x in subset]
                summaries[source] = {
                    'n': len(subset),
                    'mean_relative_full_margin': mean(full),
                    'mean_relative_action_margin': mean(action),
                    'mean_relative_nonaction_margin': mean(residual),
                    'median_relative_full_margin': median(full),
                    'median_relative_action_margin': median(action),
                    'positive_relative_full_fraction': mean([x > 0 for x in full]),
                    'positive_relative_action_fraction': mean([x > 0 for x in action]),
                    'correlation_action_with_full_margin': pearson(action, full),
                    'correlation_full_margin_with_response_tokens': pearson(full, [x['response_tokens'] for x in subset]),
                    'correlation_action_margin_with_action_tokens': pearson(action, [x['action_tokens'] for x in subset]),
                }
            run[str(update)] = summaries
        endpoint = json.loads((root / 'student/validation-000174.json').read_text())['examples']
        slices = {}
        for pair_type in sorted({panels[x['id']]['pair_type'] for x in endpoint}):
            subset = [x for x in endpoint if panels[x['id']]['pair_type'] == pair_type]
            slices[pair_type] = {
                'n': len(subset),
                'full_ranking': mean([x['chosen']['full'] > x['rejected']['full'] for x in subset]),
                'action_ranking': mean([x['chosen']['action'] > x['rejected']['action'] for x in subset]),
            }
        report['runs'][objective] = {'updates': run, 'endpoint_by_pair_type': slices}
    write_json(OUT / 'offline-objective-audit.json', report)
    return report


def unavailable(record):
    return not record['valid']


def reason(record):
    if record.get('error_type'):
        return f"judge:{record['error_type']}"
    value = str(record.get('terminate_reason'))
    if '400 Bad Request' in value:
        return 'generation:context_overflow_400'
    if 'timed out' in value:
        return 'generation:timeout'
    return value


def availability_audit():
    records = {}
    for objective in ['sft', 'dpo']:
        root = RUNS / f'joint-v2-{objective}-2gpu-r2/evaluation/all300-records.json'
        records[objective] = json.loads(root.read_text())
    invalid = {name: {task for task, row in rows.items() if unavailable(row)}
               for name, rows in records.items()}
    report = {
        'counts': {},
        'overlap': {
            'both_unavailable': len(invalid['sft'] & invalid['dpo']),
            'sft_only_unavailable': len(invalid['sft'] - invalid['dpo']),
            'dpo_only_unavailable': len(invalid['dpo'] - invalid['sft']),
            'both_valid': len(set(records['sft']) - invalid['sft'] - invalid['dpo']),
        },
    }
    for objective, rows in records.items():
        bad = [row for row in rows.values() if unavailable(row)]
        overflow = [row for row in bad if reason(row) == 'generation:context_overflow_400']
        turns = [row.get('metadata', {}).get('num_turns_in_trajectory') for row in overflow]
        report['counts'][objective] = {
            'valid': sum(row['valid'] for row in rows.values()),
            'unavailable': len(bad),
            'unavailable_reasons': dict(Counter(reason(row) for row in bad)),
            'context_overflow_turns': turns,
            'context_overflow_median_turn': median(turns),
        }
    write_json(OUT / 'availability-audit.json', report)
    return report


if __name__ == '__main__':
    offline = offline_audit()
    availability = availability_audit()
    print(json.dumps({'offline': offline['runs']['dpo']['updates']['174']['all'],
                      'availability': availability}, indent=2))
