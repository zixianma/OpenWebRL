#!/usr/bin/env python3
"""Exploratory valid-only audit of the historical 124-state pilot picks."""
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
import random

ROOT = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/selection-pairwise-teacher-20261014')
OLD = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/selection-head-20261008')
PANELS = OLD / 'branch124/panels.jsonl'


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def estimate(rows, values, names, n_boot=10000, seed=0):
    by_group = defaultdict(list)
    for row in rows:
        by_group[row['task_group']].append(row['state_id'])
    groups = list(by_group.values())
    point = {name: sum(values[name][r['state_id']] for r in rows) / len(rows) for name in names}
    delta = {r['state_id']: .5 * sum(
        values[f'pairwise_teacher-seed{seed_id}'][r['state_id']]
        - values[f'soft_ce-seed{seed_id}'][r['state_id']]
        for seed_id in (0, 1)) for r in rows}
    rng = random.Random(seed)
    draws = []
    for _ in range(n_boot):
        sample = [sid for _ in groups for sid in rng.choice(groups)]
        draws.append(sum(delta[sid] for sid in sample) / len(sample))
    draws.sort()
    return {'states': len(rows), 'task_groups': len(groups),
            'success_percent': {name: 100 * point[name] for name in names},
            'pairwise_minus_soft_ce_pp': 100 * sum(delta.values()) / len(delta),
            'pairwise_minus_soft_ce_95pct_task_bootstrap_pp':
                [100 * draws[int(.025 * n_boot)], 100 * draws[int(.975 * n_boot)]]}


def main():
    rows = [json.loads(line) for line in PANELS.open()]
    assert len(rows) == 124 and len({r['state_id'] for r in rows}) == 124
    paths = {}
    picks = {}
    for objective in ('soft_ce', 'pairwise_teacher'):
        for seed_id in (0, 1):
            name = f'{objective}-seed{seed_id}'
            path = ROOT / 'results' / name / 'branch124' / 'metrics.json'
            paths[name] = path
            picks[name] = json.loads(path.read_text())['branch_head_picks']
            assert set(picks[name]) == {r['state_id'] for r in rows}
    piotr_path = OLD / 'piotr_arm.jsonl'
    paths['piotr'] = piotr_path
    picks['Piotr ARM'] = {r['state_id']: r['selected_index'] for r in map(json.loads, piotr_path.open())}
    assert set(picks['Piotr ARM']) == {r['state_id'] for r in rows}
    common = [r for r in rows if all(o['valid'] > 0 for o in r['outcomes'])]
    values = {name: {} for name in (*picks, 'Uniform', 'Actor first')}
    for row in common:
        sid = row['state_id']
        rate = [o['successes'] / o['valid'] for o in row['outcomes']]
        assert all(0 <= x <= 1 for x in rate)
        for name, selected in picks.items():
            assert 0 <= selected[sid] < 5
            values[name][sid] = rate[selected[sid]]
        values['Uniform'][sid] = sum(rate) / 5
        values['Actor first'][sid] = rate[0]
    names = list(values)
    report = {'scope': 'exploratory, original historical three-draw panels; invalid draws excluded from denominators',
              'input_sha256': {'panels': digest(PANELS), **{name: digest(path) for name, path in paths.items()}},
              'records': {'states': 124, 'all_five_with_valid_draw': len(common),
                          'excluded_states': 124-len(common),
                          'invalid_draw_slots': sum(o['trials']-o['valid'] for r in rows for o in r['outcomes'])},
              'common_valid': estimate(common, values, names),
              'no_piotr_task_overlap': estimate([r for r in common if not r['piotr_task_overlap']], values, names),
              'all_five_fully_valid': estimate([r for r in common if all(o['valid']==o['trials'] for o in r['outcomes'])], values, names)}
    out = ROOT / 'valid-only-historical124.json'
    out.write_text(json.dumps(report, indent=2, sort_keys=True)+'\n')
    print(json.dumps({'output': str(out), 'common_valid': report['common_valid']}, sort_keys=True))


if __name__ == '__main__':
    main()
