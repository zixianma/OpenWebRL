#!/usr/bin/env python3
"""Audit paired fixed-action outcomes after original/shared reasoning replay."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(Path(path).read_text())


def main(root, output):
    plan = read(root / 'plan.json')
    picks = {row['state_id']: row for row in read(root / 'model/frozen-picks.json')}
    panels = {row['state_id']: row for row in
              (json.loads(line) for line in (root / 'data/panels.jsonl').open())}
    import sys
    sys.path.insert(0, plan['source'])
    from openwebrl.arm_branch_state import BranchState
    state = BranchState(root, allow_unapproved_for_tests=True)
    conditions = defaultdict(dict)
    coverage = dict(planned=640, recorded=0, valid=0, invalid=0, complete_panels=0,
                    rejected_panels=0, unfinished_panels=0)
    for info in plan['states']:
        panel_id = info['id']
        if (root / 'states' / panel_id / 'reject.json').exists():
            coverage['rejected_panels'] += 1
        elif (root / 'states' / panel_id / 'complete.json').exists():
            coverage['complete_panels'] += 1
        else:
            coverage['unfinished_panels'] += 1
        outcomes = defaultdict(lambda: [0, 0])
        per_candidate = defaultdict(list)
        for item in plan['schedule']:
            if item['state'] != panel_id:
                continue
            row, attempts = state.item(item['item_id'])
            if row['status'] != 'finished' or not attempts:
                continue
            last = attempts[-1]
            if last['status'] != 'finished':
                continue
            value = json.loads(last['result_json'])
            if not value['released']:
                continue
            coverage['recorded'] += 1
            coverage['valid' if value['valid'] else 'invalid'] += 1
            name = panels[info['source_state']]['actions'][item['candidate']]['name']
            per_candidate[item['candidate']].append(value)
            if value['valid']:
                outcomes[name][0] += int(value['reward'] > 0)
                outcomes[name][1] += 1
        conditions[info['source_state']][info['condition']] = dict(
            complete=(root / 'states' / panel_id / 'complete.json').exists(),
            names={name: dict(success=wins, valid=valid,
                              rate=wins / valid if valid else None)
                   for name, (wins, valid) in outcomes.items()},
            recorded=sum(map(len, per_candidate.values())),
            candidates={str(i): dict(valid=sum(bool(x['valid']) for x in rows),
                                     invalid=sum(not x['valid'] for x in rows))
                        for i, rows in per_candidate.items()})
    detail, scoreable = [], []
    for sid, arms in sorted(conditions.items()):
        pick = picks[sid]
        row = dict(state_id=sid, task_group=pick['task_group'],
                   actor_name=pick['actor_name'], value_name=pick['value_name'], arms=arms)
        detail.append(row)
        if set(arms) != {'original', 'shared'} or not all(x['complete'] for x in arms.values()):
            continue
        names = {a['name'] for a in panels[sid]['actions']}
        if any(set(arm['names']) != names or any(arm['names'][name]['valid'] == 0 for name in names)
               for arm in arms.values()):
            continue
        for condition, arm in arms.items():
            scoreable.append(dict(state_id=sid, task_group=pick['task_group'],
                                  condition=condition,
                                  actor=arm['names'][pick['actor_name']]['rate'],
                                  value=arm['names'][pick['value_name']]['rate'],
                                  changed=pick['actor_name'] != pick['value_name']))
    def summarize(condition):
        rows = [row for row in scoreable if row['condition'] == condition]
        if not rows:
            return dict(states=0, task_groups=0)
        delta = np.array([row['value'] - row['actor'] for row in rows])
        groups = sorted({row['task_group'] for row in rows})
        rng = np.random.default_rng(20261018)
        by_group = {g: np.array([r['value'] - r['actor'] for r in rows
                                 if r['task_group'] == g]) for g in groups}
        boot = [np.concatenate([by_group[g] for g in rng.choice(groups, len(groups))]).mean()
                for _ in range(10000)]
        return dict(states=len(rows), task_groups=len(groups),
                    actor_percent=100 * np.mean([row['actor'] for row in rows]),
                    value_percent=100 * np.mean([row['value'] for row in rows]),
                    value_minus_actor_pp=100 * delta.mean(),
                    task_bootstrap_95_pp=[float(x) for x in 100 * np.quantile(boot, [.025, .975])],
                    changed_states=sum(row['changed'] for row in rows))
    paired = []
    for sid in sorted({row['state_id'] for row in scoreable}):
        arms = {row['condition']: row for row in scoreable if row['state_id'] == sid}
        if set(arms) == {'original', 'shared'}:
            paired.append(dict(task_group=arms['original']['task_group'],
                               actor=arms['shared']['actor'] - arms['original']['actor'],
                               value=arms['shared']['value'] - arms['original']['value']))
    paired_shift = dict(states=len(paired), task_groups=len({x['task_group'] for x in paired}),
                        actor_shared_minus_original_pp=100 * np.mean([x['actor'] for x in paired])
                        if paired else None,
                        value_shared_minus_original_pp=100 * np.mean([x['value'] for x in paired])
                        if paired else None)
    aggregate = dict(status='complete' if coverage['complete_panels'] == 16 else 'partial',
                     design='Eight archived Piotr-matched states, same five executable actions, '
                            'independent original/shared reasoning continuation panels, '
                            'eight draws per action and arm; outcomes valid-only',
                     coverage=coverage, comparable_states=len(scoreable) // 2,
                     original=summarize('original'), shared=summarize('shared'),
                     paired_prefix_shift=paired_shift,
                     caution='Task bootstrap has at most five groups; exploratory mechanism check only.')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(aggregate, indent=2) + '\n')
    (root / 'results' / 'analysis-private.json').write_text(json.dumps(detail, indent=2) + '\n')
    print(json.dumps(aggregate, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    main(args.root, args.output)
