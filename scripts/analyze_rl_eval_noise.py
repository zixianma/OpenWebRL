#!/usr/bin/env python3
"""Checkpoint noise and paired comparisons for full-300 local GPT-4.1/T0 RL evaluations.

Reads saved per-task outcome records (rollouts/*.json) of each lineage's checkpoint
evaluations, cross-checks counts against the published audits, and reports:
  * per-checkpoint counts, and the mean of the last three evaluated checkpoints;
  * the adjacent-checkpoint noise floor (task flip rates, success-rate deltas);
  * paired task-level bootstrap CIs between lineages on the last-three mean.
Overall rate counts invalid episodes as failures; valid-only uses valid episodes.
"""
import argparse
import json
from pathlib import Path

import numpy as np

EVALS = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/evaluations')
AUDITS = Path(__file__).resolve().parents[1]/'openwebrl/docs/arm_results/rl_integration'
MIXED = {'mixed-bonus': {10: 'iter10-335699', 20: 'iter20-335699', 30: 'iter30-335699', 40: 'iter40-335700',
                         50: 'iter50-335700', 60: 'iter60-337983', 70: 'iter70-338904', 80: 'iter80-338904',
                         90: 'iter90-340425'},
         'mixed-reweight': {10: 'iter10-335697', 20: 'iter20-335697', 30: 'iter30-335697', 40: 'iter40-335698',
                            50: 'iter50-335698', 60: 'iter60-335698', 70: 'iter70-338905', 80: 'iter80-338905',
                            90: 'iter90-340423'}}
LINEAGES = {
    'outcome56': {i: EVALS/f'outcome-only-56groups-20261006-iter{i}' for i in range(10, 100, 10)},
    'uniform8': {i: EVALS/f'outcome-uniform8-original2102-20261006-iter{i}' for i in range(10, 70, 10)},
    **{k: {i: EVALS/f'arm-{k}-{s}' for i, s in v.items()} for k, v in MIXED.items()},
}


def load(directory):
    """Return {task_id: (success, valid)} from the saved per-task outcome records."""
    rows = {}
    for path in (directory/'rollouts').glob('*.json'):
        r = json.loads(path.read_text())
        m = r['metrics']
        if r['task_id'] in rows:
            raise ValueError(f'Duplicate task {r["task_id"]} in {directory}')
        rows[r['task_id']] = (int(m['successes'] > 0), int(m['valid_trajectories'] > 0))
    if len(rows) != 300:
        raise ValueError(f'{directory}: {len(rows)} tasks, expected 300')
    return rows


def audit_counts(lineage, iteration):
    path = AUDITS/f'{lineage}-iteration{iteration}-audit.json'
    if not path.is_file():
        return None
    a = json.loads(path.read_text())
    f = a.get('full300') or a.get('evaluation') or a
    return f.get('successes'), f.get('valid')


def bootstrap(diff, draws, rng):
    idx = rng.integers(0, len(diff), size=(draws, len(diff)))
    means = diff[idx].mean(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--draws', type=int, default=20000)
    args = parser.parse_args()
    rng = np.random.default_rng(0)
    data, checkpoints, mismatches = {}, [], []
    for lineage, evals in LINEAGES.items():
        data[lineage] = {}
        for it, d in sorted(evals.items()):
            rows = load(d)
            s = sum(v[0] for v in rows.values()); n = sum(v[1] for v in rows.values())
            expected = audit_counts(lineage, it)
            if expected and expected != (s, n):
                mismatches.append(dict(lineage=lineage, iteration=it, recount=[s, n], audit=list(expected)))
            data[lineage][it] = rows
            checkpoints.append(dict(lineage=lineage, iteration=it, successes=s, valid=n,
                                    overall=s/300, valid_only=s/n, audit_match=expected == (s, n) if expected else None))
    tasks = sorted(set.intersection(*(set(r) for v in data.values() for r in v.values())))
    if len(tasks) != 300:
        raise ValueError(f'Task sets differ across cohorts: {len(tasks)} common')

    # Adjacent-checkpoint noise floor within each lineage.
    noise = []
    for lineage, evals in data.items():
        its = sorted(evals)
        for a, b in zip(its, its[1:]):
            x = np.array([evals[a][t][0] for t in tasks]); y = np.array([evals[b][t][0] for t in tasks])
            noise.append(dict(lineage=lineage, pair=[a, b], flips=int((x != y).sum()),
                              gained=int(((x == 0) & (y == 1)).sum()), lost=int(((x == 1) & (y == 0)).sum()),
                              delta_pp=100*(y.mean()-x.mean())))
    deltas = np.array([abs(n['delta_pp']) for n in noise])

    # Last-three-checkpoint per-task success rate; paired bootstrap over tasks.
    last3 = {}
    for lineage, evals in data.items():
        its = sorted(evals)[-3:]
        succ = np.array([[evals[i][t][0] for t in tasks] for i in its])
        val = np.array([[evals[i][t][1] for t in tasks] for i in its])
        last3[lineage] = dict(iterations=its, per_task=succ.mean(0),
                              overall=float(succ.mean()), valid_only=float(succ.sum()/val.sum()),
                              per_checkpoint_overall=[float(x) for x in succ.mean(1)])
    # Matched-iteration comparison at 40/50/60, which every lineage has evaluated.
    matched = {}
    for lineage, evals in data.items():
        succ = np.array([[evals[i][t][0] for t in tasks] for i in (40, 50, 60)])
        matched[lineage] = succ.mean(0)
    pairs = []
    names = list(data)
    for i, a in enumerate(names):
        for b in names[i+1:]:
            for label, source in (('last3', {k: v['per_task'] for k, v in last3.items()}), ('iter40-60', matched)):
                diff = source[a]-source[b]
                lo, hi = bootstrap(diff, args.draws, rng)
                pairs.append(dict(comparison=f'{a} - {b}', window=label, delta_pp=100*float(diff.mean()),
                                  ci95_pp=[100*lo, 100*hi]))
    result = dict(
        schema_version=1, protocol='local browser, GPT-4.1 action_history judge, T0, full 300',
        definitions=dict(overall='successes/300 (invalid counted as failure)', valid_only='successes/valid',
                         ci='paired task-level percentile bootstrap over 300 tasks; checkpoints fixed',
                         last3='mean of each lineage\'s last three evaluated checkpoints'),
        audit_mismatches=mismatches, checkpoints=checkpoints,
        noise_floor=dict(adjacent_pairs=noise, mean_abs_delta_pp=float(deltas.mean()),
                         max_abs_delta_pp=float(deltas.max()),
                         mean_flips=float(np.mean([n['flips'] for n in noise]))),
        last3={k: {kk: vv for kk, vv in v.items() if kk != 'per_task'} for k, v in last3.items()},
        paired=pairs)
    args.output.write_text(json.dumps(result, indent=1))
    print(json.dumps(dict(mismatches=mismatches, noise=result['noise_floor'] | dict(adjacent_pairs=len(noise)),
                          last3=result['last3'], paired=pairs), indent=1))


if __name__ == '__main__':
    main()
