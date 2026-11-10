#!/usr/bin/env python3
"""Iteration-90 repeat evaluations (job 385593): per-repeat counts and paired comparisons.

Three contemporaneous full-300 repeats per checkpoint (local browser, GPT-4.1 action_history,
T0, 30 turns) on one harness. Per-task success is averaged over the repeats; paired percentile
bootstrap over tasks (20,000 draws) and a paired sign-flip permutation test (20,000 draws),
Holm-corrected over the three comparisons. Invalid episodes count as failures overall.
"""
import argparse
import glob
import json
from pathlib import Path
import random

EVALS = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/evaluations')
RUNS = {'outcome48': 'Outcome-only 48 groups', 'outcome56': 'Outcome-only 56 groups', 'gate-b': 'Gate B (ARM)'}
ORIGINAL = {'outcome48': None, 'outcome56': 'outcome-only-56groups-20261006-iter90', 'gate-b': 'arm-gate-b-iter90-335729'}
PAIRS = [('gate-b', 'outcome56'), ('gate-b', 'outcome48'), ('outcome56', 'outcome48')]
DRAWS = 20000


def load(d):
    out = {}
    for f in glob.glob(str(EVALS/d/'rollouts'/'*.json')):
        r = json.load(open(f))
        out[r['task_id']] = (r['metrics']['successes'] > 0, r['metrics']['valid_trajectories'] > 0)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    reps = {k: [load(f'repeat-iter90-{k}-r{i}-385593-a1') for i in (1, 2, 3)] for k in RUNS}
    tasks = sorted(set.intersection(*[set(x) for v in reps.values() for x in v]))
    assert len(tasks) == 300, len(tasks)
    summary = dict(protocol='local browser, GPT-4.1 action_history judge, actor T0, 30 turns, full 300; job 385593',
                   tasks=len(tasks), runs={}, comparisons=[])
    mean = {}
    for k, v in reps.items():
        mean[k] = {t: sum(x[t][0] for x in v)/3 for t in tasks}
        succ = [sum(x[t][0] for t in tasks) for x in v]
        valid = [sum(x[t][1] for t in tasks) for x in v]
        orig = None
        if ORIGINAL[k]:
            m = json.load(open(EVALS/ORIGINAL[k]/'metrics.json'))
            pre = 'eval/online-mind2web-monitor/task/'
            orig = dict(cohort=ORIGINAL[k], successes=m[pre+'successes'], valid=m[pre+'valid_trajectories'])
        summary['runs'][k] = dict(label=RUNS[k], successes=succ, valid=valid,
                                  overall_mean_pct=100*sum(succ)/(3*len(tasks)),
                                  valid_only_mean_pct=100*sum(succ)/sum(valid), original_evaluation=orig)
    rng = random.Random(0)
    raw = []
    for x, y in PAIRS:
        d = [mean[x][t]-mean[y][t] for t in tasks]
        n = len(d)
        obs = 100*sum(d)/n
        boots = sorted(100*sum(d[rng.randrange(n)] for _ in range(n))/n for _ in range(DRAWS))
        perm = sum(abs(sum(v if rng.random() < .5 else -v for v in d)) >= abs(sum(d))-1e-12 for _ in range(DRAWS))
        raw.append(dict(comparison=f'{x} - {y}', delta_pp=obs, ci95_pp=[boots[int(.025*DRAWS)], boots[int(.975*DRAWS)-1]],
                        p_perm=(perm+1)/(DRAWS+1)))
    order = sorted(range(len(raw)), key=lambda i: raw[i]['p_perm'])
    running = 0
    for rank, i in enumerate(order):
        running = max(running, min(1, (len(raw)-rank)*raw[i]['p_perm']))
        raw[i]['p_holm'] = running
    summary['comparisons'] = raw
    a.out.write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
