#!/usr/bin/env python3
"""RL behavior across training: held-out evaluations and training rollouts (CPU only).

--eval: summarizes compact evaluation episodes written by analyze_rl_eval_failures.py
  (one JSONL per checkpoint) into per-checkpoint behavior and failure-category rates, plus
  task-level transitions between consecutive checkpoints of a lineage.
--train: reads the outcome-only 48-group run's completed training-rollout archives (every
  completed group, accepted or not; iterations 24-100) and summarizes per-iteration
  behavior. Archives keep actions but not tool results, so tool-error rates are omitted.

Episode features: steps, action-type shares, done/turn-limit, repeated non-wait action
(>= 3 identical), impossibility claim in the final answer, answer length, first action.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import gzip
import json
import os
from pathlib import Path
import re

from analyze_rl_eval_failures import INFEASIBLE, TOOL_CALL, classify

ARCHIVE_RUNS = sorted(Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/runs').glob('openwebrl-4b-reference-*/completed_rollout_archive'))


def features(calls, success, valid, steps):
    names = [c.get('name') for c in calls]
    keys = Counter(json.dumps(c, sort_keys=True) for c in calls if c.get('name') not in ('done', 'wait'))
    done = next((c for c in reversed(calls) if c.get('name') == 'done'), None)
    answer = json.dumps(done.get('arguments', '')) if done else ''
    return dict(success=success, valid=valid, steps=steps, actions=Counter(names), called_done=done is not None,
                repeat_loop=bool(keys) and keys.most_common(1)[0][1] >= 3,
                claims_infeasible=bool(done) and bool(INFEASIBLE.search(answer)),
                answer_chars=len(answer), first_action=names[0] if names else None)


def aggregate(rows):
    valid = [r for r in rows if r['valid']]
    actions = Counter()
    for r in valid:
        actions.update(r['actions'])
    total_actions = sum(actions.values()) or 1
    mean = lambda xs: sum(xs)/len(xs) if xs else None  # noqa: E731
    fails = [r for r in valid if not r['success']]
    return dict(
        episodes=len(rows), valid=len(valid), successes=sum(r['success'] for r in rows),
        success_rate_all=mean([r['success'] for r in rows]), success_rate_valid=mean([r['success'] for r in valid]),
        invalid_rate=1-len(valid)/len(rows) if rows else None,
        mean_steps_valid=mean([r['steps'] for r in valid]), mean_steps_success=mean([r['steps'] for r in valid if r['success']]),
        mean_steps_failure=mean([r['steps'] for r in fails]),
        done_rate_valid=mean([r['called_done'] for r in valid]),
        repeat_loop_rate_valid=mean([r['repeat_loop'] for r in valid]),
        claims_infeasible_rate_valid=mean([r['claims_infeasible'] for r in valid]),
        mean_answer_chars_done=mean([r['answer_chars'] for r in valid if r['called_done']]),
        action_share={k: v/total_actions for k, v in actions.most_common()},
        first_action_share={k: v/len(valid) for k, v in Counter(r['first_action'] for r in valid).most_common(8)} if valid else {})


def eval_rows(path, max_steps):
    rows = []
    for line in Path(path).read_text().splitlines():
        r = json.loads(line)
        calls = [s['call'] for s in r.get('steps') or [] if s.get('call')]
        f = features(calls, r['success'], r['valid'], r.get('total_steps') or len(r.get('steps') or []))
        f.update(task_id=r['task_id'], category=classify(r, max_steps))
        rows.append(f)
    return rows


def run_eval(args):
    out = {}
    for spec in args.cohorts:  # lineage:iteration=jsonl
        key, path = spec.split('=', 1)
        lineage, iteration = key.split(':')
        rows = eval_rows(path, args.max_steps)
        summary = aggregate(rows)
        summary['categories'] = dict(Counter(r['category'] for r in rows).most_common())
        out.setdefault(lineage, {})[int(iteration)] = dict(summary=summary, outcomes={r['task_id']: (r['success'], r['valid']) for r in rows})
    result = {}
    for lineage, its in out.items():
        order = sorted(its)
        transitions = []
        for a, b in zip(order, order[1:]):
            x, y = its[a]['outcomes'], its[b]['outcomes']
            common = set(x) & set(y)
            transitions.append(dict(pair=[a, b], gained=sum(not x[t][0] and y[t][0] for t in common),
                                    lost=sum(x[t][0] and not y[t][0] for t in common),
                                    valid_to_invalid=sum(x[t][1] and not y[t][1] for t in common),
                                    invalid_to_valid=sum(not x[t][1] and y[t][1] for t in common)))
        ever = {t for i in order for t, v in its[i]['outcomes'].items() if v[0]}
        always = set.intersection(*({t for t, v in its[i]['outcomes'].items() if v[0]} for i in order)) if order else set()
        result[lineage] = dict(checkpoints={i: its[i]['summary'] for i in order}, transitions=transitions,
                               tasks_ever_solved=len(ever), tasks_solved_at_every_checkpoint=len(always))
    return result


def archive_iterations():
    found = {}
    for root in ARCHIVE_RUNS:
        for d in root.glob('iteration_*'):
            found.setdefault(int(d.name.split('_')[1]), d)  # first archive per reward iteration
    return dict(sorted(found.items()))


def train_iteration(directory):
    groups, rows = Counter(), []
    for f in sorted(Path(directory).glob('group_*.json.gz')):
        g = json.load(gzip.open(f))
        groups[g.get('classification')] += 1
        for t in g['trajectories']:
            calls = []
            for turn in t['turns']:
                m = TOOL_CALL.search(turn.get('response') or '')
                if m:
                    try:
                        calls.append(json.loads(m.group(1)))
                    except json.JSONDecodeError:
                        calls.append({'name': 'unparsable'})
            meta = (t['turns'][-1].get('metadata') or {}) if t['turns'] else {}
            r = features(calls, bool(t.get('success')), bool(t.get('valid')), meta.get('total_steps') or len(t['turns']))
            r['terminate_reason'] = re.split(r'[:\[]', str(meta.get('terminate_reason')))[0][:40]
            rows.append(r)
    s = aggregate(rows)
    s['group_classification'] = dict(groups)
    s['terminate_reasons'] = dict(Counter(r['terminate_reason'] for r in rows).most_common(8))
    return s


def run_train(workers):
    its = archive_iterations()
    with ProcessPoolExecutor(workers) as pool:
        summaries = dict(zip(its, pool.map(train_iteration, its.values())))
    return dict(archives={i: str(d) for i, d in its.items()}, iterations=summaries)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--eval', action='store_true')
    p.add_argument('--train', action='store_true')
    p.add_argument('--cohorts', nargs='*', default=[], help='lineage:iteration=/path/to/compact.jsonl')
    p.add_argument('--max-steps', type=int, default=30)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = {}
    if a.eval:
        result['evaluations'] = run_eval(a)
    if a.train:
        result['training_outcome48'] = run_train(int(os.getenv('SLURM_CPUS_PER_TASK', '1')))
    a.output.write_text(json.dumps(result, indent=1, default=str))
    print('wrote', a.output)


if __name__ == '__main__':
    main()
