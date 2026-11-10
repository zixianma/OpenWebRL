#!/usr/bin/env python3
"""Repeated-action behavior of RL policies on held-out Online-Mind2Web (aggregate only).

Reads analyze_rl_eval_failures.py JSONL (private, runtime) and writes
rl_results/rl_repetition.{png,pdf}, the standalone iteration-90 panel
rl_results/rl_repetition_iter90.{png,pdf}, plus the aggregate counts it plots to the docs evidence
folder. Needs matplotlib (venv-llamafactory).

Two measures per full-300 evaluation:
  repeat-loop failures: episodes in the 'repeat_loop' failure category (first-match rules of
    analyze_rl_eval_failures.classify), counted out of 300 tasks;
  repetition rate: share of valid episodes, successes included, that issue an identical
    non-wait, non-done action >= 3 times. This does not depend on the judge.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import statistics

import matplotlib
import numpy as np
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from analyze_rl_eval_failures import classify  # noqa: E402

DOCS = Path(__file__).resolve().parents[1]/'openwebrl/docs'
EVIDENCE = DOCS/'arm_results/rl_integration/behavior'
ITERS = range(10, 101, 10)
# key: (label, color, linestyle, cohort name pattern with {i} for the iteration)
RUNS = {
    'outcome56': ('Outcome-only 56 groups', '#1565c0', '-', 'outcome-only-56groups-20261006-iter{i}'),
    'g8': ('Outcome-only G8 (8 rollouts)', '#6a1b9a', '-', 'outcome-uniform8-original2102-20261006-iter{i}'),
    'outcome48': ('Outcome-only 48 groups', '#0097a7', '-', 'outcome48-iter{i}'),
    'mixed_bonus': ('Mixed-only ARM bonus', '#2e7d32', '--', 'arm-mixed-bonus-iter{i}-*'),
    'mixed_reweight': ('Mixed-only ARM outcome reweight', '#ef6c00', '--', 'arm-mixed-reweight-iter{i}-*'),
    'action_only': ('Mixed-only ARM action-only', '#4f46e5', ':', 'arm-mixed-action-only-iter{i}-*'),
}
REPEATS = [('outcome48', 'Outcome-only\n48 groups', '#0097a7'), ('outcome56', 'Outcome-only\n56 groups', '#1565c0'),
           ('gate-b', 'Gate B\n(ARM, failure-additive)', '#2e7d32')]


def repeats_action(r):
    calls = [s['call'] for s in r.get('steps') or [] if s.get('call')]
    keys = Counter(json.dumps(c, sort_keys=True) for c in calls if c.get('name') not in ('done', 'wait'))
    return bool(keys) and keys.most_common(1)[0][1] >= 3


def measure(path):
    rows = [json.loads(x) for x in path.read_text().splitlines()]
    if len(rows) != 300 or not any(r.get('steps') for r in rows):
        raise ValueError(f'{path.name}: needs 300 tasks with trajectories')
    valid = [r for r in rows if r['valid']]
    rep = sum(map(repeats_action, valid))
    return dict(tasks=len(rows), successes=sum(r['success'] for r in rows), valid=len(valid),
                repeat_loop_failures=sum(classify(r, 30) == 'repeat_loop' for r in rows),
                valid_with_repeat=rep, repetition_rate=rep/len(valid))


def loop_by_task(compact_dir, key):
    """Per task, the share of the three repeats that ended as a repeated-action failure."""
    out = {}
    files = sorted(compact_dir.glob(f'repeat-iter90-{key}-r[123]-*.jsonl'))
    for f in files:
        for line in f.read_text().splitlines():
            r = json.loads(line)
            out.setdefault(str(r['task_id']), []).append(classify(r, 30) == 'repeat_loop')
    if len(files) != 3 or any(len(v) != 3 for v in out.values()):
        raise ValueError(f'{key}: expected three complete repeats')
    return {t: sum(v)/3 for t, v in out.items()}


def paired_differences(compact_dir, draws=20000, seed=0):
    """Gate B minus each outcome-only run: repeat-mean repeated-action failures per 300 tasks,
    paired percentile bootstrap over tasks."""
    rng = np.random.default_rng(seed)
    gate = loop_by_task(compact_dir, 'gate-b')
    out = {}
    for key in ('outcome56', 'outcome48'):
        other = loop_by_task(compact_dir, key)
        tasks = sorted(gate)
        if set(tasks) != set(other):
            raise ValueError('task sets differ')
        d = np.array([gate[t]-other[t] for t in tasks])*len(tasks)
        boots = d[rng.integers(0, len(d), (draws, len(d)))].mean(1)
        out[f'gate-b - {key}'] = dict(delta=float(d.mean()), ci95=[float(np.percentile(boots, 2.5)),
                                                                  float(np.percentile(boots, 97.5))])
    return out


def draw_repeats(ax, repeats, paired=None):
    for x, (key, label, color) in enumerate(REPEATS):
        vals = [m['repeat_loop_failures'] for m in repeats[key]]
        if not vals:
            continue
        ax.bar(x, statistics.mean(vals), color=color, alpha=.35, width=.6)
        ax.scatter([x]*len(vals), vals, color=color, zorder=3, s=22)
        ax.text(x+.33, statistics.mean(vals), f'{statistics.mean(vals):.1f}', va='center', fontsize=9)
    ax.set_xticks(range(len(REPEATS)))
    ax.set_xticklabels([label for _, label, _ in REPEATS], fontsize=8)
    ax.set_ylabel('Repeated-action failures (of 300 OM2W tasks)')
    ax.set_title('Iteration 90, three matched evaluations each')
    ax.set_ylim(0, 80)
    ax.grid(axis='y', alpha=.3)
    if paired:
        text = '\n'.join(f"Gate B − {k.split('outcome')[1]} groups: {v['delta']:+.1f} [{v['ci95'][0]:+.1f}, {v['ci95'][1]:+.1f}]"
                         for k, v in paired.items())
        ax.text(.98, .97, text+'\n(paired task bootstrap, 95% CI)', transform=ax.transAxes, ha='right', va='top',
                fontsize=7.5, bbox=dict(facecolor='white', edgecolor='#ccc'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--compact-dir', type=Path, required=True)
    a = p.parse_args()
    curves = {}
    for key, (*_, pattern) in RUNS.items():
        for i in ITERS:
            found = sorted(a.compact_dir.glob(pattern.format(i=i)+'.jsonl'))
            if found:
                curves.setdefault(key, {})[i] = dict(cohort=found[0].stem, **measure(found[0]))
    repeats = {key: [dict(cohort=f.stem, **measure(f)) for f in
                     sorted(a.compact_dir.glob(f'repeat-iter90-{key}-r[123]-*.jsonl'))] for key, *_ in REPEATS}
    paired = paired_differences(a.compact_dir)
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    (EVIDENCE/'rl-repetition-20261028.json').write_text(json.dumps(dict(
        note='Held-out Online-Mind2Web full-300 evaluations, local GPT-4.1/action_history T0 protocol. '
             'repeat_loop_failures: first-match failure category (analyze_rl_eval_failures.classify). '
             'repetition_rate: valid episodes (successes included) with an identical non-wait, non-done '
             'action issued >= 3 times. Iteration-90 repeats were collected on one harness on matched dates.',
        curves=curves, iteration90_repeats=repeats, iteration90_paired=paired), indent=1))

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), gridspec_kw=dict(width_ratios=[1.25, 1.25, 1]))
    for metric, ax, ylabel in (('repeat_loop_failures', axes[0], 'Repeated-action failures (of 300 tasks)'),
                               ('repetition_rate', axes[1], 'Valid episodes repeating an action ≥3× (%)')):
        for key, (label, color, ls, _) in RUNS.items():
            pts = sorted(curves.get(key, {}).items())
            if not pts:
                continue
            xs = [i for i, _ in pts]
            ys = [m[metric]*(100 if metric == 'repetition_rate' else 1) for _, m in pts]
            ax.plot(xs, ys, ls, marker='o', ms=4, color=color, label=label)
        ax.set_xlabel('Training iteration')
        ax.set_ylabel(ylabel)
        ax.set_xticks(list(ITERS))
        ax.set_ylim(bottom=0)
        ax.grid(alpha=.3)
    axes[0].set_title('Episodes that fail in a repeated-action loop')
    axes[1].set_title('Any action repeated ≥3× (judge-independent)')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=6, fontsize=8, frameon=False)

    draw_repeats(axes[2], repeats)
    fig.tight_layout(rect=(0, .07, 1, 1))
    out = DOCS/'rl_results/rl_repetition'
    fig.savefig(out.with_suffix('.png'), dpi=160)
    fig.savefig(out.with_suffix('.pdf'))
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    draw_repeats(ax, repeats, paired)
    fig.tight_layout()
    out = DOCS/'rl_results/rl_repetition_iter90'
    fig.savefig(out.with_suffix('.png'), dpi=160)
    fig.savefig(out.with_suffix('.pdf'))
    print(json.dumps(dict(curves={k: {i: (m['repeat_loop_failures'], round(100*m['repetition_rate'], 1))
                                      for i, m in v.items()} for k, v in curves.items()},
                          repeats={k: [(m['repeat_loop_failures'], round(100*m['repetition_rate'], 1)) for m in v]
                                   for k, v in repeats.items()}), indent=1))


if __name__ == '__main__':
    main()
