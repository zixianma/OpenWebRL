#!/usr/bin/env python3
"""RL behavior figures (aggregate only; no task payloads). Needs matplotlib (venv-llamafactory).

1. rl_behavior_categories_56.png: outcome-only 56-group held-out episodes by full category
   (needs saved trajectories).
2. rl_behavior_categories_56_vs_gateb.png: the same run and Gate B by coarse category, which
   uses only fields saved for every evaluation (Gate B trajectories were pruned).
3. rl_behavior_difficulty_56_vs_gateb.png: success rate by OM2W difficulty over training.

--compact-dir points at analyze_rl_eval_failures.py JSONL (private, runtime); the script
also writes the aggregate counts it plots to the docs evidence folder.
"""
import argparse
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

DOCS = Path(__file__).resolve().parents[1]/'openwebrl/docs'
EVIDENCE = DOCS/'arm_results/rl_integration/behavior'
TASKS = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/reference-arm-stealth90-o4-t06-20260929/online_mind2web_monitor.jsonl')
FULL = [('success', 'Success', '#2e7d32'), ('wrong_answer', 'Wrong answer', '#e57373'),
        ('repeat_loop', 'Repeated action', '#ffb74d'), ('claims_infeasible', 'Claims impossible', '#ba68c8'),
        ('turn_limit', 'Turn limit', '#64b5f6'), ('other_valid', 'Other valid failure', '#bdbdbd'),
        ('invalid:env_step_error', 'Invalid: step error', '#616161'),
        ('invalid:start_page_failed', 'Invalid: start page', '#9e9e9e'), ('invalid:other', 'Invalid: other', '#e0e0e0')]
COARSE = [('success', 'Success', '#2e7d32'), ('finished_failure', 'Finished, judged failure', '#e57373'),
          ('turn_limit', 'Turn limit (30 turns)', '#64b5f6'), ('other_valid', 'Other valid failure (length cut)', '#bdbdbd'),
          ('invalid_step', 'Invalid: step error', '#616161'), ('invalid_start', 'Invalid: start page / timeout', '#9e9e9e')]
RUNS = {'outcome56': ('Outcome-only 56 groups', '#1565c0', 'outcome-only-56groups-20261006-iter'),
        'gate-b': ('Gate B (ARM)', '#2e7d32', 'arm-gate-b-iter')}


def coarse(r):
    """Coarse category from fields saved for every evaluation (calibrated on the 56-group run)."""
    if not r['valid']:
        return 'invalid_start' if (r.get('turns') or 0) <= 1 else 'invalid_step'
    if r['success']:
        return 'success'
    status = r.get('terminal_status')
    if status == 'failed' or (r.get('turns') or 0) >= 30:
        return 'turn_limit'
    return 'finished_failure' if status == 'completed' else 'other_valid'


def difficulty():
    out = {}
    for line in TASKS.read_text().splitlines():
        m = json.loads(line)['metadata']
        n = int(m['reference_length'])
        out[str(m['task_id'])] = 'easy' if n <= 5 else 'medium' if n <= 10 else 'hard'
    return out


def load(compact):
    runs = {}
    for key, (_, _, prefix) in RUNS.items():
        for f in sorted(compact.glob(prefix+'*.jsonl')):
            it = int(re.search(r'iter(\d+)', f.name).group(1))
            runs.setdefault(key, {})[it] = [json.loads(x) for x in f.read_text().splitlines()]
    return {k: dict(sorted(v.items())) for k, v in runs.items()}


def stacked(ax, rows, cats, title):
    its = list(rows)
    bottom = [0]*len(its)
    for key, label, color in cats:
        vals = [rows[i].get(key, 0) for i in its]
        ax.bar([str(i) for i in its], vals, bottom=bottom, color=color, label=label, width=.75)
        bottom = [b+v for b, v in zip(bottom, vals)]
    ax.set(title=title, xlabel='Training iteration', ylabel='Episodes (of 300)', ylim=(0, 300))


def save(fig, name, out):
    for ext in ('png', 'pdf'):
        fig.savefig(out/f'{name}.{ext}', dpi=160, bbox_inches='tight')
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--compact-dir', type=Path, required=True)
    p.add_argument('--out-dir', type=Path, default=DOCS/'rl_results')
    a = p.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    cats = json.loads((EVIDENCE/'rl-eval-failure-categories-20261024.json').read_text())['cohorts']
    full = {}
    for name, v in cats.items():
        if name.startswith('outcome-only-56groups'):
            c = dict(v['categories'])
            c['other_valid'] = c.pop('no_done_other', 0)+c.pop('tool_errors', 0)
            c['invalid:other'] = sum(c.pop(k) for k in list(c) if k.startswith('invalid:')
                                     and k not in ('invalid:env_step_error', 'invalid:start_page_failed'))
            full[int(name.rsplit('iter', 1)[1])] = c
    fig, ax = plt.subplots(figsize=(8, 5))
    stacked(ax, dict(sorted(full.items())), FULL, 'Outcome-only 56 groups: held-out episodes by category')
    ax.legend(fontsize=8, ncol=3, loc='upper center', bbox_to_anchor=(.5, -.13), frameon=False)
    save(fig, 'rl_behavior_categories_56', a.out_dir)

    runs, bins = load(a.compact_dir), difficulty()
    evidence = dict(coarse_rule='success; invalid split by turns<=1 (start page/timeout) vs later (step error); '
                                 'turn limit = status failed or 30 turns; finished failure = completed, judged failure; other = truncated',
                    difficulty_rule='human reference length 1-5 easy, 6-10 medium, 11+ hard (80/141/79 tasks)', runs={})
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, (key, (label, _, _)) in zip(axes, RUNS.items()):
        rows = {}
        for it, records in runs[key].items():
            c = {}
            for r in records:
                c[coarse(r)] = c.get(coarse(r), 0)+1
            rows[it] = c
            d = {}
            for band in ('easy', 'medium', 'hard'):
                sub = [r for r in records if bins[r['task_id']] == band]
                d[band] = dict(tasks=len(sub), successes=sum(r['success'] for r in sub),
                               valid=sum(r['valid'] for r in sub))
            evidence['runs'].setdefault(key, {})[it] = dict(categories=c, difficulty=d)
        stacked(ax, rows, COARSE, f'{label}: held-out episodes by coarse category')
    fig.tight_layout(rect=(0, .08, 1, 1))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=8, ncol=6, loc='lower center', frameon=False)
    save(fig, 'rl_behavior_categories_56_vs_gateb', a.out_dir)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), sharey=True)
    for ax, band in zip(axes, ('easy', 'medium', 'hard')):
        for key, (label, color, _) in RUNS.items():
            pts = sorted(evidence['runs'][key].items())
            xs = [i for i, _ in pts]
            ys = [100*v['difficulty'][band]['successes']/v['difficulty'][band]['tasks'] for _, v in pts]
            ax.plot(xs, ys, 'o-', color=color, label=label, alpha=.85)
            if len(ys) >= 3:  # centered three-checkpoint mean to show the trend through checkpoint noise
                ax.plot(xs[1:-1], [(ys[k-1]+ys[k]+ys[k+1])/3 for k in range(1, len(ys)-1)], '--', color=color, lw=2.5, alpha=.5)
        n = next(iter(evidence['runs']['outcome56'].values()))['difficulty'][band]['tasks']
        ax.set(title=f'{band.capitalize()} tasks (n = {n})', xlabel='Training iteration', ylim=(0, 100))
        ax.grid(alpha=.3)
    axes[0].set_ylabel('Success % (invalid counted as failure)')
    axes[0].legend(fontsize=9, loc='lower left')
    fig.suptitle('Held-out OM2W success by difficulty · local GPT-4.1/T0 · solid: per checkpoint, dashed: 3-checkpoint mean', fontsize=10)
    fig.tight_layout()
    save(fig, 'rl_behavior_difficulty_56_vs_gateb', a.out_dir)
    (EVIDENCE/'rl-eval-coarse-difficulty-20261024.json').write_text(json.dumps(evidence, indent=1))
    print('ok')


if __name__ == '__main__':
    main()
