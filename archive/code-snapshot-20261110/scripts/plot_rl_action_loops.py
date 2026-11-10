#!/usr/bin/env python3
"""Repeated-action loops: ARM RL versus outcome-only RL (paper figure and statistics).

Inputs: compact held-out trajectories from analyze_rl_eval_failures.py --extract (full-300 Online-Mind2Web,
local browser, GPT-4.1/action_history judge, actor T0, 30 turns) and, optionally, the training-label
aggregate from analyze_arm_label_repeats.py. Outputs (aggregate only):
  rl_results/rl_action_loops.{pdf,png}        three-panel paper figure
  rl_results/rl_action_loops_outcomes.{pdf,png} where outcome-only loop tasks go under Gate B
  arm_results/rl_integration/behavior/rl-action-loops-20261103.json

Definitions (judge-independent except "loop failure"):
  loop episode: a valid episode whose agent issues one identical non-wait, non-done action (name and
    arguments) >= 3 times; successes included.
  loop rate: loop episodes / valid episodes.
  loop failure: first-match failure category 'repeat_loop' of analyze_rl_eval_failures.classify, out of 300.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

import matplotlib
import numpy as np
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from analyze_rl_eval_failures import classify  # noqa: E402

DOCS = Path(__file__).resolve().parents[1]/'openwebrl/docs'
EVIDENCE = DOCS/'arm_results/rl_integration/behavior'
# key: label, colour (Okabe-Ito), marker, cohort file pattern
RUNS = {
    'outcome56': ('Outcome-only, 56 groups', '#555555', 'o', 'outcome-only-56groups-20261006-iter{i}'),
    'uniform8': ('Outcome-only (G8)', '#999999', 's', 'outcome-uniform8-original2102-20261006-iter{i}'),
    'mixed_bonus': ('ARM bonus', '#E69F00', '^', 'arm-mixed-bonus-iter{i}-*'),
    'mixed_reweight': ('ARM reweight', '#009E73', 'D', 'arm-mixed-reweight-iter{i}-*'),
    'g40m16f': ('ARM Gate B 40M+16F', '#56B4E9', 'v', 'arm-gate-b-40m16f-iter{i}-*'),
    'gate_b': ('ARM Gate B', '#D55E00', 'P', None),
    'original': ('ARM bonus, strict gate', '#0072B2', 'X', 'arm-original-iter{i}-*'),
}
REPEATS = (('outcome48', 'Outcome\n48 groups', '#777777'), ('outcome56', 'Outcome\n56 groups', '#555555'),
           ('gate-b', 'ARM\nGate B', '#D55E00'))
LATE = (70, 80, 90)


def max_repeat(r):
    calls = [s['call'] for s in r.get('steps') or [] if s.get('call')]
    keys = Counter(json.dumps(c, sort_keys=True) for c in calls if c.get('name') not in ('done', 'wait'))
    return keys.most_common(1)[0][1] if keys else 0


def consecutive_repeat(r):
    calls = [json.dumps(s['call'], sort_keys=True) for s in r.get('steps') or []
             if s.get('call') and s['call'].get('name') not in ('done', 'wait')]
    best = run = 1 if calls else 0
    for a, b in zip(calls, calls[1:]):
        run = run+1 if a == b else 1
        best = max(best, run)
    return best


def load(path):
    rows = [json.loads(x) for x in path.read_text().splitlines()]
    if len(rows) != 300 or not any(r.get('steps') for r in rows):
        return None
    out = {}
    for r in rows:
        out[str(r['task_id'])] = dict(valid=bool(r['valid']), success=bool(r['success']), rep=max_repeat(r),
                                      cons=consecutive_repeat(r), cat=classify(r, 30))
    return out


def summary(c):
    valid = [x for x in c.values() if x['valid']]
    return dict(valid=len(valid), successes=sum(x['success'] for x in c.values()),
                loop_episodes=sum(x['rep'] >= 3 for x in valid), loop_rate=sum(x['rep'] >= 3 for x in valid)/len(valid),
                loop_failures=sum(x['cat'] == 'repeat_loop' for x in c.values()),
                sensitivity=dict(ge4=sum(x['rep'] >= 4 for x in valid)/len(valid), ge5=sum(x['rep'] >= 5 for x in valid)/len(valid),
                                 consecutive3=sum(x['cons'] >= 3 for x in valid)/len(valid)))


def per_task(cohorts, tasks, metric):
    """Per-task arrays (n_cohorts x tasks): loop indicator and validity."""
    if metric == 'loop_failure':
        return np.array([[c[t]['cat'] == 'repeat_loop' for t in tasks] for c in cohorts], float), None
    loop = np.array([[c[t]['valid'] and c[t]['rep'] >= 3 for t in tasks] for c in cohorts], float)
    valid = np.array([[c[t]['valid'] for t in tasks] for c in cohorts], float)
    return loop, valid


def boot_diff(a_loop, a_valid, b_loop, b_valid, rng, draws=20000):
    """Paired task bootstrap of a difference in loop rate (ratio of sums) or per-300 loop failures."""
    n = a_loop.shape[1]
    idx = rng.integers(0, n, (draws, n))
    if a_valid is None:
        da = a_loop.mean(0)-b_loop.mean(0)
        est, bs = 300*da.mean(), 300*da[idx].mean(1)
    else:
        sa, va, sb, vb = a_loop.sum(0), a_valid.sum(0), b_loop.sum(0), b_valid.sum(0)
        est = 100*(sa.sum()/va.sum()-sb.sum()/vb.sum())
        bs = 100*(sa[idx].sum(1)/va[idx].sum(1)-sb[idx].sum(1)/vb[idx].sum(1))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    p = min(1.0, 2*min((bs <= 0).mean(), (bs >= 0).mean()))
    return dict(delta=float(est), ci95=[float(lo), float(hi)], p_boot=float(p))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--compact-dir', type=Path, required=True)
    ap.add_argument('--labels', type=Path)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8.5, 'axes.labelsize': 8, 'legend.fontsize': 7,
                         'xtick.labelsize': 7, 'ytick.labelsize': 7, 'pdf.fonttype': 42, 'axes.spines.top': False,
                         'axes.spines.right': False})

    curves = {}
    for key, (*_, pat) in RUNS.items():
        for i in range(10, 101, 10):
            if pat is None:
                continue
            f = sorted(a.compact_dir.glob(pat.format(i=i)+'.jsonl'))
            c = load(f[0]) if f else None
            if c:
                curves.setdefault(key, {})[i] = c
    g10 = sorted(a.compact_dir.glob('gate-b-iter10-full300-*.jsonl'))
    if g10:
        curves.setdefault('gate_b', {})[10] = load(g10[0])
    reps = {k: [load(f) for f in sorted(a.compact_dir.glob(f'repeat-iter90-{k}-r[123]-*.jsonl'))] for k, *_ in REPEATS}
    tasks = sorted(curves['outcome56'][10])

    res = dict(definitions=__doc__.split('Definitions')[1].strip(), curves={}, late_window={}, iteration90={})
    for k, v in curves.items():
        res['curves'][k] = {i: summary(c) for i, c in sorted(v.items())}
    for k, v in reps.items():
        res['iteration90'][k] = [summary(c) for c in v]

    # late window: every run with 70/80/90 trajectories, paired against outcome-only (mean of 56 and G8)
    late = {k: [curves[k][i] for i in LATE] for k in curves if all(i in curves[k] for i in LATE)}
    ref_l = np.concatenate([per_task(late[k], tasks, 'rate')[0] for k in ('outcome56', 'uniform8')])
    ref_v = np.concatenate([per_task(late[k], tasks, 'rate')[1] for k in ('outcome56', 'uniform8')])
    ref_f = np.concatenate([per_task(late[k], tasks, 'loop_failure')[0] for k in ('outcome56', 'uniform8')])
    for k, cs in late.items():
        l, v = per_task(cs, tasks, 'rate'); f, _ = per_task(cs, tasks, 'loop_failure')
        row = dict(loop_rate=float(l.sum()/v.sum()), loop_failures_per300=float(f.sum(1).mean()),
                   loop_failures=[int(x) for x in f.sum(1)])
        if k not in ('outcome56', 'uniform8'):
            row['rate_minus_outcome_pp'] = boot_diff(l, v, ref_l, ref_v, rng)
            row['failures_minus_outcome_per300'] = boot_diff(f, None, ref_f, None, rng)
        res['late_window'][k] = row
    ol, ov = per_task(late['outcome56'], tasks, 'rate'); gl, gv = per_task(late['uniform8'], tasks, 'rate')
    res['late_window']['uniform8']['rate_minus_outcome56_pp'] = boot_diff(gl, gv, ol, ov, rng)

    # iteration-90 matched repeats
    pair = {}
    gl, gv = per_task(reps['gate-b'], tasks, 'rate'); gf, _ = per_task(reps['gate-b'], tasks, 'loop_failure')
    for k in ('outcome56', 'outcome48'):
        l, v = per_task(reps[k], tasks, 'rate'); f, _ = per_task(reps[k], tasks, 'loop_failure')
        pair[f'gate_b - {k}'] = dict(rate_pp=boot_diff(gl, gv, l, v, rng), failures_per300=boot_diff(gf, None, f, None, rng))
    res['iteration90_paired'] = pair

    # are loops failures? pooled over every cohort with trajectories
    allc = [c for v in curves.values() for c in v.values()]+[c for v in reps.values() for c in v]
    lv = [x for c in allc for x in c.values() if x['valid'] and x['rep'] >= 3]
    nv = [x for c in allc for x in c.values() if x['valid'] and x['rep'] < 3]
    res['success_given_loop'] = dict(loop_episodes=len(lv), loop_success=float(np.mean([x['success'] for x in lv])),
                                     other_episodes=len(nv), other_success=float(np.mean([x['success'] for x in nv])))

    # where do outcome-only loop tasks go under Gate B (iteration-90 repeats, same dates)
    base = reps['outcome56']+reps['outcome48']
    loop_tasks = [t for t in tasks if sum(c[t]['cat'] == 'repeat_loop' for c in base) >= 3]
    def dist(cs, ts):
        cnt = Counter()
        for c in cs:
            for t in ts:
                x = c[t]
                cnt['success' if x['success'] else 'invalid' if not x['valid'] else
                    {'repeat_loop': 'loop', 'wrong_answer': 'wrong answer', 'claims_infeasible': 'claims impossible',
                     'turn_limit': 'turn limit'}.get(x['cat'], 'other failure')] += 1
        tot = sum(cnt.values())
        return {k: v/tot for k, v in cnt.items()}, tot
    res['outcome_loop_tasks'] = dict(rule='looped in >= 3 of the 6 outcome-only iteration-90 repeats (48 and 56 groups)',
                                     tasks=len(loop_tasks), outcome_only=dist(base, loop_tasks)[0],
                                     gate_b=dist(reps['gate-b'], loop_tasks)[0])

    lab = json.loads(a.labels.read_text()) if a.labels and a.labels.is_file() else None
    if lab:
        res['training_labels'] = {k: v.get('all') for k, v in lab['lineages'].items()}

    # ---------------- figure: three panels
    fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.9), gridspec_kw=dict(width_ratios=[1.3, .9, 1.2]))
    x0 = ax[0]
    for k, (label, color, marker, _) in RUNS.items():
        pts = sorted(res['curves'].get(k, {}).items())
        if k == 'gate_b':
            continue
        if not pts:
            continue
        x0.plot([i for i, _ in pts], [100*m['loop_rate'] for _, m in pts], marker=marker, ms=3.2, lw=1.1,
                color=color, ls='-' if k.startswith(('outcome', 'uniform')) else '--', label=label)
    x0.set_xlabel('Training iteration'); x0.set_ylabel('Valid episodes with a loop (%)')
    x0.set_xticks(range(10, 100, 20)); x0.set_ylim(0, None); x0.grid(alpha=.25, lw=.5)
    x0.set_title('(a) Loop rate during training')
    h, l = x0.get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=3, frameon=False, fontsize=6.5, handlelength=1.8,
               bbox_to_anchor=(.5, 0))

    x1 = ax[1]
    for j, (k, label, color) in enumerate(REPEATS):
        vals = [m['loop_failures'] for m in res['iteration90'][k]]
        x1.bar(j, np.mean(vals), color=color, alpha=.35, width=.62)
        x1.scatter([j]*len(vals), vals, color=color, s=10, zorder=3)
    x1.set_xticks(range(len(REPEATS))); x1.set_xticklabels([l for _, l, _ in REPEATS])
    x1.set_ylabel('Loop failures (of 300 tasks)'); x1.set_ylim(0, 80); x1.grid(axis='y', alpha=.25, lw=.5)
    p56 = pair['gate_b - outcome56']['failures_per300']
    x1.text(.98, .98, f"Gate B − 56 groups:\n{p56['delta']:+.1f} [{p56['ci95'][0]:+.1f}, {p56['ci95'][1]:+.1f}]",
            transform=x1.transAxes, ha='right', va='top', fontsize=6.3)
    x1.set_title('(b) Iteration 90, 3 matched repeats')

    x2 = ax[2]
    if lab:
        order = [k for k in ('gate_b', 'mixed_bonus', 'mixed_reweight', 'g40m16f', 'original', 'additive')
                 if k in res['training_labels']]
        names = {'gate_b': 'Gate B', 'mixed_bonus': 'Bonus', 'mixed_reweight': 'Re-\nweight', 'g40m16f': '40M\n+16F',
                 'original': 'Bonus', 'additive': 'Additive'}
        w = .38
        for j, k in enumerate(order):
            m = res['training_labels'][k]
            for off, key, n_key, col, lab_ in ((-w/2, 'p_exec_new', 'exec_new_n', '#0072B2', 'executed action is new'),
                                               (w/2, 'p_exec_repeat', 'exec_repeat_n', '#CC79A7', 'executed action repeats')):
                p_, n_ = m[key], m[n_key]
                se = 1.96*(p_*(1-p_)/n_)**.5
                x2.bar(j+off, 100*p_, w, yerr=100*se, color=col, error_kw=dict(lw=.7, capsize=1.5),
                       label=lab_ if j == 0 else None)
        x2.axvline(3.5, color='#bbb', lw=.6)
        x2.text(1.5, 28.5, 'relaxed gate', ha='center', fontsize=6); x2.text(4.5, 28.5, 'strict gate', ha='center', fontsize=6)
        x2.axhline(20, color='k', lw=.8, ls=':'); x2.text(5.75, 20.4, 'chance', fontsize=5.5, ha='right')
        x2.set_xticks(range(len(order))); x2.set_xticklabels([names[k] for k in order], fontsize=5.8)
        x2.set_ylabel('ARM selects executed action (%)'); x2.set_ylim(0, 30)
        x2.legend(loc='lower left', frameon=True, framealpha=.9, edgecolor='none', fontsize=5.5, borderaxespad=.2)
        x2.set_title('(c) ARM labels during RL')
        x2.grid(axis='y', alpha=.25, lw=.5)
    fig.tight_layout(pad=.4, w_pad=1.0, rect=(0, .14, 1, 1))
    out = DOCS/'rl_results/rl_action_loops'
    fig.savefig(out.with_suffix('.pdf')); fig.savefig(out.with_suffix('.png'), dpi=300)

    # ---------------- supplementary: outcomes on outcome-only loop tasks
    fig, axs = plt.subplots(figsize=(3.4, 1.7))
    cats = [('success', '#009E73'), ('wrong answer', '#E69F00'), ('claims impossible', '#F0E442'), ('loop', '#CC79A7'),
            ('turn limit', '#56B4E9'), ('other failure', '#999999'), ('invalid', '#DDDDDD')]
    ol = res['outcome_loop_tasks']
    for y, (name, d) in enumerate((('ARM Gate B', ol['gate_b']), ('Outcome-only', ol['outcome_only']))):
        left = 0
        for c, col in cats:
            v = 100*d.get(c, 0)
            axs.barh(y, v, left=left, color=col, height=.6, label=c if y == 0 else None)
            if v >= 8:
                axs.text(left+v/2, y, f'{v:.0f}', ha='center', va='center', fontsize=6)
            left += v
    axs.set_yticks([0, 1]); axs.set_yticklabels(['ARM Gate B', 'Outcome-only']); axs.set_xlim(0, 100)
    axs.set_xlabel(f"Episodes on the {ol['tasks']} outcome-only loop tasks (%)")
    axs.legend(ncol=4, fontsize=5.5, frameon=False, loc='lower center', bbox_to_anchor=(.45, 1.0), handlelength=1)
    fig.tight_layout(pad=.3)
    out = DOCS/'rl_results/rl_action_loops_outcomes'
    fig.savefig(out.with_suffix('.pdf')); fig.savefig(out.with_suffix('.png'), dpi=300)

    EVIDENCE.mkdir(parents=True, exist_ok=True)
    (EVIDENCE/'rl-action-loops-20261103.json').write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ('late_window', 'iteration90_paired', 'success_given_loop', 'outcome_loop_tasks')}, indent=1))


if __name__ == '__main__':
    main()
