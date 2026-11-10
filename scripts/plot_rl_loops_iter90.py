#!/usr/bin/env python3
"""Standalone paper figure: repeated-action loop failures at iteration 90, three matched repeats.

Reads the aggregate written by plot_rl_action_loops.py (no trajectories needed) and writes
rl_results/rl_action_loops_iter90.{pdf,png}. Bars are the mean of three back-to-back full-300
evaluations of each fixed checkpoint on one harness (job 385593); dots are the repeats; brackets give
the paired task-bootstrap 95% CI of Gate B minus each outcome-only run.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

DOCS = Path(__file__).resolve().parents[1]/'openwebrl/docs'
SRC = DOCS/'arm_results/rl_integration/behavior/rl-action-loops-20261103.json'
INK, INK2, GRID, SURFACE = '#0b0b0b', '#52514e', '#e4e3df', '#ffffff'
RUNS = (('outcome48', 'Outcome-only\n(48 groups)', '#b4b2ac'),
        ('outcome56', 'Outcome-only\n(56 groups)', '#8e8c86'),
        ('gate-b', 'ARM\n(Gate B)', '#2a78d6'))


def main():
    d = json.loads(SRC.read_text())
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    fig, ax = plt.subplots(figsize=(3.3, 2.6))
    fig.patch.set_facecolor(SURFACE)
    w = .5
    means = {}
    for x, (key, label, color) in enumerate(RUNS):
        vals = [m['loop_failures'] for m in d['iteration90'][key]]
        mean = sum(vals)/len(vals)
        means[key] = mean
        ax.bar(x, mean, width=w, color=color, zorder=2, linewidth=0)
        offsets = (-.13, 0, .13)
        ax.scatter([x+o for o in offsets], vals, s=22, color=INK, edgecolor=SURFACE, linewidth=1.2, zorder=3)
        ax.text(x+w/2+.04, mean, f'{mean:.1f}', ha='left', va='center', fontsize=7.5, color=INK, zorder=4)
    # paired-difference brackets
    pair = d['iteration90_paired']
    top = max(m['loop_failures'] for k in d['iteration90'] for m in d['iteration90'][k])
    for i, (ref, xr) in enumerate((('outcome56', 1), ('outcome48', 0))):
        p = pair[f'gate_b - {ref}']['failures_per300']
        y = top + 7 + i*13
        ax.plot([xr, xr, 2, 2], [y-2, y, y, y-2], color=INK2, lw=.8, zorder=1)
        ax.text((xr+2)/2, y+1.2, f"Δ {p['delta']:+.1f}  [{p['ci95'][0]:+.1f}, {p['ci95'][1]:+.1f}]".replace('-', '\u2212'),
                ha='center', va='bottom', fontsize=7, color=INK)
    ax.set_xticks(range(len(RUNS)))
    ax.set_xticklabels([l for _, l, _ in RUNS], fontsize=7.5, color=INK)
    ax.set_xlim(-.55, 2.75)
    ax.set_ylim(0, top+30)
    ax.set_ylabel('Loop failures per 300 tasks', color=INK)
    ax.yaxis.grid(True, color=GRID, lw=.6, zorder=0)
    ax.set_axisbelow(True)
    for s in ('top', 'right', 'left'):
        ax.spines[s].set_visible(False)
    ax.spines['bottom'].set_color(INK2)
    ax.tick_params(axis='y', length=0, colors=INK2, labelsize=7)
    ax.tick_params(axis='x', length=0, pad=4)
    fig.tight_layout(pad=.3)
    out = DOCS/'rl_results/rl_action_loops_iter90'
    fig.savefig(out.with_suffix('.pdf'))
    fig.savefig(out.with_suffix('.png'), dpi=300)
    print(out, means)


if __name__ == '__main__':
    main()
