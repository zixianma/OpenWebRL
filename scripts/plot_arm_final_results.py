#!/usr/bin/env python3
"""Plot the main-paper ARM RL figures from the published aggregate JSON.

Reads openwebrl/docs/arm_results/rl_final_20261108/arm_final_results_data.json only (no
runtime paths or trajectories) and writes validation and training-success
small multiples as PNG and PDF. Each panel holds at most three series, so the
categorical colors stay distinguishable under color-vision deficiency.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "openwebrl/docs"))
import paper_plot_style  # noqa: F401
from paper_plot_palette import COLORS as C

OUT = Path(__file__).resolve().parents[1] / 'openwebrl/docs/arm_results/rl_final_20261108'
DATA = json.loads((OUT / 'arm_final_results_data.json').read_text())
RUNS = {r['key']: r for r in DATA['runs']}

# Shared identities: outcome-only controls neutral; ADM reference and ablations purple.
OUTCOME48, OUTCOME56, GATE_B, ABLATION = C["BASELINE_LIGHT"], C["BASELINE_DARK"], C["ADM_DEFAULT"], C["ADM_DEFAULT"]
INK, MUTED, GRID = C["INK"], C["MUTED"], C["BORDER"]
ABLATIONS = [
    ('additive', 'Strict gate (D = 5)'),
    ('mixed_bonus', 'No failure groups'),
    ('mixed_reweight', 'No failure groups, reweight credit'),
    ('all_failure', 'Failure groups replace mixed'),
    ('g40m16f', 'More failure groups: 40M + 16F'),
    ('g48m16f', 'More failure groups: 48M + 16F'),
]


def style(ax, title, ylabel):
    ax.set_title(title, fontsize=9.5, color=INK, loc='left', pad=6)
    ax.set_xlim(0, 99.5)
    ax.set_xticks(range(0, 91, 30))
    ax.grid(axis='y', color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8.5, color=MUTED)


def validation_series(key):
    v = RUNS[key]['validation']
    xs = sorted(int(k) for k in v)
    return xs, [100 * v[str(x)]['successes'] / 300 for x in xs]


def training_series(key, window=5):
    v = RUNS[key]['training_success_valid']
    xs = sorted(int(k) for k in v)
    raw = [100 * v[str(x)] for x in xs]
    smooth = [sum(raw[max(0, i - window + 1):i + 1]) / len(raw[max(0, i - window + 1):i + 1])
              for i in range(len(raw))]
    return xs, raw, smooth


def draw(ax, key, color, label, kind, dashed=False, width=2.0):
    ls = (0, (4, 2)) if dashed else '-'
    if kind == 'validation':
        xs, ys = validation_series(key)
        ax.plot(xs, ys, color=color, linewidth=width, linestyle=ls, label=label,
                marker='o', markersize=4.2, markeredgecolor='white', markeredgewidth=1.0, zorder=3)
    else:
        xs, raw, smooth = training_series(key)
        ax.plot(xs, raw, color=color, linewidth=0.8, alpha=0.22, linestyle='-', zorder=2)
        ax.plot(xs, smooth, color=color, linewidth=width, linestyle=ls, label=label, zorder=3)


def figure(kind):
    fig, axes = plt.subplots(2, 4, figsize=(13.2, 6.0), sharex=True, sharey=True)
    axes = axes.ravel()
    ylabel = 'Full-300 success (%)' if kind == 'validation' else 'Training success, valid episodes (%)'
    main = axes[0]
    draw(main, 'outcome48', OUTCOME48, 'Outcome-only, 48 groups', kind, dashed=True)
    draw(main, 'outcome56', OUTCOME56, 'Outcome-only, 56 groups', kind)
    draw(main, 'gate_b', GATE_B, 'Gate B (ours)', kind, width=2.4)
    if kind == 'validation':
        m = DATA['matched_iter90_local']['runs']
        for key, color, dx in [('outcome48', OUTCOME48, 3), ('outcome56', OUTCOME56, 5), ('gate_b', GATE_B, 7)]:
            mean = 100 * sum(m[key]['successes']) / 900
            main.scatter([90 + dx], [mean], marker='X', s=70, color=color, edgecolor='white',
                         linewidth=1.0, zorder=5)
    style(main, 'Main: Gate B vs outcome-only', ylabel)
    for i, (key, title) in enumerate(ABLATIONS, start=1):
        ax = axes[i]
        draw(ax, 'outcome56', OUTCOME56, 'Outcome-only, 56 groups', kind, width=1.4)
        draw(ax, 'gate_b', GATE_B, 'Gate B', kind, width=1.6)
        draw(ax, key, ABLATION, RUNS[key]['label'], kind, dashed=True, width=2.2)
        style(ax, title, ylabel if i == 4 else None)
    legend = axes[7]
    legend.axis('off')
    handles = [
        plt.Line2D([], [], color=OUTCOME48, linestyle=(0, (4, 2)), linewidth=2, label='Outcome-only, 48 groups'),
        plt.Line2D([], [], color=OUTCOME56, linewidth=2, label='Outcome-only, 56 groups'),
        plt.Line2D([], [], color=GATE_B, linewidth=2.4, label='Gate B (ours)'),
        plt.Line2D([], [], color=ABLATION, linestyle=(0, (4, 2)), linewidth=2.2, label='Ablation named in panel title'),
    ]
    if kind == 'validation':
        handles.append(plt.Line2D([], [], color=MUTED, marker='X', linestyle='none', markersize=8,
                                  label='Iteration 90, mean of 3 same-day repeats'))
        note = ('One full-300 evaluation per checkpoint\n(local browser, GPT-4.1 judge, T0).\n'
                'Repeat evaluations of one checkpoint\ndiffer by up to ~4 pp.')
    else:
        note = ('Success among valid training rollouts collected\neach iteration, before dynamic sampling.\n'
                'Bold: trailing 5-iteration mean; faint: raw.')
    legend.legend(handles=handles, loc='upper left', frameon=False, fontsize=8.5, labelcolor=INK)
    legend.text(0.02, 0.18, note, transform=legend.transAxes, fontsize=8, color=MUTED, va='bottom')
    for ax in [axes[3], *axes[4:7]]:
        ax.set_xlabel('Training iteration', fontsize=8.5, color=MUTED)
    axes[3].tick_params(labelbottom=True)
    ymin = 20 if kind == 'validation' else 30
    main.set_ylim(ymin, 46 if kind == 'validation' else 70)
    fig.tight_layout(w_pad=1.2, h_pad=1.6)
    name = 'validation_success' if kind == 'validation' else 'training_success'
    for ext in ('png', 'pdf'):
        fig.savefig(OUT / f'{name}.{ext}', dpi=200, facecolor='white')
    plt.close(fig)


if __name__ == '__main__':
    figure('validation')
    figure('training')
