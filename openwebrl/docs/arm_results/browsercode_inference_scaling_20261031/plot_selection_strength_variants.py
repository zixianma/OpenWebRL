#!/usr/bin/env python3
"""Two alternative action-selection views using frozen public aggregates only.

Keeps the existing compact figure unchanged. Reuses audited paired intervals;
no new bootstrap, model calls, task payloads or experimental runs are needed.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/openwebrl-selection-strength-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# Shared paper typography, including bundled Albert Sans fonts.
import sys as _paper_sys
from pathlib import Path as _PaperPath
_paper_sys.path.insert(0, str(next(p for p in _PaperPath(__file__).resolve().parents
                                if (p / "paper_plot_style.py").is_file())))
import paper_plot_style as _paper_style

from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('scaling_families', HERE / 'plot_browsercode_scaling_families.py')
F = importlib.util.module_from_spec(spec)
spec.loader.exec_module(F)
BLUE, RED, GRAY, INK = '#3187B3', '#B41620', '#8C929A', '#24272B'
BENCHMARKS = [('online_mind2web', 'OM2W'), ('webvoyager', 'WebVoyager'), ('deepshop', 'DeepShop')]
PAIRS = [(30, 'parallel5_select_high', 'luna_single_high'),
         (60, 'parallel5_select_high_60steps', 'luna_single_high_60steps')]
plt.rcParams.update({'font.family': _paper_style.FONT_FAMILY, 'font.size': 7,
                     'axes.labelsize': 7, 'axes.titlesize': 8,
                     'xtick.labelsize': 7, 'ytick.labelsize': 7,
                     'svg.fonttype': 'none', 'pdf.fonttype': 42,
                     'svg.hashsalt': 'selection-strength-20261103',
                     'savefig.facecolor': 'white'})


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def style(ax):
    ax.grid(False, which='both')
    ax.spines[['top', 'right']].set_visible(False)
    for side in ['left', 'bottom']:
        ax.spines[side].set_color('#555555')
        ax.spines[side].set_linewidth(.6)
    ax.tick_params(length=3, width=.6, pad=2)


def save(fig, stem):
    for ext in ['png', 'svg', 'pdf']:
        metadata = {'Date': None} if ext == 'svg' else {'CreationDate': None, 'ModDate': None} if ext == 'pdf' else {}
        path = HERE / f'{stem}.{ext}'
        fig.savefig(path, dpi=300, bbox_inches='tight', pad_inches=.025, metadata=metadata)
        if ext == 'svg':
            path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines()) + '\n')
    plt.close(fig)


def interaction_view(points, contrasts):
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.55), gridspec_kw={'width_ratios': [1.5, 1]})
    fig.subplots_adjust(left=.075, right=.98, top=.78, bottom=.21, wspace=.49)
    ax = axes[0]
    highlight = {
        'luna_single_high': ('N1 / 30', BLUE, 'o', True, (4, -10)),
        'luna_single_high_60steps': ('N1 / 60', BLUE, 'o', False, (4, -10)),
        'luna_single_high_120steps': ('N1 / 120', BLUE, 'D', False, (4, 5)),
        'parallel5_select_high': ('N5 / 30', RED, 'o', True, (4, 5)),
        'parallel5_select_high_60steps': ('N5 / 60', RED, 'o', False, (4, 5)),
    }
    for method, point in points.items():
        if method in highlight:
            continue
        ax.scatter(point['browser_commands'], 100 * point['success_rate'], s=19,
                   marker='s' if method == 'serial_high' else 'o',
                   facecolor=GRAY, edgecolor='white', linewidth=.4, zorder=2)
        if method == 'serial_high':
            ax.annotate('Sol / 30', (point['browser_commands'], 100 * point['success_rate']),
                        xytext=(4, 3), textcoords='offset points', color=GRAY, fontsize=7)
    for budget, left, right in PAIRS:
        start, end = points[right], points[left]
        # An observed configuration contrast, never an interpolated scaling curve.
        ax.annotate('', (end['browser_commands'], 100 * end['success_rate']),
                    (start['browser_commands'], 100 * start['success_rate']),
                    arrowprops={'arrowstyle': '->', 'color': RED, 'lw': .8,
                                'linestyle': '-' if budget == 30 else '--',
                                'shrinkA': 5, 'shrinkB': 5}, zorder=1)
    for method, (label, color, marker, filled, offset) in highlight.items():
        point = points[method]
        ax.scatter(point['browser_commands'], 100 * point['success_rate'], s=32, marker=marker,
                   facecolors=color if filled else 'white', edgecolors=color, linewidth=.9, zorder=3)
        ax.annotate(label, (point['browser_commands'], 100 * point['success_rate']),
                    xytext=offset, textcoords='offset points', color=color, fontsize=7)
    ax.set(xlim=(11.3, 16.5), ylim=(79, 88.2), xticks=[12, 14, 16], yticks=[80, 84, 88],
           xlabel='Executed browser batches / task', ylabel='Common-valid success (%)')
    ax.set_title('a)  Success vs browser interaction', loc='left', pad=9)
    ax = axes[1]
    ax.axvline(0, color='#A1A6AD', linewidth=.7, linestyle='--', zorder=0)
    for y, (budget, left, right) in zip([1, 0], PAIRS):
        result = contrasts[left, right]
        gain = result['difference_pp']
        low, high = result['pointwise_paired_bootstrap_95_interval_pp']
        ax.errorbar(gain, y, xerr=[[gain - low], [high - gain]], fmt='o',
                    mfc=RED if budget == 30 else 'white', mec=RED, color=RED,
                    markersize=4.5, capsize=3, linewidth=.9, zorder=3)
    ax.set(xlim=(-2, 7), ylim=(-.55, 1.55), xticks=[-2, 0, 2, 4, 6],
           yticks=[1, 0], yticklabels=['30 decisions', '60 decisions'],
           xlabel='N5 − N1 success (pp; paired 95% CI)')
    ax.set_title('b)  Gain at a fixed decision limit', loc='left', pad=9)
    for ax in axes:
        style(ax)
    handles = [Line2D([], [], marker='o', linestyle='none', color=BLUE, label='N1 single'),
               Line2D([], [], marker='o', linestyle='none', color=RED, label='N5 selection'),
               Line2D([], [], marker='o', linestyle='none', color=GRAY, label='Other settings')]
    fig.legend(handles=handles, loc='upper right', bbox_to_anchor=(.99, 1.02),
               frameon=False, ncol=3, fontsize=7, handletextpad=.35, columnspacing=1.3)
    save(fig, 'browsercode_selection_interaction')


def paired_outcomes_view(data, groups):
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.55), sharex=True, sharey=True)
    fig.subplots_adjust(left=.155, right=.98, top=.78, bottom=.22, wspace=.16)
    exported = []
    weights = {b: row['planned'] / 1045 for b, row in data['benchmark_counts'].items()}
    for ax, (budget, left, right) in zip(axes, PAIRS):
        pair = groups[left, right]
        rows = []
        for benchmark, label in BENCHMARKS:
            row = pair[benchmark]
            rows.append((label, row['left_only_success'] / row['common_valid'],
                         row['right_only_success'] / row['common_valid']))
        weighted_win = sum(weights[b] * row['left_only_success'] / row['common_valid'] for b, row in pair.items())
        weighted_loss = sum(weights[b] * row['right_only_success'] / row['common_valid'] for b, row in pair.items())
        rows = [('Weighted mean', weighted_win, weighted_loss)] + rows
        for y, (label, win, loss) in zip([3, 2, 1, 0], rows):
            ax.barh(y, 100 * win, height=.45, color=RED, zorder=2)
            ax.barh(y, -100 * loss, height=.45, color=BLUE, zorder=2)
            exported.append(dict(decision_limit=budget, population=label,
                                 n5_only_success_percent=100 * win,
                                 n1_only_success_percent=100 * loss,
                                 net_gain_pp=100 * (win - loss)))
        ax.axvline(0, color='#A1A6AD', linewidth=.7, zorder=0)
        ax.set(xlim=(-15, 15), xticks=[-15, -10, -5, 0, 5, 10, 15],
               ylim=(-.55, 3.55), yticks=[3, 2, 1, 0],
               yticklabels=['Weighted mean', 'OM2W', 'WebVoyager', 'DeepShop'],
               xlabel='Share of the same valid tasks (%)')
        ax.set_xticklabels(['15', '10', '5', '0', '5', '10', '15'])
        ax.set_title(f'{"a" if budget == 30 else "b"})  {budget} decisions', loc='left', pad=9)
        style(ax)
    fig.legend(handles=[Patch(facecolor=BLUE, label='N1 succeeds, N5 fails'),
                        Patch(facecolor=RED, label='N5 succeeds, N1 fails')],
               loc='upper right', bbox_to_anchor=(.99, 1.02), frameon=False,
               ncol=2, fontsize=7, handletextpad=.4, columnspacing=1.5)
    save(fig, 'browsercode_selection_paired_outcomes')
    return exported


def main():
    data, averaged, groups = F.load()
    summary = json.loads((HERE / 'browsercode_scaling_family_summary.json').read_text())
    points = {row['method']: row for row in averaged['points']}
    assert points == {row['method']: row for row in summary['points']}
    contrasts = {(row['left_method'], row['right_method']): row for row in summary['paired_contrasts']}
    for _, left, right in PAIRS:
        assert abs(100 * (points[left]['success_rate'] - points[right]['success_rate'])
                   - contrasts[left, right]['difference_pp']) < 1e-10
    interaction_view(points, contrasts)
    outcomes = paired_outcomes_view(data, groups)
    for row in outcomes:
        if row['population'] == 'Weighted mean':
            pair = next((left, right) for budget, left, right in PAIRS if budget == row['decision_limit'])
            assert abs(row['net_gain_pp'] - contrasts[pair]['difference_pp']) < 1e-10
    report = dict(schema='browsercode-selection-strength-variants-v1',
                  source_sha256={name: sha(HERE / name) for name in ['aggregate_common_valid.json',
                      'browsercode_scaling_family_pairs.json', 'browsercode_scaling_family_summary.json']},
                  renderer_sha256=sha(__file__), paired_outcomes=outcomes,
                  population='Same 744 all-method common-valid tasks: OM2W 226, WebVoyager 387, DeepShop 131',
                  weighting='Fixed planned benchmark weights 300/1045, 595/1045, 150/1045; weighted mean is not pooled task frequency',
                  intervals='Reused 50,000-draw pointwise paired task-bootstrap intervals; no new uncertainty calculations',
                  batch_scope='Selected nonempty BrowserCode executions, not individual browser operations; excludes unselected proposals and model calls',
                  interpretation='Selection includes candidate generation and selection jointly; live-window and protocol differences remain. No random-selection/equal-generation control isolates selector quality.',
                  efficiency='Browser batch counts are not browser-resource measurements: no CPU/RAM, session-time, traffic or billed-resource saving is inferred. N5 costs more actor dollars and time than its same-horizon N1 baseline; this view is supplementary, not practical-efficiency evidence.')
    (HERE / 'browsercode_selection_strength_variants.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
