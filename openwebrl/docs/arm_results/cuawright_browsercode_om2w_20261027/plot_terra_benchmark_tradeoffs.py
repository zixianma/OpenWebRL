#!/usr/bin/env python3
"""Render four benchmark/harness rows from a public shared-Terra aggregate.

Requires Python and matplotlib. Reads only the aggregate JSON: no credentials,
private trajectories, original campaign directories, API, or scheduler needed.
All methods use high actor effort and the shared Terra terminal-success judge.
Success and actor metrics use each method's valid Terra tasks; valid subsets,
collection windows, actor limits, and native endpoint evidence availability differ.
Crossbenchmark shared-Terra judgments preserve earlier benchmark-native results.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/openwebrl-crossbench-paper-matplotlib')

PROTOCOL = {'model': 'gpt-5.6-terra', 'reasoning_effort': 'medium',
            'max_completion_tokens': 4096, 'seed': 42,
            'version': 'gpt56-terra-terminal-success-v1'}

METHODS = ('serial_high', 'luna_single_high', 'parallel3_select_high', 'parallel5_select_high')

CROSS_METHODS_V1 = ('serial_high', 'luna_single_high', 'parallel5_select_high')
CROSS_METHODS = METHODS

STYLES = {'serial_high': ('Sol single', 's', '#08519C', True),
          'luna_single_high': ('Luna single', 'o', '#08519C', True),
          'parallel3_select_high': ('Luna N3', 'o', '#A50F15', False),
          'parallel5_select_high': ('Luna N5', 'o', '#A50F15', True)}

METRICS = (('mean_model_usd_per_task', 'model_usd_excluding_judge', 'Model cost ($, log)', 'log'),
           ('mean_browser_steps', 'selected_browser_code_batches', 'Browser commands', 'linear'),
           ('mean_task_seconds', 'task_latency_seconds', 'Actor latency (s, log)', 'log'))

def need(condition, message):
    if not condition:
        raise ValueError(message)

def render(data, output_dir, stem='shared_terra_high_four_row_paper'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator, NullFormatter, StrMethodFormatter
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7.5, 'axes.labelsize': 7.5,
                         'axes.titlesize': 8, 'axes.linewidth': .6, 'svg.fonttype': 'none',
                         'svg.hashsalt': 'openwebrl-shared-terra-four-row-v1', 'pdf.fonttype': 42})
    panels = [('online_mind2web', 'browsercode', 'OM2W · BrowserCode'),
              ('online_mind2web', 'native', 'OM2W · Native CUAWright'),
              ('webvoyager', 'browsercode', 'WebVoyager · BrowserCode'),
              ('deepshop', 'browsercode', 'DeepShop · BrowserCode')]
    fig, axes = plt.subplots(4, 3, figsize=(7, 5.5), sharey='row', squeeze=False)
    fig.subplots_adjust(left=.075, right=.815, top=.91, bottom=.085, hspace=.73, wspace=.25)
    for index, (benchmark, harness, title) in enumerate(panels):
        rows = [p for p in data['points'] if p['benchmark'] == benchmark and p['harness'] == harness]
        expected_count = 4 if index < 2 or data['schema'].endswith('-v2') else 3
        need(len(rows) == expected_count, 'Panel method count differs')
        rates = [100 * p['success_rate'] for p in rows]
        low = max(0, 5 * math.floor((min(rates) - .5) / 5))
        high = min(100, 5 * math.ceil((max(rates) + .5) / 5))
        axes[index, 0].set_title(f'({"abcd"[index]}) {title}', loc='left', weight='bold', pad=4)
        for column, (metric, _, label, scale) in enumerate(METRICS):
            ax = axes[index, column]
            ax.set_xscale(scale)
            ax.set_ylim(low, high)
            ax.set_yticks(range(low, high + 1, 5))
            ax.spines[['top', 'right']].set_visible(False)
            ax.tick_params(length=2.5, width=.6, pad=2)
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.xaxis.set_major_formatter(StrMethodFormatter('{x:g}'))
            ticks = {
                ('online_mind2web', 'browsercode'): ([0.01, 0.05, 0.2], [15, 16], [100, 150, 200]),
                ('online_mind2web', 'native'): ([0.1, 0.3, 1], [28, 30, 32], [600, 900, 1200]),
                ('webvoyager', 'browsercode'): ([0.01, 0.05, 0.2], [11, 12, 13], [100, 150, 200]),
                ('deepshop', 'browsercode'): ([0.02, 0.1, 0.4], [18, 20, 22], [150, 200, 300]),
            }
            ax.set_xticks(ticks[(benchmark, harness)][column])
            if column == 0:
                ax.set_ylabel('Success (%)', labelpad=3)
            if index == 3:
                ax.set_xlabel(label, labelpad=2)
            elif index == 0:
                ax.text(.5, 1.33, label, transform=ax.transAxes, ha='center', va='bottom', fontsize=7.5)
            for point in rows:
                need(point[metric] is not None and math.isfinite(point[metric]) and
                     (point[metric] > 0 if scale == 'log' else point[metric] >= 0), 'Missing/nonpositive plotted mean')
                _, marker, color, filled = STYLES[point['method']]
                ax.scatter(point[metric], 100 * point['success_rate'], s=25, marker=marker,
                           edgecolors=color, facecolors=color if filled else 'white', linewidths=.9, zorder=3)
    handles = [Line2D([], [], linestyle='none', marker=marker, markeredgecolor=color,
                      markerfacecolor=color if filled else 'white', markeredgewidth=.9,
                      markersize=4.7, label=label) for label, marker, color, filled in STYLES.values()]
    fig.legend(handles=handles, loc='center left', bbox_to_anchor=(.835, .52), frameon=False,
               fontsize=7.5, handletextpad=.3, labelspacing=.7, borderaxespad=0)
    for extension in ('png', 'svg', 'pdf'):
        metadata = {'Date': None} if extension == 'svg' else ({'CreationDate': None, 'ModDate': None} if extension == 'pdf' else {})
        fig.savefig(output_dir / f'{stem}.{extension}', dpi=300, metadata=metadata)
        if extension == 'svg':
            path = output_dir / f'{stem}.svg'
            path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines()) + '\n')
    plt.close(fig)
    fields = [key for key, value in data['points'][0].items() if not isinstance(value, (dict, list))]
    fields = list(dict.fromkeys(fields + [key for point in data['points'] for key, value in point.items() if not isinstance(value, (dict, list))]))
    with (output_dir / f'{stem}_coordinates.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows({key: value for key, value in point.items() if key in fields} for point in data['points'])

def read_aggregate(path):
    data = json.loads(Path(path).read_text())
    need(data.get('schema') in ('shared-terra-high-four-row-paper-aggregate-v1',
                                'shared-terra-high-four-row-paper-aggregate-v2'), 'Unknown paper aggregate schema')
    need(all(type(data['judge_protocol'].get(key)) is type(value) and data['judge_protocol'][key] == value
             for key, value in PROTOCOL.items()), 'Every point must use the exact shared Terra protocol')
    need(data.get('metric_population') == 'resolved_valid_tasks', 'Actor metrics must use valid Terra tasks')
    points = data['points']
    expected = {('online_mind2web', harness, method) for harness in ('browsercode', 'native') for method in METHODS}
    cross_methods = CROSS_METHODS if data['schema'].endswith('-v2') else CROSS_METHODS_V1
    expected |= {(benchmark, 'browsercode', method) for benchmark in ('webvoyager', 'deepshop') for method in cross_methods}
    actual = [(row['benchmark'], row['harness'], row['method']) for row in points]
    need(type(points) is list and len(points) == len(expected) and len(set(actual)) == len(actual) and set(actual) == expected,
         'Exact high-effort benchmark/harness/method identities required for this schema')
    for point in points:
        requested = 595 if point['benchmark'] == 'webvoyager' else 150 if point['benchmark'] == 'deepshop' else 300
        need(type(point['requested']) is int and point['requested'] == requested, 'Original benchmark roster differs')
        for key in ('original_valid', 'original_invalid_preserved', 'eligible', 'recorded', 'graded_valid',
                    'graded_invalid', 'successes', 'missing_eligible_judgments', 'original_valid_missing_evidence'):
            need(type(point[key]) is int and point[key] >= 0, 'Exact nonnegative disposition counts required: ' + key)
        need(point['original_valid'] + point['original_invalid_preserved'] == requested,
             'Original invalids must remain explicit in the planned roster')
        need(point['eligible'] + point['original_valid_missing_evidence'] == point['original_valid'],
             'Original-valid evidence exclusions must remain explicit')
        need(point['recorded'] == point['eligible'] == point['graded_valid'] + point['graded_invalid']
             and point['missing_eligible_judgments'] == 0, 'All eligible Terra endpoints must have recorded dispositions')
        need(0 < point['graded_valid'] <= point['eligible'] <= requested and point['successes'] <= point['graded_valid'],
             'Inconsistent valid success counts')
        need(type(point['success_rate']) in (int, float) and math.isfinite(point['success_rate']) and
             math.isclose(point['success_rate'], point['successes'] / point['graded_valid'], rel_tol=0, abs_tol=1e-12),
             'Success must use the exact valid Terra denominator')
        need(point.get('metric_population') == 'resolved_valid_tasks', 'Point metric population differs')
        for metric, _, _, scale in METRICS:
            value = point[metric]
            need(type(value) in (int, float) and math.isfinite(value) and (value > 0 if scale == 'log' else value >= 0),
                 'Finite actor means required: ' + metric)
            observed = point[metric + '_observed_tasks']
            missing = point[metric + '_missing_valid_tasks']
            need(type(observed) is int and type(missing) is int and 0 < observed <= point['graded_valid']
                 and missing >= 0 and observed + missing == point['graded_valid'],
                 'Metric-specific valid-task coverage must remain explicit: ' + metric)
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--aggregate', type=Path, default=Path(__file__).with_name('shared_terra_high_four_row_aggregate.json'))
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).parent)
    parser.add_argument('--stem', default='shared_terra_high_four_row_paper')
    args = parser.parse_args()
    need(args.stem and Path(args.stem).name == args.stem and args.stem not in ('.', '..'), 'Output stem must be a filename')
    data = read_aggregate(args.aggregate)
    render(data, args.output_dir, args.stem)
    print(json.dumps({'points': len(data['points']), 'axes': 12, 'width_inches': 7, 'height_inches': 5.5}))


if __name__ == '__main__':
    main()
