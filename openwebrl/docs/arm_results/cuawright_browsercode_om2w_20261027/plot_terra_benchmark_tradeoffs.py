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
# Shared semantic palette; source measurements and geometry are unchanged.
from pathlib import Path as _PalettePath
import sys as _palette_sys
_palette_sys.path.insert(0, str(next(p for p in _PalettePath(__file__).resolve().parents
                                     if (p / "paper_plot_palette.py").is_file())))
from paper_plot_palette import COLORS as PALETTE, tint as palette_tint
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

STYLES = {'serial_high': ('Sol single', 's', PALETTE["SINGLE_CALL_MAIN"], True),
          'luna_single_high': ('Luna single', 'o', PALETTE["SINGLE_CALL_MAIN"], True),
          'parallel3_select_high': ('Luna N3', 'o', palette_tint("ADM_DEFAULT", .25), True),
          'parallel5_select_high': ('Luna N5', 'o', PALETTE["ADM_DEFAULT"], True)}

METRICS = (('mean_model_usd_per_task', 'model_usd_excluding_judge', 'Model cost ($, log)', 'log'),
           ('mean_browser_steps', 'selected_browser_code_batches', 'Browser batches / steps', 'linear'),
           ('mean_task_seconds', 'task_latency_seconds', 'Actor latency (s, log)', 'log'))

def need(condition, message):
    if not condition:
        raise ValueError(message)

def render(data, output_dir, stem='shared_terra_high_four_row_paper'):
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
    from matplotlib.ticker import MaxNLocator, NullFormatter, StrMethodFormatter
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': "Albert Sans", 'font.size': 7.5, 'axes.labelsize': 7.5,
                         'axes.titlesize': 8, 'axes.linewidth': .6, 'svg.fonttype': 'none',
                         'svg.hashsalt': 'openwebrl-shared-terra-four-row-v1', 'pdf.fonttype': 42})
    panels = [('online_mind2web', 'browsercode', 'OM2W · BrowserCode'),
              ('online_mind2web', 'native', 'OM2W · Native CUAWright · cumulative recovery'),
              ('webvoyager', 'browsercode', 'WebVoyager · BrowserCode'),
              ('deepshop', 'browsercode', 'DeepShop · BrowserCode')]
    fig, axes = plt.subplots(4, 3, figsize=(7, 5.5), sharey='row', squeeze=False)
    fig.subplots_adjust(left=.075, right=.815, top=.91, bottom=.105, hspace=.73, wspace=.25)
    for index, (benchmark, harness, title) in enumerate(panels):
        rows = [p for p in data['points'] if p['benchmark'] == benchmark and p['harness'] == harness]
        expected_count = 4 if index < 2 or data['schema'].endswith(('-v2', '-v3')) else 3
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
                ('online_mind2web', 'native'): ([0.1, 0.3, 1], [30, 40, 50], [800, 1200, 1600]),
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
    if data['schema'].endswith('-v3'):
        native = [p for p in data['points'] if p['harness'] == 'native']
        names = {'serial_high': 'Sol', 'luna_single_high': 'single', 'parallel3_select_high': 'N3', 'parallel5_select_high': 'N5'}
        coverage = 'Native qualified / 300\n' + '\n'.join(f"{names[p['method']]}: {p['graded_valid']}" for p in native)
        fig.text(.835, .23, coverage + '\nN3 / N5 partial', fontsize=6.5, va='top')
        fig.text(.075, .012, 'Native recovered-task cost, browser records and latency include original + recovery attempts; judge fees excluded.\nValid Terra subsets, timing boundaries, record units and collection windows differ.', fontsize=6, va='bottom')
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

def validate_native_recovery(p):
    """Keep historical qualification, claimed exclusions and unclaimed cells distinct."""
    names = ['requested', 'original_valid', 'original_invalid_preserved', 'historical_terra_eligible', 'historical_terra_valid', 'historical_terra_successes', 'original_valid_missing_evidence', 'recovery_requested', 'recovery_actor_claims', 'recovery_actor_dispositions', 'recovery_execution_valid', 'recovery_execution_invalid', 'recovery_endpoint_eligible', 'recovery_endpoint_unqualified', 'recovery_terra_recorded', 'recovery_terra_valid', 'recovery_terra_invalid', 'recovery_successes', 'recovery_valid_from_original_invalid', 'recovery_valid_from_missing_original_evidence', 'eligible', 'recorded', 'graded_valid', 'graded_invalid', 'missing_eligible_judgments', 'ungraded_no_judgment', 'successes', 'recovery_unclaimed', 'recovery_claimed', 'recovery_disposed', 'recovery_original_endpoint_eligible', 'recovery_derived_endpoint_eligible']
    need(all(type(p.get(k)) is int and p[k] >= 0 for k in names), 'Exact native recovery counts required')
    need(p['historical_terra_eligible'] == p['historical_terra_valid']
         and p['historical_terra_valid'] + p['original_valid_missing_evidence'] == p['original_valid'],
         'Historical native evidence population changed')
    need(p['recovery_requested'] == p['original_invalid_preserved'] + p['original_valid_missing_evidence']
         and p['recovery_claimed'] == p['recovery_disposed'] == p['recovery_actor_claims'] == p['recovery_actor_dispositions']
         and p['recovery_claimed'] + p['recovery_unclaimed'] == p['recovery_requested'], 'Native claimed/unclaimed recovery roster differs')
    need(p['recovery_execution_valid'] + p['recovery_execution_invalid'] == p['recovery_claimed']
         and p['recovery_endpoint_eligible'] + p['recovery_endpoint_unqualified'] == p['recovery_execution_valid'],
         'Native recovery execution/endpoint populations differ')
    need(p['recovery_original_endpoint_eligible'] + p['recovery_derived_endpoint_eligible'] == p['recovery_endpoint_eligible']
         and p['recovery_terra_recorded'] == p['recovery_endpoint_eligible'] == p['recovery_terra_valid'] + p['recovery_terra_invalid'],
         'Every native eligible endpoint must have its first Terra disposition')
    need(p['eligible'] == p['historical_terra_valid'] + p['recovery_endpoint_eligible']
         and p['recorded'] == p['historical_terra_valid'] + p['recovery_terra_recorded']
         and p['graded_valid'] == p['historical_terra_valid'] + p['recovery_terra_valid']
         and p['graded_invalid'] == p['recovery_terra_invalid'], 'Native historical/recovery qualification sums differ')
    need(p['historical_terra_successes'] <= p['historical_terra_valid'] and p['recovery_successes'] <= p['recovery_terra_valid']
         and p['successes'] == p['historical_terra_successes'] + p['recovery_successes']
         and p['recovery_valid_from_original_invalid'] + p['recovery_valid_from_missing_original_evidence'] == p['recovery_terra_valid'],
         'Native success/source denominators differ')
    need(type(p.get('resolved_invalid_recorded_tasks')) is int and type(p.get('not_resolved_valid_tasks')) is int
         and p['resolved_invalid_recorded_tasks'] == p['recovery_claimed'] - p['recovery_terra_valid']
         and p['not_resolved_valid_tasks'] == p['requested'] - p['graded_valid'] == p['resolved_invalid_recorded_tasks'] + p['recovery_unclaimed'],
         'Never-rerun native cells cannot be counted as recorded invalids')
    need(type(p.get('recovery_roster_complete')) is bool and p['recovery_roster_complete'] == (p['recovery_unclaimed'] == 0),
         'Native complete-roster flag differs')
    stop = p.get('source_stop_classification')
    need(stop in ('complete_roster', 'planned_cutoff_partial', 'operational_partial')
         and (stop == 'complete_roster') == p['recovery_roster_complete'], 'Native partial stop cannot become complete')

def read_aggregate(path):
    data = json.loads(Path(path).read_text())
    need(data.get('schema') in ('shared-terra-high-four-row-paper-aggregate-v1',
                                'shared-terra-high-four-row-paper-aggregate-v2',
                                'shared-terra-high-four-row-paper-aggregate-v3'), 'Unknown paper aggregate schema')
    need(all(type(data['judge_protocol'].get(key)) is type(value) and data['judge_protocol'][key] == value
             for key, value in PROTOCOL.items()), 'Every point must use the exact shared Terra protocol')
    need(data.get('metric_population') == 'resolved_valid_tasks', 'Actor metrics must use valid Terra tasks')
    points = data['points']
    expected = {('online_mind2web', harness, method) for harness in ('browsercode', 'native') for method in METHODS}
    cross_methods = CROSS_METHODS if data['schema'].endswith(('-v2', '-v3')) else CROSS_METHODS_V1
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
        if data['schema'].endswith('-v3') and point['harness'] == 'native':
            validate_native_recovery(point)
        else:
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
    if data['schema'].endswith('-v3'):
        native = [p for p in points if p['harness'] == 'native']
        summary = data.get('native_recovery_summary', {})
        expected_summary = {
            'requested_recovery_cells': sum(p['recovery_requested'] for p in native),
            'claimed_recovery_cells': sum(p['recovery_claimed'] for p in native),
            'unclaimed_recovery_cells': sum(p['recovery_unclaimed'] for p in native),
            'qualified_historical_labels': sum(p['historical_terra_valid'] for p in native),
            'qualified_standard_recovery_labels': sum(p['recovery_terra_valid'] - p['recovery_derived_endpoint_eligible'] for p in native),
            'qualified_separate_terminal_labels': sum(p['recovery_derived_endpoint_eligible'] for p in native),
            'total_qualified': sum(p['graded_valid'] for p in native),
            'total_planned_task_arm_cells': sum(p['requested'] for p in native),
            'claimed_cells_remaining_unqualified': sum(p['resolved_invalid_recorded_tasks'] for p in native),
            'full379_recovery_complete': all(p['recovery_roster_complete'] for p in native),
            'all_eligible_first_judgments_known': True,
        }
        need(summary == expected_summary and all(type(summary[k]) is type(v) for k, v in expected_summary.items()),
             'Native summary must retain exact recorded/unclaimed and qualification counts')
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
