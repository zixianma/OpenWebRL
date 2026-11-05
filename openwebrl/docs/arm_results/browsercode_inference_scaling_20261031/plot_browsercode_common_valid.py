#!/usr/bin/env python3
"""Render the all-method common-valid alternative from sanitized aggregates.

The same task intersection supplies success and every actor metric for all
eleven methods within each benchmark. Benchmark weights stay 300/595/150,
matching the original task-weighted figure. No task payloads or paid calls.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/openwebrl-common-valid-matplotlib')
HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location('_common_valid_styles', HERE/'plot_browsercode_scaling.py')
SCALING = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCALING)
METRICS = ('actor_usd', 'browser_commands', 'actor_seconds')
POPULATION = 'same_all_method_common_valid_tasks_within_each_benchmark'
CAVEAT = ('Common-valid conditioning changes the task mix; live collection windows and protocols differ. '
          'Pairing task identity does not remove these differences.')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate(data):
    need = SCALING.need
    keys = {'schema', 'methods', 'judge_protocol', 'metric_population', 'benchmark_counts',
            'source_aggregate_sha256', 'task_reconstruction_sha256', 'comparison_caveat',
            'prompt_limit_caveat', 'browser_command_definition', 'points'}
    need(type(data) is dict and set(data) == keys, 'Only documented aggregate fields are allowed')
    need(data['schema'] == 'browsercode-common-valid-scaling-v1', 'Unknown common-valid schema')
    need(data['methods'] == list(SCALING.ALL_METHODS), 'Intersection must include all eleven plotted methods')
    need(data['judge_protocol'] == SCALING.PROTOCOL, 'Shared Terra protocol required')
    need(data['metric_population'] == POPULATION and data['comparison_caveat'] == CAVEAT,
         'Explicit intersection population and comparability caveat required')
    need(data['browser_command_definition'] == SCALING.COMMAND_DEFINITION,
         'Selected browser-code batch definition differs')
    original = json.loads((HERE/'aggregate.json').read_text())
    SCALING.validate(original)
    need(data['source_aggregate_sha256'] == sha(HERE/'aggregate.json'),
         'Common-valid alternative must bind the exact original figure aggregate')
    need(data['prompt_limit_caveat'] == original['extension_metadata']['prompt_limit_caveat'],
         'Original prompt/decision comparability caveat required')
    need(SCALING.digest(data['task_reconstruction_sha256']), 'Task reconstruction evidence hash required')
    counts = data['benchmark_counts']
    need(type(counts) is dict and set(counts) == set(SCALING.BENCHMARKS), 'Three benchmark counts required')
    for benchmark, (_, planned) in SCALING.BENCHMARKS.items():
        row = counts[benchmark]
        need(type(row) is dict and set(row) == {'planned', 'common_valid', 'excluded'},
             'Keep planned, common-valid and excluded counts separate')
        need(all(SCALING.integer(v) for v in row.values()) and row['planned'] == planned and
             row['common_valid'] > 0 and row['common_valid'] + row['excluded'] == planned,
             'Intersection coverage counts disagree')
    expected = {(b, m) for b in SCALING.BENCHMARKS for m in SCALING.ALL_METHODS}
    original_points = {(p['benchmark'], p['method']): p for p in original['points']}
    points = data['points']
    need(type(points) is list and len(points) == len(expected), 'All 33 common-valid points required')
    identities = [(p.get('benchmark'), p.get('method')) for p in points]
    need(len(set(identities)) == len(expected) and set(identities) == expected,
         'Unique complete benchmark/method matrix required')
    for p in points:
        need(set(p) == {'benchmark', 'method', 'common_valid', 'successes', 'success_rate', 'metrics'},
             'Only aggregate point fields are allowed')
        n = counts[p['benchmark']]['common_valid']
        source = original_points[p['benchmark'], p['method']]
        need(type(p['common_valid']) is int and p['common_valid'] == n and
             SCALING.integer(p['successes']) and p['successes'] <= n,
             'Every method must use the same benchmark intersection')
        need(n <= source['valid'] and p['successes'] <= source['successes'] and
             n-p['successes'] <= source['valid']-source['successes'],
             'Intersection successes/failures must fit the original valid cohort')
        need(SCALING.number(p['success_rate']) and abs(p['success_rate']-p['successes']/n) < 1e-12,
             'Success denominator differs from the common-valid population')
        need(type(p['metrics']) is dict and set(p['metrics']) == set(METRICS), 'Three actor metrics required')
        limit = 120 if p['method'] == 'luna_single_high_120steps' else 60 if p['method'].endswith('60steps') else 30
        for name, metric in p['metrics'].items():
            need(type(metric) is dict and set(metric) == {'mean', 'observed_tasks', 'missing_valid_tasks'},
                 'Metric mean and exact coverage required')
            need(type(metric['observed_tasks']) is int and metric['observed_tasks'] == n and
                 type(metric['missing_valid_tasks']) is int and metric['missing_valid_tasks'] == 0,
                 'Every metric must cover every task in the same intersection')
            need(SCALING.number(metric['mean']) and metric['mean'] >= 0,
                 'Missing measurements cannot be imputed')
            if name in ('actor_usd', 'actor_seconds'):
                need(metric['mean'] > 0, 'Positive observed actor cost and latency required')
            if name == 'browser_commands':
                need(metric['mean'] <= limit, 'Selected batches exceed the decision limit')
    return data


def average(data):
    validate(data)
    weights = {b: n for b, (_, n) in SCALING.BENCHMARKS.items()}
    total = sum(weights.values())
    lookup = {(p['benchmark'], p['method']): p for p in data['points']}
    points = []
    for method in SCALING.ALL_METHODS:
        cells = [lookup[b, method] for b in weights]
        points.append({'method': method,
            'success_rate': math.fsum(w*p['success_rate'] for w, p in zip(weights.values(), cells))/total,
            **{name: math.fsum(w*p['metrics'][name]['mean'] for w, p in zip(weights.values(), cells))/total
               for name in METRICS}})
    return {'schema': 'browsercode-common-valid-task-weighted-average-v1',
            'judge_protocol': data['judge_protocol'],
            'metric_population': POPULATION, 'benchmark_counts': data['benchmark_counts'],
            'benchmark_weights': {b: f'{w}/{total}' for b, w in weights.items()},
            'weighting_definition': 'Fixed original planned benchmark counts, applied identically to success and all actor means. Not a pooled common-valid task rate.',
            'comparison_caveat': CAVEAT, 'prompt_limit_caveat': data['prompt_limit_caveat'],
            'uncertainty': 'Descriptive point estimates; no confidence intervals or superiority claims inferred.',
            'points': points}


def csv_write(path, rows, fields):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def render(data, output, stem):
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

    result = average(data)
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': _paper_style.FONT_FAMILY, 'font.size': 7.5,
        'axes.linewidth': .6, 'svg.fonttype': 'none', 'pdf.fonttype': 42,
        'svg.hashsalt': 'browsercode-common-valid-average-v1'})
    fig, axes = plt.subplots(1, 3, figsize=(7, 2.55), sharey=True)
    fig.subplots_adjust(left=.075, right=.755, top=.83, bottom=.29, wspace=.35)
    fig.suptitle('Common-valid tasks · task-weighted average', x=.075, y=.98, ha='left', fontsize=9)
    rates = [100*p['success_rate'] for p in result['points']]
    low = max(0, 5*math.floor((min(rates)-1)/5))
    high = min(100, 5*math.ceil((max(rates)+1)/5))
    labels = ('Mean actor cost ($, log)', 'Mean browser-code batches', 'Mean actor latency (s)')
    for ax, name, label in zip(axes, METRICS, labels):
        ax.set_ylim(low, high)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(length=2.5, width=.6, pad=2)
        ax.grid(axis='y', linewidth=.35, color='#dddddd', zorder=0)
        if name == 'actor_usd':
            ax.set_xscale('log')
        else:
            ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
        ax.xaxis.set_major_formatter(StrMethodFormatter('{x:g}'))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel(label, labelpad=4, fontsize=7.2)
        for p in result['points']:
            _, marker, color, filled = SCALING.STYLES[p['method']]
            ax.scatter(p[name], 100*p['success_rate'], s=27, marker=marker,
                edgecolors=color, facecolors=color if filled else 'white', linewidths=.9, zorder=3)
    axes[0].set_ylabel('Mean common-valid success (%)', labelpad=4, fontsize=7.2)
    handles = [Line2D([], [], linestyle='none', marker=SCALING.STYLES[m][1],
        color=SCALING.STYLES[m][2], markerfacecolor=SCALING.STYLES[m][2]
        if SCALING.STYLES[m][3] else 'white', markeredgewidth=.9, markersize=4.8,
        label=SCALING.STYLES[m][0]) for m in SCALING.ALL_METHODS]
    fig.legend(handles=handles, loc='center left', bbox_to_anchor=(.777, .55), frameon=False,
        fontsize=6.6, handletextpad=.3, labelspacing=.65, borderaxespad=0)
    coverage = ' · '.join(f'{label}: {data["benchmark_counts"][b]["common_valid"]}'
                          for b, (label, _) in SCALING.BENCHMARKS.items())
    fig.text(.075, .095, 'Valid in all 11 methods — '+coverage, fontsize=7, color='#555555')
    fig.text(.075, .035, 'Planned weights: 300/595/150 · Terra judged · Actor metrics exclude Terra',
             fontsize=7, color='#555555')
    for extension in ('png', 'svg', 'pdf'):
        metadata = {'Date': None} if extension == 'svg' else {'CreationDate': None, 'ModDate': None} if extension == 'pdf' else {}
        path = output/f'{stem}.{extension}'
        fig.savefig(path, dpi=300, metadata=metadata)
        if extension == 'svg':
            path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
    plt.close(fig)
    (output/f'{stem}.json').write_text(json.dumps(result, indent=2)+'\n')
    csv_write(output/f'{stem}_coordinates.csv', result['points'], ['method', 'success_rate', *METRICS])
    rows = [{'benchmark':p['benchmark'], 'method':p['method'], 'common_valid':p['common_valid'],
             'successes':p['successes'], 'success_rate':p['success_rate'],
             **{name:p['metrics'][name]['mean'] for name in METRICS}} for p in data['points']]
    csv_write(output/f'{stem}_per_benchmark_coordinates.csv', rows,
              ['benchmark', 'method', 'common_valid', 'successes', 'success_rate', *METRICS])
    csv_write(output/f'{stem}_benchmark_counts.csv',
              [{'benchmark':b, **counts} for b, counts in data['benchmark_counts'].items()],
              ['benchmark', 'planned', 'common_valid', 'excluded'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--aggregate', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--stem', default='browsercode_scaling_common_valid_weighted_average')
    args = parser.parse_args()
    SCALING.need(args.stem and Path(args.stem).name == args.stem and args.stem not in ('.', '..'),
                 'Output stem must be a filename')
    raw = args.aggregate.read_bytes()
    data = json.loads(raw)
    render(data, args.output_dir, args.stem)
    suffixes = ('.png', '.svg', '.pdf', '.json', '_coordinates.csv',
                '_per_benchmark_coordinates.csv', '_benchmark_counts.csv')
    provenance = {'schema':'browsercode-common-valid-average-render-v1',
        'input_sha256':hashlib.sha256(raw).hexdigest(), 'renderer_sha256':sha(__file__),
        'source_styles_and_validator_sha256':sha(HERE/'plot_browsercode_scaling.py'),
        'matplotlib_version':__import__('matplotlib').__version__, 'dimensions_inches':[7, 2.55],
        'methods':11, 'per_benchmark_points':33, 'benchmark_counts':data['benchmark_counts'],
        'output_sha256':{s:sha(args.output_dir/(args.stem+s)) for s in suffixes}}
    (args.output_dir/f'{args.stem}_render_provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    print(json.dumps(provenance))


if __name__ == '__main__':
    main()
