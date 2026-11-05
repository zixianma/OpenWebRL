#!/usr/bin/env python3
"""Benchmark averages from public BrowserCode scaling aggregates.

Average the three valid-task success rates and the three within-benchmark actor
means separately. Use equal weights or fixed original benchmark task counts.
No task pooling, confidence intervals, runtime imports or paid calls.
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

os.environ.setdefault('MPLCONFIGDIR', '/tmp/openwebrl-scaling-matplotlib')
SOURCE = Path(__file__).with_name('plot_browsercode_scaling.py')
SPEC = importlib.util.spec_from_file_location('_browsercode_scaling_source', SOURCE)
SCALING = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCALING)
METRICS = ('actor_usd', 'browser_commands', 'actor_seconds')


def average(data, weighting='equal'):
    SCALING.validate(data)
    SCALING.need(weighting in ('equal', 'task-count'), 'Unknown benchmark weighting')
    methods = SCALING.profile_methods(data['profile'])
    benchmarks = tuple(SCALING.BENCHMARKS)
    weights = [1 if weighting == 'equal' else SCALING.BENCHMARKS[b][1] for b in benchmarks]
    total_weight = sum(weights)
    lookup = {(p['benchmark'], p['method']): p for p in data['points']}
    result = []
    for method in methods:
        cells = [lookup[b, method] for b in benchmarks]
        for cell in cells:
            for name in METRICS:
                metric = cell['metrics'][name]
                SCALING.need(metric['observed_tasks'] == cell['valid'] and
                             metric['missing_valid_tasks'] == 0,
                             'Every benchmark must have complete valid-task metric coverage')
        result.append({
            'method': method,
            'benchmark_count': len(benchmarks),
            'success_rate': math.fsum(w * p['successes'] / p['valid'] for w, p in zip(weights, cells)) / total_weight,
            'metrics': {name: math.fsum(w * p['metrics'][name]['mean'] for w, p in zip(weights, cells)) / total_weight
                        for name in METRICS},
            'benchmark_counts': {b: {
                'requested': p['requested'], 'native_saved': p.get('native_saved', p['recorded']),
                'valid': p['valid'], 'invalid': p['invalid'], 'successes': p['successes'],
                'missing_native': p.get('claimed_missing_native', 0),
            } for b, p in zip(benchmarks, cells)},
        })
    return {
        'schema': 'browsercode-equal-benchmark-average-v1' if weighting == 'equal' else 'browsercode-task-weighted-benchmark-average-v1',
        'judge_protocol': data['judge_protocol'],
        'harness': 'browsercode_local',
        'benchmark_weights': {b: f'{w}/{total_weight}' for b, w in zip(benchmarks, weights)},
        'success_definition': 'Arithmetic mean of the three successes/native-valid rates; not pooled task success.' if weighting == 'equal' else 'Fixed planned-task-count-weighted mean of the three successes/native-valid rates; not pooled native-valid task success. The same 300/595/150 weights apply to every method.',
        'actor_metric_definition': 'Arithmetic mean of the three within-benchmark valid-task means, including valid failures and excluding Terra.' if weighting == 'equal' else 'Fixed planned-task-count-weighted mean of the three within-benchmark valid-task means, including valid failures and excluding Terra; weights are 300/1045, 595/1045 and 150/1045 for every method.',
        'comparison_caveat': data['comparison_caveat'],
        'missing_outcomes': 'Never imputed; native, valid, invalid and missing counts remain separate for each benchmark.',
        'uncertainty': 'No interval inferred from aggregate means; no matched common-valid effect claimed.',
        'points': result,
    }


def render(data, output_dir, stem):
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

    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': _paper_style.FONT_FAMILY, 'font.size': 7.5,
                         'axes.linewidth': .6, 'svg.fonttype': 'none',
                         'pdf.fonttype': 42, 'svg.hashsalt': 'browsercode-benchmark-average-v1'})
    fig, axes = plt.subplots(1, 3, figsize=(7, 2.55), sharey=True)
    fig.subplots_adjust(left=.075, right=.755, top=.83, bottom=.25, wspace=.35)
    weighted = data['schema'] == 'browsercode-task-weighted-benchmark-average-v1'
    title = 'Task-weighted average' if weighted else 'Equal-weight average'
    fig.suptitle(title + ': OM2W · WebVoyager · DeepShop',
                 x=.075, y=.98, ha='left', fontsize=9)
    labels = ('Mean actor cost ($, log)', 'Mean browser-code batches', 'Mean actor latency (s)')
    rates = [100 * p['success_rate'] for p in data['points']]
    low = max(0, 5 * math.floor((min(rates) - 1) / 5))
    high = min(100, 5 * math.ceil((max(rates) + 1) / 5))
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
        for p in data['points']:
            _, marker, color, filled = SCALING.STYLES[p['method']]
            ax.scatter(p['metrics'][name], 100 * p['success_rate'], s=27,
                       marker=marker, edgecolors=color,
                       facecolors=color if filled else 'white', linewidths=.9, zorder=3)
    axes[0].set_ylabel('Mean valid-only success (%)', labelpad=4, fontsize=7.2)
    handles = [Line2D([], [], linestyle='none', marker=SCALING.STYLES[p['method']][1],
                      color=SCALING.STYLES[p['method']][2],
                      markerfacecolor=SCALING.STYLES[p['method']][2]
                      if SCALING.STYLES[p['method']][3] else 'white',
                      markeredgewidth=.9, markersize=4.8,
                      label=SCALING.STYLES[p['method']][0]) for p in data['points']]
    fig.legend(handles=handles, loc='center left', bbox_to_anchor=(.777, .55),
               frameon=False, fontsize=6.6, handletextpad=.3, labelspacing=.65, borderaxespad=0)
    note = 'Weights: 300/595/150 planned tasks · Terra judged · Own valid subsets differ' if weighted else 'Each benchmark: 1/3 weight · Terra judged · Own valid subsets differ'
    fig.text(.075, .045, note,
             fontsize=7, color='#555555')
    for extension in ('png', 'svg', 'pdf'):
        metadata = {'Date': None} if extension == 'svg' else {'CreationDate': None, 'ModDate': None} if extension == 'pdf' else {}
        path = output_dir / f'{stem}.{extension}'
        fig.savefig(path, dpi=300, metadata=metadata)
        if extension == 'svg':
            path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines()) + '\n')
    plt.close(fig)
    (output_dir / f'{stem}.json').write_text(json.dumps(data, indent=2) + '\n')
    fields = ['method', 'benchmark_count', 'success_rate', *METRICS]
    fields += [f'{b}_{key}' for b in SCALING.BENCHMARKS for key in
               ('requested', 'native_saved', 'valid', 'invalid', 'successes', 'missing_native')]
    with (output_dir / f'{stem}_coordinates.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        for point in data['points']:
            row = {k: point[k] for k in ('method', 'benchmark_count', 'success_rate')}
            row.update(point['metrics'])
            row.update({f'{b}_{key}': value for b, counts in point['benchmark_counts'].items()
                        for key, value in counts.items()})
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--aggregate', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--weighting', choices=('equal', 'task-count'), default='equal',
                        help='Equal benchmark weights, or fixed original task counts 300/595/150')
    parser.add_argument('--stem', help='Default: browsercode_scaling_average, or browsercode_scaling_weighted_average for task-count weighting')
    args = parser.parse_args()
    if args.stem is None:
        args.stem = 'browsercode_scaling_average' if args.weighting == 'equal' else 'browsercode_scaling_weighted_average'
    SCALING.need(args.stem and Path(args.stem).name == args.stem and args.stem not in ('.', '..'),
                 'Output stem must be a filename')
    raw = args.aggregate.read_bytes()
    data = average(json.loads(raw), args.weighting)
    render(data, args.output_dir, args.stem)
    extensions = ('.png', '.svg', '.pdf', '.json', '_coordinates.csv')
    provenance = {
        'schema': 'browsercode-benchmark-average-render-v1',
        'input_sha256': hashlib.sha256(raw).hexdigest(),
        'renderer_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'source_validator_and_styles_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        'matplotlib_version': __import__('matplotlib').__version__,
        'dimensions_inches': [7, 2.55], 'points': len(data['points']),
        'benchmark_weights': data['benchmark_weights'],
        'output_sha256': {ext: hashlib.sha256((args.output_dir / (args.stem + ext)).read_bytes()).hexdigest()
                          for ext in extensions},
    }
    (args.output_dir / f'{args.stem}_render_provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps(provenance))


if __name__ == '__main__':
    main()
