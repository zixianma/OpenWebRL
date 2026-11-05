#!/usr/bin/env python3
"""Add observed Pareto overlays to copies of the frozen BrowserCode plots.

Reuses the original renderers/layouts and public validators without changing
original plots or inputs. Own-valid and common-valid populations stay separate.
No convex-hull fit, smoothing, uncertainty calculation, model or browser calls.
"""
import argparse
from pathlib import Path
from tempfile import mkdtemp
from unittest.mock import patch
import hashlib
import importlib.util
import json
import math
import os

os.environ.setdefault('MPLCONFIGDIR', '/tmp/openwebrl-pareto-previews-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
METRICS = ('actor_usd', 'browser_commands', 'actor_seconds')
FRONTIER = '#596675'


def module(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


SCALING = module('pareto_scaling', 'plot_browsercode_scaling.py')
AVERAGE = module('pareto_average', 'plot_browsercode_average.py')
COMMON = module('pareto_common', 'plot_browsercode_common_valid.py')
PAPER = module('pareto_paper_style', 'browsercode_pareto_paper_style.py')
SOL60_SOURCE = HERE.parent / 'browsercode_sol_high60_20261104' / 'aggregate.json'
SOL60_METHOD = 'sol_single_high60'
SOL60_STYLE = ('Sol high · 60 steps', 's', '#08519c', False)
SCALING.STYLES[SOL60_METHOD] = SOL60_STYLE
AVERAGE.SCALING.STYLES[SOL60_METHOD] = SOL60_STYLE


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def coordinate(row, metric):
    if metric in row:
        return row[metric]
    value = row['metrics'][metric]
    return value['mean'] if isinstance(value, dict) else value


def sol60_weighted_point():
    """Join the completed own-valid cohort; never insert it in matched plots."""
    data = json.loads(SOL60_SOURCE.read_text())
    assert data['schema'] == 'sol60-full-three-benchmark-sanitized-results-v1'
    assert data['status'] == 'VERIFIED_COVERAGE_COMPLETE'
    assert data['protocol'] == dict(
        actor_deadline_seconds=900, actor_model='gpt-6-sol', candidates=1,
        harness='browsercode', judge='gpt-5.6-terra', judge_effort='medium',
        judge_max_output_tokens=4096, judge_seed=42, max_output_tokens=12288,
        maximum_decisions_advertised=60, maximum_decisions_enforced=60,
        reasoning_effort='high')
    counts, cells, weights = {}, [], []
    for benchmark, (_, planned) in SCALING.BENCHMARKS.items():
        source_name = 'online-mind2web' if benchmark == 'online_mind2web' else benchmark
        cell = data['benchmarks'][source_name]
        assert all(type(cell[k]) is int and cell[k] >= 0 for k in
                   ('planned', 'saved', 'valid', 'invalid', 'successes', 'unclaimed'))
        assert cell['planned'] == cell['saved'] == planned and cell['unclaimed'] == 0
        assert cell['valid'] + cell['invalid'] == planned
        assert 0 <= cell['successes'] <= cell['valid'] and cell['valid'] > 0
        assert math.isclose(cell['valid_only_success'], cell['successes'] / cell['valid'], abs_tol=1e-12)
        for metric in METRICS:
            value = cell['valid_task_efficiency'][metric]
            assert value['n'] == cell['valid']
            assert math.isfinite(value['mean']) and value['mean'] > 0
        assert cell['valid_task_efficiency']['browser_commands']['mean'] <= 60
        counts[benchmark] = dict(requested=planned, native_saved=cell['saved'],
                                 valid=cell['valid'], invalid=cell['invalid'],
                                 successes=cell['successes'], missing_native=0)
        cells.append(cell)
        weights.append(planned)
    for source_key, cell_key in [('planned', 'planned'), ('saved', 'saved'),
                                 ('valid', 'valid'), ('invalid', 'invalid'),
                                 ('successes', 'successes'), ('unclaimed', 'unclaimed')]:
        assert data[source_key] == sum(c[cell_key] for c in cells)
    assert data['planned'] == data['saved'] == sum(weights) == 1045
    return dict(
        method=SOL60_METHOD, benchmark_count=3,
        success_rate=math.fsum(w * c['successes'] / c['valid']
                               for w, c in zip(weights, cells)) / sum(weights),
        metrics={metric: math.fsum(w * c['valid_task_efficiency'][metric]['mean']
                                  for w, c in zip(weights, cells)) / sum(weights)
                 for metric in METRICS},
        benchmark_counts=counts)


def frontier(rows, metric):
    """Lower x and higher success; dominance needs at least one strict gain."""
    selected = []
    for p in rows:
        x, y = coordinate(p, metric), p['success_rate']
        if not any(coordinate(q, metric) <= x and q['success_rate'] >= y
                   and (coordinate(q, metric) < x or q['success_rate'] > y)
                   for q in rows):
            selected.append(p)
    return sorted(selected, key=lambda p: (coordinate(p, metric), -p['success_rate'], p['method']))


def capture(renderer, *args):
    # Suppress the original saves/close; retain its exact axes and method points.
    # Any renderer-produced coordinate sidecars go only to the temporary folder.
    def placeholder_save(_fig, path, *args, **kwargs):
        Path(path).write_bytes(b'')
    with patch.object(Figure, 'savefig', new=placeholder_save), patch.object(plt, 'close'):
        renderer(*args)
        fig = plt.gcf()
    return fig


def overlay(fig, panels, *, method_count=11):
    assert len(fig.axes) == len(panels)
    report = []
    for ax, (population, rows, metric) in zip(fig.axes, panels):
        assert len(ax.collections) == len(rows) == method_count
        assert len({p['method'] for p in rows}) == method_count
        actual = sorted(tuple(map(float, collection.get_offsets()[0])) for collection in ax.collections)
        expected = sorted((coordinate(p, metric), 100 * p['success_rate']) for p in rows)
        assert actual == expected, 'Overlay must match the original plotted population exactly'
        members = frontier(rows, metric)
        xy = sorted({(coordinate(p, metric), 100 * p['success_rate']) for p in members})
        if len(xy) > 1:
            ax.plot([x for x, _ in xy], [y for _, y in xy], linestyle='--',
                    color=FRONTIER, linewidth=.95, zorder=1.8,
                    label='_observed_pareto_frontier')
        # Rings show the frontier members even when only one point survives.
        ax.scatter([x for x, _ in xy], [y for _, y in xy], s=57,
                   facecolors='none', edgecolors=FRONTIER, linewidths=.6, zorder=2.4)
        report.append(dict(population=population, metric=metric, method_count=method_count,
                           frontier=[dict(method=p['method'], x=coordinate(p, metric),
                                          success_rate=p['success_rate']) for p in members],
                           dominated_methods=sorted(p['method'] for p in rows if p not in members)))
    fig.legend(handles=[Line2D([], [], linestyle='--', color=FRONTIER, linewidth=.95,
                              marker='o', markerfacecolor='white', markeredgewidth=.6,
                              markersize=4.4, label='Observed frontier')],
               loc='upper left', bbox_to_anchor=(.777, .985), frameon=False,
               fontsize=6.6, handlelength=1.8, handletextpad=.35, borderaxespad=0)
    return report



# Per-benchmark and own-valid weighted previews share grouped bottom legends.
# Original plots and the common-valid preview retain their existing layouts.
SINGLE_LEGEND = (
    ('luna_single_high', 'Luna baseline'),
    ('luna_single_xhigh_12288', 'Luna xhigh effort'),
    ('serial_high', 'Sol high · 30 steps'),
    ('luna_single_high_6144', 'Luna 0.5× max tokens'),
    ('luna_single_high_24576', 'Luna 2× max tokens'),
    ('luna_single_high_60steps', 'Luna 60 steps'),
    ('luna_single_high_120steps', 'Luna 120 steps'),
)
SELECTION_LEGEND = (
    ('parallel2_select_high', 'N=2 · 30 steps'),
    ('parallel3_select_high', 'N=3 · 30 steps'),
    ('parallel5_select_high', 'N=5 · 30 steps'),
    ('parallel5_select_high_60steps', 'N=5 · 60 steps'),
)


def grouped_bottom_legends(fig, *, weighted=False):
    for legend in list(fig.legends):
        legend.remove()
    # Remove the original side note, whose population/protocol caveats are in
    # the document caption; preserve all axes, coordinates and frontier lines.
    for note in list(fig.texts):
        note.remove()
    if weighted:
        fig.set_size_inches(7, 2.65)
        fig.subplots_adjust(left=.077, right=.985, top=.97, bottom=.45, wspace=.35)
    else:
        fig.set_size_inches(7, 4.7)
        fig.subplots_adjust(left=.077, right=.985, top=.94, bottom=.29,
                            hspace=.60, wspace=.35)
    legend_top = .32 if weighted else .185
    for ax in fig.axes:
        if ax.get_xlabel() in ('Browser-code batches', 'Mean browser-code batches'):
            ax.xaxis.label.set_text('Browser steps')
        for text in ax.texts:
            if text.get_text() == 'Browser-code batches':
                text.set_text('Browser steps')

    def handles(entries):
        result = []
        for method, label in entries:
            _, marker, color, filled = SCALING.STYLES[method]
            result.append(Line2D([], [], linestyle='none', marker=marker,
                                 color=color, markerfacecolor=color if filled else 'white',
                                 markeredgewidth=.9, markersize=4.8, label=label))
        return result

    single_entries = SINGLE_LEGEND
    if weighted:
        single_entries = tuple(row for row in SINGLE_LEGEND if row[0] != 'serial_high')
        single_entries += (('serial_high', 'Sol high · 30 steps'),
                           (SOL60_METHOD, SOL60_STYLE[0]))
    single = fig.legend(handles=handles(single_entries), title='Single-call scaling (N=1)',
                        loc='upper left', bbox_to_anchor=(.077, legend_top), ncol=3,
                        fontsize=6.5, title_fontsize=7.2, frameon=False,
                        handlelength=.8, handletextpad=.4, columnspacing=1.2,
                        labelspacing=.45, borderaxespad=0, alignment='left')
    selection = fig.legend(handles=handles(SELECTION_LEGEND), title='Best-of-N action scaling',
                           loc='upper left', bbox_to_anchor=(.64, legend_top), ncol=2,
                           fontsize=6.5, title_fontsize=7.2, frameon=False,
                           handlelength=.8, handletextpad=.4, columnspacing=1.2,
                           labelspacing=.45, borderaxespad=0, alignment='left')
    fig.legend(handles=[Line2D([], [], linestyle='--', color=FRONTIER, linewidth=.95,
                              marker='o', markerfacecolor='white', markeredgewidth=.6,
                              markersize=4.4, label='Observed frontier')],
               loc='lower left', bbox_to_anchor=(.077, .02), frameon=False,
               fontsize=6.5, handlelength=1.8, handletextpad=.35, borderaxespad=0)
    fig.text(.32, .026,
             'Luna defaults: high effort · max tokens = 12,288 · 30-step limit. N = candidates.',
             fontsize=6.2, color='#555555', va='bottom')
    return single, selection


def save(fig, stem):
    for extension in ('png', 'svg', 'pdf'):
        metadata = {'Date': None} if extension == 'svg' else {'CreationDate': None, 'ModDate': None} if extension == 'pdf' else {}
        path = HERE / f'{stem}.{extension}'
        fig.savefig(path, dpi=300, metadata=metadata)
        if extension == 'svg':
            path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines()) + '\n')
    plt.close(fig)


def build():
    temporary = Path(mkdtemp(prefix='openwebrl-pareto-render-', dir='/tmp'))
    raw = SCALING.validate(json.loads((HERE / 'aggregate.json').read_text()))
    weighted = AVERAGE.average(raw, 'task-count')
    published_weighted = json.loads((HERE / 'browsercode_scaling_weighted_average.json').read_text())
    assert weighted == published_weighted
    weighted = {**weighted, 'points': [*weighted['points'], sol60_weighted_point()]}
    common = COMMON.validate(json.loads((HERE / 'aggregate_common_valid.json').read_text()))
    common_average = COMMON.average(common)
    assert common_average == json.loads((HERE / 'browsercode_scaling_common_valid_weighted_average.json').read_text())
    figures = []
    fig = capture(SCALING.render, raw, temporary, 'source_per_benchmark')
    panels = [(b, [p for p in raw['points'] if p['benchmark'] == b], metric)
              for b in SCALING.BENCHMARKS for metric in METRICS]
    report = overlay(fig, panels)
    grouped_bottom_legends(fig)
    PAPER.apply_style(fig, [SCALING.ALL_METHODS] * len(fig.axes), SCALING.STYLES,
                      {label: method for method, label in SINGLE_LEGEND + SELECTION_LEGEND}, weighted=False)
    figures.append((fig, 'browsercode_scaling_pareto_preview', report))
    fig = capture(AVERAGE.render, weighted, temporary, 'source_weighted')
    panels = [('weighted_own_valid', weighted['points'], metric) for metric in METRICS]
    report = overlay(fig, panels, method_count=12)
    grouped_bottom_legends(fig, weighted=True)
    figures.append((fig, 'browsercode_scaling_weighted_pareto_preview', report))
    fig = capture(COMMON.render, common, temporary, 'source_common_valid')
    panels = [('weighted_common_valid', common_average['points'], metric) for metric in METRICS]
    figures.append((fig, 'browsercode_scaling_common_valid_pareto_preview', overlay(fig, panels)))
    return figures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--per-benchmark-only', action='store_true',
                        help='Save only the per-benchmark preview; preserve weighted figure files')
    parser.add_argument('--grouped-only', action='store_true',
                        help='Save grouped per-benchmark and own-valid weighted previews only')
    parser.add_argument('--weighted-only', action='store_true',
                        help='Save only the own-valid weighted preview')
    args = parser.parse_args()
    figures = build()
    reports = {}
    for fig, stem, panels in figures:
        selected = not args.per_benchmark_only or stem == 'browsercode_scaling_pareto_preview'
        selected = selected and (not args.grouped_only or stem != 'browsercode_scaling_common_valid_pareto_preview')
        selected = selected and (not args.weighted_only or stem == 'browsercode_scaling_weighted_pareto_preview')
        if selected:
            save(fig, stem)
        else:
            plt.close(fig)
        reports[stem] = panels
    report = dict(schema='browsercode-observed-pareto-previews-v1',
                  definition='Minimize the displayed x metric and maximize success. Dominance needs at least one strict improvement. All eleven historical methods compete in every panel; the own-valid weighted preview also includes completed Sol high/60 (twelve methods).',
                  rendering='Dashed straight segments join consecutive observed nondominated coordinates. No smoothing, convex hull, interpolation attainability or statistical superiority is implied. Thin rings identify frontier nodes; a single-node frontier has no segment.',
                  populations='Per-benchmark (eleven methods) and own-valid weighted (twelve methods, including Sol high/60) previews retain each method own-valid cohort; the common-valid weighted preview retains eleven methods and the same 744 valid tasks in every method. Fixed planned weights 300/1045, 595/1045, 150/1045. Sol high/60 is not inserted into the historical matched cohort.',
                  weighted_preview_additions=[dict(point=sol60_weighted_point(),
                      source='../browsercode_sol_high60_20261104/aggregate.json',
                      source_sha256=sha(SOL60_SOURCE),
                      population='Own valid tasks, including valid failures; separate live windows. Actor means exclude Terra. No paired gain or uncertainty inferred.')],
                  batch_interpretation='Browser-code batches are command counts, not CPU/RAM, session time, network traffic or browser-resource costs.',
                  uncertainty='Observed point estimates only; validity differences, live windows and historical prompt differences can change membership. No uncertainty bands or significance inferred.',
                  source_sha256={name: sha(HERE / name) for name in ['aggregate.json', 'aggregate_common_valid.json',
                      'browsercode_scaling_weighted_average.json', 'browsercode_scaling_common_valid_weighted_average.json',
                      'plot_browsercode_scaling.py', 'plot_browsercode_average.py', 'plot_browsercode_common_valid.py',
                      'browsercode_pareto_paper_style.py',
                      '../browsercode_sol_high60_20261104/aggregate.json']},
                  per_benchmark_legends=dict(
                      single_call=[dict(method=m, label=l) for m, l in SINGLE_LEGEND],
                      selection=[dict(method=m, label=l) for m, l in SELECTION_LEGEND],
                      applies_to=['browsercode_scaling_pareto_preview', 'browsercode_scaling_weighted_pareto_preview'],
                      weighted_single_call=[dict(method=m, label=l) for m, l in
                          tuple(row for row in SINGLE_LEGEND if row[0] != 'serial_high') +
                          (('serial_high', 'Sol high · 30 steps'), (SOL60_METHOD, SOL60_STYLE[0]))],
                      placement='Bottom; original point encodings preserved',
                      selection_title='Best-of-N action scaling',
                      browser_step_definition='Executed nonempty BrowserCode batches; each batch can contain multiple browser operations. Setup step limits cap agent decisions.',
                      defaults='Luna high effort, 12,288 output tokens per call, 30 steps; N = candidates',
                      cap_definition='Ceiling on output tokens per call, including reasoning; not realized token use'),
                  per_benchmark_paper_style=dict(hollow='30 steps', filled='60/120 steps',
                      axes_font_pt=9, legend_font_pt=8, title_font_pt=10, footer_font_pt=7.2,
                      horizontal_spacing=.12, grid=False, focus_label='Luna ADM'),
                  renderer_sha256=sha(__file__), previews=reports)
    (HERE / 'browsercode_pareto_previews.json').write_text(json.dumps(report, indent=2) + '\n')
    for stem, panels in reports.items():
        print(stem)
        for panel in panels:
            print(panel['population'], panel['metric'], [p['method'] for p in panel['frontier']])


if __name__ == '__main__':
    main()
