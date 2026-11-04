#!/usr/bin/env python3
"""Add observed Pareto overlays to copies of the frozen BrowserCode plots.

Reuses the original renderers/layouts and public validators without changing
original plots or inputs. Own-valid and common-valid populations stay separate.
No convex-hull fit, smoothing, uncertainty calculation, model or browser calls.
"""
from pathlib import Path
from tempfile import mkdtemp
from unittest.mock import patch
import hashlib
import importlib.util
import json
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


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def coordinate(row, metric):
    if metric in row:
        return row[metric]
    value = row['metrics'][metric]
    return value['mean'] if isinstance(value, dict) else value


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


def overlay(fig, panels):
    assert len(fig.axes) == len(panels)
    report = []
    for ax, (population, rows, metric) in zip(fig.axes, panels):
        assert len(ax.collections) == len(rows) == 11
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
        report.append(dict(population=population, metric=metric,
                           frontier=[dict(method=p['method'], x=coordinate(p, metric),
                                          success_rate=p['success_rate']) for p in members],
                           dominated_methods=sorted(p['method'] for p in rows if p not in members)))
    fig.legend(handles=[Line2D([], [], linestyle='--', color=FRONTIER, linewidth=.95,
                              marker='o', markerfacecolor='white', markeredgewidth=.6,
                              markersize=4.4, label='Observed frontier')],
               loc='upper left', bbox_to_anchor=(.777, .985), frameon=False,
               fontsize=6.6, handlelength=1.8, handletextpad=.35, borderaxespad=0)
    return report


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
    common = COMMON.validate(json.loads((HERE / 'aggregate_common_valid.json').read_text()))
    common_average = COMMON.average(common)
    assert common_average == json.loads((HERE / 'browsercode_scaling_common_valid_weighted_average.json').read_text())
    figures = []
    fig = capture(SCALING.render, raw, temporary, 'source_per_benchmark')
    panels = [(b, [p for p in raw['points'] if p['benchmark'] == b], metric)
              for b in SCALING.BENCHMARKS for metric in METRICS]
    figures.append((fig, 'browsercode_scaling_pareto_preview', overlay(fig, panels)))
    fig = capture(AVERAGE.render, weighted, temporary, 'source_weighted')
    panels = [('weighted_own_valid', weighted['points'], metric) for metric in METRICS]
    figures.append((fig, 'browsercode_scaling_weighted_pareto_preview', overlay(fig, panels)))
    fig = capture(COMMON.render, common, temporary, 'source_common_valid')
    panels = [('weighted_common_valid', common_average['points'], metric) for metric in METRICS]
    figures.append((fig, 'browsercode_scaling_common_valid_pareto_preview', overlay(fig, panels)))
    return figures


def main():
    figures = build()
    reports = {}
    for fig, stem, panels in figures:
        save(fig, stem)
        reports[stem] = panels
    report = dict(schema='browsercode-observed-pareto-previews-v1',
                  definition='Minimize the displayed x metric and maximize success. Dominance needs at least one strict improvement. All eleven methods compete in every panel.',
                  rendering='Dashed straight segments join consecutive observed nondominated coordinates. No smoothing, convex hull, interpolation attainability or statistical superiority is implied. Thin rings identify frontier nodes; a single-node frontier has no segment.',
                  populations='Per-benchmark and own-valid weighted previews retain each method own-valid cohort; common-valid weighted preview uses the same 744 valid tasks in every method. Fixed planned weights 300/1045, 595/1045, 150/1045.',
                  batch_interpretation='Browser-code batches are command counts, not CPU/RAM, session time, network traffic or browser-resource costs.',
                  uncertainty='Observed point estimates only; validity differences, live windows and historical prompt differences can change membership. No uncertainty bands or significance inferred.',
                  source_sha256={name: sha(HERE / name) for name in ['aggregate.json', 'aggregate_common_valid.json',
                      'browsercode_scaling_weighted_average.json', 'browsercode_scaling_common_valid_weighted_average.json',
                      'plot_browsercode_scaling.py', 'plot_browsercode_average.py', 'plot_browsercode_common_valid.py']},
                  renderer_sha256=sha(__file__), previews=reports)
    (HERE / 'browsercode_pareto_previews.json').write_text(json.dumps(report, indent=2) + '\n')
    for stem, panels in reports.items():
        print(stem)
        for panel in panels:
            print(panel['population'], panel['metric'], [p['method'] for p in panel['frontier']])


if __name__ == '__main__':
    main()
