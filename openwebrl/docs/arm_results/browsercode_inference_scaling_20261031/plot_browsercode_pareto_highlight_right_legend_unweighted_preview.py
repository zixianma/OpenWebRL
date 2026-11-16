#!/usr/bin/env python3
"""Equal-benchmark, own-valid version of the sixteen-setting Pareto preview.

Reaggregate all three x metrics and successes/valid separately within each
benchmark, then average benchmarks equally. Preserve method populations, colors,
shapes, fill and cache shading; recompute observed frontiers. Saved aggregates
only; no experiments, task payloads, credentials or API calls.
"""
from pathlib import Path
import copy
import importlib.util
import json
import math
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/openwebrl-unweighted-pareto-mpl')
os.environ.setdefault('XDG_CACHE_HOME', '/tmp/openwebrl-unweighted-pareto-cache')
HERE = Path(__file__).resolve().parent
SOURCE = HERE / 'plot_browsercode_pareto_highlight_right_legend_preview.py'
spec = importlib.util.spec_from_file_location('right_legend_weighted_template', SOURCE)
RIGHT = importlib.util.module_from_spec(spec)
spec.loader.exec_module(RIGHT)
ORIGINAL, BASE = RIGHT.ORIGINAL, RIGHT.ORIGINAL.BASE
STEM = 'browsercode_scaling_unweighted_pareto_highlight_right_legend_preview'
BENCH = tuple(BASE.SCALING.BENCHMARKS)


def equal_points(source_rows):
    raw = BASE.SCALING.validate(json.loads((HERE / 'aggregate.json').read_text()))
    cells = {(p['method'],p['benchmark']): p for p in raw['points']}
    cached = json.loads(ORIGINAL.CACHED_SOURCE.read_text())
    for p in cached['own_valid_points']:
        if p['method'] in ORIGINAL.CACHED:
            cells[p['method'],p['benchmark']] = p
    sol = json.loads(ORIGINAL.SOL_SOURCE.read_text())
    for method in ORIGINAL.SOL:
        for b,p in sol['benchmarks'][method].items():
            cells[method,b] = p
    sol60 = json.loads(BASE.SOL60_SOURCE.read_text())
    for b in BENCH:
        name = 'online-mind2web' if b == 'online_mind2web' else b
        p = sol60['benchmarks'][name]
        cells[BASE.SOL60_METHOD,b] = {**p, 'metrics':p['valid_task_efficiency']}
    rows = copy.deepcopy(source_rows)
    for original, row in zip(source_rows,rows):
        method = row['method']
        selected = {b:cells[method,b] for b in BENCH}
        for b,p in selected.items():
            counts = original['benchmark_counts'][b]
            assert p['valid'] == counts['valid'] > 0
            assert p['successes'] == counts['successes']
            for metric in BASE.METRICS:
                assert math.isfinite(p['metrics'][metric]['mean']) and p['metrics'][metric]['mean'] >= 0
        # Reproduce the source weighted coordinates before changing weights.
        weighted_success = math.fsum(BASE.SCALING.BENCHMARKS[b][1] * p['successes']/p['valid'] for b,p in selected.items())/1045
        assert math.isclose(weighted_success,original['success_rate'],abs_tol=1e-12)
        for metric in BASE.METRICS:
            weighted = math.fsum(BASE.SCALING.BENCHMARKS[b][1]*p['metrics'][metric]['mean'] for b,p in selected.items())/1045
            assert math.isclose(weighted,original['metrics'][metric],abs_tol=1e-12)
        row['success_rate'] = math.fsum(p['successes']/p['valid'] for p in selected.values())/3
        row['metrics'] = {metric:math.fsum(p['metrics'][metric]['mean'] for p in selected.values())/3 for metric in BASE.METRICS}
        row['benchmark_metric_means'] = {b:{m:p['metrics'][m]['mean'] for m in BASE.METRICS} for b,p in selected.items()}
        assert row['benchmark_counts'] == original['benchmark_counts']
    return rows


def build():
    fig, _, source_rows = RIGHT.build()
    frozen = json.loads((HERE / (RIGHT.STEM+'.json')).read_text())
    assert source_rows == frozen['points']
    rows = equal_points(source_rows)
    frontier = []
    for ax, metric in zip(fig.axes,BASE.METRICS):
        assert len(ax.collections) == 16
        for line in list(ax.lines):
            if line.get_label() == '_observed_pareto_frontier':
                line.remove()
        for row,point in zip(rows,ax.collections):
            point.set_offsets([(row['metrics'][metric],100*row['success_rate'])])
        focus = next(r for r in rows if r['method'] == ORIGINAL.FOCUS)
        for text in ax.texts:
            if text.get_text() == 'Luna ADM':
                text.xy = (focus['metrics'][metric],100*focus['success_rate'])
        members = BASE.frontier(rows,metric)
        xy = sorted({(r['metrics'][metric],100*r['success_rate']) for r in members})
        if len(xy)>1:
            ax.plot([x for x,y in xy],[y for x,y in xy],ls='--',color=BASE.FRONTIER,lw=.95,zorder=1.8,label='_observed_pareto_frontier')
        frontier.append(dict(metric=metric,frontier=[dict(method=r['method'],x=r['metrics'][metric],success_rate=r['success_rate']) for r in members]))
        xs = [r['metrics'][metric] for r in rows]
        if metric == 'actor_usd':
            ax.set_xlim(min(xs)/1.15,max(xs)*1.15)
        else:
            pad=max(.5,(max(xs)-min(xs))*.08)
            ax.set_xlim(max(0,min(xs)-pad),max(xs)+pad)
        low=5*math.floor((min(100*r['success_rate'] for r in rows)-1)/5)
        high=5*math.ceil((max(100*r['success_rate'] for r in rows)+1)/5)
        ax.set_ylim(low,high)
        assert len(ax.collections)==16  # no frontier rings
    for text in fig.texts:
        if text.get_text() in ('Success (%)','Mean success (%)'):
            text.set_text('Mean success (%)')
    # Keep the compact original footer. Weighting and validity definitions are
    # stated next to the figure in its caption and recorded in the sidecar.
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    for ax in fig.axes:
        for row,point in zip(rows,ax.collections):
            assert tuple(point.get_offsets()[0])==(row['metrics'][BASE.METRICS[list(fig.axes).index(ax)]],100*row['success_rate'])
    for legend in fig.legends:
        box=legend.get_window_extent(renderer)
        assert not box.overlaps(fig.texts[0].get_window_extent(renderer))
        assert all(not box.overlaps(ax.get_tightbbox(renderer)) for ax in fig.axes)
    return fig,frontier,rows,frozen


def main():
    fig,frontier,rows,source=build()
    BASE.save(fig,STEM)
    report=dict(schema='browsercode-equal-benchmark-own-valid-pareto-v1',
        benchmark_weights={b:'1/3' for b in BENCH},
        success_definition='Arithmetic mean of each benchmark successes / own native-valid tasks; not pooled task success.',
        x_definition='Arithmetic mean of the three within-benchmark own-valid actor means, including valid failures. Cost excludes judge, browser hosting and invalid-attempt spending; latency is elapsed actor-loop time.',
        comparison_caveats=source['comparison_caveats'],points=rows,frontier=frontier,
        styles={**source['styles'], 'frontier':'Recomputed with equal weights for all x and y coordinates; dashed links only, no rings'},source_sha256={str(p.relative_to(HERE.parent)):BASE.sha(p) for p in
            (SOURCE,Path(__file__),HERE/(RIGHT.STEM+'.json'),HERE/'aggregate.json',ORIGINAL.CACHED_SOURCE,ORIGINAL.SOL_SOURCE,BASE.SOL60_SOURCE)},
        output_sha256={ext:BASE.sha(HERE/(STEM+'.'+ext)) for ext in ('png','pdf','svg')})
    (HERE/(STEM+'.json')).write_text(json.dumps(report,indent=2)+'\n')
    print('Verified all sixteen weighted source coordinates; reaggregated success and all x axes with equal benchmark weights.')
    RIGHT.plt.close(fig)

if __name__=='__main__':
    main()
