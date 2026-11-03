#!/usr/bin/env python3
"""Separate controlled scaling families; consume only public aggregate counts.

Bootstrap differences resample paired outcomes within each benchmark, then
apply the original fixed 300/595/150 weights. Intervals are pointwise and
conditional on the common-valid cohort; they exclude live-window uncertainty.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import os

os.environ.setdefault('MPLCONFIGDIR', '/tmp/openwebrl-scaling-families-mpl')
HERE = Path(__file__).resolve().parent
OUTPUT = HERE
spec = importlib.util.spec_from_file_location('_common', HERE/'plot_browsercode_common_valid.py')
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)
BASE = 'luna_single_high'
SELECTION = [BASE, 'parallel2_select_high', 'parallel3_select_high', 'parallel5_select_high']
FAMILIES = [
    ('Output cap', ["luna_single_high_6144", BASE, "luna_single_high_24576"], ['0.5×', '1×', '2×']),
    ('Decision limit', [BASE, 'luna_single_high_60steps', 'luna_single_high_120steps'], ['30', '60', '120']),
    ('Candidate count', SELECTION, ['N1', 'N2', 'N3', 'N5']),
]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load():
    data = C.validate(json.loads((HERE/'aggregate_common_valid.json').read_text()))
    paired = json.loads((HERE/'browsercode_scaling_family_pairs.json').read_text())
    assert set(paired) == {'schema', 'aggregate_common_valid_sha256', 'private_reconstruction_sha256', 'population', 'pairs'}
    assert paired['schema'] == 'browsercode-common-valid-paired-contingencies-v1'
    assert paired['aggregate_common_valid_sha256'] == sha(HERE/'aggregate_common_valid.json')
    assert paired['population'] == data['metric_population']
    points = {(p['benchmark'], p['method']): p for p in data['points']}
    groups = {}
    for p in paired['pairs']:
        assert set(p) == {'benchmark', 'left_method', 'right_method', 'common_valid', 'both_success', 'left_only_success', 'right_only_success', 'both_failure'}
        n = data['benchmark_counts'][p['benchmark']]['common_valid']
        counts = [p[k] for k in ('both_failure', 'left_only_success', 'right_only_success', 'both_success')]
        assert all(type(v) is int and v >= 0 for v in counts)
        assert sum(counts) == p['common_valid'] == n
        assert p['both_success'] + p['left_only_success'] == points[p['benchmark'], p['left_method']]['successes']
        assert p['both_success'] + p['right_only_success'] == points[p['benchmark'], p['right_method']]['successes']
        group = groups.setdefault((p['left_method'], p['right_method']), {})
        assert p['benchmark'] not in group
        group[p['benchmark']] = p
    assert all(set(g) == set(data['benchmark_counts']) for g in groups.values())
    return data, C.average(data), groups


def contrasts(data, groups, draws=50000):
    import numpy as np
    weights = {b: n['planned']/1045 for b, n in data['benchmark_counts'].items()}
    result = []
    for (left, right), rows in sorted(groups.items()):
        seed = int.from_bytes(hashlib.sha256((left+'|'+right).encode()).digest()[:8], 'big')
        rng = np.random.default_rng(seed)
        sample = np.zeros(draws)
        point = 0
        for b, p in sorted(rows.items()):
            n = p['common_valid']
            counts = np.array([p['both_failure'], p['left_only_success'], p['right_only_success'], p['both_success']])
            boot = rng.multinomial(n, counts/n, size=draws)
            sample += weights[b]*(boot[:, 1]-boot[:, 2])/n
            point += weights[b]*(p['left_only_success']-p['right_only_success'])/n
        low, high = np.quantile(100*sample, [.025, .975])
        result.append(dict(left_method=left, right_method=right, difference_pp=100*point,
                           pointwise_paired_bootstrap_95_interval_pp=[float(low), float(high)]))
    return result


def theme():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7,
        'axes.linewidth': .6, 'svg.fonttype': 'none', 'pdf.fonttype': 42,
        'svg.hashsalt': 'browsercode-scaling-families-v1'})
    return plt


def style(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', color='#e4e4e4', linewidth=.4, zorder=0)
    ax.tick_params(length=2.5, width=.6, pad=2)


def dot(ax, point, key, method, label=None, offset=(3, 5)):
    _, marker, color, filled = C.SCALING.STYLES[method]
    ax.scatter(point[key], 100*point['success_rate'], s=25, marker=marker,
               edgecolors=color, facecolors=color if filled else 'white', linewidths=.9, zorder=4)
    if label:
        ax.annotate(label, (point[key], 100*point['success_rate']), xytext=offset,
                    textcoords='offset points', fontsize=6.5, color=color)


def save(fig, stem):
    for ext in ('png', 'svg', 'pdf'):
        metadata = {'Date': None} if ext == 'svg' else {'CreationDate': None, 'ModDate': None} if ext == 'pdf' else {}
        fig.savefig(OUTPUT/f'{stem}.{ext}', dpi=300, metadata=metadata)
        if ext == 'svg':
            path = OUTPUT/f'{stem}.{ext}'
            path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')


def render_selection(data, points, paired):
    from matplotlib.lines import Line2D
    from matplotlib.ticker import NullFormatter
    plt = theme()
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.65))
    fig.subplots_adjust(left=.075, right=.86, top=.78, bottom=.30, wspace=.55)
    fig.suptitle('Action selection at a fixed 30-decision limit', x=.075, y=.98, ha='left', fontsize=9)
    fig.text(.075, .87, 'Luna high · 12K output cap · one selected action per decision', fontsize=7.2, color='#555555')
    gain = {p['left_method']: p for p in paired if p['right_method'] == BASE}
    ax = axes[0]
    ax.axhline(0, color='#999999', linewidth=.7)
    ns = [1, 2, 3, 5]
    gains = [0]+[gain[m]['difference_pp'] for m in SELECTION[1:]]
    ax.plot(ns, gains, color='#c94b4b', linewidth=.9, zorder=2)
    for n, m, y in zip(ns, SELECTION, gains):
        color = C.SCALING.STYLES[m][2]
        if m != BASE:
            lo, hi = gain[m]['pointwise_paired_bootstrap_95_interval_pp']
            ax.errorbar(n, y, yerr=[[y-lo], [hi-y]], color=color, capsize=2, linewidth=.8, zorder=3)
        ax.scatter(n, y, s=24, color=color, zorder=4)
    ax.set_xticks(ns)
    ax.set_xlabel('Candidates per decision')
    ax.set_ylabel('Success gain over N1 (pp)')
    ax.set_title('(a) Paired gain with 95% interval', loc='left', fontsize=7.2, pad=8)
    labels = ['N1', 'N2', 'N3', 'N5']
    offsets = [(2, -11), (-8, 6), (-4, -11), (2, 5)]
    for ax, key, title, xlabel in zip(axes[1:], ['actor_usd', 'actor_seconds'],
            ['(b) Cost trade-off', '(c) Latency trade-off'], ['Actor cost ($/task, log)', 'Actor latency (s/task)']):
        ax.plot([points[m][key] for m in SELECTION], [100*points[m]['success_rate'] for m in SELECTION],
                color='#c94b4b', linewidth=.9, zorder=2)
        for m, label, off in zip(SELECTION, labels, offsets):
            dot(ax, points[m], key, m, label, off)
        dot(ax, points['serial_high'], key, 'serial_high', 'Sol', (2, 4))
        ax.set_title(title, loc='left', fontsize=7.2, pad=8)
        ax.set_xlabel(xlabel)
        ax.set_ylim(79, 87)
        ax.set_ylabel('Common-valid success (%)')
        if key == 'actor_usd':
            ax.set_xscale('log'); ax.set_xticks([.01, .05, .2], ['.01', '.05', '.20'])
            ax.xaxis.set_minor_formatter(NullFormatter()); ax.set_xlim(.0065, .32)
        else:
            ax.set_xticks([90, 135, 180]); ax.set_xlim(80, 200)
    for ax in axes:
        style(ax)
    handles = [Line2D([], [], marker='o', linestyle='none', color='#4299c4', label='N1 single'),
               Line2D([], [], marker='o', linestyle='-', color='#b41620', label='Selection'),
               Line2D([], [], marker='s', linestyle='none', color='#125aa0', label='Sol high')]
    fig.legend(handles=handles, loc='center left', bbox_to_anchor=(.87, .52), frameon=False,
               fontsize=6.3, handlelength=1.2, handletextpad=.3, borderaxespad=0)
    fig.text(.075, .12, 'Same 744 common-valid tasks: OM2W 226 · WebVoyager 387 · DeepShop 131', fontsize=6.8, color='#555555')
    fig.text(.075, .045, 'Fixed planned weights 300/595/150 · Terra judged · Actor metrics exclude judging', fontsize=6.8, color='#555555')
    save(fig, 'browsercode_selection_scaling')
    plt.close(fig)


def render_families(data, points):
    from matplotlib.ticker import NullFormatter, MaxNLocator
    plt = theme()
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.2), sharey=True)
    fig.subplots_adjust(left=.085, right=.97, top=.81, bottom=.21, wspace=.38, hspace=.48)
    fig.suptitle('Different scaling controls, compared on the same tasks', x=.085, y=.98, ha='left', fontsize=9)
    fig.text(.085, .90, 'Luna high · separate intended controls · shared cost and latency axes', fontsize=7, color='#555555')
    for col, (name, methods, labels) in enumerate(FAMILIES):
        subtitle = ['N1 · 30 decisions', 'N1 and N5 · 12K cap', '30 decisions · 12K cap'][col]
        axes[0, col].set_title(name+'\n'+subtitle, fontsize=7.6, pad=8)
        for row, key in enumerate(['actor_usd', 'actor_seconds']):
            ax = axes[row, col]
            ax.plot([points[m][key] for m in methods], [100*points[m]['success_rate'] for m in methods],
                    color='#c94b4b' if col == 2 else '#4385b4', linewidth=.9, zorder=2)
            for m, label in zip(methods, labels):
                off = (3, -11) if m == BASE else (3, 4)
                if m == 'parallel3_select_high': off = (-9, -11)
                if m == 'luna_single_high_60steps': off = (3, -11) if row == 0 else (-18, -6)
                dot(ax, points[m], key, m, label, off)
            if col == 1:
                extra = ['parallel5_select_high', 'parallel5_select_high_60steps']
                ax.plot([points[m][key] for m in extra], [100*points[m]['success_rate'] for m in extra],
                        color='#b41620', linewidth=.9, zorder=2)
                for m, label in zip(extra, ['N5/30', 'N5/60']):
                    dot(ax, points[m], key, m, label, (3, 4))
            ax.set_ylim(79, 88)
            style(ax)
            if col == 0: ax.set_ylabel('Common-valid success (%)')
            if row == 0:
                ax.set_xscale('log'); ax.set_xlabel('Actor cost ($/task, log)')
                ax.xaxis.set_minor_formatter(NullFormatter())
                ax.set_xlim(.0065, .085); ax.set_xticks([.01, .03, .06], ['.01', '.03', '.06'])
            else:
                ax.set_xlabel('Actor latency (s/task)'); ax.xaxis.set_major_locator(MaxNLocator(3))
                ax.set_xlim(85, 200); ax.set_xticks([90, 135, 180])
    fig.text(.085, .075, 'Common-valid OM2W 226 / WebVoyager 387 / DeepShop 131 · Fixed planned weights 300/595/150', fontsize=6.8, color='#555555')
    fig.text(.085, .025, 'Historical single/60 enforced 60 but advertised 30 · Live windows differ · Actor metrics exclude judging', fontsize=6.8, color='#555555')
    save(fig, 'browsercode_scaling_by_family')
    plt.close(fig)


def main():
    global OUTPUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=HERE)
    args = parser.parse_args()
    OUTPUT = args.output_dir
    OUTPUT.mkdir(parents=True, exist_ok=True)
    data, average, groups = load()
    paired = contrasts(data, groups)
    points = {p['method']: p for p in average['points']}
    for p in paired:
        expected = 100*(points[p['left_method']]['success_rate']-points[p['right_method']]['success_rate'])
        assert abs(p['difference_pp']-expected) < 1e-10
    render_selection(data, points, paired)
    render_families(data, points)
    summary = dict(schema='browsercode-scaling-family-summary-v1', benchmark_counts=data['benchmark_counts'],
        benchmark_weights=average['benchmark_weights'], points=average['points'], paired_contrasts=paired,
        uncertainty='50000 paired task bootstrap replicates within benchmark; fixed planned weights; pointwise percentile95 intervals without multiplicity correction. Conditional on common validity; task dependence, live-window and protocol changes are not covered.',
        causal_limit='N>1 adds proposals and selection jointly. Without random-selection/equal-generation controls these comparisons do not isolate selector quality.',
        scope='Published historical cohort; active cached reruns, gap repairs and new Sol60/xhigh are not merged into this analysis.')
    (OUTPUT/'browsercode_scaling_family_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    provenance = dict(schema='browsercode-scaling-family-render-v1', renderer_sha256=sha(__file__),
        aggregate_sha256=sha(HERE/'aggregate_common_valid.json'), paired_counts_sha256=sha(HERE/'browsercode_scaling_family_pairs.json'),
        matplotlib_version=__import__('matplotlib').__version__, numpy_version=__import__('numpy').__version__, bootstrap_draws=50000,
        artifacts={p.name:sha(p) for stem in ['browsercode_selection_scaling', 'browsercode_scaling_by_family']
                   for ext in ['png', 'pdf', 'svg'] for p in [OUTPUT/f'{stem}.{ext}']})
    (OUTPUT/'browsercode_scaling_family_render_provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    for p in paired:
        if (p['left_method'], p['right_method']) in [('parallel5_select_high', BASE), ('parallel5_select_high', 'serial_high'), ('parallel5_select_high_60steps', 'luna_single_high_60steps')]:
            print(json.dumps(p))


if __name__ == '__main__':
    main()
