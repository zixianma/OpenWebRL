#!/usr/bin/env python3
"""Plot selector design targets from existing aggregate evidence; no inference."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / 'openwebrl/docs/arm_results/inference_interaction_tradeoff_20261009'


def frontier(points):
    """Upper concave envelope: independent mixtures match expected cost."""
    candidates = []
    for x, y in sorted(points, key=lambda p: (p[0], -p[1])):
        if candidates and (x == candidates[-1][0] or y <= candidates[-1][1]):
            continue
        candidates.append((x, y))
    hull = []
    for point in candidates:
        while len(hull) > 1:
            a, b = hull[-2:]
            if (b[1]-a[1])*(point[0]-b[0]) >= (point[1]-b[1])*(b[0]-a[0]):
                break
            hull.pop()
        hull.append(point)
    return hull


def cost_at_success(points, success):
    hull = frontier(points)
    for (x0, y0), (x1, y1) in zip(hull, hull[1:]):
        if y0 <= success <= y1:
            return x0 + (x1-x0)*(success-y0)/(y1-y0)
    raise ValueError('Target outside measured success range; no extrapolation')


def analyze(data):
    c = data['corrected']
    success = c['guided']['success']['mean']
    targets = {}
    for metric in ('decoder_observed_kv_flops', 'decoder_cold_flops', 'browser_step_calls'):
        current = c['guided']['costs'][metric]['mean']
        targets[metric] = {'current': current}
        for mode in ('run_all_costs', 'oracle_stop_costs'):
            points = [(a[mode][metric]['mean'], a['success']['mean']) for a in c['actor']]
            cost = cost_at_success(points, success)
            targets[metric][mode] = {'break_even_at_current_success': cost,
                                   'fraction_reduction_from_current': 1-cost/current,
                                   'measured_retry_envelope': frontier(points)}
    return {
        'corrected_tasks': c['tasks'], 'repetitions_per_method': 3,
        'current_success': success, 'targets': targets,
        'largest_measured_retry_k': 3,
        'success_of_largest_measured_retry_k': c['actor'][-1]['success']['mean'],
        'interpretation': 'Point-estimate targets versus measured policies, not a global frontier or significance test.',
        'mixtures': 'Choose k independently of task and outcome; expected costs, not hard per-task budgets. No skip policy included.',
        'compute': 'All actor proposals plus selector decoder prefill/decode. Vision and unknown failed-request work excluded.',
        'oracle_stop': 'Free perfect terminal-success recognition. Saved additive receipts, not remeasured cache workloads.',
        'new_model_calls': 0, 'new_browser_calls': 0, 'new_judge_calls': 0,
    }


def render(data, result, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    c = data['corrected']
    blue, teal, orange, green = '#315c96', '#168b8b', '#c45628', '#21804a'
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10.5,
                         'svg.fonttype': 'none', 'axes.spines.top': False,
                         'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.8), sharey=True)
    modes = [('run_all_costs', blue, '-'), ('oracle_stop_costs', teal, '--')]
    current_y = c['guided']['success']['mean']*100
    for ax, metric, scale in zip(axes, ('decoder_observed_kv_flops', 'browser_step_calls'), (1e15, 1)):
        curves = {}
        for mode, color, style in modes:
            xs = [a[mode][metric]['mean']/scale for a in c['actor']]
            ys = [a['success']['mean']*100 for a in c['actor']]
            curves[mode] = list(zip(xs, ys))
            ax.plot(xs, ys, style, color=color, linewidth=2, marker='o', markersize=6, zorder=4)
            for a, x, y in zip(c['actor'], xs, ys):
                if mode == 'oracle_stop_costs' and a['k'] == 1:
                    continue
                lo, hi = a['success']['task_bootstrap95']
                ax.errorbar(x, y, yerr=[[y-lo*100], [hi*100-y]], fmt='none',
                            color=color, capsize=3, alpha=.5, zorder=2)
                if mode == 'run_all_costs':
                    ax.annotate('k='+str(a['k']), (x, y), xytext=(7, 6),
                                textcoords='offset points', color=color, fontsize=10)
        current_x = c['guided']['costs'][metric]['mean']/scale
        ci = c['guided']['success']['task_bootstrap95']
        ax.errorbar(current_x, current_y, yerr=[[current_y-ci[0]*100], [ci[1]*100-current_y]],
                    fmt='D', color=orange, capsize=4, markersize=9, zorder=7)
        ax.set_ylim(26, 61)
        ax.grid(axis='both', alpha=.15)
        if metric == 'decoder_observed_kv_flops':
            ax.set_xlim(.28, 2.12)
            hull = frontier(curves['oracle_stop_costs'])
            hx, hy = zip(*hull)
            ax.fill_between(hx, hy, 61, color=green, alpha=.08)
            max_measured_cost = curves['run_all_costs'][-1][0]
            ax.axvspan(max_measured_cost, 2.12, color='#8a929c', alpha=.055)
            ax.text(.315, 59.6, 'Above both measured retry curves', color=green, fontsize=10)
            ax.text(1.22, 28.0, 'Beyond measured retry budgets:\nhigher k is unknown', color='#606874', fontsize=10)
            costs = result['targets'][metric]
            all_target = costs['run_all_costs']['break_even_at_current_success']/scale
            stop_target = costs['oracle_stop_costs']['break_even_at_current_success']/scale
            ax.plot([stop_target, current_x], [current_y, current_y], ':', color=orange, alpha=.7)
            ax.scatter([all_target, stop_target], [current_y, current_y], marker='o',
                       facecolors='white', edgecolors=[blue, teal], s=85, linewidths=2, zorder=8)
            ax.annotate('Same 38.67% success:\n≤0.620 all-attempt target\n≤0.560 perfect-stop target',
                        (stop_target, current_y), xytext=(.32, 48.2), fontsize=10,
                        arrowprops={'arrowstyle': '->', 'color': '#59616d'},
                        bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': .9})
            ax.annotate('', (all_target+.04, current_y-.45), (current_x-.05, current_y-.45),
                        arrowprops={'arrowstyle': '->', 'color': orange, 'lw': 1.5})
            ax.text(1.12, 35.4, '68–71% less total compute', color=orange, fontsize=10)
            ax.annotate('Current N=5\n1.919; 38.67%', (current_x, current_y),
                        xytext=(1.71, 31.0), color=orange, fontsize=10,
                        arrowprops={'arrowstyle': '-', 'color': orange})
            ax.annotate('At today’s cost: >48.67% is necessary\nto beat measured k≤3; k>3 remains unknown',
                        (current_x, ys[-1]), xytext=(1.10, 55.2), fontsize=9.5,
                        arrowprops={'arrowstyle': '->', 'color': '#59616d'},
                        bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': .9})
            ax.scatter([current_x], [ys[-1]], marker='^', s=65, facecolors='none',
                       edgecolors='#59616d', zorder=5)
            ax.set_title('Compute frontier: substantial improvement needed', weight='bold', pad=14)
            ax.set_xlabel('Actor + selector decoder work per task (×10¹⁵ FLOPs)\nRecorded actor KV-cache hits; all proposals included')
        else:
            ax.set_xlim(12.4, 47.0)
            known = curves['run_all_costs']+curves['oracle_stop_costs']+[(current_x, current_y)]
            hx, hy = zip(*frontier(known))
            ax.plot(hx, hy, color=green, alpha=.6, linewidth=5, zorder=1)
            ax.fill_between(hx, hy, 61, color=green, alpha=.07)
            ax.text(13.7, 59.6, 'Envelope including current selection + oracle stopping',
                    color=green, fontsize=10)
            ax.annotate('Current N=5\n15.87 calls; 38.67%\nAlready on this measured frontier',
                        (current_x, current_y), xytext=(13.6, 49.7), color=orange, fontsize=10,
                        arrowprops={'arrowstyle': '->', 'color': orange},
                        bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': .9})
            ax.text(25.0, 28.0, 'Compute and browser budgets\nanswer different questions.',
                    color='#606874', fontsize=11)
            ax.set_title('Browser frontier: current selection is competitive', weight='bold', pad=14)
            ax.set_xlabel('Mean browser-step calls per task\nCalls include done/errors; omit reset and observation work')
    axes[0].set_ylabel('Judged success (%)')
    legend = [Line2D([0], [0], color=blue, marker='o', label='Oracle pass@k: all attempts charged'),
              Line2D([0], [0], color=teal, linestyle='--', marker='o', label='Oracle pass@k: free perfect stopping'),
              Line2D([0], [0], color=orange, marker='D', linestyle='none', label='Current pre-action selection, N=5')]
    fig.legend(handles=legend, loc='upper center', bbox_to_anchor=(.5, .902), ncol=3, frameon=False)
    fig.suptitle('Where a new action selector needs to land', fontsize=20, weight='bold', y=.99)
    fig.text(.5, .935, 'Corrected protocol · 300 tasks · three repetitions per method · no new inference', ha='center', fontsize=11)
    fig.text(.5, .07, 'Lines permit task/outcome-independent mixtures: expected budgets, not hard per-task caps. Frontiers use point estimates; bars show 95% task-bootstrap intervals.', ha='center', fontsize=9)
    fig.text(.5, .035, 'Compute excludes vision and unknown failed-request work. Oracle retries have free success recognition. No guarantee beyond measured policies; no safety/latency claim.', ha='center', fontsize=9)
    fig.subplots_adjust(top=.79, bottom=.22, left=.065, right=.985, wspace=.13)
    for ext in ('png', 'svg', 'pdf'):
        fig.savefig(output/f'selector_frontier.{ext}', dpi=180, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=DEFAULT)
    args = parser.parse_args()
    source = args.directory/'aggregate.json'
    data = json.loads(source.read_text())
    result = analyze(data)
    result['aggregate_sha256'] = hashlib.sha256(source.read_bytes()).hexdigest()
    result['plot_source_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (args.directory/'selector_frontier.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    render(data, result, args.directory)
    print(json.dumps(result['targets'], indent=2))


if __name__ == '__main__':
    main()
