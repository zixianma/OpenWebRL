#!/usr/bin/env python3
"""Aggregate-only figures and Markdown. No task payloads, APIs or publication.

Final use requires root --qualification approving the exact aggregate SHA.
--preview permits partial private inputs with a visible preview label.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/openwebrl-claude-tradeoff-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, StrMethodFormatter

METHODS = {
    'opus_single': ('Opus single', '#0072B2', 's'),
    'sonnet_single': ('Sonnet single', '#0072B2', 'o'),
    'sonnet_n3': ('Sonnet N = 3', '#E5B000', 'o'),
    'sonnet_n5': ('Sonnet N = 5', '#D55E00', 'o'),
}
METRICS = (('mean_model_usd', 'Model cost / task ($)'),
           ('mean_browser_batches', 'Browser batches / task'),
           ('mean_task_latency_seconds', 'Task latency (s)'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_inputs(path, preview=False):
    data = json.loads(Path(path).read_text())
    if data.get('metric_checks_passed') is not True or data.get('errors'):
        raise ValueError('Native metric integration checks must pass')
    if not preview and data.get('all_four_arms_300_recorded') is not True:
        raise ValueError('Final plots require all300 outcomes in allfour arms')
    selected = {}
    for scope in ('arm_valid_only', 'four_arm_common_valid'):
        rows = [r for r in data['aggregate_rows'] if r['scope'] == scope]
        if len(rows) != 4 or {r['arm'] for r in rows} != set(METHODS):
            raise ValueError('Exact four methods required')
        for row in rows:
            n, valid, recorded = row['denominator'], row['valid'], row['recorded']
            if type(n) is not int or n <= 0 or type(valid) is not int or valid <= 0 or row['invalid'] + valid != recorded:
                raise ValueError('Invalid outcome counts')
            if not preview and (recorded != 300 or row['fixed_tasks'] != 300):
                raise ValueError('Final300 roster missing')
            if row['successes'] > n or abs(row['success_rate'] - row['successes'] / n) > 1e-12:
                raise ValueError('Success denominator mismatch')
            if scope == 'arm_valid_only' and n != valid:
                raise ValueError('Primary means must use exactly each arm valid cohort')
            for key in ('model_cost_measurements', 'latency_measurements', 'browser_batch_measurements'):
                if row[key] != n:
                    raise ValueError('Incomplete or mismatched metric coverage')
            for key, _ in METRICS:
                if type(row[key]) not in (int, float) or not math.isfinite(row[key]) or row[key] < 0:
                    raise ValueError('Unobserved or nonfinite metric')
        if scope == 'four_arm_common_valid' and len({r['denominator'] for r in rows}) != 1:
            raise ValueError('Common cohort size differs')
        selected[scope] = sorted(rows, key=lambda r: list(METHODS).index(r['arm']))
    return data, selected


def frontier(rows, key):
    """Observed nondominance: lower x and higher success; collapse equal points."""
    selected = []
    for row in sorted(rows, key=lambda r: (r[key], -r['success_rate'])):
        if not selected or row['success_rate'] > selected[-1]['success_rate']:
            selected.append(row)
    return selected


def draw(ax, rows, key, label):
    points = frontier(rows, key)
    if len(points) > 1:
        ax.plot([r[key] for r in points], [100 * r['success_rate'] for r in points],
                color='#777777', linewidth=.7, zorder=1)
    for row in rows:
        name, color, marker = METHODS[row['arm']]
        ax.scatter(row[key], 100 * row['success_rate'], s=33, marker=marker,
                   color=color, edgecolors='white', linewidths=.55, zorder=3, label=name)
    xs = [r[key] for r in rows]
    ys = [100 * r['success_rate'] for r in rows]
    xp = max((max(xs) - min(xs)) * .13, .02 if key == 'mean_model_usd' else .2)
    yp = max((max(ys) - min(ys)) * .16, .8)
    ax.set_xlim(max(0, min(xs) - xp), max(xs) + xp)
    ax.set_ylim(max(0, min(ys) - yp), min(100, max(ys) + yp))
    ax.set_xlabel(label)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    if key == 'mean_model_usd':
        ax.xaxis.set_major_formatter(StrMethodFormatter('${x:.2f}'))
    ax.grid(False)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(length=3, width=.6, pad=2)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color('#555555')
        ax.spines[side].set_linewidth(.6)


def figure(rows, output, stem, preview):
    fig, axes = plt.subplots(1, 3, figsize=(7, 2.12), sharey=True)
    for i, (key, label) in enumerate(METRICS):
        draw(axes[i], rows, key, f'({"abc"[i]}) {label}')
    axes[0].set_ylabel('Valid-only success (%)')
    fig.subplots_adjust(left=.075, right=.985, bottom=.22, top=.73, wspace=.18)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.52, .92),
               ncol=4, frameon=False, fontsize=7.7, handletextpad=.4, columnspacing=1.3)
    fig.suptitle('Claude BrowserCode · OM2W · high effort', fontsize=9, y=1.015)
    if preview:
        fig.text(.5, .03, 'PRIVATE PREVIEW — final audit pending',
                 ha='center', fontsize=7, color='#A00000')
    for ext in ('png', 'svg', 'pdf'):
        fig.savefig(output / (stem + '.' + ext), dpi=300, bbox_inches='tight', pad_inches=.025)
    plt.close(fig)


def table(rows):
    lines = ['| Method | Recorded | Valid | Invalid | Successes | Valid-only success | Model $ / task | Latency (s) | Browser batches |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {METHODS[r['arm']][0]} | {r['recorded']} | {r['valid']} | {r['invalid']} | {r['successes']} | {100*r['success_rate']:.1f}% | {r['mean_model_usd']:.3f} | {r['mean_task_latency_seconds']:.1f} | {r['mean_browser_batches']:.2f} |")
    return '\n'.join(lines)


def markdown(rows, common, preview, invalid_reasons=None):
    n = common[0]['denominator']
    lead = '**PRIVATE PREVIEW: final audit qualification pending; counts reflect recorded outcomes.**\n\n' if preview else ''
    text = lead + '<a id="claude-browsercode-om2w300"></a>\n\n## Claude BrowserCode: high effort on OM2W\n\n' + table(rows)
    text += '''

Successes divide by valid outcomes in each arm; invalid outcomes stay in the separate count column. Model cost, latency and browser batches use exactly that arm's valid cohort, including valid failures. Cohorts can differ between arms and exclude cap-constrained episodes; these conditional rates do not establish a causal model-quality gain.

![Claude BrowserCode performance versus model cost, browser batches and latency](arm_results/claude_browsercode_om2w300_20261028/claude_tradeoffs.png)

*Opus 5.5 and Sonnet 5.5; 12,288 output tokens and 30 decisions. N = 3/5 samples candidates independently and executes only the selected batch. The thin gray line joins observed Pareto points within each panel; it implies neither interpolation nor statistical superiority. Axes are cropped; no uncertainty is inferred from aggregate points. Collections occur on live sites.*

Model cost includes all candidate and selector calls in each included episode, including thinking output and conservative unknown reservations. Native latency measures actor-loop wall time, including online waits and browser execution, excluding browser setup, final judge-only capture, cleanup and external judging. A browser batch is one selected BrowserCode block and can contain multiple browser operations. GPT-5.6 Terra grades terminal success; historical GPT figures using another judge remain separately labeled.

[Coordinates (CSV)](arm_results/claude_browsercode_om2w300_20261028/claude_coordinates.csv) · [Aggregate definitions](arm_results/claude_browsercode_om2w300_20261028/aggregate.json) · [SVG](arm_results/claude_browsercode_om2w300_20261028/claude_tradeoffs.svg) · [PDF](arm_results/claude_browsercode_om2w300_20261028/claude_tradeoffs.pdf) · [Plot source](arm_results/claude_browsercode_om2w300_20261028/plot_claude_tradeoffs.py)

<details>
<summary>Controlled common-valid alternative and accounting scope</summary>

'''
    text += '**Validity and budget exclusions.** Frozen all-attempt caps can deny admission of a full conservative request reservation. Such episodes remain invalid, not judged failures. Shared-halt consequences also remain invalid. Valid-only comparisons can therefore exclude cap-constrained episodes. The common-valid intersection does not remove this conditioning bias; neither view establishes a causal model-quality gain. All invalid and unknown charges remain in the separate all-attempt accounting.\n\n'
    if invalid_reasons:
        text += '| Method | Invalid reason | Count |\n|---|---|---:|\n'
        for arm in METHODS:
            for reason, count in sorted(invalid_reasons.get(arm, {}).items()):
                text += f'| {METHODS[arm][0]} | {reason.replace("_", " ")} | {count} |\n'
        text += '\n'
    text += f'All four methods are valid on {n} common tasks. The following rates and all three metric means use that identical intersection; conditioning on validity excludes invalid outcomes and can alter the task mix.\n\n'
    text += '| Method | Common tasks | Successes | Success | Model $ / task | Latency (s) | Browser batches |\n|---|---:|---:|---:|---:|---:|---:|\n'
    for r in common:
        text += f"| {METHODS[r['arm']][0]} | {n} | {r['successes']} | {100*r['success_rate']:.1f}% | {r['mean_model_usd']:.3f} | {r['mean_task_latency_seconds']:.1f} | {r['mean_browser_batches']:.2f} |\n"
    text += '''
[Common-valid figure (SVG)](arm_results/claude_browsercode_om2w300_20261028/claude_tradeoffs_common_valid.svg). This OM2W-only result does not extend the earlier equal-weight three-benchmark mean. Valid-cohort costs above are distinct from all-attempt campaign spend, which includes every invalid/interrupted prior attempt and remains in the independently audited accounting record. External judge charges are excluded throughout the model-cost plots.

</details>
'''
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--aggregate', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--qualification', type=Path)
    parser.add_argument('--preview', action='store_true')
    args = parser.parse_args()
    data, scopes = read_inputs(args.aggregate, args.preview)
    if not args.preview:
        if args.qualification is None:
            raise ValueError('Final root publication qualification is mandatory')
        qualification = json.loads(args.qualification.read_text())
        if not (qualification.get('approved_for_final_metric_plot_publication') is True
                and qualification.get('aggregate_sha256') == sha(args.aggregate)
                and len(qualification.get('final_audit', {}).get('sha256', '')) == 64
                and qualification.get('final_audit', {}).get('public_description')):
            raise ValueError('Missing exact final audit and aggregate qualification')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'xtick.labelsize': 7,
                        'ytick.labelsize': 7, 'axes.labelpad': 3, 'svg.fonttype': 'none',
                        'pdf.fonttype': 42, 'savefig.facecolor': 'white'})
    prefix = 'preview_' if args.preview else ''
    for scope, rows in scopes.items():
        figure(rows, args.output_dir, prefix + 'claude_tradeoffs' +
               ('_common_valid' if scope != 'arm_valid_only' else ''), args.preview)
    public = {'benchmark': 'Online-Mind2Web', 'definitions': data['definitions'],
              'aggregate_rows': data['aggregate_rows'], 'source_aggregate_sha256': sha(args.aggregate),
              'final_qualified': not args.preview, 'common_valid_tasks': data['common_valid_tasks'],
              'all_four_arms_300_recorded': data['all_four_arms_300_recorded'],
              'metric_checks_passed': data['metric_checks_passed'],
              'invalid_reason_counts': data.get('invalid_reason_counts', {}),
              'invalid_reason_caveat': data.get('invalid_reason_caveat'),
              'no_task_payloads': True}
    (args.output_dir / 'aggregate.json').write_text(json.dumps(public, indent=2) + '\n')
    with (args.output_dir / 'claude_coordinates.csv').open('w', newline='') as stream:
        rows = data['aggregate_rows']
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / 'results-section-proposal.md').write_text(markdown(scopes['arm_valid_only'], scopes['four_arm_common_valid'], args.preview, data.get('invalid_reason_counts')))
    (args.output_dir / 'provenance.json').write_text(json.dumps({
        'input_sha256': sha(args.aggregate), 'plot_script_sha256': sha(__file__),
        'qualification_sha256': sha(args.qualification) if args.qualification else None,
        'private_preview': args.preview,
        'frontier': 'Observed weak-cost/strict-success nondominated points, descriptive only',
        'plot_order': ['model cost', 'selected browser batches', 'native latency'],
    }, indent=2) + '\n')


if __name__ == '__main__':
    main()
