#!/usr/bin/env python3
"""Verify the complete cohort and plot cost, latency, and token trade-offs."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from openwebrl.luna_qwen_metrics import MODES, LABELS, PRICING, digest, summarize, write_json


def plot(report, destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    colors = dict(zip(MODES, ['#6b7280', '#2563eb', '#7c3aed', '#ea580c']))

    def point(ax, mode, x, xlabel, upper=None):
        row = report['by_mode'][mode]
        y = row['overall'] * 100
        lower_y, upper_y = [v * 100 for v in row['success_95']]
        if x is None:
            return
        ax.errorbar(x, y, yerr=[[y - lower_y], [upper_y - y]], fmt='o', markersize=8,
                    capsize=3, color=colors[mode], label=LABELS[mode])
        if upper is not None and upper != x:
            ax.plot([x, upper], [y, y], color=colors[mode], linewidth=2)
            ax.plot(upper, y, marker='|', color=colors[mode])
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Overall judged success (%)')
        ax.set_ylim(0, 100)
        ax.grid(alpha=.2)

    def save(fig, name, subtitle):
        fig.suptitle(subtitle, fontsize=11)
        fig.tight_layout(rect=(0, .03, 1, .93))
        for extension in ('png', 'svg'):
            fig.savefig(destination / f'{name}.{extension}', dpi=170, bbox_inches='tight')
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    missing = []
    for mode in MODES:
        row = report['by_mode'][mode]
        if row['cost_upper_mean_usd'] is None:
            missing.append(LABELS[mode])
            continue
        lo, hi = row['cost_lower_mean_usd'], row['cost_upper_mean_usd']
        point(ax, mode, lo, 'Estimated model-serving USD per attempted task', upper=hi)
    ax.legend(loc='best', fontsize=9)
    note = f"Local rate: ${report['pricing']['h200_hour']:.2f}/H200-hour; judge and research overhead separate."
    if missing:
        note += '\nUnbounded missing usage: ' + ', '.join(missing)
    fig.text(.5, .005, note, ha='center', fontsize=8)
    save(fig, 'cost', 'Success versus serving cost\nHorizontal spans: cache-price bounds; vertical bars: task-bootstrap 95% intervals')

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for mode in MODES:
        latency = report['by_mode'][mode]['metrics']['episode_seconds']
        point(ax, mode, latency['p50'], 'Agent episode seconds (offline judge excluded)', upper=latency['p95'])
    ax.legend(loc='best', fontsize=9)
    save(fig, 'latency', 'Success versus episode latency\nDot: median; horizontal span: median to p95, not a confidence interval')

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for ax, metric, label in zip(axes, ['model_input_tokens', 'model_output_tokens'], ['Input', 'Output']):
        for mode in MODES:
            row = report['by_mode'][mode]
            if row['unknown_usage_requests']:
                continue
            point(ax, mode, row['metrics'][metric]['mean'], f'{label} tokens per attempted task')
        ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
    axes[0].legend(loc='best', fontsize=8)
    fig.text(.5, .005, 'All candidates counted; tokenizers differ. Subsets counted once. Modes with missing usage are omitted.',
             ha='center', fontsize=8)
    save(fig, 'tokens', 'Success versus recorded model tokens\nProvider/model/role breakdowns are retained in the aggregate report')


def report(root, destination, gpu_rate=None):
    root, destination = Path(root), Path(destination)
    plan = json.loads((root / 'plan.json').read_text())
    records = []
    pricing = dict(plan['pricing'])
    if gpu_rate is not None:
        if gpu_rate < 0:
            raise ValueError('GPU price must be nonnegative')
        pricing['h200_hour'] = gpu_rate
    for path in (root / 'pilot/records').glob('*.json'):
        row = json.loads(path.read_text())
        if row['protocol_sha256'] != digest(plan['protocol']):
            raise ValueError('Mixed protocols')
        if hashlib.sha256(Path(row['artifact']).read_bytes()).hexdigest() != row['artifact_sha256']:
            raise ValueError('Original result was modified')
        from openwebrl.luna_qwen_metrics import episode_usage
        usage = episode_usage(Path(row['artifact']).parent, row['mode'], row['episode_seconds'], pricing)
        for name in ('model_input_tokens', 'model_output_tokens', 'model_usage_missing_requests'):
            if usage[name] != row[name]:
                raise ValueError('Receipt totals disagree with committed episode')
        records.append(dict(row, **{k: v for k, v in usage.items()}))
    aggregate = summarize(records, [r['task_id'] for r in plan['schedule']], pricing)
    aggregate['protocol'] = plan['protocol']
    aggregate['pilot_not_full300'] = len(plan['schedule']) != 300
    physical = []
    missing_timing = 0
    for identity in (root / 'pilot/attempts').glob('*/*/identity.json'):
        measured = identity.parent / 'physical-usage.json'
        if measured.exists():
            physical.append(json.loads(measured.read_text()))
        else:
            # SIGKILL can precede finalization. Preserve its request usage;
            # allocation accounting, not a fabricated episode timer, covers it.
            info = json.loads(identity.read_text())
            physical.append(dict(episode_usage(identity.parent, info['mode'], 0., pricing), interrupted=True))
            missing_timing += 1
    aggregate['physical_collection'] = dict(attempts=len(physical), committed=len(records),
        interrupted=sum(r['interrupted'] for r in physical),
        attempts_missing_final_timing=missing_timing,
        model_input_tokens=sum(r['model_input_tokens'] for r in physical),
        model_output_tokens=sum(r['model_output_tokens'] for r in physical),
        unknown_usage_requests=sum(r['model_usage_missing_requests'] for r in physical))
    aggregate['api_ledgers'] = {role: json.loads((root / f'pilot/{role}-budget/ledger.json').read_text())
                              for role in ('luna', 'judge') if (root / f'pilot/{role}-budget/ledger.json').exists()}
    destination.mkdir(parents=True, exist_ok=True)
    write_json(destination / 'summary.json', aggregate)
    with (destination / 'metrics.csv').open('w', newline='') as stream:
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow(['mode', 'success', 'valid', 'tasks', 'cost_lower_usd', 'cost_upper_usd',
                         'median_seconds', 'p95_seconds', 'mean_input_tokens', 'mean_output_tokens',
                         'mean_browser_steps', 'mean_local_gpu_reserved_seconds', 'mean_qwen_flops_lower', 'mean_qwen_flops_upper'])
        for mode in MODES:
            d = aggregate['by_mode'][mode]
            m = d['metrics']
            writer.writerow([mode, d['overall'], d['valid'], d['episodes'], d['cost_lower_mean_usd'], d['cost_upper_mean_usd'],
                m['episode_seconds']['p50'], m['episode_seconds']['p95'], m['model_input_tokens']['mean'],
                m['model_output_tokens']['mean'], m['browser_steps']['mean'], m['local_gpu_reserved_seconds']['mean'],
                m['qwen_flops_lower']['mean'], m['qwen_flops_upper']['mean']])
    plot(aggregate, destination)
    return aggregate


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--gpu-usd-hour', type=float)
    args = parser.parse_args()
    result = report(args.root, args.output, args.gpu_usd_hour)
    print(json.dumps(dict(complete=result['complete'], tasks=result['scheduled_tasks'], plots=['cost', 'latency', 'tokens'])))
