"""Summarize the adaptive-compute campaign: success, cost, latency, escalation.

Each (task, arm) pair resolves to its first valid outcome, else its latest invalid
one. Success is valid-only per benchmark; the overall mean weights the three
benchmarks equally. Paired differences use tasks valid in both arms, with a
task-level bootstrap (benchmark-stratified, equal benchmark weights).
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ARMS = ('luna_single_high', 'luna_n5_high', 'adaptive_agree', 'adaptive_self', 'adaptive_self_agree')
BENCHMARKS = ('online-mind2web', 'webvoyager', 'deepshop')
REFERENCES = ('luna_n5_high', 'luna_single_high')


def read(path):
    return json.loads(Path(path).read_text())


def outcomes(root, shards):
    resolved = {}
    for shard in shards:
        for result in sorted((root / 'shards' / shard).glob('attempts/*/episodes/*/result.json')):
            row = read(result)
            key = (row['task_id'], row['mode'])
            decisions = [read(p) for p in sorted((result.parent / 'decisions').glob('*.json'))]
            groups = defaultdict(int)
            for request in (result.parent / 'requests').glob('luna-*.json'):
                groups[read(request).get('decision_group_id')] += 1
            for d in decisions:
                d['_calls'] = groups.get(d['decision_group_id'], 0)
            row['_decisions'] = decisions
            previous = resolved.get(key)
            if previous is None or (not previous['valid'] and (row['valid'] or row['started_epoch'] > previous['started_epoch'])):
                resolved[key] = row
    return resolved


def cost(row):
    return row['usage_accounting']['model_excluding_judge']['charged_or_reserved_usd']


def paired(rows, tasks, arm, reference, draws, seed):
    """Equal-benchmark-weight paired difference in valid-only success over common-valid tasks."""
    groups = {}
    for b in BENCHMARKS:
        ids = [t for t in tasks[b] if rows.get((t, arm), {}).get('valid') and rows.get((t, reference), {}).get('valid')]
        if ids:
            groups[b] = np.array([(rows[(t, arm)]['reward'] == 1) - (rows[(t, reference)]['reward'] == 1) for t in ids], float)
    if not groups:
        return None
    rng = np.random.default_rng(seed)
    point = float(np.mean([g.mean() for g in groups.values()]))
    boot = np.mean([np.array([rng.choice(g, len(g)).mean() for _ in range(draws)]) for g in groups.values()], axis=0)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {'diff_pp': 100 * point, 'ci95_pp': [100 * float(lo), 100 * float(hi)],
            'common_valid': {b: int(len(g)) for b, g in groups.items()},
            'per_benchmark_pp': {b: 100 * float(g.mean()) for b, g in groups.items()}}


def summarize(root, shards, draws=2000, seed=0):
    plan = read(root / 'campaign-plan.json')
    arms = [x['arm'] for x in plan['arms']]
    meta = {t['task_id']: t for t in read(root / 'tasks.json')}
    wanted = {t for s in shards for t in plan['shards'][s]}
    tasks = {b: sorted(t for t in wanted if meta[t]['benchmark'] == b) for b in BENCHMARKS}
    rows = outcomes(root, shards)
    table = {}
    for arm in arms:
        per = {}
        for b in BENCHMARKS:
            recorded = [rows[(t, arm)] for t in tasks[b] if (t, arm) in rows]
            valid = [r for r in recorded if r['valid']]
            decisions = [d for r in valid for d in r['_decisions']]
            calls = [d['_calls'] for d in decisions]
            per[b] = {'planned': len(tasks[b]), 'recorded': len(recorded), 'valid': len(valid), 'invalid': len(recorded) - len(valid),
                      'successes': sum(r['reward'] == 1 for r in valid),
                      'valid_only_success': sum(r['reward'] == 1 for r in valid) / len(valid) if valid else None,
                      'model_usd_per_valid_task': float(np.mean([cost(r) for r in valid])) if valid else None,
                      'latency_s_per_valid_task': float(np.mean([r['task_latency_seconds'] for r in valid if r.get('task_latency_seconds') is not None])) if valid else None,
                      'decisions_per_valid_task': float(np.mean([len(r['_decisions']) for r in valid])) if valid else None,
                      'escalated_decision_share': float(np.mean([bool(d.get('escalated')) for d in decisions])) if decisions else None,
                      'model_calls_per_decision': float(np.mean(calls)) if calls else None,
                      'triggers': {str(k): sum(d.get('escalation_trigger') == k for d in decisions) for k in sorted({d.get('escalation_trigger') for d in decisions}, key=str)}}
        present = [b for b in BENCHMARKS if per[b]['valid']]
        overall = {'mean_valid_only_success': float(np.mean([per[b]['valid_only_success'] for b in present])) if present else None,
                   'mean_overall_success': float(np.mean([per[b]['successes'] / per[b]['planned'] for b in present])) if present else None,
                   'mean_model_usd_per_valid_task': float(np.mean([per[b]['model_usd_per_valid_task'] for b in present])) if present else None,
                   'mean_latency_s': float(np.mean([per[b]['latency_s_per_valid_task'] for b in present])) if present else None}
        table[arm] = {'benchmarks': per, 'overall': overall,
                      'paired': {ref: paired(rows, tasks, arm, ref, draws, seed) for ref in REFERENCES if ref != arm}}
    spend = {}
    for shard in shards:
        budget = root / 'shards' / shard / 'budget'
        for ledger in budget.glob('*/model/ledger.json'):
            spend.setdefault(ledger.parent.parent.name, 0.0)
            spend[ledger.parent.parent.name] += float(read(ledger)['charged_or_reserved_usd'])
        if (budget / 'judge/ledger.json').exists():
            spend['judge'] = spend.get('judge', 0.0) + float(read(budget / 'judge/ledger.json')['charged_or_reserved_usd'])
    return {'shards': shards, 'arms': table, 'all_attempt_spend_usd': spend, 'all_attempt_spend_total_usd': sum(spend.values())}


def markdown(summary):
    lines = ['| Method | OM2W | WebVoyager | DeepShop | Mean overall | Mean valid-only | $ / task | Latency (s) | Escalated decisions | vs N5 (pp, 95% CI) | vs single (pp, 95% CI) |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for arm, row in summary['arms'].items():
        cells = []
        for b in BENCHMARKS:
            x = row['benchmarks'][b]
            cells.append(f"{100 * x['valid_only_success']:.1f}% ({x['successes']}/{x['valid']})" if x['valid'] else '–')
        o = row['overall']
        esc = [row['benchmarks'][b]['escalated_decision_share'] for b in BENCHMARKS if row['benchmarks'][b]['escalated_decision_share'] is not None]
        diffs = []
        for ref in REFERENCES:
            d = row['paired'].get(ref)
            diffs.append(f"{d['diff_pp']:+.1f} [{d['ci95_pp'][0]:+.1f}, {d['ci95_pp'][1]:+.1f}]" if d else '–')
        fmt = lambda v, f: f.format(v) if v is not None else '–'
        lines.append(f"| {arm} | " + ' | '.join(cells) + f" | {fmt(o['mean_overall_success'] and 100 * o['mean_overall_success'], '{:.1f}%')} | {fmt(o['mean_valid_only_success'] and 100 * o['mean_valid_only_success'], '{:.1f}%')} | "
                     f"{fmt(o['mean_model_usd_per_valid_task'], '${:.4f}')} | {fmt(o['mean_latency_s'], '{:.0f}')} | {fmt(np.mean(esc) * 100 if esc else None, '{:.0f}%')} | " + ' | '.join(diffs) + ' |')
    return '\n'.join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--shards', nargs='*')
    p.add_argument('--output', type=Path)
    p.add_argument('--draws', type=int, default=2000)
    a = p.parse_args()
    plan = read(a.root / 'campaign-plan.json')
    shards = a.shards or [s for s in plan['shards'] if s != 'smoke']
    summary = summarize(a.root, shards, a.draws)
    if a.output:
        a.output.write_text(json.dumps(summary, indent=2) + '\n')
    print(markdown(summary))
    print(json.dumps({'all_attempt_spend_usd': summary['all_attempt_spend_usd'], 'total': summary['all_attempt_spend_total_usd']}, indent=1))


if __name__ == '__main__':
    main()
