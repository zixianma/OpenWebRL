#!/usr/bin/env python3
"""Offline comparison of saved retries and pre-action selection; no inference.

Task records and source manifests remain private. Public outputs are aggregates.
Reuses verified cohorts without changing their outcomes or completion pointers.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import sys

for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = '1'
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from openwebrl.arm_inference_flops import decoder_flops
from openwebrl.inference_retry_analysis import summarize_task

RUNTIME = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime')
REPEATS = RUNTIME/'evaluations/selectionarm-piotr-repeats-20261006'
OLD = RUNTIME/'arm-turn-bonus-preparation/controlled-inference-20261004'
BASE = ('browser_steps', 'browser_step_seconds', 'elapsed_seconds',
        'actor_input_tokens', 'actor_output_tokens', 'selector_input_tokens',
        'selector_output_tokens', 'actor_requests', 'selector_calls')
RENAMES = {'browser_steps': 'browser_step_calls',
           'elapsed_seconds': 'recorded_episode_seconds'}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def base_episode(record):
    if type(record['valid']) is not bool:
        raise ValueError('Invalid validity field')
    if record['valid'] and record['reward'] not in (0, 1):
        raise ValueError('Invalid binary outcome')
    costs = {RENAMES.get(k, k): record[k] for k in BASE}
    return dict(success=record['valid'] and record['reward'] == 1, valid=record['valid'], costs=costs)


def corrected_episode(item):
    record_path, actor_config, selector_config = item
    record = read(record_path)
    sources = {str(record_path): sha(record_path)}
    episode = base_episode(record)
    cold = observed = 0
    inp = out = failed = received = retractions = 0
    receipts = list(Path(record['artifact']).parent.glob('actor-request-*.json'))
    if len(receipts) != record['actor_requests']:
        raise ValueError('Actor receipt count mismatch')
    for path in receipts:
        sources[str(path)] = sha(path)
        receipt = read(path)
        if receipt['status'] != 'received':
            failed += 1
            continue
        meta = receipt['meta']
        p, g = meta['prompt_tokens'], meta['completion_tokens']
        cold += decoder_flops(actor_config, p, g)['total']
        observed += decoder_flops(actor_config, p, g, meta['cached_tokens'])['total']
        inp += p
        out += g
        received += 1
        retractions += meta.get('total_retractions', 0)
    if (inp, out) != (record['actor_input_tokens'], record['actor_output_tokens']):
        raise ValueError('Actor receipt tokens do not reconcile')
    sin = sout = calls = panels = proposals = 0
    for entry in record['trace_files']:
        path = Path(entry['path'])
        sources[str(path)] = sha(path)
        if sources[str(path)] != entry['sha256']:
            raise ValueError('Decision trace hash changed')
        for line in path.read_text().splitlines():
            trace = json.loads(line)
            n = 5 if record['mode'] == 'arm' else 1
            if len(trace['candidates']) != n or trace.get('fallback'):
                raise ValueError('Candidate count or selector fallback changed')
            panels += 1
            proposals += n
            if record['mode'] == 'arm':
                telemetry = trace['selector_telemetry']
                p, g = telemetry['input_tokens'], telemetry['output_tokens']
                value = decoder_flops(selector_config, p, g)['total']
                cold += value
                observed += value
                sin += p
                sout += g
                calls += 1
    if (sin, sout, calls) != (record['selector_input_tokens'],
                            record['selector_output_tokens'], record['selector_calls']):
        raise ValueError('Selector trace tokens do not reconcile')
    episode['costs'].update(decoder_cold_flops=cold, decoder_observed_kv_flops=observed)
    audit = dict(received_actor_requests=received, unmetered_failed_actor_requests=failed,
                 retractions=retractions, panels=panels, proposals=proposals,
                 unselected_proposals=proposals-panels)
    return record, episode, sources, audit


def load_corrected():
    completion = read(REPEATS/'completion.json')
    if not completion['verified_complete']:
        raise ValueError('Corrected study not verified complete')
    combined = Path(completion['combined_aggregate_path'])
    if sha(combined) != completion['combined_aggregate_sha256']:
        raise ValueError('Combined audit changed')
    inputs = {str(REPEATS/'completion.json'): sha(REPEATS/'completion.json'),
              str(combined): sha(combined)}
    groups = defaultdict(lambda: {'actor': [], 'guided': [], 'valid': []})
    audit = defaultdict(int)
    roots = [REPEATS/rep for rep in ('repeat-1-seed43', 'repeat-2-seed44')]
    roots.append(RUNTIME/'evaluations/selectionarm-piotr-sameday30-20261007')
    scientific_protocol = None
    for root in roots:
        rep = root.name
        end = read(root/'completion.json')
        aggregate = Path(end.get('aggregate_path', end.get('aggregate')))
        if not end.get('verified_complete', end.get('stage') == 'verified_complete'):
            raise ValueError('Repeat completion is not verified')
        if 'aggregate_sha256' in end and sha(aggregate) != end['aggregate_sha256']:
            raise ValueError('Repeat aggregate changed')
        if 'artifacts_sha256' in end and sha(aggregate) != end['artifacts_sha256']['aggregate.json']:
            raise ValueError('Same-day aggregate changed')
        inputs[str(aggregate)] = sha(aggregate)
        inputs[str(root/'completion.json')] = sha(root/'completion.json')
        report = read(aggregate)
        expected = report.get('cohort', report)['by_mode']
        items = []
        pin_by_path = {}
        for shard in (0, 1):
            plan_path = root/f'shard-{shard}/plan.json'
            plan = read(plan_path)
            inputs[str(plan_path)] = sha(plan_path)
            settings = {k: plan['protocol'][k] for k in ('actor_revision', 'selector_revision',
                'actor_system_policy_sha256', 'temperature', 'top_p', 'max_new_tokens',
                'max_steps', 'context_length', 'context_num_screenshots', 'history',
                'judge', 'judge_protocol', 'judge_response_tokens', 'candidates')}
            if scientific_protocol is None:
                scientific_protocol = settings
            if settings != scientific_protocol:
                raise ValueError('Scientific settings differ across corrected repetitions')
            configs = {}
            for role in ('actor', 'selector'):
                path = Path(plan[role])/'config.json'
                inputs[str(path)] = sha(path)
                configs[role] = read(path)['text_config']
            pin = hashlib.sha256(json.dumps(plan['protocol'], sort_keys=True).encode()).hexdigest()
            for path in (root/f'shard-{shard}/collection/records').glob('*.json'):
                items.append((path, configs['actor'], configs['selector']))
                pin_by_path[str(path)] = pin
        if len(items) != 600:
            raise ValueError('Expected 600 committed records per repeat')
        seen = set()
        sums = defaultdict(lambda: defaultdict(float))
        with ThreadPoolExecutor(max_workers=8) as pool:
            for (record, episode, sources, counts), item in zip(pool.map(corrected_episode, items), items):
                identity = record['task_id'], record['mode']
                if identity in seen or record['protocol_sha256'] != pin_by_path[str(item[0])]:
                    raise ValueError('Duplicate identity or protocol drift')
                seen.add(identity)
                role = 'guided' if record['mode'] == 'arm' else 'actor'
                if record['mode'] not in ('actor0', 'arm'):
                    raise ValueError('Unexpected mode')
                groups[record['task_id']][role].append(episode)
                groups[record['task_id']]['valid'].append(record['valid'])
                inputs.update(sources)
                for k, v in counts.items():
                    audit[k] += v
                sums[record['mode']]['successes'] += episode['success']
                for k, v in episode['costs'].items():
                    sums[record['mode']][k] += v
        for mode, total in sums.items():
            if total['successes'] != expected[mode]['successes']:
                raise ValueError('Existing outcome audit mismatch')
            for metric in ('decoder_cold_flops', 'decoder_observed_kv_flops'):
                if metric in expected[mode].get('mean', {}) and abs(total[metric]/300/expected[mode]['mean'][metric]-1) > 1e-10:
                    raise ValueError('Existing decoder-work audit mismatch')
            for metric in BASE:
                target = expected[mode].get('totals', expected[mode])[metric]
                if abs(sums[mode][RENAMES.get(metric, metric)]-target) > max(1e-7, abs(target)*1e-10):
                    raise ValueError('Existing cost audit mismatch')
        print('Verified corrected repeat:', rep, flush=True)
    if len(groups) != 300 or any(len(g['actor']) != 3 or len(g['guided']) != 3 for g in groups.values()):
        raise ValueError('Corrected cohort coverage changed')
    return groups, inputs, dict(audit)


def load_historical():
    end = read(OLD/'completion.json')
    if not end['complete'] or sha(OLD/'final-report.json') != end['final_report_sha256']:
        raise ValueError('Historical completion audit changed')
    inputs = {str(OLD/n): sha(OLD/n) for n in ('completion.json', 'final-report.json', 'flops-audit.json')}
    groups = defaultdict(lambda: {'actor': [], 'guided': [], 'valid': []})
    metrics = ('cold_request_flops', 'observed_kv_lower_flops', 'observed_kv_upper_flops')
    seen = set()
    for path in sorted(OLD.glob('shard-*/records/*.json')):
        record = read(path)
        ident = record['task_id'], record['mode']
        if ident in seen:
            raise ValueError('Duplicate historical identity')
        seen.add(ident)
        index_path = OLD/'flops-indices'/(record['key']+'.json')
        index = read(index_path)
        if any(index[k] != record[k] for k in ('key', 'task_id', 'mode', 'artifact_sha256')):
            raise ValueError('Historical FLOP index binding changed')
        inputs[str(path)] = sha(path)
        inputs[str(index_path)] = sha(index_path)
        episode = base_episode(record)
        episode['costs'].update({k: index[k] for k in metrics})
        role = 'guided' if record['mode'] == 'arm' else 'actor'
        if record['mode'] not in ('arm', *[f'actor{i}' for i in range(5)]):
            raise ValueError('Unexpected historical mode')
        groups[record['task_id']][role].append(episode)
        groups[record['task_id']]['valid'].append(record['valid'])
    if len(groups) != 300 or any(len(g['actor']) != 5 or len(g['guided']) != 1 for g in groups.values()):
        raise ValueError('Historical cohort coverage changed')
    for mode, role in (('arm', 'guided'), ('actor', 'actor')):
        audit = read(OLD/'flops-audit.json')['curve']
        target = audit['arm'] if mode == 'arm' else audit['actor'][-1]
        for metric in metrics:
            value = sum(sum(e['costs'][metric] for e in g[role]) for g in groups.values())/300
            if abs(value/target[metric]-1) > 1e-10:
                raise ValueError('Historical aggregate FLOPs changed')
    return groups, inputs, read(OLD/'flops-audit.json')['counts']


def summarize(groups, draws):
    import numpy as np
    ids = sorted(groups)
    tasks = [summarize_task(groups[t]['actor'], groups[t]['guided']) for t in ids]
    n = len(tasks)
    weights = np.random.default_rng(20261009).multinomial(n, np.full(n, 1/n), size=draws)/n

    def estimate(values):
        x = np.asarray(values, dtype=float)
        return dict(mean=float(x.mean()), task_bootstrap95=np.quantile(weights@x, [.025, .975]).tolist())

    guided_y = np.array([t['guided']['success'] for t in tasks])
    metrics = list(tasks[0]['guided']['costs'])
    guided_cost = {m: np.array([t['guided']['costs'][m] for t in tasks]) for m in metrics}
    guided = dict(success=estimate(guided_y), costs={m: estimate(v) for m, v in guided_cost.items()})
    actor = []
    for k in range(len(tasks[0]['actor'])):
        y = np.array([t['actor'][k]['success'] for t in tasks])
        row = dict(k=k+1, success=estimate(y), guided_minus_actor=estimate(guided_y-y))
        for treatment in ('run_all_costs', 'oracle_stop_costs'):
            row[treatment] = {}
            row[treatment+'_guided_ratio'] = {}
            for metric in metrics:
                x = np.array([t['actor'][k][treatment][metric] for t in tasks])
                row[treatment][metric] = estimate(x)
                a, b = weights@guided_cost[metric], weights@x
                positive = b > 0
                row[treatment+'_guided_ratio'][metric] = dict(
                    ratio_of_means=float(guided_cost[metric].mean()/x.mean()) if x.mean() else None,
                    task_bootstrap95=np.quantile(a[positive]/b[positive], [.025, .975]).tolist() if positive.any() else None)
        actor.append(row)
    # Equal expected browser budget: randomize k independently of task/outcome.
    # No per-task cap or feasibility claim, and never extrapolate beyond saved k.
    def matched(costs, scores, target):
        if target < costs[0] or target > costs[-1]:
            return None
        for i in range(len(costs)-1):
            if costs[i] <= target <= costs[i+1] and costs[i+1] > costs[i]:
                w = (target-costs[i])/(costs[i+1]-costs[i])
                return scores[i]*(1-w)+scores[i+1]*w, i+1, i+2, w
        return None
    scores = np.array([[a['success'] for a in t['actor']] for t in tasks])
    target = guided_cost['browser_step_calls']
    equal = {}
    for treatment in ('run_all_costs', 'oracle_stop_costs'):
        costs = np.array([[t['actor'][k][treatment]['browser_step_calls'] for k in range(len(actor))] for t in tasks])
        point = matched(costs.mean(0), scores.mean(0), target.mean())
        diffs = []
        for c, s, b, y in zip(weights@costs, weights@scores, weights@target, weights@guided_y):
            value = matched(c, s, b)
            if value is not None:
                diffs.append(y-value[0])
        value = dict(covered=point is not None, covered_bootstrap_draws=len(diffs),
                     interval_scope='Conditional on saved k range covering the resampled mean budget; no extrapolation.')
        if point:
            value.update(actor_success=point[0], lower_k=point[1], upper_k=point[2], upper_k_probability=point[3],
                         guided_minus_actor=float(guided_y.mean()-point[0]),
                         task_bootstrap95=np.quantile(diffs, [.025, .975]).tolist())
        equal[treatment] = value
    all_valid = [i for i, t in enumerate(ids) if all(groups[t]['valid'])]
    sensitivity = dict(tasks=len(all_valid), guided_success=float(guided_y[all_valid].mean()) if all_valid else None,
                       actor_success=[float(scores[all_valid, k].mean()) for k in range(len(actor))] if all_valid else [])
    counts = {role: dict(episodes=sum(len(g[role]) for g in groups.values()),
                        valid=sum(e['valid'] for g in groups.values() for e in g[role]),
                        successes=sum(e['success'] for g in groups.values() for e in g[role])) for role in ('actor', 'guided')}
    counts['any_actor_success_tasks'] = sum(any(e['success'] for e in g['actor']) for g in groups.values())
    return dict(tasks=n, counts=counts, guided=guided, actor=actor,
                equal_expected_browser_calls=equal, common_valid=sensitivity), dict(zip(ids, tasks))


def plot(result, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.size': 10, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    colors = {'actor': '#406a9c', 'guided': '#c45b27', 'stop': '#75a2b5'}
    for row, cohort in enumerate(('corrected', 'historical')):
        data = result[cohort]
        for col, metric in enumerate(('browser_step_calls', 'compute')):
            ax = axes[row, col]
            if metric == 'compute':
                low, high = (('decoder_observed_kv_flops',)*2 if row == 0 else ('observed_kv_lower_flops', 'observed_kv_upper_flops'))
                scale = 1e15
            else:
                low = high = metric
                scale = 1
            def coordinates(point, cost_key):
                c = point[cost_key]
                lo, hi = c[low]['mean']/scale, c[high]['mean']/scale
                return (lo+hi)/2, (hi-lo)/2
            xs, ys = [], []
            for point in data['actor']:
                x, xe = coordinates(point, 'run_all_costs')
                y = point['success']['mean']*100
                ci = point['success']['task_bootstrap95']
                ax.errorbar(x, y, xerr=xe, yerr=[[y-ci[0]*100], [ci[1]*100-y]], color=colors['actor'], fmt='o', capsize=3)
                ax.annotate('k='+str(point['k']), (x, y), xytext=(4, 7), textcoords='offset points', fontsize=9)
                xs.append(x); ys.append(y)
            ax.plot(xs, ys, '--', color=colors['actor'], label='Ordinary oracle pass@k; run all k')
            if metric != 'compute':
                stop_x = [coordinates(p, 'oracle_stop_costs')[0] for p in data['actor']]
                ax.plot(stop_x, ys, ':s', color=colors['stop'], markersize=4, label='Free perfect stopping (sensitivity)')
            x, xe = coordinates(data['guided'], 'costs')
            y = data['guided']['success']['mean']*100
            ci = data['guided']['success']['task_bootstrap95']
            ax.errorbar(x, y, xerr=xe, yerr=[[y-ci[0]*100], [ci[1]*100-y]], color=colors['guided'], fmt='D', markersize=8, capsize=4, label='One N=5 selection episode')
            ax.set_ylim(24, 67)
            ax.set_xlim(left=0)
            ax.grid(alpha=.2)
            ax.set_ylabel('Judged success (%)')
            labels = {'browser_step_calls': 'Recorded browser-step calls per task',
                      'compute': ('Known decoder work (10¹⁵ FLOPs)\nVision and failed requests excluded' if row == 0 else 'Actor + selector work (10¹⁵ FLOPs)\nHorizontal bars: vision-cache bounds'),
                      'recorded_episode_seconds': 'Sum of recorded episode seconds per task\nNot measured multi-attempt deployment latency'}
            ax.set_xlabel(labels[metric])
            if col == 0:
                ax.legend(fontsize=8, loc='lower right')
    axes[0, 0].set_title('Corrected policy: three repetitions / 300 tasks', pad=15, weight='bold')
    axes[0, 1].set_title('Corrected policy: known decoder work', pad=15, weight='bold')
    axes[1, 0].set_title('Historical: actor browser policy omitted', pad=15, weight='bold')
    axes[1, 1].set_title('Historical: observed KV + bounded vision', pad=15, weight='bold')
    fig.suptitle('Retries and pre-action selection spend different resources', fontsize=16, weight='bold', y=.995)
    fig.text(.5, .01, 'Vertical bars: 95% task bootstrap. Invalid episodes count as zero. Oracle retries have free success recognition.\nBrowser calls include done/errors, omit reset/observation work, and may contain several tools. No harmful-consequence endpoint was measured.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .055, 1, .96), h_pad=3)
    for extension in ('png', 'svg', 'pdf'):
        fig.savefig(output/f'tradeoff.{extension}', dpi=180, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=REPO/'openwebrl/docs/arm_results/inference_interaction_tradeoff_20261009')
    parser.add_argument('--private', type=Path, default=RUNTIME/'analysis/inference-interaction-tradeoff-20261009')
    parser.add_argument('--draws', type=int, default=10000)
    parser.add_argument('--render-only', action='store_true', help='Render an existing audited aggregate without rereading trajectories')
    args = parser.parse_args()
    if args.render_only:
        plot(read(args.output/'aggregate.json'), args.output)
        return
    if args.draws < 1000:
        raise ValueError('At least 1000 bootstrap draws required')
    corrected, corrected_sources, corrected_audit = load_corrected()
    historical, historical_sources, historical_audit = load_historical()
    if set(corrected) != set(historical):
        raise ValueError('Benchmark task identities differ; report separately before comparing')
    results = dict(schema_version=1, new_model_calls=0, new_browser_calls=0, new_judge_calls=0,
                   bootstrap=dict(draws=args.draws, seed=20261009, unit='task; all attempts and methods retained'),
                   definitions=dict(guided='Uniform mean of ONE guided episode, not any-success across guided repeats.',
                       run_all='Uniform average over all ordered k-subsets of saved ordinary episodes.',
                       oracle_stop='Stop after first native judged success, with free perfect recognition. Costs conditional on saved episodes/order and observed cache receipts.',
                       browser_step_calls='Outer env.step invocations, including done and failing calls; not individual dispatched tools. Reset/navigation/observation work excluded.',
                       recorded_episode_seconds='Recorded episode duration includes startup, shared queues, judge and artifact work. Sums are descriptive workload, not measured sequential or parallel deployment latency.',
                       flops='Analytic dominant forward operations; multiply-add counts as two. All selected/unselected proposals and selector received work included. Unknown failed-request work is excluded, not assumed free.',
                       equal_expected_browser='Outcome-independent random mixture of adjacent k matching mean browser calls; no hard per-task budget claim.',
                       safety='No recorded harmful-consequence endpoint or backend-state rollback verification; fewer calls do not establish safer outcomes.'),
                   caveats=['Historical Oct4 cohort omitted actor browser system policy; kept separate from corrected repeats.',
                            'Corrected compute is decoder-only; historical compute includes bounded vision work. Axes are not interchangeable.',
                            'Native evaluator permits partial progress; success is not strict complete-task or consequence-free success.',
                            'Task bootstrap conditions on saved repetitions and sites; does not include judge error, serving alias drift or website drift.',
                            'Policy curves cover committed episodes. Every interrupted attempt/startup remains in separate research accounting.'])
    for name, groups, sources, audit in [('corrected', corrected, corrected_sources, corrected_audit),
                                        ('historical', historical, historical_sources, historical_audit)]:
        summary, private_rows = summarize(groups, args.draws)
        results[name] = summary
        results[name]['audit'] = audit
        write(args.private/f'{name}-task-analysis.json', private_rows)
        write(args.private/f'{name}-sources.json', sources)
        results[name]['source_manifest_sha256'] = sha(args.private/f'{name}-sources.json')
    results['analysis_source_sha256'] = {str(p.relative_to(REPO)): sha(p) for p in
        (Path(__file__), REPO/'openwebrl/inference_retry_analysis.py', REPO/'openwebrl/arm_inference_flops.py')}
    write(args.output/'aggregate.json', results)
    plot(results, args.output)
    print(json.dumps({n: {k: results[n][k] for k in ('counts', 'equal_expected_browser_calls', 'common_valid')} for n in ('corrected', 'historical')}, indent=2))


if __name__ == '__main__':
    main()
