#!/usr/bin/env python3
"""Verify the historical three-arm cohort and render aggregate-only reports.

Default mode requires all 900 episodes and final independent shard accounting.
--partial is a preview on tasks with all three committed outcomes. It never
writes a completion pointer or silently substitutes that subset for full300.
No model calls, browser calls, allocations, or external publication occur here.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from openwebrl.arm_controlled_inference import verify_record
from openwebrl.arm_inference_flops import decoder_flops
from openwebrl.arm_paired_inference import MODES, PROTOCOL, schedule, summarize
from openwebrl.controlled_sft_state import digest

TERMINAL = {'COMPLETED', 'FAILED', 'TIMEOUT', 'CANCELLED', 'NODE_FAIL',
            'OUT_OF_MEMORY', 'PREEMPTED'}
LABELS = {'actor0': 'SFT alone', 'piotr': 'SFT + Piotr ARM', 'arm': 'SFT + RL-task ARM'}
ORDER = ('actor0', 'piotr', 'arm')
RESOURCES = dict(gpus=2, gpu_type='H200', cpus=16, memory_gib=240, total_seconds=14400)
LIMITS = dict(judge_usd=10., judge_calls=1980)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def matched_records(records, task_ids):
    # Validate every record, including unmatched/invalid ones, before subsetting.
    summarize(records, task_ids)
    index = {(r['task_id'], r['mode']): r for r in records}
    common = [t for t in task_ids if all((t, m) in index for m in MODES)]
    return common, [index[t, m] for t in common for m in MODES]


def shard_accounting(approval, scheduler, ledger, receipts, require_final):
    if (approval.get('approved') is not True or approval['resources'] != RESOURCES
            or approval['limits'] != LIMITS):
        raise ValueError('Independent shard approval changed')
    jobs = [str(a['job_id']) for a in approval['attempts']]
    if len(jobs) != len(set(jobs)) or {r['job_id'] for r in scheduler} != set(jobs):
        raise ValueError('Missing or duplicate scheduler attempts')
    if len(scheduler) != len(jobs):
        raise ValueError('Duplicate scheduler accounting')
    seconds = sum(r['elapsed_seconds'] for r in scheduler)
    if seconds > 14400 or ledger['calls'] > 1980 or ledger['charged_or_reserved_usd'] > 10 + 1e-8:
        raise ValueError('Independent shard cap exceeded')
    pending = sum(r['status'] != 'received' for r in receipts)
    final = all(r['state'].split()[0] in TERMINAL for r in scheduler) and not pending
    settled = sum(r.get('charged_usd', 0.) for r in receipts if r['status'] == 'received')
    if settled > ledger['charged_or_reserved_usd'] + 1e-8:
        raise ValueError('Receipt spend exceeds ledger')
    if final and (len(receipts) != ledger['calls'] or abs(settled - ledger['charged_or_reserved_usd']) > 1e-8):
        raise ValueError('Final judge receipts do not reconcile')
    if require_final and not final:
        raise ValueError('Final scheduler/API accounting is not available')
    return dict(attempts=scheduler, allocation_seconds=seconds, allocation_seconds_cap=14400,
                allocated_gpu_hours=seconds * 2 / 3600, judge=ledger,
                settled_judge_usd=settled, unsettled_judge_receipts=pending,
                final_accounting_verified=final)


def episode_decoder_work(record, actor_config, selector_config):
    """Receipt-derived decoder-only work; no invented visual/hardware cost.

    The cold view charges every candidate's full prefill identically. The KV
    view uses observed actor cache hits. Both include actual selector decoder
    work. Failed/unmetered requests are counted separately, not assigned zero
    compute, and all their elapsed allocation time remains in accounting.
    """
    receipts = [read(p) for p in Path(record['artifact']).parent.glob('actor-request-*.json')]
    if len(receipts) != record['actor_requests']:
        raise ValueError('Missing actor request receipts')
    cold = cached = 0
    inp = out = failed = 0
    for receipt in receipts:
        sampling = receipt.get('sampling_params')
        if sampling is not None:
            expected = dict(temperature=.7, top_p=.9, max_new_tokens=1024,
                            sampling_seed=receipt['sampling_seed'])
            if sampling != expected:
                raise ValueError('Actor decoding receipt differs from historical protocol')
        elif receipt.get('artifact_io_policy', 'synchronous_receipts_v0') != 'synchronous_receipts_v0':
            raise ValueError('New actor receipt lacks sampling metadata')
        if receipt['status'] != 'received':
            failed += 1
            continue
        m = receipt['meta']; p, g = m['prompt_tokens'], m['completion_tokens']
        cold += decoder_flops(actor_config, p, g)['total']
        cached += decoder_flops(actor_config, p, g, m['cached_tokens'])['total']
        inp += p; out += g
    if (inp, out) != (record['actor_input_tokens'], record['actor_output_tokens']):
        raise ValueError('Actor token totals disagree with receipts')
    sin = sout = calls = 0
    if record['mode'] != 'actor0':
        for entry in record['trace_files']:
            for line in Path(entry['path']).read_text().splitlines():
                trace = json.loads(line)
                if trace.get('fallback'):
                    raise ValueError('Selector fallback prevents strict comparison')
                telemetry = trace['selector_telemetry']
                p, g = telemetry['input_tokens'], telemetry['output_tokens']
                value = decoder_flops(selector_config, p, g)['total']
                cold += value; cached += value; sin += p; sout += g; calls += 1
    if (sin, sout, calls) != (record['selector_input_tokens'], record['selector_output_tokens'], record['selector_calls']):
        raise ValueError('Selector token totals disagree with traces')
    return dict(decoder_cold_flops=cold, decoder_observed_kv_flops=cached,
                unmetered_failed_actor_requests=failed)


def load(root, require_final):
    root = Path(root); all_records = []; accounting = []; models = None; task_ids = None
    protocol_sha = hashlib.sha256(json.dumps(PROTOCOL, sort_keys=True).encode()).hexdigest()
    for shard in (0, 1):
        r = root / f'shard-{shard}'; plan = read(r/'plan.json'); approval = read(r/'approval.json')
        if approval['plan_sha256'] != digest(plan) or plan['protocol'] != PROTOCOL:
            raise ValueError('Frozen protocol/approval hash changed')
        manifest = read(r/'experiment.json'); tasks = Path(plan['task_file'])
        if manifest['protocol'] != PROTOCOL or manifest['task_file_sha256'] != sha(tasks):
            raise ValueError('Frozen tasks or protocol changed')
        ids = [json.loads(line)['metadata']['task_id'] for line in tasks.read_text().splitlines()]
        if len(ids) != 300 or (task_ids is not None and ids != task_ids):
            raise ValueError('Expected identical full300 task files')
        task_ids = ids
        sched = read(r/'schedule.json')
        if sched != schedule(ids) or manifest['schedule_sha256'] != sha(r/'schedule.json'):
            raise ValueError('Frozen three-arm schedule changed')
        slots = {a['key']: (t, a) for t in sched if t['shard'] == shard for a in t['attempts']}
        collection = r/plan.get('collection_directory','collection')
        if collection.parent != r:
            raise ValueError('Collection must be an immediate shard subdirectory')
        for path in (collection/'records').glob('*.json'):
            record = read(path); item, attempt = slots[record['key']]
            verify_record(record, item, attempt, collection, protocol_sha)
            # Preserve and verify invalid trajectories too.
            for entry in record['trace_files']:
                path = Path(entry['path'])
                if not path.resolve().is_relative_to(collection.resolve()) or sha(path) != entry['sha256']:
                    raise ValueError('Decision trace hash/path changed')
                for line in path.read_text().splitlines():
                    trace = json.loads(line)
                    size = 1 if record['mode'] == 'actor0' else 5
                    if len(trace['candidates']) != size or not 0 <= trace['selected_index'] < size or trace.get('fallback'):
                        raise ValueError('Candidate count, selection index or fallback changed')
            all_records.append(record)
        jobs = [str(a['job_id']) for a in approval['attempts']]
        raw = subprocess.check_output(['sacct', '-X', '-n', '-P', '-j', ','.join(jobs),
                                       '--format=JobIDRaw,State,ElapsedRaw,AllocTRES'], text=True)
        scheduler = []
        for line in raw.splitlines():
            jid, state, seconds, tres = line.split('|')[:4]
            if jid in jobs:
                shape = dict(x.split('=', 1) for x in tres.split(',') if '=' in x)
                if (shape.get('cpu') != '16' or shape.get('gres/gpu') != '2'
                        or shape.get('gres/gpu:h200') != '2'
                        or shape.get('mem') not in ('240G', '245760M')):
                    raise ValueError('Actual allocation shape differs from independent shard approval')
                scheduler.append(dict(job_id=jid, state=state, elapsed_seconds=int(seconds), allocated_tres=tres))
        budget = r/'collection/judge-budget'
        receipts = [read(p) for p in budget.glob('[0-9][0-9][0-9][0-9][0-9].json')]
        ledger = read(budget/'ledger.json')  # Read after receipts while collection is live.
        account = shard_accounting(approval, scheduler, ledger, receipts, require_final)
        for p in budget.glob('*-response.json'):
            if read(p)['response_model'] != 'o4-mini-2025-04-16':
                raise ValueError('Judge model changed')
        current = read(r/'current.json'); journal = r/'scheduler-attempts'/current['job_id']
        if require_final and not read(journal/'teardown.json')['verified']:
            raise ValueError('Final owned-process teardown is not verified')
        expected_models = {k: plan[v] for k, v in [('actor', 'actor'), ('selector', 'selector'), ('piotr', 'piotr_selector')]}
        for name, path in expected_models.items():
            card = read(journal/f'{name}-identity.json')
            if card.get('model_path', card.get('model')) != path:
                raise ValueError('Served model identity changed')
        if models is not None and models != expected_models:
            raise ValueError('Shards served different models')
        models = expected_models; accounting.append(dict(shard=shard, **account))
    summarize(all_records, task_ids, require_complete=require_final)
    return task_ids, all_records, accounting, models


def report(root, partial=False):
    import numpy as np
    task_ids, records, accounting, models = load(root, not partial)
    common, matched = matched_records(records, task_ids)
    if not common:
        raise ValueError('No complete three-arm task blocks to compare')
    cohort = summarize(matched, common, require_complete=True)
    configs = {k: read(Path(v)/'config.json')['text_config'] for k, v in models.items()}
    costs = {m: [] for m in MODES}
    for row in matched:
        selector = configs['piotr' if row['mode'] == 'piotr' else 'selector']
        costs[row['mode']].append(episode_decoder_work(row, configs['actor'], selector))
    # Use the same task-resampling weights for all three arms and all axes.
    weights = np.random.default_rng(42).multinomial(len(common), np.full(len(common), 1/len(common)), size=10000)/len(common)
    for mode, rows in costs.items():
        summary = cohort['by_mode'][mode]
        summary['mean'] = {k: summary[k]/len(common) for k in (
            'actor_input_tokens', 'actor_output_tokens', 'selector_input_tokens', 'selector_output_tokens',
            'elapsed_seconds', 'actor_client_request_seconds', 'selector_service_seconds', 'browser_steps', 'browser_step_seconds')}
        summary['mean'].update({k: sum(r[k] for r in rows)/len(common) for k in rows[0]})
        outcomes = [r for r in matched if r['mode'] == mode]
        values = np.array([[int(r['valid'] and r['reward'] == 1), c['decoder_cold_flops'], c['decoder_observed_kv_flops'],
                            r['elapsed_seconds'], sum(r[k] for k in (
                                'actor_input_tokens', 'actor_output_tokens', 'selector_input_tokens', 'selector_output_tokens'))]
                           for r, c in zip(outcomes, rows)])
        intervals = np.quantile(weights @ values, [.025, .975], axis=0)
        summary['bootstrap95'] = {name: intervals[:, i].tolist() for i, name in enumerate(
            ('success', 'decoder_cold_flops', 'decoder_observed_kv_flops', 'elapsed_seconds', 'total_tokens'))}
    invalid_causes = {m: Counter() for m in MODES}
    for row in records:
        if row['valid']:
            continue
        artifact = read(row['artifact']); reason = artifact.get('terminate_reason', '')
        if not row['actor_requests'] and not row['browser_steps']:
            cause = 'before_first_actor_request_or_browser_step'
        elif reason == 'env_step_error':
            cause = 'browser_step_error'
        elif "400 Bad Request" in reason:
            cause = 'actor_http400_rejection'
        elif artifact.get('metadata', {}).get('judge_timeout'):
            cause = 'judge_timeout'
        else:
            cause = 'other_invalid'
        invalid_causes[row['mode']][cause] += 1
    return dict(epoch=time.time(), status='partial_matched_preview' if partial else 'ready_for_final_artifact_review',
        protocol=PROTOCOL, recorded_episodes=len(records), expected_episodes=900,
        matched_tasks=len(common), matched_episodes=len(matched), unmatched_episodes=len(records)-len(matched),
        cohort=cohort, accounting=accounting,
        invalid_causes_all_recorded={m: dict(v) for m, v in invalid_causes.items()},
        collection_target_verified=len(records)==900 and len(common)==300,
        final_accounting_verified=all(a['final_accounting_verified'] for a in accounting),
        verified_complete=False,
        caveats=[
            'Preview comparisons use only complete three-arm task blocks; all invalid outcomes count as zero. This subset can be selected by task duration and is not full300.',
            'Decoder FLOPs are analytical workload estimates, excluding vision encoders, elementwise operations, recomputation, failed/unmetered requests and hardware overhead. They are not total compute or a dollar price.',
            'Cold prefill charges each request equally. Observed actor KV reuse is reported separately; selector decoder work is included in both views.',
            'Tokens count actual received actor and selector input/output, including cached input. They are not billing-equivalent across services.',
            'Latency is complete episode wall time. Concurrent request seconds must not be summed as wall time. Shared GPU time is reported at campaign/shard level only.',
            'All-attempt GPU hours and judge spend include failed and interrupted attempts. Evaluation judge spend is not actor/selector deployment cost.',
            'Artifacts must be reviewed and remote publication verified before marking this experiment complete.'])


def plots(data, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = ('#475569', '#d97706', '#2563eb')
    prefix = f"{'Partial matched preview' if data['status'].startswith('partial') else 'Full cohort'}: {data['matched_tasks']} tasks"
    if data.get('harness_correction', {}).get('actor_system_policy_missing'):
        prefix = f"Archived blank-policy preview: {data['matched_tasks']} tasks"
    axes = [('cost', [
                ('decoder_cold_flops', 1e15, 'Cold-prefill decoder work (PFLOPs/task)'),
                ('decoder_observed_kv_flops', 1e15, 'Observed-KV decoder work (PFLOPs/task)')]),
            ('latency', [('elapsed_seconds', 1, 'Mean episode wall time (seconds/task)')]),
            ('tokens', [(None, 1e3, 'Mean actor + selector input/output tokens (thousands/task)')])]
    paths = []
    for name, panels in axes:
        fig, grid = plt.subplots(1, len(panels), figsize=(14 if len(panels) == 2 else 8, 5), squeeze=False)
        for ax, (metric, scale, label) in zip(grid[0], panels):
            for mode, color in zip(ORDER, colors):
                row = data['cohort']['by_mode'][mode]; means = row['mean']
                x = means[metric] if metric else sum(means[k] for k in (
                    'actor_input_tokens', 'actor_output_tokens', 'selector_input_tokens', 'selector_output_tokens'))
                lo, hi = row['bootstrap95'][metric or 'total_tokens']; slo, shi = row['bootstrap95']['success']
                ax.errorbar(x/scale, row['overall']*100,
                            xerr=[[max(0, x-lo)/scale], [max(0, hi-x)/scale]],
                            yerr=[[max(0, row['overall']-slo)*100], [max(0, shi-row['overall'])*100]],
                            color=color, fmt='o', markersize=7, capsize=3, label=LABELS[mode])
            ax.set(xlabel=label, ylabel='Success (%) — invalids count as zero', ylim=(0, 100))
            ax.margins(x=.1); ax.set_xlim(left=0); ax.grid(alpha=.2)
            ax.spines[['top', 'right']].set_visible(False); ax.legend(loc='upper left', fontsize=9)
        fig.suptitle(prefix)
        note = '95% task-bootstrap intervals; three fresh episodes/task. No claim of equal compute budgets.'
        if data.get('harness_correction', {}).get('actor_system_policy_missing'):
            note = 'Actor browser system policy omitted. Excluded from corrected historical-protocol cohort.\n' + note
        if name == 'cost': note += '\nDecoder estimates exclude vision and failed/unmetered requests; total GPU hours remain separately accounted.'
        fig.text(.5, .025, note, ha='center', fontsize=8); fig.tight_layout(rect=(0, .09, 1, 1))
        for ext in ('png', 'svg'):
            path = output/f'{name}.{ext}'; fig.savefig(path, dpi=180); paths.append(path)
        plt.close(fig)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--partial', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Use a new output directory to preserve previous report artifacts')
    data = report(args.root, args.partial)
    args.output.mkdir(parents=True, exist_ok=False)
    paths = plots(data, args.output)
    data['plots'] = [{'name': p.name, 'sha256': sha(p)} for p in paths]
    (args.output/'aggregate.json').write_text(json.dumps(data, indent=2)+'\n')
    print(json.dumps({k: data[k] for k in ('status', 'recorded_episodes', 'matched_tasks', 'final_accounting_verified')}))


if __name__ == '__main__':
    main()
