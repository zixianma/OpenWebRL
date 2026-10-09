"""Usage accounting for the Qwen/Luna four-arm inference comparison.

Reasoning and cached tokens are subsets, never additions to token totals.
Unknown usage stays unknown. Model-serving dollars exclude terminal judging.
"""
from __future__ import annotations

from collections import defaultdict
import fcntl
import hashlib
import json
import math
from pathlib import Path
import random

MODES = ('qwen', 'qwen_luna5', 'qwen_luna10', 'luna')
LABELS = dict(qwen='Qwen alone', qwen_luna5='Qwen + Luna (N=5)',
              qwen_luna10='Qwen + Luna (N=10)', luna='Luna alone',
              sft_luna5='Official OpenWebRL SFT + Luna (N=5)')
SEED = 20261004
PRICING = dict(luna_input=0.10, luna_cached=0.01, luna_cache_write=0.125,
               luna_output=0.50, judge_input=1.10, judge_cached=0.275,
               judge_output=4.40, h200_hour=0.90)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def schedule(task_ids):
    if len(set(task_ids)) != len(task_ids):
        raise ValueError('Duplicate task IDs')
    result = []
    for task in sorted(task_ids, key=lambda t: digest([SEED, t, 'task'])):
        modes = list(MODES)
        random.Random(digest([SEED, task, 'modes'])).shuffle(modes)
        result.append(dict(task_id=task, modes=modes))
    return result


def claimed_tasks(root, items, worker):
    """Keep each task's four arms on one worker; never duplicate committed work."""
    root = Path(root)
    (root / 'task-locks').mkdir(parents=True, exist_ok=True)
    for item in items:
        complete = lambda: all((root / 'records' / (digest([item['task_id'], mode])[:24] + '.json')).exists()
                               for mode in item['modes'])
        with (root / 'task-locks' / (digest(item['task_id']) + '.lock')).open('a+') as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                continue
            if complete():
                continue
            handle.seek(0)
            handle.truncate()
            handle.write(str(worker))
            handle.flush()
            yield item


def tokens(usage, provider):
    if provider == 'qwen':
        result = dict(input=usage['prompt_tokens'], output=usage['completion_tokens'],
                      cached_input=usage.get('cached_tokens'), reasoning_output=None,
                      cache_write_input=None, image_input=None)
    else:
        inp = usage.get('input_tokens', usage.get('prompt_tokens'))
        out = usage.get('output_tokens', usage.get('completion_tokens'))
        details = usage.get('input_tokens_details', usage.get('prompt_tokens_details')) or {}
        output_details = usage.get('output_tokens_details', usage.get('completion_tokens_details')) or {}
        result = dict(input=inp, output=out, cached_input=details.get('cached_tokens'),
                      reasoning_output=output_details.get('reasoning_tokens'),
                      cache_write_input=details.get('cache_write_tokens'),
                      image_input=details.get('image_tokens'))
    for key, value in result.items():
        if value is not None and (type(value) is not int or value < 0):
            raise ValueError(f'Invalid {key} token count')
    if result['input'] is None or result['output'] is None:
        raise ValueError('Missing total input/output usage')
    for subset, total in [('cached_input', 'input'), ('cache_write_input', 'input'),
                          ('image_input', 'input'), ('reasoning_output', 'output')]:
        if result[subset] is not None and result[subset] > result[total]:
            raise ValueError(f'{subset} exceeds {total}')
    if (result['cached_input'] is not None and result['cache_write_input'] is not None
            and result['cached_input'] + result['cache_write_input'] > result['input']):
        raise ValueError('Cache reads plus writes exceed input')
    return result


def api_cost(usage, role, pricing=PRICING):
    """Dollar bounds if the provider omits cache-read/write telemetry."""
    t = tokens(usage, 'api')
    if role not in ('actor', 'selector', 'judge'):
        raise ValueError('Unknown API role')
    prefix = 'judge' if role == 'judge' else 'luna'
    inp, out = t['input'], t['output']
    if prefix == 'luna' and inp > 272000:
        raise ValueError('Long-context price tier outside the frozen experiment')
    cached = t['cached_input']
    cached_low, cached_high = (0, inp) if cached is None else (cached, cached)
    write_rate = pricing.get(prefix + '_cache_write', pricing[prefix + '_input'])
    writes = t['cache_write_input']
    if prefix == 'judge':
        writes = 0
    if cached is None and writes is not None:
        cached_high = inp - writes
    lower = ((inp - cached_high) * pricing[prefix + '_input']
             + cached_high * pricing[prefix + '_cached'] + out * pricing[prefix + '_output'])
    upper = ((inp - cached_low) * pricing[prefix + '_input']
             + cached_low * pricing[prefix + '_cached'] + out * pricing[prefix + '_output'])
    if writes is None:
        upper += (inp - cached_low) * (write_rate - pricing[prefix + '_input'])
    else:
        surcharge = writes * (write_rate - pricing[prefix + '_input'])
        lower += surcharge
        upper += surcharge
    return dict(lower_usd=lower / 1e6, upper_usd=upper / 1e6,
                cache_write_telemetry_known=writes is not None)


def quantile(values, fraction):
    if not values:
        return None
    values = sorted(values)
    index = (len(values) - 1) * fraction
    low = math.floor(index)
    high = math.ceil(index)
    return values[low] + (values[high] - values[low]) * (index - low)


def episode_usage(directory, mode, episode_seconds, pricing=PRICING):
    """Include every returned request, even discarded or failed generations."""
    rows = [json.loads(p.read_text()) for p in Path(directory).glob('requests/*.json')]
    by_role = defaultdict(lambda: dict(input=0, output=0, cached_input=0,
        reasoning_output=0, requests=0, usage_missing=0, cached_missing=0,
        reasoning_missing=0, cache_write_input=0, cache_write_missing=0,
        image_input=0, image_missing=0, cost_lower_usd=0., cost_upper_usd=0.))
    flops_lower = flops_upper = qwen_request_seconds = 0.
    flops_missing = 0
    for row in rows:
        key = row['provider'] + '_' + row['role']
        target = by_role[key]
        target['requests'] += 1
        if row['provider'] == 'qwen':
            qwen_request_seconds += row.get('seconds', 0.)
            if row.get('flops'):
                flops_lower += row['flops']['lower']
                flops_upper += row['flops']['upper']
            else:
                flops_missing += 1
        if not row.get('usage'):
            target['usage_missing'] += 1
            continue
        t = tokens(row['usage'], row['provider'])
        for name in ('input', 'output'):
            target[name] += t[name]
        for name, missing in [('cached_input', 'cached_missing'), ('reasoning_output', 'reasoning_missing'),
                              ('cache_write_input', 'cache_write_missing'), ('image_input', 'image_missing')]:
            if t[name] is None:
                target[missing] += 1
            else:
                target[name] += t[name]
        if row['provider'] != 'qwen':
            cost = api_cost(row['usage'], row['role'], pricing)
            target['cost_lower_usd'] += cost['lower_usd']
            target['cost_upper_usd'] += cost['upper_usd']
    # One exclusive episode per GPU. Qwen's GPU remains reserved while its
    # browser and Luna selector wait. Luna alone requires no local model GPU.
    gpu_seconds = episode_seconds if mode != 'luna' else 0.
    local_cost = gpu_seconds / 3600 * pricing['h200_hour']
    model_roles = [v for k, v in by_role.items() if not k.endswith('_judge')]
    missing = sum(v['usage_missing'] for v in model_roles)
    return dict(by_role=dict(by_role), model_input_tokens=sum(v['input'] for v in model_roles),
                model_output_tokens=sum(v['output'] for v in model_roles),
                model_usage_missing_requests=missing, local_gpu_reserved_seconds=gpu_seconds,
                qwen_request_seconds_sum_overlapping=qwen_request_seconds,
                qwen_flops_lower=flops_lower, qwen_flops_upper=flops_upper if not flops_missing else None,
                qwen_compute_missing_requests=flops_missing,
                model_cost_lower_usd=local_cost + sum(v['cost_lower_usd'] for v in model_roles),
                model_cost_upper_usd=(local_cost + sum(v['cost_upper_usd'] for v in model_roles)) if not missing else None,
                judge_cost_upper_usd=sum(v['cost_upper_usd'] for k, v in by_role.items() if k.endswith('_judge')))


def summarize(records, task_ids, pricing=PRICING, require_complete=True, modes=MODES):
    modes = tuple(modes)
    if not modes or len(set(modes)) != len(modes):
        raise ValueError('Invalid comparison modes')
    index = {(r['task_id'], r['mode']): r for r in records}
    expected = {(t, m) for t in task_ids for m in modes}
    if not task_ids or len(set(task_ids)) != len(task_ids) or len(index) != len(records) or set(index) - expected:
        raise ValueError('Duplicate or unexpected episode')
    complete = set(index) == expected
    if require_complete and not complete:
        raise ValueError('Incomplete cohort: performance plots require every arm/task')
    metrics = ('episode_seconds', 'browser_steps', 'browser_step_seconds', 'model_input_tokens',
               'model_output_tokens', 'local_gpu_reserved_seconds', 'qwen_flops_lower', 'qwen_flops_upper')
    result = dict(complete=complete, scheduled_tasks=len(task_ids), recorded_episodes=len(records),
                  pricing=pricing, by_mode={}, paired_differences={},
                  cost_scope='Estimated exclusive model-serving cost; browser CPU not separately priced. Judge and research startup/idle costs are separate.',
                  token_scope='Provider token totals include image tokens and reasoning where supported; model tokenizers differ. Subsets are never added twice.')
    for mode in modes:
        rows = [r for r in records if r['mode'] == mode]
        successes = sum(r['valid'] and r['reward'] == 1 for r in rows)
        data = dict(episodes=len(rows), successes=successes, valid=sum(r['valid'] for r in rows),
                    overall=successes / len(task_ids), metrics={})
        for metric in metrics:
            vals = [r[metric] for r in rows]
            known = bool(vals) and all(v is not None for v in vals)
            data['metrics'][metric] = dict(mean=sum(vals) / len(vals) if known else None,
                p50=quantile(vals, .5) if known else None, p95=quantile(vals, .95) if known else None,
                missing=sum(v is None for v in vals))
        cost = [r['model_cost_upper_usd'] for r in rows]
        data['cost_lower_mean_usd'] = sum(r['model_cost_lower_usd'] for r in rows) / len(rows) if rows else None
        data['cost_upper_mean_usd'] = sum(cost) / len(cost) if cost and all(c is not None for c in cost) else None
        data['unknown_usage_requests'] = sum(r['model_usage_missing_requests'] for r in rows)
        roles = defaultdict(lambda: defaultdict(float))
        for row in rows:
            for role, values in row['by_role'].items():
                for key, value in values.items():
                    roles[role][key] += value
        data['role_totals'] = {k: dict(v) for k, v in roles.items()}
        result['by_mode'][mode] = data
    if complete:
        rng = random.Random(SEED)
        reference = 'qwen' if 'qwen' in modes else modes[0]
        outcomes = {m: [int(index[t, m]['valid'] and index[t, m]['reward'] == 1) for t in task_ids] for m in modes}
        boot = {m: [] for m in modes}
        diffs = {m: [] for m in modes if m != reference}
        for _ in range(5000):
            ids = [rng.randrange(len(task_ids)) for _ in task_ids]
            rates = {m: sum(outcomes[m][i] for i in ids) / len(ids) for m in modes}
            for m in modes:
                boot[m].append(rates[m])
            for m in diffs:
                diffs[m].append(rates[m] - rates[reference])
        for m in modes:
            result['by_mode'][m]['success_95'] = [quantile(boot[m], .025), quantile(boot[m], .975)]
        for m in diffs:
            result['paired_differences'][m + '_minus_' + reference] = dict(
                mean=result['by_mode'][m]['overall'] - result['by_mode'][reference]['overall'],
                task_bootstrap_95=[quantile(diffs[m], .025), quantile(diffs[m], .975)])
    return result
