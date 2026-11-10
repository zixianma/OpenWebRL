#!/usr/bin/env python3
"""CPU audit of saved inference cost proxies and empirical episode pass@k.

No inference/API calls. Protocols are reported separately, never pooled.
Historical text retokenization is not exact generation-token/GPU accounting.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import time

RUNTIME = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime')
HISTORICAL = RUNTIME/'arm-reproduction/runs/dedicated-282209-20260908T005547Z/full'
POOL = RUNTIME/'arm-turn-bonus-preparation/task-pool-expansion-20260922/curation-v3-20260929'
TOKENIZER = Path('/gpfs/scrubbed/zixianma/checkpoints/web/OpenWebRL-4B-SFT/tokenizer.json')


def pass_at_k(n, c, k):
    if not 0 <= c <= n or not 1 <= k <= n:
        raise ValueError('Invalid pass@k counts')
    return 1 - math.comb(n-c, k)/math.comb(n, k) if n-c >= k else 1.0


def success(row):
    return row['valid'] is True and row['reward'] == 1


def screen_summary(records, expected_n=5):
    groups = defaultdict(list)
    for row in records:
        groups[str(row['task_id'])].append(row)
    for rows in groups.values():
        if len(rows) != expected_n or len({r['attempt'] for r in rows}) != expected_n:
            raise ValueError('Incomplete or duplicate attempt set')
    histogram = Counter(sum(success(r) for r in rows) for rows in groups.values())
    valid_complete = [rows for rows in groups.values() if all(r['valid'] for r in rows)]
    costs = {k: sum(r[k] for r in records) for k in ('actor_requests', 'actor_output_tokens', 'steps')}
    table = []
    for k in range(1, expected_n+1):
        values = [pass_at_k(expected_n, sum(success(r) for r in rows), k) for rows in groups.values()]
        table.append(dict(k=k, overall_pass_at_k=sum(values)/len(values),
            five_valid_tasks_pass_at_k=sum(pass_at_k(expected_n, sum(success(r) for r in rows), k)
                                          for rows in valid_complete)/len(valid_complete) if valid_complete else None,
            expected_random_subset_cost_per_task={name: v*k/expected_n/len(groups) for name, v in costs.items()}))
    return dict(tasks=len(groups), attempts=len(records), valid_attempts=sum(r['valid'] for r in records),
        successful_attempts=sum(success(r) for r in records), success_count_histogram=dict(sorted(histogram.items())),
        five_valid_tasks=len(valid_complete), cost_totals=costs, pass_at_k=table,
        definition='Equal-task mean of 1-comb(5-c,k)/comb(5,k). Invalid attempts count as no success; no replacement.',
        cost_definition='Expected cost of a uniformly sampled k-subset of the five saved attempts; run all k, no early stopping.',
        limitation='Selected training-task pool; not OM2W. Empirical finite-sample coverage, not a fresh-run guarantee.')


def historical(tokenizer):
    modes = {}; identities = None
    for mode in ('baseline', 'scalar', 'selection'):
        root = HISTORICAL/mode
        manifest = json.loads((root/'manifest.json').read_text())
        # Result payloads can contain whole conversations/images. Keep only
        # denominator fields and release each large payload before the next.
        results = []
        for f in sorted((root/'results').glob('*.json')):
            payload = json.loads(f.read_text())
            results.append({k: payload[k] for k in ('task_id', 'valid', 'reward')})
            del payload
        ids = {r['task_id'] for r in results}
        if len(results) != 300 or len(ids) != 300 or (identities is not None and identities != ids):
            raise ValueError('Historical task cohort mismatch')
        identities = ids
        traces = 0; candidates = 0; length = 0; chosen_length = 0; seen = set(); duplicate = 0; fallbacks = 0
        trace_hash = hashlib.sha256(); trace_tasks = set(); finish = Counter()
        for f in sorted((root/'selections').glob('*.jsonl')):
            raw = f.read_bytes(); trace_hash.update(raw)
            for line in raw.splitlines():
                r = json.loads(line); traces += 1
                key = (r['task_id'], r['turn']); duplicate += key in seen; seen.add(key)
                if r['task_id'] not in ids:
                    raise ValueError('Trace outside result cohort')
                trace_tasks.add(r['task_id']); fallbacks += bool(r['fallback'])
                if len(r['candidates']) != manifest['candidate_count']:
                    raise ValueError('Candidate count changed')
                for i, c in enumerate(r['candidates']):
                    n = sum(len(tokenizer.encode(c[field], add_special_tokens=False).ids) for field in ('thought','action'))
                    candidates += 1; length += n
                    chosen_length += n if i == r['selected_index'] else 0
                finish.update(str(x) for x in r['finish_types'])
        modes[mode] = dict(tasks=len(results), successes=sum(success(r) for r in results),
            valid=sum(r['valid'] for r in results), overall=sum(success(r) for r in results)/len(results),
            tasks_with_saved_decisions=len(trace_tasks), saved_decisions=traces, duplicate_task_turns=duplicate,
            candidate_responses=candidates, actor_text_token_proxy=length, selected_text_token_proxy=chosen_length,
            selector_decisions=0 if mode=='baseline' else traces,
            selector_scored_candidates=traces*5 if mode=='scalar' else None,
            fallback_decisions=fallbacks, finish_types=dict(finish), trace_content_sha256=trace_hash.hexdigest(),
            protocol={k: manifest[k] for k in ('sampling','max_steps','judge','judge_protocol','candidate_count')})
    baseline = modes['baseline']
    for r in modes.values():
        r['actor_text_token_ratio_to_baseline'] = r['actor_text_token_proxy']/baseline['actor_text_token_proxy']
        r['saved_decision_ratio_to_baseline'] = r['saved_decisions']/baseline['saved_decisions']
    return dict(modes=modes,
        cost_definition='Retokenized saved thought + action text, excluding stripped delimiters and missing requests; selected and unselected candidates included.',
        limitation='Not exact generated tokens or GPU cost: no prefill, cache, image encoder, selector tokens, failed unsaved requests, browser/API timing. Resumes and concurrency differ; no per-mode GPU-hour causal comparison.',
        pass_at_k_control='Absent: only one ordinary episode per OM2W task in this historical comparison.')


def load_records(root):
    return [json.loads(f.read_text()) for f in sorted(root.glob('shard-*/records/*.json'))]


def main(output):
    started = time.monotonic()
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(TOKENIZER))
    old = historical(tokenizer)
    records = load_records(POOL/'actor-screen-min5-2000/sharded-20261002')
    if len(records) != 10000:
        raise ValueError('Expected complete 2,000-task screen')
    screen = screen_summary(records)
    rescues = load_records(POOL/'actor-arm-allfailure-20261002')
    if len(rescues) != 682 or len({r['task_id'] for r in rescues}) != 682:
        raise ValueError('Expected one guided rescue per eligible task')
    rescue_ids = {r['task_id'] for r in rescues}
    conditioning = [r for r in records if r['task_id'] in rescue_ids]
    if len(conditioning) != 682*5 or any(not r['valid'] or success(r) for r in conditioning):
        raise ValueError('Guided cohort does not match five-valid-failure screen')
    rescue = dict(tasks=len(rescues), successes=sum(success(r) for r in rescues),
        valid=sum(r['valid'] for r in rescues),
        guided_cost={k: sum(r[k] for r in rescues) for k in ('actor_requests','actor_output_tokens','steps','selector_calls')},
        conditioning_five_failures_cost={k: sum(r[k] for r in conditioning) for k in ('actor_requests','actor_output_tokens','steps')},
        fresh_actor_control_available=False,
        limitation='The five earlier failures selected this cohort; comparing their zero yield to the fresh guided retry is selection bias.')
    piotr = RUNTIME/'arm-reproduction/joint-data-inputs-0d83b48'
    eps = defaultdict(set); state_count = 0
    for line in (piotr/'openwebrl_actor/states_full.jsonl').open():
        r = json.loads(line); eps[str(r['episode_id'])].add(r['turn']); state_count += 1
    candidate_audit = json.loads((piotr/'candidate-identity-audit.json').read_text())
    report = dict(historical_om2w=old, screening_2000=screen, guided_682=rescue,
        piotr_openwebrl=dict(states=state_count, episodes=len(eps), adjacent_demonstration_pairs=sum(t+1 in turns for turns in eps.values() for t in turns),
            **{k:candidate_audit[k] for k in ('rows','unique_draw_ids','conflicting_duplicate_records')},
            limitation='Adjacency belongs to the source demo_action, not the subsequently sampled candidates. No per-candidate rollout return in these files.'),
        resources=dict(gpus=0, api_calls=0, wall_seconds=round(time.monotonic()-started,2)))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=RUNTIME/'arm-turn-bonus-preparation/outcome-reward-20260927/critic-comparison-20261004/inference-scaling.json')
    args = p.parse_args()
    main(args.output)
