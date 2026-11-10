#!/usr/bin/env python3
"""Offline rubric-difficulty/pass@k analysis of the immutable five-attempt screen.

Writes aggregate evidence only. Does not run models, browsers, or judges.
"""
import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = '1'

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from math import comb
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
from scipy.stats import pearsonr, spearmanr

RUNTIME = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime')
POOL = RUNTIME/'arm-turn-bonus-preparation/task-pool-expansion-20260922/curation-v3-20260929'


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def curve(successes, attempts=5):
    return [1 - (comb(attempts-successes, k) if attempts-successes >= k else 0)
            / comb(attempts, k) for k in range(1, attempts+1)]


def analyze(output, replicates=2000):
    native_path = POOL/'cohorts-20260930/comparison-min5-weighted-2000/weighted-tasks.jsonl'
    frozen_path = RUNTIME/'reference-arm-task-screen-sharded-20261002-v2/screen-tasks.jsonl'
    old = POOL/'actor-screen-min5-2000/sharded-20261002'
    native = read_rows(native_path)
    by_text = {(r['task_name'], r['website']): r for r in native}
    assert len(native) == len(by_text) == 2000
    frozen = read_rows(frozen_path)
    metadata = {}
    for row in frozen:
        m = row['metadata']
        task = by_text[(m['task'], m['start_url'])]
        metadata[m['task_id']] = task
    assert len(frozen) == len(metadata) == 2000
    rubric_matches = sum(r['difficulty'] == sum(len(g['facts']) for g in r['evaluator_reference'])
                         for r in native)
    assert rubric_matches == 2000, 'Difficulty definition differs from rubric-fact count'
    records = defaultdict(dict)
    manifest = hashlib.sha256()
    for path in sorted(old.glob('shard-*/records/*.json')):
        data = path.read_bytes()
        r = json.loads(data)
        task, attempt = r['task_id'], r['attempt']
        assert task in metadata and attempt in range(5) and attempt not in records[task]
        assert r['policy_id'] == 'OpenWebRL-4B-SFT:iteration0'
        assert r['judge_id'] == 'gpt-4.1/action_history'
        assert type(r['valid']) is bool and (not r['valid'] or r['reward'] in (-1, 0, 1))
        records[task][attempt] = r
        manifest.update(str(path.relative_to(old)).encode() + b'\0' + hashlib.sha256(data).digest())
    assert len(records) == 2000 and all(set(rs) == set(range(5)) for rs in records.values())
    ids = sorted(records)
    x = np.array([metadata[t]['difficulty'] for t in ids], dtype=float)
    c = np.array([sum(r['valid'] and r['reward'] == 1 for r in records[t].values()) for t in ids])
    valid = np.array([sum(r['valid'] for r in records[t].values()) for t in ids])
    y = np.array([curve(int(v)) for v in c])
    sites = np.array([urlsplit(metadata[t]['website']).hostname.lower().removeprefix('www.') for t in ids])
    sources = np.array([metadata[t]['benchmark_name'] for t in ids])
    assert c.sum() == 3597 and (c > 0).sum() == 1293 and valid.sum() == 9936
    assert np.allclose(y.mean(0), [.3597, .49285, .5654, .6128, .6465])

    def summary(mask):
        return dict(tasks=int(mask.sum()), attempts=int(5*mask.sum()),
                    valid_attempts=int(valid[mask].sum()),
                    all_five_valid_tasks=int((valid[mask] == 5).sum()),
                    tasks_with_success=int((c[mask] > 0).sum()),
                    zero_success_tasks=int((c[mask] == 0).sum()),
                    success_count_histogram={str(j): int((c[mask] == j).sum()) for j in range(6)},
                    pass_at_k={str(k): float(y[mask, k-1].mean()) for k in range(1, 6)})

    def correlations(mask):
        return {str(k): dict(pearson=float(pearsonr(x[mask], y[mask, k-1]).statistic),
                             spearman=float(spearmanr(x[mask], y[mask, k-1]).statistic))
                for k in range(1, 6)}

    # Resample whole websites to avoid treating same-site tasks as independent.
    rng = np.random.default_rng(20261006)
    groups = [np.flatnonzero(sites == s) for s in sorted(set(sites))]
    samples = np.empty((replicates, 5, 2))
    for b in range(replicates):
        chosen = np.concatenate([groups[j] for j in rng.integers(len(groups), size=len(groups))])
        for k in range(5):
            samples[b,k,0] = pearsonr(x[chosen], y[chosen,k]).statistic
            samples[b,k,1] = spearmanr(x[chosen], y[chosen,k]).statistic
    corr = correlations(np.ones(len(x), dtype=bool))
    for k in range(5):
        for j, name in enumerate(('pearson','spearman')):
            corr[str(k+1)][name+'_website_bootstrap_95ci'] = np.quantile(samples[:,k,j], [.025,.975]).tolist()

    # Demeaning within source/site checks whether gross differences are only
    # associated with website and dataset composition. It is not causal control.
    residual_x, residual_y = x.copy(), y.copy()
    strata = [(source, site) for source, site in zip(sources, sites)]
    for stratum in set(strata):
        mask = np.array([s == stratum for s in strata])
        residual_x[mask] -= x[mask].mean()
        residual_y[mask] -= y[mask].mean(0)
    result = dict(schema_version=1, tasks=2000, primary_attempts=10000,
        actor='OpenWebRL/OpenWebRL-4B-SFT:iteration0',
        protocol=dict(temperature=.8, top_p=1., top_k=-1, max_response_tokens=1024,
                      max_turns=15, browser='local', judge='gpt-4.1/action_history'),
        difficulty_definition='Number of evaluator-reference rubric facts; verified on all 2000 tasks',
        estimator='mean_task[1 - C(5 - successes, k) / C(5,k)], invalids count as no success',
        input_hashes=dict(native_tasks_sha256=hashlib.sha256(native_path.read_bytes()).hexdigest(),
                         frozen_tasks_sha256=hashlib.sha256(frozen_path.read_bytes()).hexdigest(),
                         ordered_record_manifest_sha256=manifest.hexdigest()),
        overall=summary(np.ones(len(x), dtype=bool)),
        difficulty_counts={str(int(v)):int((x == v).sum()) for v in sorted(set(x))},
        by_difficulty={str(int(v)): summary(x == v) for v in sorted(set(x))},
        by_difficulty_group={**{str(v):summary(x == v) for v in (5,6,7)}, '8+':summary(x >= 8)},
        correlations=corr,
        bootstrap=dict(unit='website', websites=len(groups), replicates=replicates, seed=20261006),
        all_five_valid=dict(**summary(valid == 5), correlations=correlations(valid == 5)),
        by_source={source:dict(**summary(sources == source), correlations=correlations(sources == source))
                   for source in sorted(set(sources))},
        within_source_website_pearson={str(k): float(pearsonr(residual_x,residual_y[:,k-1]).statistic)
                                      for k in range(1,6)},
        rescue_partition=dict(original_success_tasks=1293, zero_success_tasks=707,
            five_valid_failure_tasks=int(((c == 0) & (valid == 5)).sum()),
            zero_success_with_invalid_tasks=int(((c == 0) & (valid < 5)).sum()),
            extension_tasks_needed_for_observed_cumulative_pass8=707, extension_primary_episodes=2121),
        limitations=['Selected training pool with minimum rubric difficulty 5, not full difficulty range or OM2W.',
            'Only 50 tasks have difficulty >=8; upper levels should not be interpreted separately.',
            'Rubric fact count does not measure interaction difficulty directly.',
            'Estimated pass@1..5 share outcomes and are not independent evidence.',
            'Website bootstrap quantifies between-task/site uncertainty, not systematic judge error.',
            'Coverage across historical five plus fresh three is not a contemporaneous eight-rollout study.'])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('difficulty_counts','by_difficulty_group','correlations',
                                          'within_source_website_pearson','bootstrap')}, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--bootstrap-replicates', type=int, default=2000)
    args = p.parse_args()
    analyze(args.output, args.bootstrap_replicates)
