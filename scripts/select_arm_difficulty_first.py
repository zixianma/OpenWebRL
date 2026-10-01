#!/usr/bin/env python3
"""Rank by rubric fact count; break the final score tie by weighted coverage."""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'

import argparse
from collections import Counter, defaultdict
import json
import mmap
from pathlib import Path
import resource
import time

import numpy as np
from threadpoolctl import threadpool_limits
from cluster_arm_task_pool import load_metadata, rows, sha, write_json
from visualize_arm_task_clusters import facility_order, quotas


def split_cutoff(records, budget):
    if budget < 1 or budget > len(records):
        raise ValueError('Invalid cohort size')
    for r in records:
        if not isinstance(r['rubric_difficulty'], int) or r['rubric_difficulty'] < 1:
            raise ValueError('Expected a positive integer rubric-fact score')
    cutoff = sorted((r['rubric_difficulty'] for r in records), reverse=True)[budget-1]
    mandatory = [r for r in records if r['rubric_difficulty'] > cutoff]
    tied = [r for r in records if r['rubric_difficulty'] == cutoff]
    return cutoff, mandatory, tied, budget-len(mandatory)


def main(root, output, budget):
    start = time.monotonic();output.mkdir(parents=True, exist_ok=False)
    plan, records, eligible_path = load_metadata(root)
    cluster_root = root/'clusters-20260930/run'
    expected = json.loads((cluster_root/'summary.json').read_text())
    assert sha(cluster_root/'assignments-r50.jsonl') == expected['artifacts']['assignments-r50.jsonl']
    assignment = {r['embedding_row']: r for r in rows(cluster_root/'assignments-r50.jsonl')}
    assert len(assignment) == len(records)
    for r in records:
        a = assignment[r['embedding_row']]
        assert (r['source'], r['task_id'], r['site']) == (a['source'], a['task_id'], a['site'])
    cutoff, mandatory, tied, remaining = split_cutoff(records, budget)
    by_site = defaultdict(list);fixed_by_site = defaultdict(list)
    for r in tied:by_site[r['site']].append(r)
    for r in mandatory:fixed_by_site[r['site']].append(r)
    # Website balancing is restricted to equal-score candidates. Never admit a
    # lower-scored task to fill a website quota.
    extra = quotas(remaining, {s:len(rs) for s,rs in by_site.items()},
                   {s:len({assignment[r['embedding_row']]['cluster_id'] for r in rs})**.5
                    for s,rs in by_site.items()})
    cache = root/'semantic-qwen8b/embeddings.npy';cache_stat = cache.stat()
    emb = np.load(cache, mmap_mode='r');assert emb.shape == (plan['total_texts'], 4096)
    chosen = [dict(r, selection_reason='above_cutoff', cutoff_selection_rank=None) for r in mandatory]
    for site in sorted(by_site):
        rs = sorted(by_site[site], key=lambda r:r['embedding_row'])
        x = np.asarray(emb[[r['embedding_row'] for r in rs]])
        assert np.isfinite(x).all() and np.max(np.abs(np.linalg.norm(x,axis=1)-1)) < 1e-4
        groups = defaultdict(list)
        for i,r in enumerate(rs):groups[assignment[r['embedding_row']]['cluster_id']].append(i)
        ids = sorted(groups)
        # Restrict each existing cluster to the cutoff tier; do not optimize
        # representation of lower-difficulty tasks outside the candidate pool.
        centers = np.asarray([x[groups[c]].mean(axis=0) for c in ids])
        centers /= np.linalg.norm(centers,axis=1,keepdims=True)
        weights = np.sqrt(np.asarray([len(groups[c]) for c in ids], dtype=np.float64));weights /= weights.sum()
        fixed = fixed_by_site[site]
        initial = (np.maximum(centers @ np.asarray(emb[[r['embedding_row'] for r in fixed]]).T,0).max(axis=1)
                   if fixed else np.zeros(len(centers)))
        sim = np.maximum(centers @ x.T,0)
        order = facility_order(sim, weights, extra[site], initial_coverage=initial)
        assert len(order) == len(set(order)) == extra[site]
        chosen.extend(dict(rs[j], selection_reason='weighted_cutoff_tie', cutoff_selection_rank=rank)
                      for rank,j in enumerate(order,1))
        del x, centers, sim
        emb._mmap.madvise(mmap.MADV_DONTNEED)
    chosen.sort(key=lambda r:(r['site'], -r['rubric_difficulty'], r['cutoff_selection_rank'] or 0,r['embedding_row']))
    counters = Counter()
    for r in chosen:
        counters[r['site']] += 1
        r.update(within_site_rank=counters[r['site']], cluster_id=assignment[r['embedding_row']]['cluster_id'],
                 selection_method='rubric_difficulty_first_weighted_cutoff_tie')
    chosen_ids = {(r['source'],r['task_id']) for r in chosen}
    assert len(chosen_ids) == len(chosen) == budget
    assert {(r['source'],r['task_id']) for r in mandatory} <= chosen_ids
    assert min(r['rubric_difficulty'] for r in chosen) == cutoff
    assert sum(r['rubric_difficulty'] == cutoff for r in chosen) == remaining
    native = []
    lookup = {(r['source'],r['task_id']):r for r in chosen}
    for r in rows(root/'candidates.jsonl'):
        key = r['benchmark_name'],str(r['task_id'])
        if key in chosen_ids:
            m = lookup[key]
            assert r['task_name'] == m['instruction'] and r['website'] == m['start_url']
            assert r['difficulty'] == sum(len(v.get('facts',[])) for v in r['evaluator_reference'])
            native.append(r)
    assert len(native) == budget
    for name,values in [('weighted-tasks.jsonl',native),('selection-manifest.jsonl',chosen)]:
        with (output/name).open('w') as f:
            for r in values:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    write_json(output/'website-quotas.json',dict(counters))
    write_json(output/'cutoff-website-quotas.json',extra)
    previous = {(r['source'],str(r['task_id'])) for r in rows(root/'cohorts-20260930/weighted-2000/selection-manifest.jsonl')}
    assert cache.stat().st_size == cache_stat.st_size and cache.stat().st_mtime_ns == cache_stat.st_mtime_ns
    summary = dict(status='prepared_for_review_and_screening_not_training',selection_complete=True,
        tasks=budget,eligible_tasks=len(records),selection_mode='difficulty_first',cutoff_score=cutoff,
        mandatory_above_cutoff=len(mandatory),cutoff_candidates=len(tied),selected_at_cutoff=remaining,
        difficulty_counts=dict(Counter(r['rubric_difficulty'] for r in chosen)),
        bands=dict(Counter('hard' if r['rubric_difficulty']>=7 else 'medium' if r['rubric_difficulty']>=4 else 'easy' for r in chosen)),
        websites=len(counters),clusters_represented=len({r['cluster_id'] for r in chosen}),
        overlap_with_original_weighted=len(previous&chosen_ids),
        cutoff_coverage='Existing fine clusters restricted to equal-score candidates; normalized centers and sqrt(tier cluster size); marginal coverage starts with retained higher-score tasks',
        cutoff_quota='One per cutoff website, then sqrt(tier cluster count) highest averages, capped by available cutoff candidates',
        difficulty_definition='Source rubric fact count; not measured actor success or browser steps',
        expected_actor_attempts_per_task=5,primary_actor_trajectories=budget*5,
        rollouts_launched=False,random_comparison_prepared=False,training_data_changed=False,
        api_calls=0,gpu_hours=0,previous_cohort_preserved=True,
        implementation_sha256=sha(Path(__file__)),
        helper_sha256=sha(Path(__file__).with_name('visualize_arm_task_clusters.py')),
        input_sha256={'eligible':sha(eligible_path),'assignments':sha(cluster_root/'assignments-r50.jsonl'),
                      'ordered_texts':plan['text_sha256']},
        artifacts={p.name:sha(p) for p in output.iterdir() if p.is_file()},
        elapsed_seconds=time.monotonic()-start,cpu_seconds=time.process_time(),
        peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        caveats=['Rubric redundancy can inflate difficulty; malformed tasks are not established hard tasks.',
                 'No website floor may override the primary difficulty ranking.',
                 'A matched random tie-break control shares all mandatory higher-score tasks.'])
    write_json(output/'summary.json',summary)
    print(json.dumps({k:summary[k] for k in ['tasks','cutoff_score','mandatory_above_cutoff','selected_at_cutoff',
        'bands','websites','clusters_represented','overlap_with_original_weighted','elapsed_seconds','peak_rss_mib']},indent=2))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--tasks',type=int,default=2000)
    a=p.parse_args();resource.setrlimit(resource.RLIMIT_CPU,(180,185))
    with threadpool_limits(limits=1):main(a.root,a.output,a.tasks)
