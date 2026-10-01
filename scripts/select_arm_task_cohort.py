#!/usr/bin/env python3
"""Freeze a screening cohort from the full eligible task pool; no rollouts."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'

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


def run(root, output, budget, minimum_difficulty=None, diagnostic=False):
    start = time.monotonic()
    output.mkdir(parents=True, exist_ok=False)
    plan, records, selected_path = load_metadata(root)
    cluster_root = root / 'clusters-20260930/run'
    expected = json.loads((cluster_root / 'summary.json').read_text())
    for name in ('clusters.jsonl', 'assignments-r50.jsonl'):
        assert sha(cluster_root / name) == expected['artifacts'][name]
    assignment = {r['embedding_row']: r for r in rows(cluster_root / 'assignments-r50.jsonl')}
    assert len(assignment) == len(records)
    full_eligible_count = len(records)
    if minimum_difficulty is not None:
        records = [r for r in records if r['rubric_difficulty'] >= minimum_difficulty]
    by_site = defaultdict(list)
    for r in records:
        a = assignment[r['embedding_row']]
        assert (r['source'], r['task_id'], r['site'], r['primary_workflow']) == (
            a['source'], a['task_id'], a['site'], a['primary_workflow'])
        by_site[r['site']].append(r)
    clusters = {c['cluster_id']: c for c in rows(cluster_root / 'clusters.jsonl')
                if '-r50-' in c['cluster_id']}
    site_clusters = {s:len({assignment[r['embedding_row']]['cluster_id'] for r in rs})
                     for s,rs in by_site.items()}
    allocation = quotas(budget, {s: len(rs) for s, rs in by_site.items()},
                        {s: site_clusters[s] ** .5 for s in by_site})
    cache = root / 'semantic-qwen8b/embeddings.npy'
    stat = cache.stat()
    emb = np.load(cache, mmap_mode='r')
    assert emb.shape == (plan['total_texts'], 4096)
    choices = []
    for ordinal, site in enumerate(sorted(by_site, key=lambda s: (-len(by_site[s]), s))):
        members = sorted(by_site[site], key=lambda r: r['embedding_row'])
        x = np.asarray(emb[[r['embedding_row'] for r in members]])
        emb._mmap.madvise(mmap.MADV_DONTNEED)
        assert np.isfinite(x).all()
        assert np.max(np.abs(np.linalg.norm(x, axis=1) - 1)) < 1e-4
        group = defaultdict(list)
        for i, r in enumerate(members):
            group[assignment[r['embedding_row']]['cluster_id']].append(i)
        ids = sorted(group)
        centers = np.asarray([x[group[c]].mean(axis=0) for c in ids])
        centers /= np.linalg.norm(centers, axis=1, keepdims=True)
        assert all(len(group[c]) <= clusters[c]['size'] for c in ids)
        if minimum_difficulty is None:
            assert all(len(group[c]) == clusters[c]['size'] for c in ids)
        weights = np.sqrt(np.asarray([len(group[c]) for c in ids], dtype=np.float64))
        weights /= weights.sum()
        sim = np.maximum(centers @ x.T, 0)
        order = facility_order(sim, weights, allocation[site])
        assert len(order) == len(set(order)) == allocation[site]
        for rank, j in enumerate(order, 1):
            r = dict(members[j])
            r.update(cluster_id=assignment[r['embedding_row']]['cluster_id'],
                     selection_method='weighted_coverage_full_pool', within_site_rank=rank)
            choices.append(r)
        if ordinal < 10:
            print(site, len(members), 'selected', allocation[site],
                  'elapsed', round(time.monotonic() - start, 2), flush=True)
        del x, centers, sim
    assert len(choices) == budget
    key = lambda r: (r['source'], str(r['task_id']))
    selected = {key(r): r for r in choices}
    assert len(selected) == budget
    assert dict(Counter(r['site'] for r in choices)) == allocation
    # Preserve the native task payload exactly; metadata is a separate manifest.
    native = []
    for r in rows(root / 'candidates.jsonl'):
        identity = (r['benchmark_name'], str(r['task_id']))
        if identity in selected:
            assert r['task_name'] == selected[identity]['instruction']
            assert r['website'] == selected[identity]['start_url']
            native.append(r)
    assert len(native) == budget
    with (output / 'weighted-tasks.jsonl').open('w') as f:
        for r in native:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    with (output / 'selection-manifest.jsonl').open('w') as f:
        for r in sorted(choices, key=lambda r: (r['site'], r['within_site_rank'])):
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    write_json(output / 'website-quotas.json', allocation)
    assert cache.stat().st_size == stat.st_size and cache.stat().st_mtime_ns == stat.st_mtime_ns
    summary = dict(
        selection_complete=True, status='diagnostic_alternative_only' if diagnostic else 'prepared_for_screening_not_training',
        tasks=budget, websites=len(allocation), eligible_tasks=len(records),
        full_eligible_tasks=full_eligible_count, minimum_rubric_difficulty=minimum_difficulty,
        diagnostic_alternative=diagnostic,
        expected_actor_attempts_per_task=5, primary_actor_trajectories=5 * budget,
        rollouts_launched=False, random_comparison_prepared=False,
        training_data_changed=False, api_calls=0, gpu_hours=0,
        candidate_pool=f'All {len(records):,} eligible tasks after the declared minimum rubric-fact filter; no diagnostic shortlist',
        objective='sum_c sqrt(n_c) max_j max(0, cosine(normalized_center_c, task_embedding_j)) within website',
        quota='One/site, then highest averages with sqrt(fine_cluster_count), capped by full site capacity',
        tie_breaking='Ascending embedding row within each site; deterministic lazy greedy',
        sources=dict(Counter(r['source'] for r in choices)),
        difficulty_counts=dict(Counter(r['rubric_difficulty'] for r in choices)),
        workflows=dict(Counter(r['primary_workflow'] for r in choices)),
        clusters_represented=len({r['cluster_id'] for r in choices}),
        implementation_sha256=sha(Path(__file__)),
        dependencies_sha256={name: sha(Path(__file__).with_name(name)) for name in
                             ['cluster_arm_task_pool.py', 'visualize_arm_task_clusters.py']},
        input_sha256={'selected_tasks': sha(selected_path),
                      'clusters': sha(cluster_root / 'clusters.jsonl'),
                      'assignments': sha(cluster_root / 'assignments-r50.jsonl'),
                      'ordered_embedding_texts': plan['text_sha256']},
        artifacts={p.name: sha(p) for p in output.iterdir() if p.is_file()},
        elapsed_seconds=time.monotonic() - start,
        cpu_seconds=time.process_time(),
        peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        caveats=['Initial page availability does not establish task quality or solvability.',
                 'This freezes a screening cohort, not a final RL training dataset.',
                 'The prior diagnostic used a shortlist; its coverage scores do not describe this new full-pool cohort.'])
    write_json(output / 'summary.json', summary)
    print(json.dumps({k: summary[k] for k in ['tasks', 'websites', 'clusters_represented',
          'elapsed_seconds', 'cpu_seconds', 'peak_rss_mib', 'primary_actor_trajectories']}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tasks', type=int, default=2000)
    parser.add_argument('--minimum-difficulty', type=int)
    parser.add_argument('--diagnostic', action='store_true')
    args = parser.parse_args()
    resource.setrlimit(resource.RLIMIT_CPU, (180, 185))
    with threadpool_limits(limits=1):
        run(args.root, args.output, args.tasks, args.minimum_difficulty, args.diagnostic)
