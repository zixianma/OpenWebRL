#!/usr/bin/env python3
"""Rebuild additional-task candidates with pinned released OpenWebRL utilities.

CPU metadata only. The public release does not include its popularity allowlist
or original semantic curation program. Outputs are candidates, never train-ready
tasks. Existing pools, browser jobs, and old review decisions are not modified.
"""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import resource
import time
from urllib.parse import urlsplit

REPO = Path(__file__).resolve().parents[1]
ROOT = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-turn-bonus-preparation/task-pool-expansion-20260922')
UPSTREAM_SHA = '9a120949aca3e58a2628f4b4e6edd0474d984873'
ALLOWED = {'insta-v3', 'pae-webvoyager'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_rows(path):
    with Path(path).open() as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def frozen_json(path, data):
    raw = json.dumps(data, indent=2, ensure_ascii=False) + '\n'
    if path.exists() and path.read_text() != raw:
        raise ValueError('Refusing to replace frozen artifact: ' + str(path))
    if not path.exists():
        path.write_text(raw)


def frozen_rows(path, rows):
    # Exclusive creation prevents silent replacement of a prior selection.
    with path.open('x') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def load_upstream(directory):
    manifest = json.loads((directory / 'source.json').read_text())
    if manifest['commit'] != UPSTREAM_SHA:
        raise ValueError('Unexpected upstream revision')
    for name, expected in manifest['sha256'].items():
        if sha(directory / name) != expected:
            raise ValueError('Upstream snapshot mismatch: ' + name)
    spec = importlib.util.spec_from_file_location('released_openwebrl_converter', directory / 'convert_benchmark_jsonl_to_parquet.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, manifest


def text(row):
    m = row.get('metadata') or {}
    return row.get('task_name') or row.get('confirmed_task') or row.get('question') or row.get('ques') or row.get('task') or m.get('task') or m.get('intent') or ''


def leakage_key(instruction):
    # Our conservative exact-match screen; not claimed as upstream code.
    return ' '.join(re.findall(r'\w+', instruction.casefold()))


def task_key(row, upstream):
    # Identical generic instructions on different pages are not identical tasks.
    # Preserve punctuation, case, path, and query. Upstream normalizes URL aliases.
    return (' '.join(row['task_name'].split()), upstream.normalize_start_url(row['website']))


def host(row):
    return (urlsplit(row['website']).hostname or '').lower().removeprefix('www.')


def stats(rows):
    return dict(tasks=len(rows), hosts=len({host(r) for r in rows}),
                sources=dict(Counter(r['benchmark_name'] for r in rows)),
                rubric_difficulty=dict(Counter('unknown' if not isinstance(r.get('difficulty'), (int, float)) else
                    'hard_7plus' if r['difficulty'] >= 7 else 'medium_4to6' if r['difficulty'] >= 4 else 'easy_1to3' for r in rows)))


def dedup_exact(rows, upstream):
    seen, kept, dropped = {}, [], []
    for row in rows:
        key = task_key(row, upstream)
        if key in seen:
            dropped.append(dict(task_id=str(row['task_id']), reason='identical_instruction_and_start_url', representative_id=seen[key]))
        else:
            seen[key] = str(row['task_id'])
            kept.append(row)
    return kept, dropped


def run(root, output, upstream_dir):
    import pyarrow.parquet as pq
    resource.setrlimit(resource.RLIMIT_CPU, (90, 95))
    started = time.monotonic()
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'audit.json').exists():
        raise ValueError('Completed output exists; choose a new version directory')
    upstream, provenance = load_upstream(upstream_dir)
    sources = json.loads((root / 'sources.json').read_text())
    for item in sources['files']:
        if sha(root / item['file']) != item['sha256']:
            raise ValueError('Input hash mismatch: ' + item['file'])
    released = list(read_rows(root / 'openwebrl-rl-tasks.jsonl'))
    heldout = list(read_rows(root / 'webgym-test.jsonl'))
    refs = [dict(source='webgym_test', task_id=str(r['task_id']), text=r['task_name']) for r in heldout]
    input_hashes = {}
    for relative in ['openwebrl/data/eval/online-mind2web.jsonl', 'openwebrl/data/eval/deepshop.jsonl',
                     'openwebrl/data/eval/webvoyager_fara.jsonl', 'openwebrl/data/online-mind2web.jsonl']:
        path = REPO / relative
        input_hashes[relative] = sha(path)
        for i, row in enumerate(read_rows(path)):
            if not text(row):
                raise ValueError('Missing heldout instruction: ' + relative)
            refs.append(dict(source=relative, task_id=str(row.get('task_id', row.get('id', i))), text=text(row)))
    validation = REPO / 'openwebrl/data/webvoyager_val.parquet'
    input_hashes[str(validation.relative_to(REPO))] = sha(validation)
    for i, row in enumerate(pq.read_table(validation).to_pylist()):
        instruction = text(row) or row.get('prompt', [{}])[-1].get('content', '')
        if not instruction:
            raise ValueError('Missing validation instruction')
        refs.append(dict(source='webvoyager_validation', task_id=str(i), text=instruction))
    heldout_names = {leakage_key(r['text']) for r in refs}
    refs += [dict(source='released_openwebrl', task_id=str(r['task_id']), text=r['task_name']) for r in released]
    frozen_rows(output / 'semantic-references.jsonl', refs)

    records = list(read_rows(root / 'webgym-train.jsonl'))
    if len({str(r['task_id']) for r in records}) != len(records):
        raise ValueError('Nonunique raw task IDs')
    counts = dict(raw=stats(records))
    rejected = []

    def screen(reason, predicate):
        nonlocal records
        kept = []
        for row in records:
            if predicate(row):
                kept.append(row)
            else:
                rejected.append(dict(task_id=str(row['task_id']), reason=reason))
        records = kept
        counts[reason] = stats(records)

    screen('source_scope', lambda r: r['benchmark_name'] in ALLOWED)
    screen('original_not_decomposed', lambda r: r.get('task_id_decomposed_from') is None)
    screen('well_formed_instruction_url', lambda r: isinstance(r.get('task_name'), str) and r['task_name'].strip()
           and isinstance(r.get('website'), str) and urlsplit(r['website']).scheme in {'http', 'https'}
           and bool(urlsplit(r['website']).hostname))
    exclusions = {'webvoyager/' + str(r['task_id']) for r in released + heldout}
    records, dropped = upstream.filter_records_by_excluded_task_ids(records, 'webvoyager', exclusions)
    rejected += [dict(task_id=str(r['task_id']), reason='released_training_or_webgym_test_id') for r in dropped]
    counts['released_training_or_webgym_test_id'] = stats(records)
    screen('exact_heldout_intent', lambda r: leakage_key(r['task_name']) not in heldout_names)
    before_blacklist = stats(records)
    blacklist = upstream.load_host_blacklist(str(upstream_dir / 'webgym_filtered_popular_blacklist_hosts.txt'))
    records, quarantined = upstream.filter_records_by_host_blacklist(records, blacklist)
    frozen_rows(output / 'upstream-blacklist-quarantine.jsonl', quarantined)
    rejected += [dict(task_id=str(r['task_id']), reason='upstream_blacklist_quarantine') for r in quarantined]
    counts['upstream_host_blacklist'] = stats(records)
    # Same upstream source file's order; no difficulty or per-site selection.
    records, exact_duplicates = dedup_exact(records, upstream)
    rejected += exact_duplicates
    counts['exact_task_dedup'] = stats(records)
    # Explicitly retain generic instructions at different URLs until semantic review.
    text_counts = Counter(' '.join(r['task_name'].split()) for r in records)
    repeated_text = sum(n for n in text_counts.values() if n > 1)

    frozen_rows(output / 'candidates.jsonl', records)
    frozen_rows(output / 'exclusions.jsonl', rejected)
    by_site = Counter(host(r) for r in records)
    frozen_rows(output / 'site-inventory.jsonl', [dict(host=h, tasks=n) for h, n in sorted(by_site.items(), key=lambda x: (-x[1], x[0]))])
    # Read-only example panel; every retained task is in candidates.jsonl.
    examples = []
    for source in sorted(ALLOWED):
        for lower, upper in [(1, 3), (4, 6), (7, 100000)]:
            group = [r for r in records if r['benchmark_name'] == source and lower <= (r.get('difficulty') or 0) <= upper]
            examples += sorted(group, key=lambda r: hashlib.sha256(('curation-v3:' + str(r['task_id'])).encode()).hexdigest())[:5]
    frozen_rows(output / 'examples-30.jsonl', examples)

    current_path = REPO / 'openwebrl/data/webgym_filtered_popular_2102_cleaned.parquet'
    current_hash = sha(current_path)
    active = pq.read_table(current_path).to_pylist()
    published = pq.read_table(upstream_dir / 'webgym_filtered_popular_2102_cleaned.parquet').to_pylist()
    active_by_id = {r['metadata']['task_id']: r for r in active}
    converted = [upstream.record_to_row(r, 'webvoyager') for r in released if 'webvoyager/' + str(r['task_id']) in active_by_id]
    converter_equal = all(r == active_by_id[r['metadata']['task_id']] for r in converted)
    upstream_ids = {str(r['metadata']['task_id']) for r in published}
    filtered_release, filtered_drop = upstream.filter_records_by_host_blacklist(released, blacklist)
    filtered_ids = {'webvoyager/' + str(r['task_id']) for r in filtered_release}
    parity = dict(active_tasks=len(active), converted_tasks=len(converted), converted_row_values_match=converter_equal,
        active_parquet_matches_published_bytes=current_hash == sha(upstream_dir / current_path.name),
        published_ids_equal_active=upstream_ids == set(active_by_id),
        released_tasks=len(released), released_after_current_blacklist=len(filtered_release),
        blacklist_rows_removed=len(filtered_drop), active_rows_blacklisted=len(set(active_by_id) - filtered_ids),
        blacklist_survivors_absent_from_active=len(filtered_ids - set(active_by_id)))
    if not converter_equal or len(converted) != len(active) or not parity['published_ids_equal_active']:
        raise ValueError('Released converter/snapshot parity failed: ' + str(parity))
    # Prove active inputs are untouched, including upstream's mutable blacklist.
    if sha(current_path) != current_hash:
        raise ValueError('Active parquet changed during metadata pass')
    artifact_names = ['candidates.jsonl', 'semantic-references.jsonl', 'upstream-blacklist-quarantine.jsonl',
                      'exclusions.jsonl', 'site-inventory.jsonl', 'examples-30.jsonl']
    audit = dict(schema=3, date='2026-09-29', status='metadata_complete_semantic_and_site_quality_pending',
        sources=sources, upstream=provenance, local_reference_sha256=input_hashes, stages=counts,
        rejection_counts=dict(Counter(r['reason'] for r in rejected)), before_blacklist=before_blacklist,
        blacklist_hosts=len(blacklist), blacklist_quarantine=stats(quarantined), upstream_parity=parity,
        repeated_instruction_rows_on_distinct_start_urls=repeated_text,
        retained_missing_rubric=sum(not r.get('evaluator_reference') for r in records),
        removed_old_rules=['existing_host_membership', 'five_tasks_per_host', 'medium_hard_only', 'token_jaccard_0.65', 'risk_ranked_panel_as_pool'],
        executed_upstream_functions=['filter_records_by_excluded_task_ids', 'load_host_blacklist',
            'filter_records_by_host_blacklist', 'normalize_start_url', 'record_to_row'],
        reconstructed_rules=['source_families', 'nondecomposed', 'well_formed_input', 'exact_heldout_intent', 'instruction_and_url_exact_dedup'],
        unreleased_original_details=['website_popularity_allowlist_and_rule', 'original_semantic_script',
            'embedding_prompt_pooling_batching_precision', 'greedy_order_and_tie_breaking', 'benchmark_overlap_implementation'],
        pending=['website_quality_popularity_review', 'Qwen3-Embedding-8B_greedy_0.95', 'semantic_heldout_overlap_review', 'live_browser_verification'],
        semantic_plan=dict(model='Qwen/Qwen3-Embedding-8B', revision='1d8ad4ca9b3dd8059ad90a75d4983776a23d44af',
            threshold=0.95, ordering='raw_train_order_after_metadata_filters', existing_and_heldout_as_exclusion_references=True,
            algorithm='blockwise_exact_cosine_greedy_against_previously_kept_candidates', exact_upstream_reproduction=False),
        training_data_changed=False, gpu_hours_used=0, api_calls=0,
        active_training_parquet_sha256=current_hash,
        artifacts={name:dict(path=str(output / name), sha256=sha(output / name)) for name in artifact_names},
        wall_seconds=round(time.monotonic() - started, 3))
    frozen_json(output / 'audit.json', audit)
    print(json.dumps({k: audit[k] for k in ['status', 'stages', 'rejection_counts', 'upstream_parity', 'wall_seconds']}, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--output', type=Path, default=ROOT / 'curation-v3-20260929')
    p.add_argument('--upstream', type=Path, default=ROOT / 'curation-v3-20260929/upstream')
    args = p.parse_args()
    run(args.root, args.output, args.upstream)
