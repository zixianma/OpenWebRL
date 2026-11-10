#!/usr/bin/env python3
"""CPU-only audit and paired rescue experiment specification; never launches jobs.

Task IDs, URLs, schedules and source records stay under the private runtime.
Only aggregate counts and experimental settings are suitable for publication.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import sys
from urllib.parse import urlsplit

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from openwebrl.arm_rescue_yield import stable_seed
from openwebrl.arm_task_screen import POLICY, JUDGE, summarize as screen_summary
from openwebrl.arm_task_rescue import ARM_REVISION, summarize as rescue_summary

RUNTIME = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime')
POOL = RUNTIME/'arm-turn-bonus-preparation/task-pool-expansion-20260922/curation-v3-20260929'
SOURCE = RUNTIME/'reference-arm-task-screen-sharded-20261002-v2'
SEED = 20261003
MODES = ('arm', 'actor0', 'actor1', 'actor2', 'actor3', 'actor4')


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze(path, value):
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n'
    path = Path(path)
    if path.exists() and path.read_text() != payload:
        raise ValueError(f'Prepared artifact changed; choose a new output directory: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload)


def host(row):
    value = urlsplit(row['metadata']['start_url']).hostname
    if not value:
        raise ValueError('Task has no start-URL hostname')
    return value.removeprefix('www.')


def validate_records(screen, guided, ids):
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate task IDs')
    for record in screen:
        if (record['policy_id'] != POLICY or record['judge_id'] != JUDGE or
                record['phase'] != 'screen' or record['mode'] != 'actor' or
                record['selector_calls'] != 0):
            raise ValueError('Screen protocol mismatch')
        if record['valid'] and record['reward'] not in (-1, 0, 1):
            raise ValueError('Unsupported native reward')
    screened = screen_summary(screen, ids, require_complete=True, verify_artifacts=False)
    eligible = [task for task in ids if screened['dispositions'][task] == 'all_failure']
    for record in guided:
        if (record['policy_id'] != POLICY or record['judge_id'] != JUDGE or
                record['phase'] != 'guided' or record['mode'] != 'arm' or
                record['selector_revision'] != ARM_REVISION or
                record['candidate_count'] != 5 or record['candidate_representation'] != 'full'):
            raise ValueError('Guided protocol mismatch')
        if record['valid'] and record['reward'] not in (-1, 0, 1):
            raise ValueError('Unsupported native reward')
    rescued = rescue_summary(guided, eligible, require_complete=True, verify_artifacts=False)
    return screened, eligible, rescued


def schedule(ids, shards=4):
    if len(ids) != len(set(ids)) or shards < 1:
        raise ValueError('Require unique task IDs and positive shard count')
    # Depends on task identity only, never historical rescue outcome or completion speed.
    order = sorted(ids, key=lambda task: (stable_seed(SEED, task, 'paired-task-order'), task))
    output = []
    for index, task in enumerate(order):
        modes = list(MODES)
        random.Random(stable_seed(SEED, task, 'paired-mode-order')).shuffle(modes)
        output.append(dict(task_id=task, shard=index % shards, attempts=[dict(
            mode=mode, attempt=0 if mode == 'arm' else int(mode[-1]),
            phase='selection_control', seed=stable_seed(SEED, POLICY, task,
                'selection_control', mode, 0 if mode == 'arm' else int(mode[-1]))) for mode in modes]))
    return output


def matched_random(rows, eligible, selected):
    by_site = defaultdict(list)
    for task in eligible:
        by_site[host(rows[task])].append(task)
    quotas = Counter(host(rows[task]) for task in selected)
    result = []
    for site, count in sorted(quotas.items()):
        # Include ARM successes in the sampling frame. Excluding them constructs
        # a known-negative control and exaggerates the effect of task selection.
        candidates = sorted(by_site[site], key=lambda task: (stable_seed(SEED, task, 'random-control'), task))
        if len(candidates) < count:
            raise ValueError('Cannot fill matched site quota')
        result.extend(candidates[:count])
    return result


def prepare(output, pool=POOL, source=SOURCE, verify_guided=True):
    screen_root = pool/'actor-screen-min5-2000/sharded-20261002'
    guided_root = pool/'actor-arm-allfailure-20261002'
    receipts = [screen_root/'agent-completion-review.json', guided_root/'agent-completion-review.json']
    receipt_values = [read(p) for p in receipts]
    if not all(r.get('verified_complete') and r.get('all_allocations_released') for r in receipt_values):
        raise ValueError('Both completed collections need verified receipts')
    if not receipt_values[0].get('every_archive_and_verdict_checked') or not receipt_values[1].get('every_archive_verdict_and_candidate_trace_checked'):
        raise ValueError('Missing independent artifact review')
    rows = [json.loads(line) for line in (source/'screen-tasks.jsonl').read_text().splitlines()]
    ids = [str(row['metadata']['task_id']) for row in rows]
    screen_paths = sorted(screen_root.glob('shard-*/records/*.json'))
    guided_paths = sorted(guided_root.glob('shard-*/records/*.json'))
    screen, guided = [list(map(read, paths)) for paths in (screen_paths, guided_paths)]
    screened, eligible, rescued = validate_records(screen, guided, ids)
    for actual, expected in [(screened, receipt_values[0]['summary']), (rescued, receipt_values[1]['summary'])]:
        for key in ('primary_attempts', 'valid_attempts', 'successes', 'invalid_attempts', 'policy_id'):
            if actual[key] != expected[key]:
                raise ValueError(f'Record/receipt mismatch: {key}')
    if verify_guided:
        rescue_summary(guided, eligible, require_complete=True)  # Sidecars + traces, not full archive CRC.
    indexed = {str(row['metadata']['task_id']): row for row in rows}
    selected = rescued['rescued_tasks']
    random_ids = matched_random(indexed, eligible, selected)
    eligible_set = set(eligible)
    failed_screen = [r for r in screen if r['task_id'] in eligible_set]
    site_counts = Counter(host(indexed[task]) for task in eligible)
    success_sites = Counter(host(indexed[task]) for task in selected)
    attempts = schedule(eligible)
    shard_seconds = [0.] * 4
    task_seconds = defaultdict(float)
    for record in failed_screen + guided:
        task_seconds[record['task_id']] += record['elapsed_seconds']
    for row in attempts:
        shard_seconds[row['shard']] += task_seconds[row['task_id']]
    provenance = dict(source_tasks_sha256=digest(source/'screen-tasks.jsonl'),
        receipts={str(p): digest(p) for p in receipts},
        screen_record_digest=hashlib.sha256(''.join(digest(p) for p in screen_paths).encode()).hexdigest(),
        guided_record_digest=hashlib.sha256(''.join(digest(p) for p in guided_paths).encode()).hexdigest(),
        verification='Record identities/counts reconciled to independent completion receipts; guided sidecars and candidate traces rechecked' if verify_guided else 'Record identities/counts reconciled; guided artifacts not rechecked')
    report = dict(eligible_tasks=len(eligible), historical_rescued_tasks=len(selected),
        historical_invalid_guided_attempts=rescued['invalid_attempts'],
        eligible_hosts=len(site_counts), rescued_hosts=len(success_sites),
        largest_seven_hosts_rescues=sum(n for _, n in success_sites.most_common(7)),
        eligible_tasks_with_native_format_failures=len({r['task_id'] for r in failed_screen if r['reward'] == -1}),
        guided_actor_output_tokens=rescued['actor_output_tokens'],
        original_five_failure_actor_output_tokens=sum(r['actor_output_tokens'] for r in failed_screen),
        guided_actor_requests=rescued['actor_requests'],
        original_five_failure_actor_requests=sum(r['actor_requests'] for r in failed_screen),
        site_matched_random_tasks=len(random_ids), random_arm_overlap=len(set(random_ids) & set(selected)),
        selected_share_expanded_pool=len(selected)/4102,
        runtime_sizing=dict(browser_slots_per_shard=8,
            ideal_hours_per_shard=[s/8/3600 for s in shard_seconds],
            caveat='Historical elapsed times include shared load; ideal slot arithmetic is a sizing reference, not a throughput prediction.'),
        limitations=['Historical five attempts define eligibility and cannot serve as an unbiased retry control.',
            'The 78 successes are discovery outcomes, not independent proof of selector gain or RL learnability.',
            'Host concentration confounds an unmatched task-selection comparison.',
            'Five actor retries approximate generation cost, not total compute including selector and browser costs.'])
    spec = dict(status='CPU specification and private schedule prepared; launcher and GPU validation pending',
        submitted=False, compute_approved=False, protocol=dict(actor=POLICY, selector_revision=ARM_REVISION,
            judge=JUDGE, browser='local', temperature=.8, top_p=1., top_k=-1,
            response_tokens=1024, context_tokens=32768, max_turns=15, candidates=5,
            candidate_representation='full reasoning and action', optimizer_updates=0),
        tasks=len(eligible), primary_attempts=len(eligible)*6, seed=SEED,
        design='Fresh ARM plus five actor retries per task; seeded within-task order; run all six regardless of success.',
        historical_78_role='Frozen discovery-selected subgroup, tested with new independent attempts; no historical outcomes reused as validation.',
        primary='Paired overall task rescue difference: fresh ARM versus actor0, all eligible tasks.',
        secondary=['ARM versus actor-any-of-five; both discordant counts and unique rescued tasks.',
            'Independent actor success on historical ARM-selected versus site-matched random tasks.',
            'Valid-only and both-valid sensitivity, invalid causes, actual tokens/requests, selector time, browser time and GPU-hours.',
            'Per-host descriptives; task bootstrap CI and host-cluster sensitivity; no training-seed claim.'],
        inference='Resample task IDs keeping all six attempts together. Missing results are incomplete; invalid attempts remain in overall denominators.',
        decision='Prefer ARM task selection only if independent yield/unique coverage and cost justify it versus actor retries; then test learning with matched training pools.',
        resources_proposal=dict(jobs=4, gpus_per_job=2, gpu_type='H200', hours_per_job=8,
            cpus_per_job=16, memory_gib_per_job=240, total_gpu_hours=64,
            judge_max_usd=50, judge_max_calls=12000, includes_all_retries=True),
        followup_training=dict(status='not prepared or approved', initial_iteration=0,
            common_recipe='outcome-only first to isolate task selection; same initialization, sampling share and optimization in all arms',
            pools=['historical ARM-rescued', 'site-matched random F0', 'site-matched actor-retry-selected'],
            matching='Equal per-host counts and total pool size. If actor-selected quotas are insufficient, reduce the corresponding quotas in every arm before training.',
            scope='Construct alternative pools; the current 4102 union already contains the 78, so appending them again does not test selection.',
            evaluation='full300 every tenth iteration; fixed endpoint, then matched stealth confirmation; no best-checkpoint selection'),
        launch_requirements=['Separate immutable frozen source with validated browser fixes and unchanged actor/judge recipe.',
            'Controller owns and awaits every worker, persists each attempt, skips only verified records on retry.',
            'Exact resource/API approval and shared per-shard retry ledgers; released older allocations provide no new authorization.',
            'Native CPU preflight, live GPU/browser startup gate, independent final artifact audit and active repair supervision.'])
    freeze(output/'audit.json', report)
    freeze(output/'provenance.json', provenance)
    freeze(output/'experiment.json', spec)
    freeze(output/'paired-schedule.json', attempts)
    freeze(output/'selection-pools.json', dict(eligible=eligible, historical_arm_selected=selected,
        site_matched_random=random_ids, actor_retry_selected=None,
        random_frame_includes_arm_successes=True, actor_selection_requires_new_outcomes=True))
    freeze(output/'task-payloads.json', [indexed[row['task_id']] for row in attempts])
    return report, spec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=POOL/'selection-control-20261003')
    args = parser.parse_args()
    report, spec = prepare(args.output)
    print(json.dumps(dict(output=str(args.output), audit=report, status=spec['status']), indent=2))


if __name__ == '__main__':
    main()
