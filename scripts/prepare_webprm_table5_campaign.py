#!/usr/bin/env python3
"""Offline, outcome-independent Table 5 cohort and observed-cost preparation.

This command never allocates resources, downloads, serves, or evaluates tasks.
It writes task identifiers and hashes, not task instructions or raw trajectories.
The resulting manifest remains pending protocol/resource review.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import statistics

RUNTIME = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-webarena-20261011')
TASKS = RUNTIME / 'task-preparation/released_single_site_155.json'
TASKS_SHA = '73b5133abd1ef7bf1d94447f6e82676419cd500a942187bcf4c748094c859cea'
PILOT = RUNTIME / 'next-stage-preparation/reddit-corrected-pilot-v3/combined-v2/campaign'
ACTORS = ('gpt-4o-mini-2024-07-18', 'gpt-4o-2024-05-13')
SELECTORS = ('no_search', 'gpt4omini', 'shepherd8b', 'arbiter7b', 'piotr_selectionarm')
COUNTS = {'shopping': 45, 'shopping_admin': 35, 'reddit': 19, 'gitlab': 30, 'map': 26}
RATES = {ACTORS[0]: (Decimal('.00000015'), Decimal('.00000060')),
         ACTORS[1]: (Decimal('.000005'), Decimal('.000015'))}
ORDER_VERSION = 'table5-task-id-paired-sha256-v1'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cohort(tasks):
    """Reject duplicate or mixed-site members before constructing the fixed grid."""
    ids, domains, records = set(), Counter(), []
    for task in tasks:
        task_id = task['task_id']
        if type(task_id) is not int or task_id < 0 or task_id in ids:
            raise ValueError('Task IDs must be distinct nonnegative integers')
        if not isinstance(task['sites'], list) or len(task['sites']) != 1 or task['sites'][0] not in COUNTS:
            raise ValueError('Exactly one supported domain required')
        ids.add(task_id)
        domain = task['sites'][0]
        domains[domain] += 1
        records.append(dict(task_id=task_id, original_task_id=task['old_task_id'], domain=domain,
                            task_config_sha256=digest(task)))
    if dict(domains) != COUNTS:
        raise ValueError('Released single-site domain counts changed')
    records.sort(key=lambda row: row['task_id'])
    cells = []
    for record in records:
        block = []
        for actor in ACTORS:
            for selector in SELECTORS:
                cell_id = f"webarena_lite_{record['task_id']:03d}__{actor}__{selector}"
                block.append(dict(record, actor=actor, selector=selector, cell_id=cell_id))
        block.sort(key=lambda row: digest([ORDER_VERSION, row['cell_id']]))
        cells.extend(block)
    return dict(schema_version=1, status='prepared_not_protocol_or_resource_approved',
        order_version=ORDER_VERSION, task_count=len(records), episode_count=len(cells),
        domain_counts=COUNTS, actors=list(ACTORS), selectors=list(SELECTORS),
        first_task_per_domain={domain: min(r['task_id'] for r in records if r['domain'] == domain) for domain in COUNTS},
        source_sha256=TASKS_SHA, paper_exact_task_membership_verified=False,
        reset='Fresh original site state and authentication for every cell; no mutated-state reuse.',
        primary_attempts_per_cell=1, replacement_task_admission=False,
        protocol_sha256=None, resource_approval_sha256=None, cells=cells,
        warning='Prepared candidate reconstruction; order is outcome-independent but not yet a committed scientific freeze.')


def dispatch_bounds(tasks=155, turns=30):
    if type(tasks) is not int or tasks < 1 or type(turns) is not int or turns < 1:
        raise ValueError('Positive integral task and turn limits required')
    conditions = tasks * len(ACTORS)
    actor = conditions * len(SELECTORS) * (3 * turns - 1)
    selector = conditions * (len(SELECTORS) - 1) * turns
    fuzzy = 39 * len(SELECTORS) * len(ACTORS) if tasks == 155 else None
    local = {'shepherd8b': conditions * (1 + 5 * turns),
             'arbiter7b': conditions * 4 * turns, 'piotr_selectionarm': conditions * turns}
    return dict(actor_paid=actor, gpt_mini_selector_paid=selector, native_fuzzy_paid=fuzzy,
        total_paid=actor+selector+fuzzy if fuzzy is not None else None,
        local=local, total_local=sum(local.values()),
        assumptions='One attempt per cell; 30 turns; cached Shepherd checklist; native auxiliary stages; '
                    '39 single-site tasks have at most one fuzzy target; excludes every startup/recovery call.')


def money(input_tokens, output_tokens, model):
    a, b = RATES[model]
    return Decimal(input_tokens) * a + Decimal(output_tokens) * b


def audit_calibration(campaign):
    """Join every paid request to its actual ledger entry; never infer token use."""
    campaign = Path(campaign)
    ledger_path = campaign / 'shared-budget.json'
    ledger = json.loads(ledger_path.read_text())
    entries = {r['id']: r for r in ledger['calls']}
    if len(entries) != len(ledger['calls']) or any(r['state'] != 'settled' for r in entries.values()):
        raise ValueError('Unique settled ledger reservations required')
    rows = defaultdict(lambda: dict(calls=0, input=0, output=0, usd=Decimal(0), inputs=[], outputs=[]))
    seen, spans, sources = set(), {}, {str(ledger_path): file_sha(ledger_path)}
    for selector in SELECTORS:
        path = campaign / 'attempt-355425/episodes/webarena_lite_005' / f'{ACTORS[0]}__{selector}' / 'api.jsonl'
        events = [json.loads(line) for line in path.read_text().splitlines()]
        sources[str(path)] = file_sha(path)
        spans[selector] = max(e['time'] for e in events) - min(e['time'] for e in events)
        for request in events:
            if request.get('event') != 'request' or request.get('local'):
                continue
            key = request['reservation']
            if key in seen or key not in entries:
                raise ValueError('Missing or repeated paid reservation in primary journals')
            seen.add(key)
            entry = entries[key]
            if entry['model'] != request['payload']['model']:
                raise ValueError('Request/ledger model identity mismatch')
            purpose = request['purpose']
            if purpose not in {'terminal', 'policy', 'state_diff', 'selector_pair', 'native_fuzzy'}:
                raise ValueError('Unreviewed primary purpose')
            row = rows[purpose]
            usage = entry['usage']
            pin, pout = usage['prompt_tokens'], usage['completion_tokens']
            if type(pin) is not int or type(pout) is not int or min(pin, pout) < 0:
                raise ValueError('Nonnegative observed integral token usage required')
            row['calls'] += 1
            row['input'] += pin
            row['output'] += pout
            row['usd'] += money(pin, pout, entry['model'])
            row['inputs'].append(pin)
            row['outputs'].append(pout)
    if len(seen) != 570 or len(entries) != 571 or set(entries)-seen != {0}:
        raise ValueError('The completed pilot must have 570 primary calls and its single startup judge')
    if entries[0]['model'] != ACTORS[1]:
        raise ValueError('Unexpected startup judge model')
    full_actor = {}
    for model in ACTORS:
        full_actor[model] = sum((money(rows[p]['input'], rows[p]['output'], model) for p in ('terminal','policy','state_diff')), Decimal(0)) * 155
    selector_cost = money(rows['selector_pair']['input'], rows['selector_pair']['output'], ACTORS[0]) * 310
    fuzzy_cost = money(rows['native_fuzzy']['input'], rows['native_fuzzy']['output'], ACTORS[1]) * Decimal(390) / 5
    observed_total = sum((money(r['usage']['prompt_tokens'], r['usage']['completion_tokens'], r['model']) for r in entries.values()), Decimal(0))
    for row in rows.values():
        for key in ('inputs', 'outputs'):
            values = row.pop(key)
            row[key+'_distribution'] = dict(min=min(values), median=statistics.median(values), max=max(values))
        row['usd'] = str(row['usd'])
    return dict(schema_version=1, source_hashes=sources, paid_calls=571, primary_paid_calls=570,
        primary_purpose_counts=dict(rows), observed_paid_usd_without_cache_discounts=str(observed_total),
        per_selector_journal_span_seconds=spans,
        same_trace_full_cohort_scenario=dict(
            actor_usd={k:str(v) for k,v in full_actor.items()}, selector_usd=str(selector_cost),
            fuzzy_usd=str(fuzzy_cost), total_usd=str(sum(full_actor.values())+selector_cost+fuzzy_cost),
            journal_span_hours=sum(spans.values())*310/3600,
            full_allocation_hours=3179*310/3600,
            exclusions='Extra startup, diagnostics, retries, additional hosting, other domains, future judge changes, actor-dependent tokens/latency.'),
        rates_usd_per_token={k:[str(x) for x in v] for k,v in RATES.items()},
        price_basis='Frozen October 11 pilot rate receipts; not a fresh provider price quotation.',
        claim_limit='One Reddit task with format failures and invalid semantic judgments. Same-trace repricing is a scenario, not an expected-cost or completion forecast.')


def exclusive_json(path, value):
    path = Path(path)
    allowed = (RUNTIME, Path('/tmp'))
    if not path.is_absolute() or path.resolve() != path or any(p.is_symlink() for p in (path,*path.parents)) or not any(p in path.parents for p in allowed):
        raise ValueError('Output must be canonical and nonsymlink below runtime or /tmp')
    raw = (json.dumps(value,sort_keys=True,indent=2)+'\n').encode()
    if len(raw) > 4*2**20:
        raise ValueError('Preparation output exceeds four MiB')
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd,'wb') as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if file_sha(TASKS) != TASKS_SHA:
        raise ValueError('Pinned released task collection changed')
    value = dict(cohort=cohort(json.loads(TASKS.read_text())), dispatch_bounds=dispatch_bounds(),
                 completed_pilot_calibration=audit_calibration(PILOT),
                 execution_accounting=dict(allocations=0, model_calls=0, browser_calls=0, downloads=0))
    exclusive_json(args.output,value)
    print(json.dumps({'output':str(args.output),'sha256':file_sha(args.output),'episodes':1550,'status':'offline_prepared'}))


if __name__ == '__main__':
    main()
