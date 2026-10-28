#!/usr/bin/env python3
"""Validate a frozen WebArena plan or execute one assigned fresh-instance cell.

No resource requests, site startup/reset or model serving occur here. The owning
allocation controller must provide a new original state lease for every cell,
enter its network namespace, await this process and verify site closure before
the next reset. Preparation defaults to no execution and spends no API budget.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from openwebrl.webprm_lifecycle import EPISODE_WATCHDOG
from openwebrl.webprm_webarena import TASK_FILE, EvaluationNotReady, digest, evaluation_requirements, verify_source
from openwebrl.webprm_webarena_episode import TaskEnvelope
from openwebrl.webprm_webarena_worker import (ACTOR, ACTOR_SNAPSHOTS, PRIMARY_SELECTORS, PROTOCOL, SELECTORS,
    protocol_definition, require_supported, run_episode, validate_frozen_sources, validate_protocol_contract, write_json)

PLANNED_SELECTORS = SELECTORS
REQUIRED_CODE = ('openwebrl/webprm_webarena_worker.py', 'openwebrl/webprm_webarena_episode.py',
    'openwebrl/webprm_webarena.py', 'openwebrl/webprm_worker.py', 'openwebrl/webprm_lifecycle.py',
    'openwebrl/webprm_arbiter_harness.py', 'openwebrl/webprm_source.py',
    'scripts/run_webprm_webarena_pilot.py', 'scripts/check_webprm_reddit_reset.py',
    'openwebrl/webprm_reddit_provenance.py', 'openwebrl/webprm_runtime.py', 'openwebrl/webprm_broker.py',
    'openwebrl/jev_eval.py', 'openwebrl/webprm_webarena_piotr.py', 'openwebrl/webprm_webarena_fuzzy.py',
    'openwebrl/webprm_webarena_sites.py')


def configured_tasks(plan, *, allow_missing_endpoints=False):
    source = Path(plan['arbiter_source'])
    if verify_source(source) != plan['source_manifest']:
        raise ValueError('Released task/evaluator source changed')
    ids = plan['task_ids']
    if not ids or any(type(value) is not int for value in ids) or len(set(ids)) != len(ids):
        raise ValueError('Prospective task IDs must be nonempty, unique released integers')
    indexed = {task['task_id']: task for task in json.loads((source / TASK_FILE).read_text())}
    selected = [indexed[value] for value in ids]
    if digest(selected) != plan['tasks_sha256']:
        raise ValueError('Prospective task membership/order changed')
    result = []
    for raw in selected:
        try:
            task = TaskEnvelope.from_release(raw, plan['endpoints'])
        except EvaluationNotReady:
            if not allow_missing_endpoints:
                raise
            task = None
        result.append((task, raw))
    return result


def validate_plan(plan, *, allow_missing_endpoints=False):
    actors, selectors = plan['actors'], plan['selectors']
    if (not actors or len(set(actors)) != len(actors) or any(actor not in ACTOR_SNAPSHOTS for actor in actors)
            or not selectors or len(set(selectors)) != len(selectors)
            or any(selector not in PLANNED_SELECTORS for selector in selectors)):
        raise ValueError('Unknown or duplicate actor/selector identity')
    if plan.get('max_turns') != 30 or type(plan.get('seed')) is not int:
        raise ValueError('Freeze the common 30-turn protocol and integer seed')
    validate_protocol_contract(plan)
    if plan.get('paper_exact_protocol') is not False:
        raise ValueError('This preparation cannot claim the exact recovered paper protocol')
    cap = plan.get('api_usd_cap')
    if type(cap) not in (int, float) or not math.isfinite(cap) or cap < 0:
        raise ValueError('Explicit finite nonnegative API cap required')
    deadline = plan.get('deadline')
    if type(deadline) not in (int, float) or not math.isfinite(deadline):
        raise ValueError('Explicit finite allocation deadline required')
    repo = Path(__file__).resolve().parents[1]
    code = plan.get('code_sha256', {})
    if any(str(repo / name) not in code for name in REQUIRED_CODE):
        raise ValueError('Complete working-code freeze required')
    for path, expected in code.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise ValueError('Frozen code changed: ' + path)
    if plan.get('native_helpers') is not None:
        from openwebrl.webprm_webarena_helper_runtime import HELPER_CONFIGURATION
        if plan['native_helpers'] != HELPER_CONFIGURATION:
            raise ValueError('Unknown explicit native helper runtime configuration')
        for name in ('openwebrl/webprm_webarena_helpers.py','openwebrl/webprm_webarena_helper_runtime.py'):
            if str(repo / name) not in code:
                raise ValueError('Native helper runtime requires a complete working-code freeze')
    if plan.get('evaluator_observations') is not None:
        from openwebrl.webprm_evaluator_transcript import validate_transcript_configuration
        validate_transcript_configuration(plan['evaluator_observations'])
        if str(repo / 'openwebrl/webprm_evaluator_transcript.py') not in code:
            raise ValueError('Native evaluator transcript requires a complete working-code freeze')
    validate_frozen_sources(plan)
    if 'piotr_selectionarm' in selectors and (plan.get('piotr_manifest') is not None or cap > 0):
        from openwebrl.webprm_webarena_piotr import validate_piotr_selectionarm
        if plan.get('piotr_manifest') != validate_piotr_selectionarm(plan):
            raise ValueError('Frozen Piotr model/prompt/action-representation manifest changed')
    paid_models = set(actors) | ({ACTOR} if 'gpt4omini' in selectors else set())
    if plan.get('fuzzy_enabled'):
        from openwebrl.webprm_webarena_fuzzy import validate_fuzzy_configuration
        paid_models.add(validate_fuzzy_configuration(plan)['model'])
    for actor in paid_models:
        rates = plan.get('prices_usd_per_token', {}).get(actor)
        if rates is None and cap == 0:
            continue  # A zero-budget preparation can report this as a blocker.
        if (not isinstance(rates, list) or len(rates) != 2 or
                any(type(rate) not in (int, float) or not math.isfinite(rate) or rate <= 0 for rate in rates)):
            raise ValueError('Explicit audited actor input/output prices required')
    return configured_tasks(plan, allow_missing_endpoints=allow_missing_endpoints)


def expected_cells(plan):
    return {(f'webarena_lite_{task_id:03d}', actor + '__' + selector)
            for task_id in plan['task_ids'] for actor in plan['actors'] for selector in plan['selectors']}


def cohort_status(records, plan):
    expected = expected_cells(plan)
    observed, attempted = set(), set()
    groups = {}
    for row in records:
        key = row['task_id'], row['arm']
        if key not in expected or key in observed:
            raise ValueError('Duplicate or out-of-plan result')
        if (type(row['valid']) is not bool or type(row['success']) is not int or row['success'] not in (0, 1)
                or row['success'] and not row['valid'] or type(row.get('attempted')) is not bool
                or row['valid'] and not row['attempted']):
            raise ValueError('Invalid result fields')
        observed.add(key)
        if row.get('attempted', True):
            attempted.add(key)
        counts = groups.setdefault(row['arm'], dict(records=0, attempted=0, valid=0, successes=0))
        counts['records'] += 1
        counts['attempted'] += row.get('attempted', True)
        counts['valid'] += row['valid']
        counts['successes'] += row['success']
    missing = sorted(expected - attempted)
    return dict(status='completed' if not missing else 'partial', planned_cells=len(expected),
        recorded_cells=len(observed), attempted_cells=len(attempted),
        valid=sum(row['valid'] for row in records), successes=sum(row['success'] for row in records),
        missing_cells=[dict(task_id=task_id, arm=arm) for task_id, arm in missing], by_arm=groups)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--task-id', type=int)
    parser.add_argument('--actor')
    parser.add_argument('--selector')
    parser.add_argument('--lease', type=Path)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    tasks = validate_plan(plan, allow_missing_endpoints=not args.execute)
    blockers = []
    if plan.get('authorization_status') != 'approved' or plan['api_usd_cap'] <= 0:
        blockers.append(dict(reason='No positive API budget and execution allocation have been approved'))
    for selector in plan['selectors']:
        if selector not in SELECTORS:
            blockers.append(dict(selector=selector, reason='Condition lacks a validated execution adapter'))
        critic = {'arbiter7b': 'arbiter', 'shepherd8b': 'shepherd', 'piotr_selectionarm': 'piotr'}.get(selector)
        if critic and any(not plan.get(critic + suffix) for suffix in ('_model', '_endpoint')):
            blockers.append(dict(selector=selector, reason='Frozen local critic model and serving route are not configured'))
        if selector == 'piotr_selectionarm' and not plan.get('piotr_manifest'):
            blockers.append(dict(selector=selector, reason='Pinned Piotr source/checkpoint/prompt manifest is not configured'))
    if not plan.get('broker_socket'):
        blockers.append(dict(reason='Owned host-namespace model broker is not configured'))
    for actor in plan['actors']:
        if not plan.get('prices_usd_per_token', {}).get(actor):
            blockers.append(dict(actor=actor, reason='Audited input/output prices not configured'))
    for task, raw in tasks:
        requirements = evaluation_requirements(raw)
        if requirements['paid_fuzzy_judge_may_be_required'] and not plan.get('fuzzy_enabled'):
            blockers.append(dict(task_id=f"webarena_lite_{raw['task_id']:03d}",
                reason='Native gpt-4-1106-preview judge is disabled; no approved available native judge transport',
                judge_access=plan.get('native_judge_access', {'status': 'unverified'})))
        if task is None:
            blockers.append(dict(task_id=f"webarena_lite_{raw['task_id']:03d}",
                reason='Explicit live site endpoints are not configured', requirements=requirements))
            continue
        for selector in plan['selectors']:
            try:
                require_supported(task, selector, plan)
            except EvaluationNotReady as error:
                blockers.append(dict(task_id=task.execution_id, selector=selector, reason=str(error)))
        from openwebrl.webprm_webarena_sites import site_adapter
        try:
            site_adapter(task, plan)
        except EvaluationNotReady as error:
            blockers.append(dict(task_id=task.execution_id, reason=str(error)))
    if not args.execute:
        print(json.dumps(dict(status='prepared_not_executed', planned_cells=len(expected_cells(plan)),
                              blockers=blockers, model_calls=0, browser_started=False)))
        return
    if (plan.get('authorization_status') != 'approved' or plan['api_usd_cap'] <= 0
            or str(plan.get('job_id')) != os.environ.get('SLURM_JOB_ID')):
        raise EvaluationNotReady('Execution requires exact approved API/compute plan inside its existing job')
    if args.lease is None or args.actor not in plan['actors'] or args.selector not in plan['selectors']:
        parser.error('Execution requires an assigned --lease, --actor and --selector')
    task, raw = next((pair for pair in tasks if pair[0].released_task_id == args.task_id), (None, None))
    if task is None:
        parser.error('Task is not in the prospective fixed cohort')
    require_supported(task, args.selector, plan)
    from scripts.check_webprm_reddit_reset import safe_path
    output = safe_path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    identity = digest(plan)
    manifest_path = output / 'manifest.json'
    manifest = dict(identity=plan, identity_sha256=identity)
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError('Output has a different frozen plan')
    else:
        from scripts.check_webprm_reddit_reset import write_new
        write_new(manifest_path, manifest)
    config = dict(plan, output=str(output), actor_model=args.actor, identity_sha256=identity,
        episode_watchdog=dict(EPISODE_WATCHDOG), lease_path=str(args.lease.resolve()),
        lease_sha256=hashlib.sha256(args.lease.read_bytes()).hexdigest())
    result = run_episode(dict(task_id=task.execution_id, released=raw), args.selector, config)
    records = []
    for task_id, cell in sorted(expected_cells(plan)):
        path = output / task_id / cell / 'result.json'
        if path.exists():
            record = json.loads(path.read_text())
            if record.get('identity_sha256') != identity:
                raise ValueError('Durable cell identity differs from plan')
            records.append(record)
    status = cohort_status(records, plan)
    write_json(output / 'results.json', records)
    write_json(output / 'status.json', status)
    print(json.dumps(dict(cell=result['arm'], task_id=result['task_id'], valid=result['valid'],
                          success=result['success'], cohort_status=status['status'])))


if __name__ == '__main__':
    main()
