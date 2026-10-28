#!/usr/bin/env python3
"""Own and await approved 12-task/six-arm supported-interface pilot workers."""
import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import hashlib
import importlib.metadata
import os
import json
import multiprocessing
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from openwebrl.jev_eval import load_tasks, write_json
from openwebrl.webprm_baselines import ARMS, paired_summary
from openwebrl.webprm_worker import fixture, run_episode, SharedBudget, ACTOR, JUDGE, PILOT_ARMS, PRICES
from openwebrl.webprm_runtime import load_api_credentials


def collection_stop_reason(config):
    """Do not create missing-cohort failures after a shared resource limit."""
    if config.get('stop_admission_path') and Path(config['stop_admission_path']).exists():
        return 'operator_requested_stop_after_draining'
    if time.time() >= config['deadline']:
        return 'pilot_deadline'
    budget = json.loads(Path(config['budget_path']).read_text())
    if budget.get('halt_reason'):
        return budget['halt_reason']
    if budget['charged'] >= budget['cap']:
        return 'shared_api_budget_exhausted'
    return None


def collect_cells(pool, cells, config, workers, record_result):
    """Keep at most four cells outstanding, including across task boundaries."""
    if workers not in (1, 2, 4):
        raise ValueError('Collection supports one, two or four concurrent workers')
    remaining = iter(cells)
    pending = {}
    exhausted = False
    stop_reason = None
    while pending or not exhausted:
        stop_reason = stop_reason or collection_stop_reason(config)
        while not stop_reason and not exhausted and len(pending) < workers:
            try:
                task, arm = next(remaining)
            except StopIteration:
                exhausted = True
                break
            pending[pool.submit(run_episode, task, arm, config)] = (task['task_id'], arm)
        if not pending:
            break
        done, _ = wait(pending, return_when=FIRST_COMPLETED)
        for future in done:
            pending.pop(future)
            result = future.result()
            record_result(result)
            if result.get('error') == 'Shared API budget exhausted':
                stop_reason = 'shared_api_budget_exhausted'
            if result.get('error_type')=='HTTPError' and any(
                    result.get('error','').startswith(f'HTTP Error {code}:') for code in (401,403)):
                stop_reason = 'provider_authorization_failed'
            if not result.get('attempted', True):
                stop_reason = stop_reason or 'worker_not_started'
    return stop_reason


def collect_task(pool, task, arms, config, workers, record_result):
    """Compatibility wrapper for collecting one task's conditions."""
    return collect_cells(pool, ((task, arm) for arm in arms), config, workers, record_result)


def collect_cohort(pool, tasks, config, workers, record_result, results,
                   quality_gate_tasks=2, seed=20261009):
    """Overlap tasks within each stage; finish the quality gate before expansion.

    Submission keeps fixed task order and the original per-task seeded arm
    shuffle. Later tasks may fill freed slots once earlier cells are submitted,
    but no task beyond the gate can start until every gate cell has resolved.
    Durable cells are skipped without changing later tasks' shuffled orders.
    """
    if not 1 <= quality_gate_tasks <= len(tasks):
        raise ValueError('Quality gate must cover a nonempty initial task group')
    collected=list(results)
    completed_keys={(row['task_id'],row['arm']) for row in collected}
    rng=random.Random(seed)
    stages=[[],[]]
    for index,task in enumerate(tasks):
        order=list(PILOT_ARMS)
        rng.shuffle(order)
        stages[index >= quality_gate_tasks].extend(
            (task,arm) for arm in order if (task['task_id'],arm) not in completed_keys)
    def save_result(result):
        collected.append(result)
        record_result(result)
    stop_reason=collect_cells(pool,stages[0],config,workers,save_result)
    if stop_reason:
        return stop_reason
    gate_ids=[task['task_id'] for task in tasks[:quality_gate_tasks]]
    gate_rows=[row for row in collected if row['task_id'] in gate_ids]
    if cohort_status(gate_rows,gate_ids)['status']!='completed':
        return 'quality_gate_incomplete'
    stop_reason=quality_gate_failure(collected,gate_ids)
    if stop_reason:
        return stop_reason
    return collect_cells(pool,stages[1],config,workers,save_result)


def cohort_status(results, task_ids):
    expected = {(task_id, arm) for task_id in task_ids for arm in PILOT_ARMS}
    keys = [(row['task_id'], row['arm']) for row in results]
    if len(set(keys)) != len(keys) or not set(keys) <= expected:
        raise ValueError('Duplicate or out-of-cohort collector result')
    attempted = {(row['task_id'], row['arm']) for row in results if row.get('attempted', True)}
    missing = sorted(expected - attempted)
    return dict(status='completed' if not missing else 'partial', planned_episodes=len(expected),
        episodes=len(results), attempted_episodes=len(attempted),
        valid=sum(row['valid'] for row in results),
        missing_episodes=[dict(task_id=task_id, arm=arm) for task_id, arm in missing])


def quality_gate_failure(results, task_ids):
    """Stop an obviously broken pilot early, without claiming model quality."""
    rows = [row for row in results if row['task_id'] in task_ids]
    if len(rows) != len(task_ids) * len(PILOT_ARMS):
        return None
    missing_valid = [arm for arm in PILOT_ARMS
                     if not any(row['arm'] == arm and row['valid'] for row in rows)]
    return 'quality_gate_no_valid_episode:' + ','.join(missing_valid) if missing_valid else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-file',default=str(Path(__file__).resolve().parents[1]/'.env'))
    parser.add_argument('--fixture', action='store_true')
    parser.add_argument('--output', required=True)
    parser.add_argument('--tasks')
    parser.add_argument('--plan')
    parser.add_argument('--workers', type=int, default=4, choices=[1,2,4])
    parser.add_argument('--budget-path', help='Existing shared API ledger; never reset across cohorts')
    parser.add_argument('--quality-gate-tasks', type=int, default=2)
    parser.add_argument('--seed', type=int, default=20261009)
    parser.add_argument('--seconds', type=int, default=6000)
    parser.add_argument('--shepherd-actor-source',default='/tmp/openwebrl-webshepherd-demo.py')
    parser.add_argument('--shepherd-source', default='/tmp/openwebrl-webshepherd-upstream')
    parser.add_argument('--arbiter-source', default='/tmp/openwebrl-webarbiter-upstream')
    parser.add_argument('--arbiter-benchmark', default='/tmp/openwebrl-webprm-benchmark.jsonl')
    parser.add_argument('--shepherd-endpoint', default='http://127.0.0.1:8011/v1')
    parser.add_argument('--arbiter-endpoint', default='http://127.0.0.1:8012/v1')
    parser.add_argument('--shepherd-model', default='WebShepherd8B')
    parser.add_argument('--arbiter-model', default='WebArbiter7B')
    args = parser.parse_args()
    if args.fixture:
        fixture(args.output,shepherd_actor_source=args.shepherd_actor_source,arbiter_source=args.arbiter_source)
        return
    if not args.tasks or not args.plan or not 0 < args.seconds:
        parser.error('Pilot requires task file, prepared plan, and a positive bounded duration')
    plan = json.loads(Path(args.plan).read_text())
    if plan.get('authorization_status') != 'approved':
        raise ValueError('Pilot requires explicit compute authorization')
    if args.seconds > plan['approval']['gpu_seconds_total']:
        raise ValueError('Pilot duration exceeds the approved total allocation budget')
    if len(plan['pilot_task_ids']) != 12 or len(set(plan['pilot_task_ids']))!=12 or plan['arms']!=list(PILOT_ARMS):
        raise ValueError('Approval covers exactly 12 unique tasks and canonical six arms')
    actual_hash = hashlib.sha256(Path(args.tasks).read_bytes()).hexdigest()
    if actual_hash != plan['task_file_sha256']:
        raise ValueError('Task file differs from prepared cohort')
    indexed = {task['task_id']:task for task in load_tasks(args.tasks)}
    tasks = [indexed[ident] for ident in plan['pilot_task_ids']]
    target=Path(args.output);target.mkdir(parents=True,exist_ok=True)
    load_api_credentials(args.env_file)
    if args.budget_path and not Path(args.budget_path).is_file():
        raise ValueError('Explicit shared API ledger is missing; refusing to reset its budget')
    if not 1 <= args.quality_gate_tasks <= 12:
        parser.error('Quality gate must cover between one and twelve fixed tasks')
    from openwebrl.webprm_source import load_shepherd,load_arbiter,load_arbiter_causal_renderer
    from openwebrl.webprm_arbiter_harness import load_arbiter_harness,load_native_observation_processor
    from openwebrl.jev_eval import JUDGE_SOURCE
    sources=dict(shepherd=load_shepherd(args.shepherd_source).manifest,
        arbiter=load_arbiter(args.arbiter_source).manifest,
        arbiter_harness=load_arbiter_harness(args.arbiter_source).manifest,
        arbiter_observation=load_native_observation_processor(args.arbiter_source,viewport_size={'width':1280,'height':720}).manifest,
        arbiter_prompt=load_arbiter_causal_renderer(args.arbiter_benchmark).manifest,
        shepherd_actor_sha256=hashlib.sha256(Path(args.shepherd_actor_source).read_bytes()).hexdigest())
    for key,expected in plan.get('sources',{}).items():
        if key in sources and sources[key]!=expected:
            raise ValueError('Staged source changed since preparation')
    repo=Path(__file__).resolve().parents[1]
    files=['scripts/run_webprm_pilot.py','openwebrl/webprm_worker.py','openwebrl/webprm_source.py',
           'openwebrl/webprm_baselines.py','openwebrl/webprm_arbiter_harness.py',
           'openwebrl/webprm_shepherd_harness.py','openwebrl/webprm_runtime.py',
           'openwebrl/webprm_lifecycle.py',
           'openwebrl/jev_eval.py',str(JUDGE_SOURCE)]
    code={name:hashlib.sha256((repo/name).read_bytes()).hexdigest() for name in files}
    dependencies={name:importlib.metadata.version(name) for name in
                  ['browsergym-core','playwright','gymnasium','numpy','pillow','pyparsing','python-dotenv']}
    episode_watchdog=dict(browser_phase_timeout_seconds=180, cleanup_timeout_seconds=20,
                          termination_grace_seconds=2, poll_seconds=.25)
    identity=dict(protocol_version='native-recovery-v6', episode_watchdog=episode_watchdog,
        task_file_sha256=actual_hash,task_ids=plan['pilot_task_ids'],arms=list(PILOT_ARMS),
        actor=ACTOR,judge=JUDGE,max_turns=30,api_usd_cap=30,prices_usd_per_token=PRICES,
        sources=sources,code_sha256=code,dependencies=dependencies,seed=args.seed,
        server_models=dict(shepherd=args.shepherd_model,arbiter=args.arbiter_model),
        model_revisions=plan.get('models'),model_source_equivalence=plan.get('model_source_equivalence'),
        sampling=dict(shepherd_raw_guided=20,shepherd_raw_unguided=1,temperature=1.,
            shepherd_top_p=1.,arbiter_top_p=.9,shepherd_max_tokens=512,arbiter_max_tokens=2048,
            output_cap_note='Explicit pilot caps; author demo has no explicit output-token cap',
            arbiter_policy_context_note='Viewer8192 is declarative; no provider truncation found; full history retained',
            shepherd_critic_context_limit=40960,arbiter_critic_context_limit=32768,critic_backend='SGLang adaptation of published vLLM',
            shepherd_critic_temperature=0.,shepherd_critic_samples=1,
            shepherd_benchmark_default_note='Released offline script usesT1,N5,GTchecklist; online pilot generated checklist,T0,N1 disclosed'),
        recovery_protocol=dict(shepherd_action_subsets=['chat','tab','nav','bid','infeas'],
            malformed_samples='filter_each_fixed_sample_with_native_parser_no_resampling',
            empty_candidate_pool='consume_turn_return_action_error_continue',
            arbiter_auxiliary='released_state_diff_then_terminal_each_observation',
            arbiter_auxiliary_model=ACTOR, arbiter_auxiliary_temperature=1.,
            arbiter_auxiliary_top_p=.9, arbiter_auxiliary_max_tokens=2048,
            arbiter_terminal='native_terminal_override_before_policy_sampling',
            arbiter_error_feedback='exact_execution_error_in_history_and_state_diff',
            arbiter_screen_geometry='restore_CDP_screen_dimensions_to_current_viewport_before_native_observation',
            initial_navigation=dict(wait_until='domcontentloaded',timeout_ms=30000),
            quality_gate_tasks=args.quality_gate_tasks),
        scheduling_policy=dict(name='bounded_cross_task_two_stage_v1',
            max_outstanding_cells=args.workers,task_order='fixed_cohort_order',
            arm_order='original_per_task_seeded_shuffle_before_resume_filter',
            initial_stage_tasks=args.quality_gate_tasks,
            expansion='only_after_all_initial_cells_attempted_and_quality_gate_passes'),
        actor_prompt_origin='native_author_Shepherd_demo_and_native_Arbiter_viewer',
        arbiter_unguided_note='native five-candidate policy call; choose first without critic',
        tournament='upstream_viewer_queue_rounds_with_leftover_bye_before_winners',
        wandb_project='openwebrl-evals',training_updates=0)
    identity_sha256=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    manifest_path=target/'manifest.json'
    previous=json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    if previous and previous.get('identity_sha256')!=identity_sha256:
        raise ValueError('Resume identity or frozen source/dependencies changed')
    config=vars(args).copy()
    config.update(output=str(target.resolve()),budget_path=str(Path(args.budget_path).resolve()) if args.budget_path else str((target/'budget.json').resolve()),
        episode_watchdog=episode_watchdog,
        stop_admission_path=str(target.resolve()/'stop-admission.json'),
        deadline=previous['config']['deadline'] if previous else time.time()+args.seconds,
        identity_sha256=identity_sha256,code_sha256=code,source_manifests=sources)
    SharedBudget(config['budget_path']).change(lambda _:None)
    write_json(manifest_path,dict(config=config,identity=identity,identity_sha256=identity_sha256))
    results=[]
    # Read durable cells first, including later tasks from an interrupted attempt.
    for task in tasks:
        for arm in PILOT_ARMS:
            path=target/task['task_id']/arm/'result.json'
            if path.exists():
                row=json.loads(path.read_text())
                if row.get('identity_sha256')!=identity_sha256:
                    raise ValueError('Episode resume identity changed')
                results.append(row)
    cohort_status(results,plan['pilot_task_ids'])
    def record_result(result):
        results.append(result)
        write_json(target/'results.json',results)
        write_json(target/'status.json',cohort_status(results,plan['pilot_task_ids']))
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        stop_reason=collect_cohort(pool,tasks,config,args.workers,record_result,results,
                                   quality_gate_tasks=args.quality_gate_tasks,seed=args.seed)
    status=cohort_status(results,plan['pilot_task_ids'])
    status.update(stop_reason=stop_reason,time=time.time(),
                  budget=json.loads(Path(config['budget_path']).read_text()))
    write_json(target/'results.json',results)
    write_json(target/'status.json',status)
    if status['status']=='completed':
        write_json(target/'summary.json',paired_summary(results,plan['pilot_task_ids']))
        write_json(target/'complete.json',status)
    else:
        write_json(target/'partial.json',status)


if __name__=='__main__':
    main()
