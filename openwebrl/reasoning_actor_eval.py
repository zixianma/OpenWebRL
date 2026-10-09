"""Matched high-reasoning API actors using the corrected full300 browser worker.

The collection loop is inherited from the frozen pixel-v2 worker. Only explicit
model/effort identities, model prices, output roots and budget routing change.
"""
from __future__ import annotations
import argparse
import asyncio
import base64
from copy import copy
import hashlib
import json
import os
from pathlib import Path
import signal
import time
from openwebrl.luna_qwen_eval import Judge, action_only_messages, verified_record
from openwebrl.luna_qwen_metrics import claimed_tasks, digest, episode_usage, write_json
from openwebrl.luna_qwen_policy import Budget
from openwebrl.luna_qwen_full_eval import verify_actor_files, verify_existing_records, sampling_for_family
from openwebrl.reasoning_actor_policy import ActorPolicy, MeteredActorAPI, MODELS, metric_prices
from openwebrl.browser_actor_protocol import validate_browser_harness_protocol


def validate_plan(plan, config):
    mode = plan.get('arm_mode')
    if mode not in MODELS or plan.get('family') != 'luna':
        raise ValueError('Unknown CPU API actor family')
    protocol = plan['protocol']
    validate_browser_harness_protocol(protocol)
    if (protocol.get('actor') != MODELS[mode] or protocol.get('reasoning_effort') != 'high'
            or protocol.get('actor_max_output_tokens') != 4096
            or protocol.get('resize_output_coords') is not False
            or protocol.get('browser_coordinate_space') != 'viewport_pixels'
            or protocol.get('judge') != 'o4-mini-2025-04-16'
            or protocol.get('max_steps') != 30):
        raise ValueError('Approved actor or browser/judge protocol changed')
    if config.get('plan_sha256') != digest(plan) or Path(config['output']) != Path(plan['output']):
        raise ValueError('Frozen worker plan changed')
    if plan['pricing'] != metric_prices(MODELS[mode]):
        raise ValueError('Incorrect actor model prices')
    tasks, schedule = plan['task_ids'], plan['schedule']
    if (len(tasks) != 300 or len(set(tasks)) != 300 or len(schedule) != 300
            or {x['task_id'] for x in schedule} != set(tasks)
            or any(x['modes'] != [mode] for x in schedule)):
        raise ValueError('Require exactly300 unique tasks for this actor')
    if plan['wandb_prefix'] != plan['experiment_id'] + '-' + mode:
        raise ValueError('Actor W&B identity mismatch')
    return 'luna'  # Inherited collection branch is CPU/API-only.


async def run(config):
    plan = json.loads(Path(config['plan']).read_text())
    family = validate_plan(plan, config)
    if os.environ.get('WANDB_PROJECT') != 'openwebrl-evals':
        raise ValueError('Separate evaluations must use openwebrl-evals')
    from openai import AsyncOpenAI
    import httpx
    import wandb
    from openwebrl import run_evaluate as evaluation, generate_browser as generation
    from openwebrl.eval import reward_online_mind2web as reward
    from openwebrl.arm_inference import load_selection_builder
    from slime.utils.http_utils import init_http_client

    if not all(hasattr(generation, name) for name in ('_luna_measured_step', '_luna_measured_reset')):
        raise ValueError('Worker requires frozen browser measurement and context hooks')
    root, worker_id = Path(config['output']), config['worker_id']
    worker_root = root / 'workers' / str(worker_id)
    worker_root.mkdir(parents=True, exist_ok=True)
    identity = dict(experiment_id=plan['experiment_id'], family=family, actor_model=plan['protocol']['actor'], reasoning_effort='high', arm_mode=plan['arm_mode'],
                    plan_sha256=config['plan_sha256'], job_id=os.getenv('SLURM_JOB_ID'))
    write_json(worker_root / 'status.json', dict(identity, complete=False, stage='starting',
                                               updated_unix=time.time()))
    protocol = plan['protocol']
    if hashlib.sha256(Path(plan['task_file']).read_bytes()).hexdigest() != plan['task_file_sha256']:
        raise ValueError('Task file changed')
    os.environ.update(SLIME_BROWSER_ENV_MODE='local_process', SLIME_BROWSER_LOCAL_PROCESS_MAX_PROCESSES='1',
        SLIME_BROWSER_ROLLOUT_CONCURRENCY='1', SLIME_BROWSER_LOCAL_PROCESS_LOG_DIR=str(worker_root / 'browser_logs'),
        SLIME_BROWSER_LOCAL_PROCESS_PORT_START=str(config['browser_port']),
        SLIME_BROWSER_LOCAL_PROCESS_PORT_END=str(config['browser_port'] + 99), VISION_FULL_HISTORY='1')
    generation._BROWSER_HOST_BLACKLIST_PATH = str(worker_root / 'navigation-failures.txt')
    loaded_tasks = evaluation.load_tasks_from_jsonl(plan['task_file'])
    tasks = {t['task_id']: t for t in loaded_tasks}
    if len(tasks) != len(loaded_tasks) or set(tasks) != set(plan['task_ids']):
        raise ValueError('Task payload differs from full frozen cohort')
    builder = load_selection_builder(plan['selector_prompt_source'])
    actor_config, processor = verify_actor_files(plan)
    verify_existing_records(plan, config['plan_sha256'])
    actor_port = config.get('actor_port', 0)
    if family != 'luna':
        if not actor_port:
            raise ValueError('Local actor requires a serving port')
        async with httpx.AsyncClient(trust_env=False) as h:
            response = await h.get(f'http://127.0.0.1:{actor_port}/get_model_info')
            response.raise_for_status()
            if Path(response.json()['model_path']).resolve() != Path(plan['actor_path']).resolve():
                raise ValueError('Wrong actor weights')
    api = AsyncOpenAI(api_key=os.environ['OPENAI_API_KEY'], base_url='https://api.openai.com/v1',
                      max_retries=0, timeout=180)
    judge_client = AsyncOpenAI(api_key=os.getenv('JUDGE_API_KEY') or os.environ['OPENAI_API_KEY'],
                              base_url='https://api.openai.com/v1', max_retries=0, timeout=180)
    budget_root = Path(plan['budget_root'])
    luna_budget = Budget(budget_root / 'actor-budget', plan['limits']['actor_usd'], plan['limits']['actor_calls'])
    judge_budget = Budget(Path(plan['judge_budget_root']), plan['limits']['judge_usd'], plan['limits']['judge_calls'])
    judge = Judge(judge_client, judge_budget)
    reward._get_openai_client = lambda **unused: judge
    current = {}

    async def terminal_reward(args, samples):
        current['episode_seconds'] = time.monotonic() - current['started']
        screenshot = args.luna_episode_observation.get('screenshot')
        if not screenshot:
            return [None] * len(samples) if isinstance(samples, list) else None
        last = samples[-1] if isinstance(samples, list) else samples
        last.metadata['full_image_list'] = [base64.b64encode(screenshot).decode()]
        (current['directory'] / 'final.png').write_bytes(screenshot)
        original_messages = last.metadata.get('messages', [])
        last.metadata['messages'] = action_only_messages(original_messages)
        started = time.monotonic()
        try:
            return await reward.reward_func(args, samples)
        finally:
            last.metadata['messages'] = original_messages
            current['judge_seconds'] = time.monotonic() - started

    evaluation.reward_func = terminal_reward
    args = evaluation.EvalArgs(sglang_router_ip='127.0.0.1', sglang_router_port=actor_port,
        hf_checkpoint=plan['actor_path'], max_steps=30, context_num_screenshots=1,
        judge_api_model=protocol['judge'], judge_api_mode='served', judge_timeout_secs=180,
        browser_response_format_mode='browser_env', turn_history_reasoning_mode='full',
        browser_include_tool_response=1, inference_step_timeout_secs=180, task_timeout_secs=1800,
        rollout_temperature=1., rollout_top_p=.95, rollout_top_k=-1, rollout_max_response_len=4096,
        rollout_max_context_len=32768, sglang_server_concurrency=1)
    init_http_client(args)
    tracking = wandb.init(project='openwebrl-evals', group=plan['wandb_group'],
        id=f'{plan["wandb_prefix"]}-w{worker_id}', resume='allow', dir=str(worker_root), config=protocol)
    protocol_hash = digest(protocol)
    records = []
    main_task = asyncio.current_task()
    asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, main_task.cancel)
    try:
        for item in claimed_tasks(root, plan['schedule'], worker_id):
            for mode in item['modes']:
                key = digest([item['task_id'], mode])[:24]
                saved = root / 'records' / (key + '.json')
                if saved.exists():
                    record = verified_record(saved, protocol_hash)
                    if any(record.get(k) != identity[k] for k in ('experiment_id', 'family', 'plan_sha256')):
                        raise ValueError('Saved record belongs to another frozen family experiment')
                    records.append(record)
                    continue
                if any((b.root / 'halt.json').exists() for b in (luna_budget, judge_budget)):
                    raise RuntimeError('Persistent API budget halt')
                if family != 'luna':
                    async with httpx.AsyncClient(trust_env=False) as h:
                        flushed = await h.post(f'http://127.0.0.1:{actor_port}/flush_cache', timeout=60)
                        flushed.raise_for_status()
                directory = root / 'attempts' / key / str(time.time_ns())
                directory.mkdir(parents=True)
                local = copy(args)
                local.path_to_save_generated_samples = str(directory / 'samples')
                local.luna_episode_observation = dict(browser_steps=0, browser_step_seconds=0.)
                policy = ActorPolicy(mode, directory, MeteredActorAPI(api, luna_budget, directory, protocol['actor'], plan['pricing']),
                                    builder, actor_config, processor)
                local.browser_action_selector = policy
                judge.directory = directory
                current.clear()
                current.update(started=time.monotonic(), directory=directory)
                write_json(worker_root / 'heartbeat.json', dict(identity, stage='episode', mode=mode,
                    task_key=key, completed=len(records), updated_unix=time.time()))
                write_json(directory / 'identity.json', dict(identity, mode=mode,
                    task_id=item['task_id'], started_epoch=time.time()))
                interrupted = False
                try:
                    result = await evaluation.evaluate_single_task(local, tasks[item['task_id']],
                                                                   sampling_for_family(family), True)
                    valid = result.get('reward') in (0, 1) and 'ABORTED' not in result.get('status', '')
                    valid = valid and not result.get('metadata', {}).get('judge_timeout')
                except asyncio.CancelledError:
                    interrupted = True
                    raise
                except Exception as exc:
                    valid = False
                    result = dict(reward=None, error_type=type(exc).__name__, error=str(exc))
                finally:
                    seconds = current.get('episode_seconds', time.monotonic() - current['started'])
                    usage = episode_usage(directory, 'luna', seconds, plan['pricing'])
                    physical = dict(identity, task_id=item['task_id'], mode=mode, worker_id=worker_id,
                        interrupted=interrupted, episode_seconds=seconds,
                        local_model_batch_seconds=policy.local_batch_seconds,
                        browser_steps=local.luna_episode_observation['browser_steps'],
                        browser_step_seconds=local.luna_episode_observation['browser_step_seconds'],
                        judge_seconds=current.get('judge_seconds', 0.), **usage)
                    write_json(directory / 'physical-usage.json', physical)
                artifact = directory / 'result.json'
                write_json(artifact, result)
                record = dict(physical, valid=valid, reward=result.get('reward') if valid else None,
                    protocol_sha256=protocol_hash, artifact=str(artifact),
                    artifact_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest())
                write_json(saved, record)
                records.append(record)
                tracking.log(dict(completed_episodes=len(records),
                                  **{mode + '/last_success': int(valid and record['reward'] == 1)}))
        write_json(worker_root / 'status.json', dict(identity, complete=True, episodes=len(records),
                    protocol_sha256=protocol_hash, updated_unix=time.time()))
        tracking.summary.update(dict(complete=True, episodes=len(records)))
    finally:
        await api.close()
        await judge_client.close()
        tracking.finish()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=Path)
    args = parser.parse_args()
    from openwebrl import run_evaluate  # Import loop-policy dependencies first.
    asyncio.run(run(json.loads(args.config.read_text())))


if __name__ == '__main__':
    main()
