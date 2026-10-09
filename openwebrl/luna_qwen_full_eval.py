"""Full Online-Mind2Web actor/selector workers, isolated from the T=1/p=.9 pilot.

Local families own one GPU per worker; the Luna family only needs CPU browser
and tokenizer resources. The controller pins this module and the pilot's
instrumented browser implementation in a new immutable source directory.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
from copy import copy, deepcopy
import hashlib
import json
import os
from pathlib import Path
import signal
import time

from openwebrl.luna_qwen_eval import Judge, PROTOCOL, action_only_messages, verified_record
from openwebrl.luna_qwen_metrics import PRICING, claimed_tasks, digest, episode_usage, write_json
from openwebrl.luna_qwen_policy import Budget, MeteredAPI, Policy
from openwebrl.luna_qwen_transport import init_candidate_http_client, flush_cache_when_idle
from openwebrl.browser_actor_protocol import validate_browser_harness_protocol

FAMILY_MODES = {
    'qwen': ('qwen', 'qwen_luna5'),
    'sft': ('sft_luna5', 'sft_luna10'),
    'luna': ('luna',),
}
ACTORS = {
    'qwen': ('Qwen/Qwen3-VL-4B-Thinking', '1de27d8c51f12e819435303b9e84c4e25ba8401e'),
    'sft': ('OpenWebRL/OpenWebRL-4B-SFT', '15e777db2ddba2e0e82080ebccd3ad8d215b7f0a'),
    'luna': ('gpt-6-luna', None),
}
SELECTION_PROMPT_SHA256 = '043ab7e98127f07ccd76b0eb736837f8a089c1328fa211c4ca4c20d771a7751c'
LOCAL_SAMPLING = dict(temperature=1., top_p=.95, top_k=-1,
                      max_new_tokens=4096, repetition_penalty=1.)


def protocol_for_family(family):
    """Return the complete scientific protocol; allocation size is not a knob."""
    if family not in FAMILY_MODES:
        raise ValueError('Unknown actor family')
    protocol = deepcopy(PROTOCOL)
    protocol.pop('workers')
    protocol.update(actor=ACTORS[family][0], actor_revision=ACTORS[family][1],
        top_p=.95, candidates={'qwen': [1, 5], 'sft': [5, 10], 'luna': [1]}[family],
        benchmark='Online-Mind2Web', benchmark_tasks=300,
        terminal_status_rule='COMPLETED: judge; FAILED including step limit: zero; ABORTED: invalid',
        candidate_selection='shuffled candidates; index only; execute chosen candidate unchanged',
        local_actor_sampling='temperature 1.0; top_p 0.95; top_k -1; repetition_penalty 1.0',
        framework_tokenizer=ACTORS['qwen' if family == 'luna' else family][0],
        framework_tokenizer_revision=ACTORS['qwen' if family == 'luna' else family][1])
    if family == 'luna':
        protocol.update(temperature=None, top_p=None, top_k=None, repetition_penalty=None,
            selector=None, concurrency_per_gpu=0, qwen_cache='no local model serving',
            local_actor_sampling='not applicable; Luna API controls sampling',
            luna_snapshot='provider-resolved snapshot saved in every receipt')
    return protocol


def sampling_for_family(family):
    if family not in FAMILY_MODES:
        raise ValueError('Unknown actor family')
    # The framework needs serialization settings even when the API supplies the
    # action. Policy's Luna branch never forwards these local sampling values.
    return dict(LOCAL_SAMPLING)


def validate_plan(plan, config):
    """Reject partial cohorts, changed recipes, cross-family modes and reuse."""
    family = plan.get('family')
    if plan.get('protocol') != protocol_for_family(family):
        raise ValueError('Full-set scientific protocol changed')
    validate_browser_harness_protocol(plan['protocol'])
    if config.get('plan_sha256') != digest(plan):
        raise ValueError('Frozen family plan changed')
    if Path(config['output']).resolve() != Path(plan['output']).resolve():
        raise ValueError('Worker output differs from frozen family plan')
    if plan.get('pricing') != PRICING:
        raise ValueError('Frozen metric pricing changed')
    experiment = plan.get('experiment_id', '')
    if not experiment or experiment == 'luna-qwen-inference-20261004':
        raise ValueError('Full-set experiment needs a separate identity')
    if plan.get('wandb_group') != experiment or plan.get('wandb_prefix') != experiment + '-' + family:
        raise ValueError('W&B identity must isolate the experiment and family')
    tasks = plan.get('task_ids', [])
    schedule = plan.get('schedule', [])
    if len(tasks) != 300 or len(set(tasks)) != 300 or len(schedule) != 300:
        raise ValueError('Full Online-Mind2Web cohort must contain 300 unique tasks')
    if {row['task_id'] for row in schedule} != set(tasks):
        raise ValueError('Schedule differs from frozen cohort')
    expected_modes = set(schedule[0]['modes'])
    if not expected_modes or not expected_modes <= set(FAMILY_MODES[family]):
        raise ValueError('Unexpected actor/selector combination')
    for row in schedule:
        if set(row['modes']) != expected_modes or len(row['modes']) != len(expected_modes):
            raise ValueError('Every task must have the same unique family arms')
    if plan.get('selector_prompt_sha256') != SELECTION_PROMPT_SHA256:
        raise ValueError('Selector prompt changed')
    if not plan.get('budget_root') or any(plan['limits'].get(k, 0) <= 0
            for k in ('luna_usd', 'luna_calls', 'judge_usd', 'judge_calls')):
        raise ValueError('Shared API budget caps missing')
    return family


def verify_actor_files(plan):
    """Check pinned config/processor identities and preverified weight stats."""
    root = Path(plan['actor_path'])
    for name, key in (('config.json', 'actor_config_sha256'),
                      ('preprocessor_config.json', 'actor_preprocessor_sha256')):
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != plan[key]:
            raise ValueError('Actor configuration changed: ' + name)
    if plan['family'] != 'luna':
        if plan.get('actor_revision') != ACTORS[plan['family']][1] or not plan.get('actor_files'):
            raise ValueError('Pinned actor revision or verified weights missing')
        for name, expected in plan['actor_files'].items():
            status = (root / name).stat()
            if status.st_size != expected['size'] or status.st_mtime_ns != expected['mtime_ns']:
                raise ValueError('Preverified actor weights changed: ' + name)
    return (json.loads((root / 'config.json').read_text()),
            json.loads((root / 'preprocessor_config.json').read_text()))


def verify_existing_records(plan, plan_sha256):
    """Completed task claims are skipped, so validate their records up front."""
    expected = {digest([row['task_id'], mode])[:24]: (row['task_id'], mode)
                for row in plan['schedule'] for mode in row['modes']}
    protocol_hash = digest(plan['protocol'])
    for file in (Path(plan['output']) / 'records').glob('*.json'):
        if file.stem not in expected:
            raise ValueError('Unexpected saved full-set record')
        record = verified_record(file, protocol_hash)
        if (record.get('task_id'), record.get('mode')) != expected[file.stem]:
            raise ValueError('Saved record identity differs from filename')
        if (record.get('experiment_id'), record.get('family'), record.get('plan_sha256')) != (
                plan['experiment_id'], plan['family'], plan_sha256):
            raise ValueError('Saved record belongs to another frozen family experiment')


class FullPolicy(Policy):
    """Use unchanged proposal/selector accounting with the extra SFT N=10 arm."""
    def __init__(self, mode, *args, **kwargs):
        if mode not in {m for modes in FAMILY_MODES.values() for m in modes}:
            raise ValueError('Unexpected full-set arm')
        super().__init__('qwen_luna10' if mode == 'sft_luna10' else mode, *args, **kwargs)
        self.mode = mode

    async def __call__(self, **kwargs):
        if self.mode != 'luna':
            sampling = kwargs['sampling_params']
            fixed = {k: v for k, v in LOCAL_SAMPLING.items() if k != 'max_new_tokens'}
            remaining = 32768 - self.prompt_tokens - 1
            # The frozen pilot passes 4096; a newer framework may apply this
            # same context-boundary clamp before reaching the policy hook.
            if any(sampling.get(k) != value for k, value in fixed.items()) or (
                    sampling.get('max_new_tokens') not in (4096, min(4096, remaining))):
                raise ValueError('Actor decoding differs from full-set protocol')
        return await super().__call__(**kwargs)


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
    identity = dict(experiment_id=plan['experiment_id'], family=family,
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
    luna_budget = Budget(budget_root / 'luna-budget', plan['limits']['luna_usd'], plan['limits']['luna_calls'])
    judge_budget = Budget(budget_root / 'judge-budget', plan['limits']['judge_usd'], plan['limits']['judge_calls'])
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
    if family != 'luna':
        init_candidate_http_client(args, init_http_client)
    else:
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
                        flushed = await flush_cache_when_idle(h, f'http://127.0.0.1:{actor_port}/flush_cache')
                        write_json(worker_root / 'cache-flushes' / f'{key}-{time.time_ns()}.json', flushed)
                directory = root / 'attempts' / key / str(time.time_ns())
                directory.mkdir(parents=True)
                local = copy(args)
                local.path_to_save_generated_samples = str(directory / 'samples')
                local.luna_episode_observation = dict(browser_steps=0, browser_step_seconds=0.)
                policy = FullPolicy(mode, directory, MeteredAPI(api, luna_budget, directory),
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
                    usage = episode_usage(directory, mode, seconds, plan['pricing'])
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
