"""Paired Sol serial/parallel live pilot; no allocation submission or automatic retry."""
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
from openwebrl.sol_serial_parallel_lineage import verify_existing_records, accepted_plan_hashes
from openwebrl.luna_qwen_eval import Judge, action_only_messages, verified_record
from openwebrl.luna_qwen_metrics import claimed_tasks, digest, episode_usage, write_json
from openwebrl.luna_qwen_policy import Budget
from openwebrl.luna_qwen_full_eval import verify_actor_files, sampling_for_family
from openwebrl.reasoning_actor_policy import metric_prices
from openwebrl.outcome_validity import classify_terminate_reason
from openwebrl.browser_actor_protocol import validate_browser_harness_protocol


MODES = ('serial_high', 'parallel3_select_low')
RESOURCES = dict(gpus=0, cpus=16, memory_gib=64, total_seconds=21600)
LIMITS = dict(actor_usd=100., actor_calls=15000, judge_usd=20., judge_calls=200, browser_sessions=200)


def paired_schedule(task_ids, seed):
    """Task order and within-task arm order frozen independently of outcomes."""
    import random
    rows = []
    for index, task in enumerate(sorted(task_ids, key=lambda t: digest([seed, t, 'task']))):
        modes = list(MODES)
        if index % 2: modes.reverse()
        rows.append(dict(task_id=task, modes=modes))
    return rows


def validate_plan(plan, config):
    protocol = plan['protocol']
    validate_browser_harness_protocol(protocol)
    expected = dict(actor='gpt-6.1-sol', serial_effort='high', serial_max_output_tokens=12288,
        parallel_effort='low', parallel_max_output_tokens=4096, selector_effort='low',
        selector_max_output_tokens=2048, candidates=3, resize_output_coords=False,
        browser_coordinate_space='viewport_pixels', judge='gpt-4o', max_steps=30)
    if protocol.get('selector_efforts') != dict(parallel3_select_low='low'):
        raise ValueError('Frozen selector effort by arm changed')
    if protocol.get('latency_endpoint') != 'episode_before_terminal_judge':
        raise ValueError('Latency endpoint changed')
    if any(protocol.get(k) != v for k, v in expected.items()):
        raise ValueError('Frozen serial/parallel browser protocol changed')
    if (plan.get('family') != 'luna' or plan.get('resources') != RESOURCES
            or plan.get('limits') != LIMITS or plan.get('workers') != 6):
        raise ValueError('CPU harness or exact proposed caps changed')
    if config.get('plan_sha256') != digest(plan) or Path(config['output']) != Path(plan['output']):
        raise ValueError('Frozen worker plan changed')
    if plan['protocol'].get('judge_protocol') != 'webvoyager' or plan['protocol'].get('judge_max_attached_imgs') != 30:
        raise ValueError('Native benchmark judge protocol changed')
    if plan['pricing'] != dict(metric_prices('gpt-6.1-sol'),judge_input=2.5,judge_cached=1.25,judge_output=10.):
        raise ValueError('Incorrect Sol prices')
    tasks = plan['task_ids']
    if (len(tasks) != 100 or len(set(tasks)) != 100
            or plan['schedule'] != paired_schedule(tasks, plan['seed'])):
        raise ValueError('Require exactly100 frozen pilot tasks and frozen randomized paired order')
    if plan.get('wandb_project') != 'openwebrl-evals':
        raise ValueError('Separate evaluations must use openwebrl-evals')
    return 'luna'


class MeteredSolAPI:
    """Reserve each physical call before network I/O; unknown charges remain reserved."""
    model = 'gpt-6.1-sol'

    def __init__(self, client, budget, directory, pricing, selector_effort):
        if selector_effort != 'low': raise ValueError('Unapproved selector effort')
        self.selector_effort = selector_effort
        self.client, self.budget = client, budget
        self.directory, self.pricing = Path(directory), pricing

    def halt(self, error):
        write_json(self.budget.root / 'halt.json', dict(reason='actor/selector protocol failure; review required',
                                                      error_type=type(error).__name__))

    async def create(self, role, **kwargs):
        from openwebrl.responses_actor_history import response_input_bound
        from openwebrl.luna_qwen_metrics import api_cost
        settings = {'actor_high': ('high', 12288), 'selector': ('low', 2048),
                    **{f'actor_low_{i}': ('low', 4096) for i in range(3)}}
        if role not in settings:
            raise ValueError('Unapproved request role')
        effort, cap = settings[role]
        if (kwargs.get('model') != self.model or kwargs.get('reasoning') != {'effort': effort}
                or kwargs.get('max_output_tokens') != cap or kwargs.get('store') is not False
                or kwargs.get('service_tier') != 'default'
                or any(k in kwargs for k in ('temperature', 'top_p'))
                or (role != 'selector' and kwargs.get('include') != ['reasoning.encrypted_content'])
                or (role == 'selector' and 'include' in kwargs)
                or (role != 'selector' and (kwargs.get('tool_choice') != 'required'
                                           or kwargs.get('parallel_tool_calls') is not False))):
            raise ValueError('Request differs from frozen actor/selector protocol')
        bound = response_input_bound(kwargs['input'], kwargs.get('tools', []))
        if bound > 272000:
            raise ValueError('Conservative input bound exceeds frozen pricing tier')
        reserve = (bound * self.pricing['luna_cache_write'] + cap * self.pricing['luna_output']) / 1e6
        ident = self.budget.update(reserve=reserve)
        path = self.directory / 'requests' / f'sol-{ident:05d}.json'
        cost_role = 'selector' if role == 'selector' else 'actor'
        row = dict(provider=self.model, role=cost_role, physical_role=role,
                   request=kwargs, status='reserved', reserved_usd=reserve, started_epoch=time.time())
        write_json(path, row)
        started = time.monotonic()
        try:
            response = await self.client.responses.create(**kwargs)
            row.update(status='received', response=response.model_dump(), usage=response.usage.model_dump())
            self.budget.update(settle=(reserve, api_cost(row['usage'], cost_role, self.pricing)['upper_usd']))
            if (self.budget.root / 'halt.json').exists():
                raise RuntimeError('Usage settlement exceeded a conservative reservation or shared halt')
            if not (response.model == self.model or response.model.startswith(self.model + '-')):
                raise ValueError('Unexpected Sol model identity')
            if response.status != 'completed':
                raise ValueError('Incomplete response; never execute a partial action')
            return response
        except BaseException as exc:
            row.update(error_type=type(exc).__name__, http_status=getattr(exc, 'status_code', None))
            if 'usage' not in row:
                row['status'] = 'usage_unknown'
            write_json(self.budget.root / 'halt.json', dict(reason='physical API failure; review before retry',
                                                          error_type=type(exc).__name__))
            raise
        finally:
            row['seconds'] = time.monotonic() - started
            write_json(path, row)


class SingleAttemptJudge(Judge):
    """Inherited reward helper may retry parsing; allow exactly one physical call/episode."""
    async def create(self, **kwargs):
        if list((self.directory / 'requests').glob('judge-*.json')):
            raise RuntimeError('Physical judge retry is prohibited')
        try:
            return await super().create(**kwargs)
        except BaseException as exc:
            write_json(self.budget.root / 'halt.json', dict(reason='physical judge failure; review required',
                                                          error_type=type(exc).__name__))
            raise


async def run(config):
    plan = json.loads(Path(config['plan']).read_text())
    family = validate_plan(plan, config)
    if os.environ.get('WANDB_PROJECT') != 'openwebrl-evals':
        raise ValueError('Separate evaluations must use openwebrl-evals')
    from openwebrl.sol_serial_parallel_policy import SolSerialParallelPolicy
    from openai import AsyncOpenAI
    import httpx
    import wandb
    from openwebrl import run_evaluate as evaluation, generate_browser as generation
    from openwebrl.eval import reward_webvoyager as reward
    from slime.utils.http_utils import init_http_client

    if not all(hasattr(generation, name) for name in ('_luna_measured_step', '_luna_measured_reset')):
        raise ValueError('Worker requires frozen browser measurement and context hooks')
    root, worker_id = Path(config['output']), config['worker_id']
    worker_root = root / 'workers' / str(worker_id)
    worker_root.mkdir(parents=True, exist_ok=True)
    identity = dict(experiment_id=plan['experiment_id'], family=family, actor_model=plan['protocol']['actor'], comparison='one high action versus three low candidates with low selection',
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
    from openwebrl.native_benchmark_judge import NativeBenchmarkJudge
    judge = NativeBenchmarkJudge(judge_client, judge_budget, plan['pricing'])
    browser_budget = Budget(budget_root / 'browser-sessions', float(LIMITS['browser_sessions']), LIMITS['browser_sessions'])
    reward._get_openai_client = lambda **unused: judge
    current = {}

    async def terminal_reward(args, samples):
        current['episode_seconds'] = time.monotonic() - current['started']
        screenshot = args.luna_episode_observation.get('screenshot')
        if not screenshot:
            return [None] * len(samples) if isinstance(samples, list) else None
        last = samples[-1] if isinstance(samples, list) else samples
        # Keep native trajectory screenshots; actor observes only current screenshot.
        if not last.metadata.get('full_image_list'):
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
        judge_api_model=protocol['judge'], judge_max_attached_imgs=30, judge_api_mode='served', judge_timeout_secs=180,
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
                    if (any(record.get(k) != identity[k] for k in ('experiment_id', 'family'))
                            or record.get('plan_sha256') not in accepted_plan_hashes(plan)):
                        raise ValueError('Saved record belongs to another frozen family experiment')
                    records.append(record)
                    continue
                if any((b.root / 'halt.json').exists() for b in (luna_budget, judge_budget)):
                    raise RuntimeError('Persistent API budget halt')
                if family != 'luna':
                    async with httpx.AsyncClient(trust_env=False) as h:
                        flushed = await h.post(f'http://127.0.0.1:{actor_port}/flush_cache', timeout=60)
                        flushed.raise_for_status()
                if list((root / 'attempts' / key).glob('*')):
                    raise ValueError('Unsettled prior episode attempt requires review; no automatic retry')
                directory = root / 'attempts' / key / str(time.time_ns())
                directory.mkdir(parents=True)
                local = copy(args)
                local.path_to_save_generated_samples = str(directory / 'samples')
                local.luna_episode_observation = dict(browser_steps=0, browser_step_seconds=0.)
                policy = SolSerialParallelPolicy(mode=mode, directory=directory,
                    api=MeteredSolAPI(api, luna_budget, directory, plan['pricing'], protocol['selector_effort']))
                local.browser_action_selector = policy
                judge.directory = directory
                current.clear()
                current.update(started=time.monotonic(), directory=directory)
                write_json(worker_root / 'heartbeat.json', dict(identity, stage='episode', mode=mode,
                    task_key=key, completed=len(records), updated_unix=time.time()))
                write_json(directory / 'identity.json', dict(identity, mode=mode,
                    task_id=item['task_id'], started_epoch=time.time()))
                interrupted = False
                os.environ['SOL_BROWSER_RESOURCE_RECEIPT'] = str(directory / 'browser-resources.json')
                try:
                    browser_budget.update(reserve=1.)
                    result = await evaluation.evaluate_single_task(local, tasks[item['task_id']],
                                                                   sampling_for_family(family), True)
                    outcome = classify_terminate_reason(result.get('status'), result.get('terminate_reason'),
                        judge_failed=bool(result.get('metadata', {}).get('judge_timeout')
                                          or result.get('metadata', {}).get('judge_invalid')
                                          or result.get('reward') not in (0, 1)))
                    valid = outcome.valid and result.get('reward') in (0, 1)
                    result['outcome_validity'] = dict(ending=outcome.ending, detail=outcome.detail, valid=valid)
                    if outcome.ending == 'judge':
                        write_json(judge_budget.root / 'halt.json',
                                   dict(reason='Missing or invalid judge verdict; review before further calls'))
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
                        interrupted=interrupted, episode_seconds=seconds, task_latency_seconds=seconds,
                        task_latency_excludes_judge=True, latency_endpoint='episode_before_terminal_judge',
                        selector_effort=protocol['selector_effort'] if mode == 'parallel3_select_low' else None,
                        local_model_batch_seconds=0.,
                        browser_steps=local.luna_episode_observation['browser_steps'],
                        browser_step_seconds=local.luna_episode_observation['browser_step_seconds'],
                        judge_seconds=current.get('judge_seconds', 0.), **usage)
                    resource_path = directory / 'browser-resources.json'
                    try:
                        physical['browser_resources'] = json.loads(resource_path.read_text()) if resource_path.exists() else None
                        physical['browser_resource_receipt_missing'] = not resource_path.exists()
                    except (OSError, ValueError) as telemetry_error:
                        physical['browser_resources'] = None
                        physical['browser_resource_receipt_error'] = type(telemetry_error).__name__
                    finally:
                        os.environ.pop('SOL_BROWSER_RESOURCE_RECEIPT', None)
                    write_json(directory / 'physical-usage.json', physical)
                artifact = directory / 'result.json'
                write_json(artifact, result)
                record = dict(physical, valid=valid, reward=result.get('reward') if valid else None,
                    protocol_sha256=protocol_hash, artifact=str(artifact),
                    artifact_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest())
                write_json(saved, record)
                records.append(record)
                if any((b.root / 'halt.json').exists() for b in (luna_budget, judge_budget, browser_budget)):
                    raise RuntimeError('Persistent budget or API halt; disposition preserved')
                tracking.log(dict(completed_episodes=len(records),
                                  **{mode + '/last_success': int(valid and record['reward'] == 1),
                                     mode + '/task_latency_seconds': record['task_latency_seconds']}))
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
