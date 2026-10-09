"""Task-parallel four-arm pilot; each worker has an exclusive one-GPU actor."""
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
from types import SimpleNamespace

from openwebrl.luna_qwen_metrics import MODES, PRICING, claimed_tasks, digest, episode_usage, write_json
from openwebrl.luna_qwen_policy import Budget, MeteredAPI, Policy
from openwebrl.browser_actor_protocol import browser_harness_protocol, validate_browser_harness_protocol

PROTOCOL = dict(actor='Qwen/Qwen3-VL-4B-Thinking', actor_revision='1de27d8c51f12e819435303b9e84c4e25ba8401e',
    temperature=1., top_p=.9, top_k=-1, repetition_penalty=1., candidates=[1,5,10],
    actor_max_output_tokens=4096, context_length=32768, max_steps=30,
    selector='gpt-6-luna', luna_reasoning='medium', luna_max_output_tokens=4096,
    luna_service_tier='default', actor_history='full', selector_history='full',
    screenshots=1, browser='local_process', concurrency_per_gpu=1, workers=4, viewport=[1280,720],
    qwen_cache='flush before each episode; retain within-episode reuse',
    task_timeout_seconds=1800, request_timeout_seconds=180,
    judge='o4-mini-2025-04-16', judge_protocol='Online-Mind2Web/AgentTrek',
    judge_evidence='executed actions and final screenshot; actor thoughts excluded for every arm',
    judge_seed=42, judge_max_output_tokens=4096, judge_max_attempts=4,
    optimizer_updates=0, image_detail='high', seed=20261004, **browser_harness_protocol())

SFT_PROTOCOL = dict(PROTOCOL, actor='OpenWebRL/OpenWebRL-4B-SFT',
    actor_revision='15e777db2ddba2e0e82080ebccd3ad8d215b7f0a', candidates=[5])


def action_only_messages(messages):
    from openwebrl.arm_inference import split_response
    return [dict(m, content=split_response(m['content'])['action'])
        if m.get('role') == 'assistant' and isinstance(m.get('content'), str) else m for m in messages]


class Judge:
    def __init__(self, client, budget):
        self.client, self.budget = client, budget
        self.directory = None
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    async def create(self, **kwargs):
        from openwebrl.luna_qwen_metrics import api_cost
        if kwargs['model'] != PROTOCOL['judge'] or kwargs.get('seed') != 42:
            raise ValueError('Judge protocol changed')
        existing = list((self.directory / 'requests').glob('judge-*.json'))
        if len(existing) >= 4:
            raise RuntimeError('Four explicit judge attempts exhausted')
        text_bytes = sum(len(str(p.get('text', '')).encode()) for m in kwargs['messages']
            for p in (m['content'] if isinstance(m['content'], list) else [{'text': m['content']}]))
        upper = ((text_bytes + 32768 + 4096) * 1.1 + 4096 * 4.4) / 1e6
        ident = self.budget.update(reserve=upper)
        request = dict(kwargs, max_completion_tokens=4096, store=False)
        path = self.directory / 'requests' / f'judge-{ident:05d}.json'
        row = dict(provider='api', role='judge', status='reserved', request=request,
                   reserved_usd=upper, started_epoch=time.time())
        write_json(path, row)
        started = time.monotonic()
        try:
            response = await self.client.chat.completions.create(**request)
            row.update(status='received', response=response.model_dump(), usage=response.usage.model_dump())
            self.budget.update(settle=(upper, api_cost(row['usage'], 'judge')['upper_usd']))
            if response.model != PROTOCOL['judge']:
                raise ValueError('Unexpected judge snapshot')
            return response
        except BaseException as exc:
            row['error_type'] = type(exc).__name__
            raise
        finally:
            row['seconds'] = time.monotonic() - started
            write_json(path, row)


def verified_record(path, protocol_hash):
    record = json.loads(path.read_text())
    if record['protocol_sha256'] != protocol_hash:
        raise ValueError('Resume protocol changed')
    result = Path(record['artifact'])
    if hashlib.sha256(result.read_bytes()).hexdigest() != record['artifact_sha256']:
        raise ValueError('Saved result changed')
    return record


async def run(config):
    from openai import AsyncOpenAI
    import httpx
    import wandb
    from openwebrl import run_evaluate as evaluation, generate_browser as generation
    from openwebrl.eval import reward_online_mind2web as reward
    from openwebrl.arm_inference import load_selection_builder
    from slime.utils.http_utils import init_http_client

    root = Path(config['output'])
    worker_id = config['worker_id']
    worker_root = root / 'workers' / str(worker_id)
    worker_root.mkdir(parents=True, exist_ok=True)
    write_json(worker_root / 'status.json', dict(complete=False, stage='starting',
        job_id=os.getenv('SLURM_JOB_ID'), updated_unix=time.time()))
    plan = json.loads(Path(config['plan']).read_text())
    protocol = plan['protocol']
    validate_browser_harness_protocol(protocol)
    if protocol not in (PROTOCOL, SFT_PROTOCOL) or os.environ.get('WANDB_PROJECT') != 'openwebrl-evals':
        raise ValueError('Experiment protocol or W&B project changed')
    if protocol == SFT_PROTOCOL and any(item['modes'] != ['sft_luna5'] for item in plan['schedule']):
        raise ValueError('Official SFT extension must use its separate named arm')
    if hashlib.sha256(Path(plan['task_file']).read_bytes()).hexdigest() != plan['task_file_sha256']:
        raise ValueError('Task file changed')
    os.environ.update(SLIME_BROWSER_ENV_MODE='local_process', SLIME_BROWSER_LOCAL_PROCESS_MAX_PROCESSES='1',
        SLIME_BROWSER_ROLLOUT_CONCURRENCY='1', SLIME_BROWSER_LOCAL_PROCESS_LOG_DIR=str(worker_root / 'browser_logs'),
        SLIME_BROWSER_LOCAL_PROCESS_PORT_START=str(config['browser_port']),
        SLIME_BROWSER_LOCAL_PROCESS_PORT_END=str(config['browser_port'] + 99))
    generation._BROWSER_HOST_BLACKLIST_PATH = str(worker_root / 'navigation-failures.txt')
    tasks = {t['task_id']: t for t in evaluation.load_tasks_from_jsonl(plan['task_file'])}
    builder = load_selection_builder(plan['selector_prompt_source'])
    os.environ['VISION_FULL_HISTORY'] = '1'
    qwen = json.loads((Path(plan['actor_path']) / 'config.json').read_text())
    processor = json.loads((Path(plan['actor_path']) / 'preprocessor_config.json').read_text())
    async with httpx.AsyncClient(trust_env=False) as h:
        response = await h.get(f'http://127.0.0.1:{config["actor_port"]}/get_model_info')
        response.raise_for_status()
        if Path(response.json()['model_path']).resolve() != Path(plan['actor_path']).resolve():
            raise ValueError('Wrong actor weights')
    api = AsyncOpenAI(api_key=os.environ['OPENAI_API_KEY'], base_url='https://api.openai.com/v1',
                      max_retries=0, timeout=180)
    judge_client = AsyncOpenAI(api_key=os.getenv('JUDGE_API_KEY') or os.environ['OPENAI_API_KEY'],
                              base_url='https://api.openai.com/v1', max_retries=0, timeout=180)
    luna_budget = Budget(root / 'luna-budget', plan['limits']['luna_usd'], plan['limits']['luna_calls'])
    judge_budget = Budget(root / 'judge-budget', plan['limits']['judge_usd'], plan['limits']['judge_calls'])
    judge = Judge(judge_client, judge_budget)
    reward._get_openai_client = lambda **unused: judge
    current = {}

    async def terminal_reward(args, samples):
        # Stop agent latency before any judge request. The generation path also
        # records the actual post-action screenshot, including terminal done.
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
    args = evaluation.EvalArgs(sglang_router_ip='127.0.0.1', sglang_router_port=config['actor_port'],
        hf_checkpoint=plan['actor_path'], max_steps=30, context_num_screenshots=1,
        judge_api_model=PROTOCOL['judge'], judge_api_mode='served', judge_timeout_secs=180,
        browser_response_format_mode='browser_env', turn_history_reasoning_mode='full',
        browser_include_tool_response=1, inference_step_timeout_secs=180, task_timeout_secs=1800,
        rollout_temperature=1., rollout_top_p=.9, rollout_top_k=-1, rollout_max_response_len=4096)
    init_http_client(args)
    tracking = wandb.init(project='openwebrl-evals', group='luna-qwen-inference-20261004',
        id=f'{plan.get("wandb_prefix", "luna-qwen-pilot-20261004")}-w{worker_id}',
        resume='allow', dir=str(worker_root), config=protocol)
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
                    records.append(verified_record(saved, protocol_hash))
                    continue
                if any((b.root / 'halt.json').exists() for b in (luna_budget, judge_budget)):
                    raise RuntimeError('Persistent API budget halt')
                if mode != 'luna':
                    async with httpx.AsyncClient(trust_env=False) as h:
                        flushed = await h.post(f'http://127.0.0.1:{config["actor_port"]}/flush_cache', timeout=60)
                        flushed.raise_for_status()
                directory = root / 'attempts' / key / str(time.time_ns())
                directory.mkdir(parents=True)
                local = copy(args)
                local.path_to_save_generated_samples = str(directory / 'samples')
                local.luna_episode_observation = dict(browser_steps=0, browser_step_seconds=0.)
                policy = Policy(mode, directory, MeteredAPI(api, luna_budget, directory), builder, qwen, processor)
                local.browser_action_selector = policy
                judge.directory = directory
                current.clear()
                current.update(started=time.monotonic(), directory=directory)
                write_json(worker_root / 'heartbeat.json', dict(stage='episode', mode=mode, task_key=key,
                    completed=len(records), updated_unix=time.time(), job_id=os.getenv('SLURM_JOB_ID')))
                write_json(directory / 'identity.json', dict(mode=mode, task_id=item['task_id'], started_epoch=time.time()))
                interrupted = False
                try:
                    result = await evaluation.evaluate_single_task(local, tasks[item['task_id']],
                        dict(temperature=1., top_p=.9, top_k=-1, max_new_tokens=4096, repetition_penalty=1.), True)
                    valid = result.get('reward') in (0,1) and 'ABORTED' not in result.get('status', '')
                    valid = valid and not result.get('metadata', {}).get('judge_timeout')
                except asyncio.CancelledError:
                    interrupted = True
                    raise
                except Exception as exc:
                    valid = False
                    result = dict(reward=None, error_type=type(exc).__name__, error=str(exc))
                finally:
                    seconds = current.get('episode_seconds', time.monotonic() - current['started'])
                    usage = episode_usage(directory, mode, seconds)
                    physical = dict(task_id=item['task_id'], mode=mode, worker_id=worker_id, interrupted=interrupted,
                        episode_seconds=seconds, local_model_batch_seconds=policy.local_batch_seconds,
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
                tracking.log(dict(completed_episodes=len(records), **{mode + '/last_success': int(valid and record['reward'] == 1)}))
        write_json(worker_root / 'status.json', dict(complete=True, episodes=len(records),
                                            protocol_sha256=protocol_hash))
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
