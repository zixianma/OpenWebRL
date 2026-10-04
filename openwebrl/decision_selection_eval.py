"""Matched SFT-only and SFT+Jev/Kev live-browser pilot; no training."""
import argparse
import asyncio
import base64
from contextvars import ContextVar
from copy import copy
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import time

from openwebrl.decision_selection import DecisionSelector, install_page_observation, write_json
from openwebrl.kev_eval import check_server, model_spec

PROTOCOL = dict(actor='OpenWebRL-4B-SFT:iteration0', optimizer_updates=0,
    temperature=.6, top_p=.95, top_k=20, max_new_tokens=4096, max_steps=30,
    context_length=32768, context_num_screenshots=1, actor_history='full',
    selector_history=5, candidate_representation='full', candidates=5, seed=42,
    browser='browser-use', proxy=False, viewport=[1280,1000], browser_expiry_minutes=12,
    concurrency=2, task_timeout_seconds=600, judge='o4-mini',
    judge_protocol='online_mind2web/AgentTrek', judge_completion_tokens=4096,
    page_text_characters=16000, selector_modality='text and interactive element geometry')
CURRENT_TASK = ContextVar('decision_selection_task')
FINAL_SCREENSHOT = ContextVar('decision_selection_final_screenshot', default=None)


def remember_screenshot(image):
    # generate_turn_sample uses wait_for, which creates a child asyncio Task.
    # Mutate the per-episode holder inherited by that task; ContextVar.set in
    # the child would not propagate back to the terminal judge's parent task.
    holder = FINAL_SCREENSHOT.get()
    if holder is None:
        raise RuntimeError('Missing per-episode final screenshot holder')
    holder['image'] = image


class PilotJudge:
    """Four attempts per task, no SDK retries; persist the exact judge evidence."""
    def __init__(self, root, client):
        self.root, self.client = Path(root), client
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    async def create(self, **kwargs):
        if kwargs['model'] != 'o4-mini':
            raise ValueError('Wrong terminal judge')
        directory = self.root / CURRENT_TASK.get()
        directory.mkdir(parents=True, exist_ok=True)
        count = len(list(directory.glob('request-*.json')))
        if count >= 4:
            raise ValueError('Task judge attempt budget exhausted')
        ident = f'{count + 1:02d}'
        request = dict(kwargs, max_completion_tokens=4096, store=False)
        write_json(directory / f'request-{ident}.json', request)
        result = await self.client.chat.completions.create(**request)
        write_json(directory / f'response-{ident}.json', result.model_dump())
        if not result.model.startswith('o4-mini'):
            raise ValueError('Unexpected terminal judge identity')
        return result


def install_final_screenshot():
    from openwebrl.env.web_env import WebEnv
    reset, step = WebEnv.reset, WebEnv.step

    async def wrapped_reset(self, *args, **kwargs):
        result = await reset(self, *args, **kwargs)
        remember_screenshot(result[0].get('screenshot'))
        return result

    async def wrapped_step(self, *args, **kwargs):
        result = await step(self, *args, **kwargs)
        remember_screenshot(result[0].get('screenshot'))
        return result

    WebEnv.reset, WebEnv.step = wrapped_reset, wrapped_step


async def run(config):
    import httpx
    from openai import AsyncOpenAI
    from openwebrl import run_evaluate as evaluation, generate_browser as generation
    from openwebrl.eval import reward_online_mind2web as reward
    from openwebrl.arm_inference import ActionSelector
    from slime.utils.http_utils import init_http_client

    root = Path(config['output']); root.mkdir(parents=True, exist_ok=True)
    if config['protocol'] != PROTOCOL or os.environ.get('WANDB_PROJECT') != 'openwebrl-evals':
        raise ValueError('Evaluation protocol or tracking project changed')
    os.environ.update(SLIME_BROWSER_ENV_MODE='browser-use', SLIME_BROWSER_ROLLOUT_CONCURRENCY='2')
    generation._BROWSER_HOST_BLACKLIST_PATH = str(root / 'navigation-failures.txt')
    tasks = evaluation.load_tasks_from_jsonl(config['tasks'])
    if [t['task_id'] for t in tasks] != config['task_ids'] or len(tasks) != 10:
        raise ValueError('Task cohort changed')
    install_page_observation(); install_final_screenshot()
    judge_client = AsyncOpenAI(api_key=os.getenv('JUDGE_API_KEY') or os.environ['OPENAI_API_KEY'],
        base_url='https://api.openai.com/v1', max_retries=0, timeout=120)
    judge = PilotJudge(root / 'judge', judge_client)
    reward._get_openai_client = lambda **unused: judge

    async def terminal_reward(args, samples):
        sample = samples[-1] if isinstance(samples, list) else samples
        final = (FINAL_SCREENSHOT.get() or {}).get('image')
        if final:
            sample.metadata['full_image_list'] = [base64.b64encode(final).decode()]
            directory = root / 'final'; directory.mkdir(exist_ok=True)
            (directory / (CURRENT_TASK.get() + '.png')).write_bytes(final)
        else:
            sample.metadata['judge_invalid'] = True
            return [None] * len(samples) if isinstance(samples, list) else None
        return await reward.reward_func(args, samples)

    evaluation.reward_func = terminal_reward
    args = evaluation.EvalArgs(sglang_router_ip='127.0.0.1', sglang_router_port=config['actor_port'],
        hf_checkpoint=config['actor'], max_steps=30, context_num_screenshots=1,
        judge_api_model='o4-mini', judge_api_mode='served', judge_prompt_variant='agenttrek',
        judge_timeout_secs=120, browser_response_format_mode='browser_env',
        turn_history_reasoning_mode='full', browser_include_tool_response=1,
        inference_step_timeout_secs=180, task_timeout_secs=600,
        rollout_temperature=.6, rollout_top_p=.95, rollout_top_k=20, rollout_max_response_len=4096)
    init_http_client(args)
    sampling = dict(temperature=.6, top_p=.95, top_k=20, max_new_tokens=4096, repetition_penalty=1.)
    sem = asyncio.Semaphore(2)
    async with httpx.AsyncClient(trust_env=False) as client:
        response = await client.get(f'http://127.0.0.1:{config["actor_port"]}/get_model_info', timeout=30)
        response.raise_for_status()
        if Path(response.json()['model_path']).resolve() != Path(config['actor']).resolve():
            raise ValueError('Wrong SFT proposer')
        mode = config['mode']
        if mode == 'sft':
            selector = ActionSelector('baseline', '', root / 'selections', seed=42)
        else:
            provider = 'jev' if mode == 'jev' else 'kev'
            identity = {'model': 'jev-1.13.0'} if provider == 'jev' else model_spec(mode.removeprefix('kev-'))
            if provider == 'kev':
                write_json(root / 'server-identity.json', check_server(config['endpoint'], identity))
            selector = DecisionSelector(provider, root / 'selections', client=client,
                endpoint=config.get('endpoint'), api_key=os.getenv('TYPESAFE_API_KEY') or os.getenv('JEV_API_KEY'),
                max_requests=300, identity=identity)
        write_json(root / 'manifest.json', config)
        import wandb
        tracking = wandb.init(project='openwebrl-evals', group='SFT-decision-selection-20261004',
            id='sft-selection-20261004-' + mode, resume='allow', dir=str(root), config=config['protocol'])

        async def one(task):
            key = hashlib.sha256(task['task_id'].encode()).hexdigest()
            path = root / 'results' / (key + '.json')
            if path.exists():
                return json.loads(path.read_text())
            async with sem:
                if getattr(selector, 'halted', False):
                    raise RuntimeError('Decision provider halted; preserve remaining tasks')
                started_path = root / 'started' / (key + '.json')
                if started_path.exists():
                    raise RuntimeError('Interrupted task requires diagnosis; no automatic extra browser session')
                write_json(started_path, dict(task_id=task['task_id'], started_unix=time.time()))
                token = CURRENT_TASK.set(key); image_token = FINAL_SCREENSHOT.set({'image': None})
                local = copy(args); local.browser_action_selector = selector
                local.path_to_save_generated_samples = str(root / 'samples' / key)
                start = time.monotonic()
                try:
                    result = await evaluation.evaluate_single_task(local, task, sampling, True)
                    result['valid'] = (result.get('reward') in (0, 1) and
                        'ABORTED' not in result.get('status', '') and
                        not result.get('metadata', {}).get('judge_invalid') and
                        not result.get('metadata', {}).get('judge_timeout'))
                except Exception as exc:
                    result = dict(task_id=task['task_id'], reward=None, valid=False, error_type=type(exc).__name__)
                finally:
                    CURRENT_TASK.reset(token); FINAL_SCREENSHOT.reset(image_token)
                result.update(mode=mode, elapsed_seconds=time.monotonic() - start)
                write_json(path, result)
                tracking.log(dict(completed=1, success=int(result['valid'] and result.get('reward') == 1)))
                print(json.dumps(dict(task_id=task['task_id'], mode=mode, valid=result['valid'],
                                      reward=result.get('reward'))), flush=True)
                return result
        try:
            rows = await asyncio.gather(*(one(t) for t in tasks), return_exceptions=True)
            if any(isinstance(r, BaseException) for r in rows):
                raise RuntimeError('One or more tasks could not start; preserve the partial cohort')
            from openwebrl.arm_eval import summarize
            summary = summarize(rows, 10)
            summary['complete'] = len(rows) == 10
            summary['provider_halted'] = getattr(selector, 'halted', False)
            if summary['provider_halted']:
                raise RuntimeError('Decision provider halted; inspect saved responses')
            from openwebrl.env.browser_use_env import _list_session_ids
            if _list_session_ids():
                raise RuntimeError('Remote browser cleanup is pending')
            write_json(root / 'summary.json', summary)
            tracking.summary.update(summary)
        finally:
            tracking.finish(); await judge_client.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    a = parser.parse_args()
    # Match the established runner's import-before-loop requirement.
    from openwebrl import run_evaluate  # noqa: F401
    asyncio.run(run(json.loads(Path(a.config).read_text())))
