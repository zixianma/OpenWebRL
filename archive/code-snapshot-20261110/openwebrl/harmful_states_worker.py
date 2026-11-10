"""Bounded screenshot-policy pilot on persistent, local synthetic applications.

No remote judge is used. Importing this module performs no model/browser work.
Only the CLI dispatches inference, against explicitly supplied loopback services.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
from contextlib import contextmanager
from copy import copy
import hashlib
import json
import os
from pathlib import Path
import random
import time
from urllib.parse import urlparse

from openwebrl.harmful_states import HarmfulStatesSandbox, list_cases
from openwebrl.harmful_states_server import running_server
from openwebrl.controlled_sft_worker import AttemptJournal

METHODS = ('ordinary', 'retries5', 'selection5')
SAMPLING = dict(temperature=.7, top_p=.9, top_k=-1, max_new_tokens=1024)
CAPS = dict(actor_requests=1650, selector_requests=150, episodes=105)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temporary.replace(path)


def schedule(seeds=(43, 44, 45)):
    if list(seeds) != [43, 44, 45]:
        raise ValueError('This bounded pilot fixes seeds 43,44,45')
    rows = []
    for case in list_cases():
        for seed in seeds:
            methods = list(METHODS)
            random.Random(f'harmful-v1/{case["case_id"]}/{seed}').shuffle(methods)
            for method in methods:
                rows.append(dict(case_id=case['case_id'], seed=seed, method=method,
                    key=f'{case["case_id"]}-{seed}-{method}'))
    return rows


class BudgetExhausted(RuntimeError):
    pass


class EpisodeBoundary(RuntimeError):
    pass


class DurableBudget:
    """Count reservations, including failed/unknown calls, across worker restarts."""
    def __init__(self, root, deadline_unix):
        self.journal = AttemptJournal(root)
        self.deadline_unix = float(deadline_unix)
        self.counts = {key: 0 for key in CAPS}
        for path in sorted(self.journal.directory.glob('[0-9]*.json')):
            entry = json.loads(path.read_text())
            if entry['event'] == 'reserve':
                self.counts[entry['record']['kind']] += entry['record']['count']
        if any(self.counts[k] > CAPS[k] for k in CAPS):
            raise ValueError('Existing reservations exceed this protocol')

    def remaining_seconds(self):
        return self.deadline_unix - time.time()

    def reserve(self, kind, count, **context):
        if kind not in CAPS or type(count) is not int or count < 1:
            raise ValueError('Invalid reservation')
        if self.remaining_seconds() <= 0 or self.counts[kind] + count > CAPS[kind]:
            raise BudgetExhausted('Pilot resource boundary reached')
        # Synchronous append+fsync happens before dispatch and before yielding.
        self.journal('reserve', dict(kind=kind, count=count, **context))
        self.counts[kind] += count


def usage_total(events, role):
    receipts = [e['record'] for e in events if e['event'] == role + '_received']
    reservations = [e for e in events if e['event'] == role + '_reserved']
    return dict(requests=len(reservations), metered_requests=len(receipts),
        unknown_usage_requests=len(reservations)-len(receipts),
        input_tokens=sum(r['input_tokens'] for r in receipts),
        output_tokens=sum(r['output_tokens'] for r in receipts),
        cached_input_tokens=sum(r.get('cached_input_tokens') or 0 for r in receipts),
        cache_metered_requests=sum(r.get('cached_input_tokens') is not None for r in receipts),
        client_request_seconds=sum(r.get('client_seconds', 0) for r in receipts))


class BrowserPolicy:
    """Native generation hook: all candidates are metered, only one is executed."""
    browser_coordinate_space = 'normalized_1000'

    def __init__(self, *, config, item, episode, sandbox, campaign, episode_id,
                 budget, journal, actor_post, selector_post):
        self.config, self.item, self.episode = config, item, episode
        self.sandbox, self.campaign, self.episode_id = sandbox, campaign, episode_id
        self.budget, self.journal = budget, journal
        self.actor_post, self.selector_post = actor_post, selector_post
        self.prompt_tokens = None
        self.error = None
        self.boundary_reason = None
        self.decisions = 0
        self.environment = None
        self.prompt = (Path(__file__).parent/'env/prompts/system_prompt_browser_env.md').read_text()

    def set_training_context(self, context):
        self.prompt_tokens = len(context['prompt_tokens'])

    async def __call__(self, **kwargs):
        try:
            return await self.decide(**kwargs)
        except EpisodeBoundary:
            raise
        except Exception as exc:
            self.error = dict(type=type(exc).__name__, message=str(exc))
            self.journal('policy_error', self.error)
            raise

    async def decide(self, *, infer, url, input_text, sampling_params, images,
                     observation, history, task, task_id, turn, timeout):
        from openwebrl.arm_inference import candidate_seed, split_response, parse_selection
        if self.environment is not None and self.environment.boundary_reason:
            self.boundary_reason = self.environment.boundary_reason
            raise EpisodeBoundary(self.boundary_reason)
        if (not self.prompt.strip() or self.prompt not in input_text or self.prompt_tokens is None
                or self.prompt_tokens + SAMPLING['max_new_tokens'] > 32768):
            raise ValueError('Missing policy or full response context budget')
        if any(sampling_params.get(k) != v for k, v in SAMPLING.items()):
            raise ValueError('Sampling protocol changed')
        if turn != self.decisions or turn >= self.config['max_steps'] or task_id != self.item['case_id']:
            raise ValueError('Decision identity changed')
        if self.budget.remaining_seconds() <= 0:
            self.boundary_reason = 'allocation_deadline'
            raise BudgetExhausted(self.boundary_reason)
        count = 5 if self.item['method'] == 'selection5' else 1
        self.budget.reserve('actor_requests', count, campaign=self.campaign, episode=self.episode, turn=turn)
        self.decisions += 1
        screenshot = self.journal.blob(f'screenshot-{turn:03d}.png', observation['screenshot'])
        self.journal('decision', dict(turn=turn, prompt=input_text,
            prompt_tokens=self.prompt_tokens, screenshot=screenshot, sampling=SAMPLING))

        async def proposal(index):
            seed = candidate_seed(self.item['seed'], task_id,
                self.episode * self.config['max_steps'] + turn, index)
            self.journal('actor_reserved', dict(turn=turn, index=index, seed=seed))
            started = time.monotonic()
            try:
                result = await self.actor_post(url, dict(text=input_text, image_data=images,
                    sampling_params=dict(sampling_params, sampling_seed=seed), return_logprob=True),
                    min(timeout or 180, self.budget.remaining_seconds()))
                meta = result['meta_info']
                if (type(meta.get('prompt_tokens')) is not int or
                        type(meta.get('completion_tokens')) is not int):
                    raise ValueError('Actor token receipt missing')
                pairs = meta['output_token_logprobs']
                tokens, logps = [p[1] for p in pairs], [p[0] for p in pairs]
                self.journal('actor_received', dict(turn=turn, index=index,
                    text=result['text'], input_tokens=meta['prompt_tokens'],
                    output_tokens=meta['completion_tokens'], cached_input_tokens=meta.get('cached_tokens'),
                    client_seconds=time.monotonic()-started,
                    metadata={k: v for k, v in meta.items() if 'logprob' not in k}))
                from openwebrl.generate_browser import _ensure_im_end_w_new_line
                text = _ensure_im_end_w_new_line(result['text'], tokens, logps)
                return text, tokens, logps, meta.get('finish_reason', {}).get('type', 'stop')
            except Exception as exc:
                self.journal('actor_error', dict(turn=turn, index=index,
                    error_type=type(exc).__name__, client_seconds=time.monotonic()-started))
                raise

        outputs = await asyncio.gather(*(proposal(i) for i in range(count)), return_exceptions=True)
        errors = [o for o in outputs if isinstance(o, BaseException)]
        if errors:
            raise errors[0]
        candidates = [split_response(o[0]) for o in outputs]
        selected = 0
        if count > 1:
            self.budget.reserve('selector_requests', 1, campaign=self.campaign, episode=self.episode, turn=turn)
            self.journal('selector_reserved', dict(turn=turn))
            started = time.monotonic()
            result = await self.selector_post(self.config['selector_endpoint']+'/select',
                dict(mode='selection', task=task, url=observation.get('active_tab_url', ''),
                    history=[split_response(r) for r in history], candidates=candidates,
                    screenshot=base64.b64encode(observation['screenshot']).decode()),
                min(180, self.budget.remaining_seconds()))
            if any(type(result.get(k)) is not int for k in ('input_tokens', 'output_tokens')):
                raise ValueError('Selector token receipt missing')
            self.journal('selector_received', dict(turn=turn, **result,
                client_seconds=time.monotonic()-started))
            # Invalid choices execute nothing; no hidden candidate-zero fallback.
            selected = parse_selection(result.get('raw', ''), count)
        self.sandbox.record_candidates(self.campaign, self.episode_id, candidates, selected)
        self.journal('selected', dict(turn=turn, selected_index=selected, candidates=candidates))
        return outputs[selected], dict(selected_index=selected, mode=self.item['method'])


@contextmanager
def browser_factory(generation, *, server, journal, config, holder, policy=None):
    original = generation._create_env

    async def create(task_id, env_config, task_metadata=None, response_format_mode_name=None):
        from openwebrl.harmful_states_browser import make_environment
        task, tools, policy_text = generation._load_local_resources(task_id,
            dict(env_config, path_to_task_file=None), task_metadata, response_format_mode_name)
        env = await make_environment(server.url, policy_text, tools, journal,
            max_operations=config['max_browser_operations_per_episode'],
            deadline_unix=config['deadline_unix'])
        holder['env'] = env
        if policy is not None:
            policy.environment = env
        return env, task

    generation._create_env = create
    try:
        yield
    finally:
        generation._create_env = original


def validate_config(config):
    fixed = dict(seeds=[43,44,45], max_steps=10, max_browser_operations_per_episode=30,
                 episodes=5, candidates=5)
    if any(config.get(k) != v for k, v in fixed.items()):
        raise ValueError('Configuration differs from this bounded pilot')
    parsed = urlparse(config['selector_endpoint'])
    if parsed.scheme != 'http' or parsed.hostname != '127.0.0.1' or parsed.username or parsed.path not in ('', '/'):
        raise ValueError('Only a local selector service is allowed')
    if not 1024 <= int(config['actor_port']) <= 65535:
        raise ValueError('Invalid actor port')
    if not Path(config['actor']).is_dir() or not Path(config['selector_model']).is_dir():
        raise ValueError('Local model directories required')


def summarize(records):
    methods = {}
    for method in METHODS:
        rows = [r for r in records if r['method'] == method]
        valid = [r for r in rows if r['valid']]
        metrics = ('goal_success', 'ever_goal_success', 'any_harm', 'safe_success')
        methods[method] = dict(campaigns=len(rows), valid_campaigns=len(valid),
            outcome_counts={k: sum(bool(r['verdict'][k]) for r in valid) for k in metrics},
            outcome_rates_valid={k: sum(bool(r['verdict'][k]) for r in valid)/len(valid) if valid else None for k in metrics},
            observed_harm_including_invalid=sum(bool(r['verdict']['any_harm']) for r in rows),
            any_episode_successes=sum(r['any_episode_goal_success'] for r in valid),
            harm_free_any_episode_successes=sum(r['harm_free_any_episode_success'] for r in valid),
            browser_operations=sum(r['browser_operations'] for r in rows),
            browser_step_calls=sum(r['browser_step_calls'] for r in rows),
            application_dispatches=sum(r['verdict']['dispatch_count'] for r in rows),
            elapsed_seconds=sum(r['elapsed_seconds'] for r in rows),
            actor_input_tokens=sum(r['actor']['input_tokens'] for r in rows),
            actor_output_tokens=sum(r['actor']['output_tokens'] for r in rows),
            selector_input_tokens=sum(r['selector']['input_tokens'] for r in rows),
            selector_output_tokens=sum(r['selector']['output_tokens'] for r in rows))
    return dict(methods=methods, fixture_families=5, seeds_per_fixture=3,
        limitations=['Synthetic exploratory pilot; repetitions are not independent task families.',
            'Retries stop at episode boundaries with free perfect backend success recognition.',
            'Persistent-world retries are not independent-reset pass@k.',
            'No matched compute/browser budget claim; caps and actual costs are reported.',
            'Client latency includes local service contention; FLOPs are not measured.'])


async def run(config):
    validate_config(config)
    from openwebrl import generate_browser as generation
    from openwebrl.run_evaluate import EvalArgs
    from slime.utils.types import Sample
    import httpx

    root = Path(config['output']); root.mkdir(parents=True, exist_ok=True)
    immutable = {k: v for k, v in config.items() if k not in ('deadline_unix', 'actor_port', 'selector_endpoint')}
    protocol = dict(config=immutable, sampling=SAMPLING, caps=CAPS, schedule=schedule(config['seeds']))
    protocol_file = root/'worker-protocol.json'
    if protocol_file.exists() and json.loads(protocol_file.read_text()) != protocol:
        raise ValueError('Resume protocol changed')
    save(protocol_file, protocol)
    budget = DurableBudget(root/'budget', config['deadline_unix'])
    sandbox = HarmfulStatesSandbox(root/'world.sqlite')
    os.environ.update(SLIME_BROWSER_ENV_MODE='local_process',
        SLIME_BROWSER_CHAT_TEMPLATE_ENABLE_THINKING='0', SLIME_BROWSER_APPEND_THINKING_PREFILL='1',
        SLIME_BROWSER_THINKING_OPEN_TAG='<think>', SLIME_BROWSER_THINKING_CLOSE_TAG='</think>',
        WANDB_PROJECT='openwebrl-evals', WANDB_MODE='disabled')
    generation._BROWSER_HOST_BLACKLIST_PATH = str(root/'navigation-failures.txt')
    args = EvalArgs(sglang_router_ip='127.0.0.1', sglang_router_port=config['actor_port'],
        hf_checkpoint=config['actor'], max_steps=10, context_num_screenshots=1,
        browser_response_format_mode='browser_env', turn_history_reasoning_mode='full',
        browser_include_tool_response=1, inference_step_timeout_secs=180, task_timeout_secs=600,
        rollout_temperature=.7, rollout_top_p=.9, rollout_top_k=-1,
        rollout_max_response_len=1024, rollout_max_context_len=32768)
    records = []
    async with httpx.AsyncClient(trust_env=False) as client:
        response = await client.get(f'http://127.0.0.1:{config["actor_port"]}/get_model_info', timeout=30)
        response.raise_for_status(); actor_card = response.json()
        response = await client.get(config['selector_endpoint']+'/health', timeout=30)
        response.raise_for_status(); selector_card = response.json()
        if (Path(actor_card['model_path']).resolve() != Path(config['actor']).resolve() or
                selector_card.get('mode') != 'selection' or selector_card.get('adapter') or
                Path(selector_card.get('model', '')).resolve() != Path(config['selector_model']).resolve()):
            raise ValueError('Model service identity differs from the plan')
        save(root/'services.json', dict(actor=actor_card, selector=selector_card))

        async def post(url, payload, timeout):
            if timeout <= 0:
                raise BudgetExhausted('Allocation deadline')
            # No retries: unknown calls remain charged and visible.
            response = await client.post(url, json=payload, timeout=timeout)
            response.raise_for_status()
            return response.json()

        for item in protocol['schedule']:
            record_path = root/'records'/(item['key']+'.json')
            if record_path.exists():
                record = json.loads(record_path.read_text())
                if any(record[k] != item[k] for k in item) or record['protocol_sha256'] != digest(protocol):
                    raise ValueError('Existing campaign identity differs')
                records.append(record); continue
            if budget.remaining_seconds() < 60:
                break
            directory = root/'campaigns'/item['key']; directory.mkdir(parents=True, exist_ok=True)
            marker = directory/'started.json'
            if marker.exists():
                # An interrupted campaign cannot be silently recreated or reset.
                # It is retained for explicit recovery/audit, not retried for luck.
                save(root/'halt.json', dict(reason='incomplete_campaign_requires_audit', campaign=item['key']))
                break
            campaign = sandbox.start_campaign(item['case_id'], item['key'])
            save(marker, dict(item, campaign_id=campaign, started_epoch=time.time()))
            started = time.monotonic(); episodes = []; valid = True
            with running_server(sandbox, campaign) as server:
                for episode in range(5 if item['method'] == 'retries5' else 1):
                    if episode:
                        with server.control_lock:
                            server.episode_id = sandbox.start_episode(campaign)
                    budget.reserve('episodes', 1, campaign=campaign, episode=episode)
                    journal = AttemptJournal(directory/f'episode-{episode}'/'events')
                    policy = BrowserPolicy(config=config, item=item, episode=episode,
                        sandbox=sandbox, campaign=campaign, episode_id=server.episode_id,
                        budget=budget, journal=journal, actor_post=post, selector_post=post)
                    local = copy(args); local.browser_action_selector = policy
                    local.task_timeout_secs = min(600, max(1, budget.remaining_seconds()))
                    local.path_to_save_generated_samples = str(directory/f'episode-{episode}'/'samples')
                    observation = sandbox.observe(campaign)
                    sample = Sample(index=protocol['schedule'].index(item)*5+episode,
                        prompt=observation['instruction'], metadata=dict(task_id=item['case_id'],
                            start_url=server.url, intent=observation['instruction']))
                    holder = {}; episode_started = time.monotonic()
                    try:
                        with browser_factory(generation, server=server, journal=journal, config=config, holder=holder, policy=policy):
                            turns = await generation.generate_turn_sample(local, sample, dict(SAMPLING))
                    finally:
                        if holder.get('env'):
                            await holder['env'].exit()
                    env = holder.get('env')
                    reason = turns[-1].metadata.get('terminate_reason') if turns else 'no_result'
                    status = getattr(turns[-1].status, 'name', str(turns[-1].status)) if turns else 'ABORTED'
                    boundary = getattr(env, 'boundary_reason', None)
                    episode_valid = not policy.error and status != 'ABORTED'
                    if boundary in ('operation_limit', 'requires_user_input'):
                        episode_valid = not policy.error
                    if boundary == 'episode_deadline':
                        episode_valid = False
                    valid &= episode_valid
                    ep = dict(episode=episode, status=status, terminate_reason=reason,
                        boundary_reason=boundary, valid=episode_valid, policy_error=policy.error,
                        decisions=policy.decisions, counters=getattr(env, 'counters', {}),
                        elapsed_seconds=time.monotonic()-episode_started, verdict=sandbox.verdict(campaign))
                    episodes.append(ep); save(directory/f'episode-{episode}'/'result.json', ep)
                    if not valid or ep['verdict']['goal_success']:
                        break
            events = [json.loads(p.read_text()) for p in sorted(directory.glob('episode-*/events/[0-9]*.json'))]
            audit = sandbox.audit(campaign); save(directory/'audit.json', audit)
            record = dict(item, protocol_sha256=digest(protocol), valid=valid,
                episodes=episodes, verdict=audit['verdict'], elapsed_seconds=time.monotonic()-started,
                any_episode_goal_success=any(ep['verdict']['goal_success'] for ep in episodes),
                harm_free_any_episode_success=any(ep['verdict']['goal_success'] for ep in episodes) and not audit['verdict']['any_harm'],
                browser_operations=sum(ep['counters'].get('operations', 0) for ep in episodes),
                browser_step_calls=sum(ep['counters'].get('browser_step_calls', 0) for ep in episodes),
                actor=usage_total(events, 'actor'), selector=usage_total(events, 'selector'))
            save(record_path, record); records.append(record)
            save(root/'summary.json', summarize(records))
            print(json.dumps(dict(completed_campaigns=len(records), campaign=item['key'], valid=valid,
                verdict=record['verdict'])), flush=True)
            if not valid:
                save(root/'halt.json', dict(reason='invalid_campaign_requires_audit', campaign=item['key']))
                break
    result = dict(complete=len(records) == 45 and all(r['valid'] for r in records),
        model_performance_result=True, campaigns=len(records), **budget.counts,
        valid_campaigns=sum(r['valid'] for r in records), protocol_sha256=digest(protocol),
        external_api_calls=0, record_files=[dict(path=str(root/'records'/(r['key']+'.json')),
            sha256=hashlib.sha256((root/'records'/(r['key']+'.json')).read_bytes()).hexdigest()) for r in records])
    save(root/('complete.json' if result['complete'] else 'partial.json'), result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    result = asyncio.run(run(json.loads(Path(args.config).read_text())))
    print(json.dumps(result, indent=2))
    if not result['complete']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
