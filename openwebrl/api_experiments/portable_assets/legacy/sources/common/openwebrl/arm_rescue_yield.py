"""Frozen-policy rescue-yield measurement; no training objective or updates."""
import asyncio
from collections import defaultdict
from copy import copy, deepcopy
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import random
import time


def stable_seed(*parts):
    text = json.dumps(parts, separators=(',', ':'), ensure_ascii=False)
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], 'big') & 0x7fffffff


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.partial')
    with temp.open('w') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n'); handle.flush(); os.fsync(handle.fileno())
    temp.replace(path)


def eligible_tasks(records, task_ids, policy_id):
    grouped = defaultdict(list)
    for record in records:
        if record['policy_id'] != policy_id or record['judge_id'] != 'gpt-4.1/action_history':
            raise ValueError('Screening actor/judge identity changed')
        grouped[record['task_id']].append(record)
    if set(grouped) != set(task_ids):
        raise ValueError('Screening did not finish the exact cohort')
    eligible = []
    for task_id in task_ids:
        rows = grouped[task_id]
        if len(rows) != 5 or {r['attempt'] for r in rows} != set(range(5)):
            raise ValueError('Screening requires five distinct attempts per task')
        if all(r['valid'] and r['reward'] == 0.0 for r in rows):
            eligible.append(task_id)
    return eligible


def summarize_retries(records, selected):
    result = dict(selected_tasks=len(selected), by_mode={}, paired={})
    groups = defaultdict(dict)
    for r in records:
        if r['task_id'] not in selected or r['mode'] in groups[r['task_id']]:
            raise ValueError('Unexpected or duplicate retry')
        groups[r['task_id']][r['mode']] = r
    modes = ['arm', *[f'actor{i}' for i in range(5)]]
    if set(groups) != set(selected) or any(set(rows) != set(modes) for rows in groups.values()):
        raise ValueError('Every selected task needs ARM and five independent actor retries')
    success = lambda r: bool(r['valid'] and r['reward'] == 1.0)
    for mode in modes:
        rows = [groups[t][mode] for t in selected]
        n, valid = len(rows), sum(r['valid'] for r in rows)
        wins = sum(success(r) for r in rows)
        result['by_mode'][mode] = dict(attempts=n, valid=valid, invalid=n-valid, successes=wins,
            success_rate=wins/n if n else None, valid_only_rate=wins/valid if valid else None,
            actor_output_tokens=sum(r['actor_output_tokens'] for r in rows),
            actor_requests=sum(r['actor_requests'] for r in rows),
            trajectory_seconds=sum(r['elapsed_seconds'] for r in rows),
            selector_calls=sum(r['selector_calls'] for r in rows),
            selector_fallback_turns=sum(r['selector_fallback_turns'] for r in rows))
    result['paired'] = dict(
        arm_only_vs_actor1=sum(success(groups[t]['arm']) and not success(groups[t]['actor0']) for t in selected),
        actor1_only_vs_arm=sum(success(groups[t]['actor0']) and not success(groups[t]['arm']) for t in selected),
        both_valid_arm_actor1=sum(groups[t]['arm']['valid'] and groups[t]['actor0']['valid'] for t in selected),
        actor5_any_success=sum(any(success(groups[t][f'actor{i}']) for i in range(5)) for t in selected),
        arm_only_vs_actor5=sum(success(groups[t]['arm']) and not any(success(groups[t][f'actor{i}']) for i in range(5)) for t in selected))
    result['actor5_output_tokens'] = sum(result['by_mode'][f'actor{i}']['actor_output_tokens'] for i in range(5))
    result['limitations'] = ('Small training-task feasibility panel, not held-out performance. Five retries are a '
        'generation-budget reference, not exact equal compute. Trajectory-seconds overlap under concurrency '
        'and are not per-mode GPU-hours. Judge success does not certify every step.')
    return result


class MeteredSelector:
    def __init__(self, selector):
        self.selector = selector
        self.actor_requests = self.actor_output_tokens = self.selector_calls = self.fallback_turns = 0

    async def __call__(self, *, infer, **kwargs):
        async def metered(*a, **k):
            self.actor_requests += 1
            output = await infer(*a, **k)
            self.actor_output_tokens += len(output[1])
            return output
        self.selector_calls += int(self.selector.mode == 'selection')
        output, metadata = await self.selector(infer=metered, **kwargs)
        self.fallback_turns += int(bool(metadata.get('fallback')))
        return output, metadata


async def trajectory(args, sample, params, mode, attempt, phase, config):
    from openwebrl.arm_inference import ActionSelector
    from openwebrl.generate_browser import generate_turn_sample
    from openwebrl.reward_browser import reward_func
    from openwebrl.eval_monitor import save_task
    from slime.utils.rollout_transport import file_back_completed_group
    from slime.utils.trajectory_metrics import trajectory_metrics
    from slime.utils.types import Sample

    task_id = str(sample.metadata['task_id'])
    key = hashlib.sha256(f'{task_id}/{phase}/{mode}/{attempt}'.encode()).hexdigest()
    output = Path(config['output'])
    artifact = output/'trajectories'/key
    seed = stable_seed(config['seed'], config['policy_id'], task_id, phase, mode, attempt)
    selector = MeteredSelector(ActionSelector('selection' if mode == 'arm' else 'baseline',
        config['selector_endpoint'], artifact/'selections', seed=seed, candidates=5,
        timeout=120, candidate_representation='full'))
    local, initial = copy(args), deepcopy(sample)
    initial.index = sample.index if phase == 'screen' else (200000 if phase == 'smoke' else 100000) + sample.index*10 + attempt
    initial.session_id = None
    local.max_steps = config.get('max_steps', 15)
    local.inference_step_timeout_secs = config.get('inference_timeout', 180)
    local.judge_timeout_secs = 120
    local.browser_action_selector = selector
    turns, error = [], None
    started = time.monotonic()
    try:
        turns = await generate_turn_sample(local, initial, dict(params, sampling_seed=seed))
        if turns and not any(t.remove_sample or t.status == Sample.Status.ABORTED for t in turns):
            rewards = await reward_func(local, turns)
            for t, reward in zip(turns, rewards, strict=True):
                t.reward = reward
    except Exception as exc:
        error = type(exc).__name__
        for t in turns:
            t.remove_sample = True
    finally:
        # Direct destination arguments avoid per-attempt environment races.
        await asyncio.to_thread(file_back_completed_group, turns)
        await asyncio.to_thread(save_task, local, artifact, task_id, turns, error)
    metrics = trajectory_metrics(local, [turns])
    valid = error is None and metrics['valid_trajectories'] == 1
    terminal = max(turns, key=lambda t: t.metadata.get('turn_index', 0)) if turns else None
    reward = terminal.get_reward_value(local) if valid else None
    record = dict(task_id=task_id, phase=phase, mode=mode, attempt=attempt,
        policy_id=config['policy_id'], judge_id='gpt-4.1/action_history', seed=seed,
        valid=valid, reward=reward, error_type=error, steps=len(turns), artifact=str(artifact),
        actor_requests=selector.actor_requests, actor_output_tokens=selector.actor_output_tokens,
        selector_calls=selector.selector_calls, selector_fallback_turns=selector.fallback_turns,
        elapsed_seconds=time.monotonic()-started)
    write_json(output/f'{phase}_records'/f'{key}.json', record)
    print('[RescueYield] '+json.dumps(record), flush=True)
    return turns


async def generate(args, sample, sampling_params, evaluation=False):
    if not evaluation:
        raise ValueError('Rescue yield is collection-only')
    config = json.loads(Path(os.environ['OPENWEBRL_RESCUE_YIELD_CONFIG']).read_text())
    if args.judge_api_model != 'gpt-4.1' or args.judge_prompt_variant != 'action_history':
        raise ValueError('Preserve the native training judge')
    phase = sample.metadata.get('rescue_phase', 'screen')
    if phase == 'screen':
        return await trajectory(args, sample, sampling_params, 'actor', sample.index % 5, phase, config)
    if phase not in ('retry', 'smoke'):
        raise ValueError('Unknown pilot phase')
    order = ['arm', *[f'actor{i}' for i in range(1 if phase == 'smoke' else 5)]]
    random.Random(stable_seed(config['seed'], sample.metadata['task_id'], 'retry-order')).shuffle(order)
    turns = []
    for mode in order:
        attempt = 5 if mode == 'arm' else int(mode[-1])
        turns.extend(await trajectory(args, sample, sampling_params, mode, attempt, phase, config))
    return turns


async def run_pilot(args, rollout_id):
    from slime.rollout.sglang_rollout import eval_rollout_single_dataset
    from slime.rollout.base_types import RolloutFnEvalOutput
    config = json.loads(Path(os.environ['OPENWEBRL_RESCUE_YIELD_CONFIG']).read_text())
    output = Path(config['output'])
    rows = [json.loads(line) for line in Path(config['tasks']).read_text().splitlines()]
    if len(args.eval_datasets) != 1:
        raise ValueError('Exactly one prespecified screening cohort is required')
    screen = args.eval_datasets[0]
    if screen.n_samples_per_eval_prompt != 5:
        raise ValueError('Require exactly five fresh actor screening trajectories')
    smoke_file = output/'smoke-tasks.jsonl'
    smoke_file.write_text(''.join(json.dumps(row)+'\n' for row in rows[:3]))
    smoke = replace(screen, name='arm-rescue-smoke', path=str(smoke_file),
        n_samples_per_eval_prompt=1, metadata_overrides=dict(screen.metadata_overrides, rescue_phase='smoke'))
    result = await eval_rollout_single_dataset(args, rollout_id, smoke)
    smoke_records = [json.loads(p.read_text()) for p in (output/'smoke_records').glob('*.json')]
    smoke_by_task = defaultdict(dict)
    for r in smoke_records:
        smoke_by_task[r['task_id']][r['mode']] = r
    usable = [task for task, attempts in smoke_by_task.items()
        if set(attempts) == {'actor0', 'arm'} and all(r['valid'] for r in attempts.values())
        and attempts['arm']['selector_calls'] > 0 and attempts['arm']['selector_fallback_turns'] == 0]
    write_json(output/'smoke-gate.json', dict(passed=bool(usable), usable_tasks=usable,
        excluded_from_yield=True, attempts=len(smoke_records)))
    if not usable:
        raise RuntimeError('No valid actor/ARM smoke pair; stop before scaling to the 64-task screen')
    start = time.monotonic()
    result.update(await eval_rollout_single_dataset(args, rollout_id, screen))
    screening_seconds = time.monotonic()-start
    records = [json.loads(p.read_text()) for p in (output/'screen_records').glob('*.json')]
    ids = [str(row['metadata']['task_id']) for row in rows]
    eligible = eligible_tasks(records, ids, config['policy_id'])
    selected = sorted(eligible, key=lambda t: stable_seed(config['seed'], t, 'panel'))[:config['max_rescue_tasks']]
    write_json(output/'eligibility.json', dict(screened_tasks=len(ids), eligible=eligible, selected=selected,
        selection='Seeded hash order after all screens finish; independent of completion speed and retry outcomes'))
    selected_file = output/'selected-tasks.jsonl'
    selected_file.write_text(''.join(json.dumps(row)+'\n' for row in rows if str(row['metadata']['task_id']) in selected))
    retry_seconds = 0.0
    if selected:
        retry = replace(screen, name='arm-rescue-retries', path=str(selected_file),
            n_samples_per_eval_prompt=1, metadata_overrides=dict(screen.metadata_overrides, rescue_phase='retry'))
        start = time.monotonic()
        result.update(await eval_rollout_single_dataset(args, rollout_id, retry))
        retry_seconds = time.monotonic()-start
    records = [json.loads(p.read_text()) for p in (output/'retry_records').glob('*.json')]
    summary = summarize_retries(records, selected)
    summary.update(screened_tasks=len(ids), screen_trajectories=len(ids)*5, eligible_tasks=len(eligible),
        screening_seconds=screening_seconds, retry_phase_seconds=retry_seconds,
        complete=True, optimizer_updates=0, policy_id=config['policy_id'])
    write_json(output/'yield-summary.json', summary)
    metrics = {'rescue_yield/'+k: v for k, v in summary.items() if type(v) in (int, float)}
    for mode, values in summary['by_mode'].items():
        metrics.update({f'rescue_yield/{mode}/{k}': v for k, v in values.items() if type(v) in (int, float)})
    return RolloutFnEvalOutput(data=result, metrics=metrics)


def generate_rollout(args, rollout_id, data_source, evaluation=False):
    if not evaluation or args.num_rollout != 0:
        raise ValueError('Rescue-yield pilot must request zero optimizer rollouts')
    from slime.utils.async_utils import run
    return run(run_pilot(args, rollout_id))
