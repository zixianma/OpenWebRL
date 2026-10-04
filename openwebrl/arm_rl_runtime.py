"""Collection-time ARM hooks for isolated mechanics pilots and their manifests."""
import asyncio
from copy import copy, deepcopy
import hashlib
import json
import os
from pathlib import Path
import time

from openwebrl.arm_rl import action_token_mask, local_advantages, rescue_eligibility, stable_seed
from openwebrl.arm_rl_collection import LocalPreferenceSelector, rescue_selector

_ROUND = None


def begin_round(args, rollout_id):
    global _ROUND
    config = json.loads(Path(os.environ['OPENWEBRL_ARM_CONFIG']).read_text())
    if config['track'] not in ('rescue', 'local_preference'):
        raise ValueError('Unknown ARM track')
    if getattr(args, 'custom_generate_function_path', None) != 'openwebrl.arm_rl_runtime.generate':
        raise ValueError('ARM collection requires its versioned generator wrapper')
    if args.judge_api_model != 'gpt-4.1' or args.judge_prompt_variant != 'action_history':
        raise ValueError('ARM collection must use the unchanged training judge')
    path = Path(os.environ['OPENWEBRL_ARM_AUX_MANIFEST'].format(rollout_id=rollout_id))
    path.parent.mkdir(parents=True, exist_ok=False)
    policy = os.environ['OPENWEBRL_ARM_POLICY_ID'] + f':round{rollout_id}'
    evaluation = [json.loads(line) for line in Path(config['evaluation_tasks']).read_text().splitlines() if line.strip()]
    _ROUND = dict(config=config, rollout_id=rollout_id, policy_id=policy, manifest=path,
        originals={}, local_files=[], audits=[],
        excluded_task_ids={str(x.get('task_id', x.get('id'))) for x in evaluation},
        excluded_intents={' '.join(str(x.get('confirmed_task', x.get('intent', ''))).casefold().split()) for x in evaluation})


def _persist_local(record):
    import torch
    from tokenizers import Tokenizer
    from scripts.audit_arm_preference_pairs import canonical, parse_action, schemas
    context = record['training_context']
    if context is None:
        raise ValueError('Missing exact image-expanded actor prefix')
    tokenizer = Tokenizer.from_file(str(Path(_ROUND['config']['tokenizer']) / 'tokenizer.json'))
    tools = schemas(record['prompt'])
    actions, masks = [], []
    for text, ids, old_logps, finish in record['outputs']:
        if finish != 'stop':
            raise ValueError('Truncated candidates are not local labels')
        parsed, spans = parse_action(text, tools)
        actions.append(canonical(parsed))
        masks.append(action_token_mask(text, ids, tokenizer, spans))
    advantages = local_advantages(actions, record['selected_index'])
    if advantages is None:
        return
    key = str(stable_seed(record['policy_id'], record['trajectory_id'], record['turn']))
    destination = _ROUND['manifest'].parent / f'local-{key}.pt'
    if destination.exists():
        raise ValueError('Duplicate local state')
    rows, metadata = [], []
    for i, output in enumerate(record['outputs']):
        rows.append(dict(tokens=torch.tensor(context['prompt_tokens'] + list(output[1]), dtype=torch.long),
            loss_mask=torch.tensor(masks[i], dtype=torch.int), response_length=len(output[1]),
            multimodal_train_inputs=context['multimodal_train_inputs'], terminal_reward=None))
        metadata.append(dict(source='arm_candidate', state_id=key,
            parent_sample_index=context['parent_sample_index'], advantage=advantages[i],
            task_id=record['task_id'], candidate_index=i))
    torch.save(rows, destination)
    _ROUND['local_files'].append(dict(path=destination, records=metadata, state_id=key,
        parent_sample_index=context['parent_sample_index']))


async def generate(args, sample, sampling_params, evaluation=False):
    from openwebrl.generate_browser import generate_turn_sample
    if evaluation:
        raise ValueError('Use the separate benchmark generator for evaluation')
    if _ROUND is None:
        raise ValueError('ARM round was not initialized')
    _ROUND['originals'][str(sample.index)] = deepcopy(sample)
    local_args = copy(args)
    selector = None
    task_id = str((sample.metadata or {}).get('task_id'))
    intent = ' '.join(str((sample.metadata or {}).get('intent', '')).casefold().split())
    excluded = task_id in _ROUND['excluded_task_ids'] or intent in _ROUND['excluded_intents']
    if _ROUND['config']['track'] == 'local_preference' and not excluded:
        async def sink(record):
            writer = asyncio.create_task(asyncio.to_thread(_persist_local, record))
            try:
                await asyncio.shield(writer)
            except asyncio.CancelledError:
                await writer
                raise
        selector = LocalPreferenceSelector(policy_id=_ROUND['policy_id'], trajectory_id=str(sample.index),
            endpoint=_ROUND['config']['selector_endpoint'], sink=sink,
            scored_fraction=_ROUND['config'].get('scored_fraction', .2),
            max_pending=_ROUND['config'].get('max_pending_per_trajectory', 2))
        local_args.browser_action_selector = selector
    try:
        turns = await generate_turn_sample(local_args, sample, sampling_params)
    except BaseException:
        if selector:
            await selector.cancel()
        raise
    if selector:
        await selector.drain()
        _ROUND['audits'].extend(selector.audit)
    for turn in turns:
        turn.metadata['arm_policy_id'] = _ROUND['policy_id']
    return turns


def _flatten(groups):
    return [s for g in groups for trajectory in g for s in (trajectory if isinstance(trajectory, list) else [trajectory])]


def _failure_group(group):
    trajectories = [x if isinstance(x, list) else [x] for x in group]
    end = trajectories[0][-1]
    return dict(task_id=str(end.metadata.get('task_id')), intent=end.metadata.get('intent', ''),
        policy_id=_ROUND['policy_id'], judge_id='gpt-4.1/action_history', trajectories=[dict(
            trajectory_id=str(turns[-1].metadata.get('trajectory_id')),
            policy_id=turns[-1].metadata.get('arm_policy_id'), judge_id='gpt-4.1/action_history',
            valid=not any(t.remove_sample for t in turns) and turns[-1].reward is not None,
            reward=turns[-1].reward) for turns in trajectories])


async def _rescue(args, group, sampling_params):
    from openwebrl.generate_browser import generate_turn_sample
    from openwebrl.reward_browser import reward_func
    failure = _failure_group(group)
    first = group[0][0] if isinstance(group[0], list) else group[0]
    initial = _ROUND['originals'][str(first.metadata['trajectory_id'])]
    identity = (_ROUND['policy_id'], failure['task_id'])
    modes = ['guided', 'ordinary']
    if stable_seed(*identity, 'order') % 2:
        modes.reverse()
    demos = []
    for mode in modes:
        started = time.monotonic()
        sample = deepcopy(initial)
        sample.index = 10**12 + stable_seed(*identity, mode)
        config = copy(args)
        config.browser_action_selector = None
        if mode == 'guided':
            config.browser_action_selector = rescue_selector(policy_id=_ROUND['policy_id'],
                trajectory_id=str(sample.index), endpoint=_ROUND['config']['selector_endpoint'],
                output=_ROUND['manifest'].parent / 'rescue-traces')
        turns = await generate_turn_sample(config, sample, dict(sampling_params,
            sampling_seed=stable_seed(*identity, mode, 'actor')))
        if turns and all(not t.remove_sample for t in turns):
            values = await reward_func(config, turns)
            for turn, value in zip(turns, values, strict=True): turn.reward = value
        valid = bool(turns) and all(not t.remove_sample for t in turns) and turns[-1].reward is not None
        success = valid and turns[-1].reward == 1.
        admitted = success and mode == 'guided' and all(
            not t.metadata.get('arm', {}).get('fallback') and
            t.metadata.get('arm', {}).get('selected_index') is not None and
            'step_tool_responses' in t.metadata for t in turns)
        # Keep ordinary retry trajectories too, in a separate control artifact.
        # Store current-turn tensors and safe metadata, not repeated full-image
        # histories or browser storage-state credentials.
        import torch
        artifact = _ROUND['manifest'].parent / f'retry-{stable_seed(*identity)}-{mode}.pt'
        saved = [dict(prompt=t.prompt, response=t.response, tokens=t.tokens,
            response_length=t.response_length, loss_mask=t.loss_mask, reward=t.reward,
            rollout_log_probs=t.rollout_log_probs, multimodal_train_inputs=t.multimodal_train_inputs,
            metadata={k:t.metadata.get(k) for k in ('task_id','turn_index','terminate_reason',
                'step_tool_responses','arm','reward')}) for t in turns]
        await asyncio.to_thread(torch.save, saved, artifact)
        _ROUND['audits'].append(dict(task_id=failure['task_id'], mode=mode, valid=valid,
            reward=turns[-1].reward if valid else None, demo_admitted=admitted,
            steps=len(turns), policy_id=_ROUND['policy_id'], artifact=artifact.name,
            elapsed_seconds=time.monotonic()-started,
            executed_response_tokens=sum(t.response_length for t in turns)))
        if admitted:
            demos.extend(turns)
    return demos


async def finish_round(args, rollout_id, all_groups, accepted_groups, sampling_params):
    import torch
    if _ROUND is None or _ROUND['rollout_id'] != rollout_id:
        raise ValueError('ARM round mismatch')
    config = _ROUND['config']
    accepted = _flatten(accepted_groups)
    accepted_ids = {s.index for s in accepted if not s.remove_sample}
    rows, records = [], []
    if config['track'] == 'local_preference':
        files = sorted(_ROUND['local_files'], key=lambda x: x['state_id'])
        files = [f for f in files if f['parent_sample_index'] in accepted_ids][:config.get('max_local_states', 32)]
        for entry in files:
            rows.extend(torch.load(entry['path'], map_location='cpu', weights_only=True))
            records.extend(entry['records'])
    else:
        groups = []
        for group in all_groups:
            candidate = _failure_group(group)
            reason = rescue_eligibility(candidate, policy_id=_ROUND['policy_id'],
                judge_id='gpt-4.1/action_history', excluded_task_ids=_ROUND['excluded_task_ids'],
                excluded_intents=_ROUND['excluded_intents'])
            if reason is None: groups.append((candidate['task_id'], group))
        demos = []
        for task_id, group in sorted(groups)[:config.get('rescue_task_cap', 2)]:
            demos.extend(await _rescue(args, group, sampling_params))
        # Deterministic task-first sampling, then cycle through each task's turns.
        by_task = {}
        for turn in demos:
            by_task.setdefault(str(turn.metadata['task_id']), []).append(turn)
        tasks = sorted(by_task, key=lambda t: stable_seed(_ROUND['policy_id'], t))
        positions = {task: 0 for task in tasks}
        # This is a cap for the mechanics pilot, not an assertion of epoch trimming.
        windows = len(accepted_ids) // args.global_batch_size
        for window in range(windows):
            for draw in range(min(32, len(demos))):
                task = tasks[(window * 32 + draw) % len(tasks)]
                turn = by_task[task][positions[task] % len(by_task[task])]; positions[task] += 1
                rows.append(dict(tokens=torch.tensor(turn.tokens, dtype=torch.long),
                    loss_mask=torch.tensor(turn.loss_mask, dtype=torch.int), response_length=turn.response_length,
                    multimodal_train_inputs=turn.multimodal_train_inputs, terminal_reward=None))
                records.append(dict(source='rescue_demo', task_id=task, optimizer_window=window))
    destination = _ROUND['manifest'].parent / 'auxiliary.pt'
    torch.save(rows, destination)
    sha = hashlib.sha256()
    with destination.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): sha.update(block)
    manifest = dict(schema_version=1, rollout_id=rollout_id, policy_id=_ROUND['policy_id'],
        training_judge='gpt-4.1/action_history', track=config['track'],
        eta=config.get('eta', 0.), **{'lambda': config.get('lambda', 0.)},
        tensor_file=destination.name, tensor_sha256=sha.hexdigest(), records=records,
        accepted_outcome_samples=len(accepted_ids), audit=_ROUND['audits'])
    _ROUND['manifest'].write_text(json.dumps(manifest, indent=2)+'\n')
