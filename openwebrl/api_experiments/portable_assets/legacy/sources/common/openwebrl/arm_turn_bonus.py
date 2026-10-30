"""Calibration and gated first-batch training with an executed-turn ARM bonus.

Only ordinary candidate zero reaches the browser. The native reward normalizer
calls post_process_rewards after outcome grouping; no actor/loss backend changes.
"""
import asyncio
import base64
from collections import Counter
from copy import copy, deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

from openwebrl.arm_inference import parse_selection, request_selection_result, split_response
from scripts.audit_arm_preference_pairs import canonical, parse_action, schemas

_STATE = None


def task_identity(record):
    metadata = record.get('metadata') or {}
    task_id = record.get('task_id') or record.get('id') or metadata.get('task_id')
    intent = (record.get('confirmed_task') or record.get('intent') or record.get('task_name')
              or metadata.get('intent') or metadata.get('task') or '')
    return str(task_id) if task_id is not None else '', ' '.join(str(intent).casefold().split())


def exclusion_sets(records):
    identities = [task_identity(record) for record in records]
    return {i for i, _ in identities if i}, {t for _, t in identities if t}


def log_progress(current, force=False):
    now = time.monotonic()
    if not force and now-current.get('last_logged', 0) < 30:
        return
    records = current['records']
    admitted = [r for r in records if r.get('eligible')]
    sampled = sum(r.get('sampled', False) for r in records)
    payload = {'arm_collection/elapsed_seconds':now-current['started'],
        'arm_collection/recorded_turns':len(records), 'arm_collection/sampled_turns':sampled,
        'arm_collection/admitted_turns':len(admitted),
        'arm_collection/admitted_tasks':len({r['task_id'] for r in admitted}),
        'arm_collection/coverage_all_completed':len(admitted)/len(records) if records else 0.,
        'arm_collection/selector_requests':sum(r.get('selector_requests',0) for r in records),
        'arm_collection/candidate_tokens':sum(r.get('candidate_response_tokens',0) for r in records),
        'arm_collection/native_failure_sentinels':current.get('native_failure_sentinels',0),
        'arm_collection/excluded_trajectories':current.get('excluded_trajectories',0)}
    for reason, count in Counter(r.get('error_type',r.get('reason','pending')) for r in records).items():
        payload['arm_collection/label_status/'+reason] = count
    write_json(Path(current['config']['output'])/'live_metrics.json', payload)
    try:
        import wandb
        if wandb.run is not None:
            wandb.log(payload)
    except Exception as exc:
        print('[ARM telemetry unavailable] '+type(exc).__name__, flush=True)
    current['last_logged'] = now


def stable_seed(*parts):
    raw = json.dumps(parts, ensure_ascii=False, separators=(',', ':')).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:4], 'big') & 0x7fffffff


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def eligible_actions(prompt, outputs):
    if len(outputs) != 5:
        raise ValueError('candidate_count')
    tools = schemas(prompt)
    actions = []
    for text, ids, logps, finish in outputs:
        if finish != 'stop' or not ids:
            raise ValueError('truncated_or_empty')
        action, _ = parse_action(text, tools)
        actions.append(canonical(action))
    if len(set(actions)) != 5:
        raise ValueError('duplicate_action')
    return actions


class ShadowSelector:
    def __init__(self, config, trajectory_id, *, request=request_selection_result):
        self.config, self.trajectory_id = config, str(trajectory_id)
        self.request, self.pending, self.records = request, set(), {}
        self.context = None

    def set_training_context(self, context):
        # The executed Sample already owns exact tokens, logps and image tensors.
        self.context = dict(parent_sample_index=context['parent_sample_index'],
                            group_index=context['group_index'])

    async def __call__(self, *, infer, url, input_text, sampling_params, images,
                       observation, history, task, task_id, turn, timeout):
        started = time.monotonic()
        original = await infer(url, input_text, sampling_params, images, timeout_secs=timeout)
        identity = (self.config['seed'], self.config.get('sampling_policy_id',self.config['policy_id']),
                    str(task_id), self.trajectory_id, turn)
        record = dict(task_id=str(task_id), trajectory_id=self.trajectory_id, turn=turn,
            policy_id=self.config['policy_id'], executed_index=0, eligible=False,
            context=self.context, executed_response_tokens=len(original[1]),
            original_inference_seconds=time.monotonic()-started,
            candidate_requests=0, candidate_response_tokens=0, selector_requests=0)
        self.records[turn] = record
        overlap = ' '.join(str(task).casefold().split()) in self.config.get('excluded_intents', ())
        sampled = not overlap and stable_seed(*identity, 'score') / 2**31 < self.config['scored_fraction']
        record['sampled'] = sampled
        if sampled and len(self.pending) < self.config['max_pending_per_trajectory']:
            state = deepcopy(dict(prompt=input_text, params=sampling_params, images=images,
                observation=observation, history=history, task=task, original=original))
            worker = asyncio.create_task(self._label(infer, url, state, identity, record, timeout))
            self.pending.add(worker)
            worker.add_done_callback(self.pending.discard)
        else:
            record['reason'] = 'evaluation_overlap' if overlap else ('queue_full' if sampled else 'not_sampled')
        return original, dict(mode='executed_turn_shadow', executed_index=0,
            policy_id=self.config['policy_id'], trajectory_id=self.trajectory_id)

    async def _label(self, infer, url, state, identity, record, timeout):
        started, workers = time.monotonic(), []
        try:
            async with asyncio.timeout(self.config['label_timeout_seconds']):
                async def candidate(i):
                    record['candidate_requests'] += 1
                    output = await infer(url, state['prompt'],
                        dict(state['params'], sampling_seed=stable_seed(*identity, i)),
                        state['images'], timeout_secs=timeout)
                    record['candidate_response_tokens'] += len(output[1])
                    return output
                workers = [asyncio.create_task(candidate(i)) for i in range(1, 5)]
                outputs = [state['original'], *await asyncio.gather(*workers)]
                record['candidate_generation_seconds'] = time.monotonic()-started
                record['candidates'] = [dict(response=o[0], response_token_ids=list(o[1]),
                    finish_type=o[3]) for o in outputs]
                record['prompt'] = state['prompt']
                record['prompt_sha256'] = hashlib.sha256(state['prompt'].encode()).hexdigest()
                screenshot = state['observation']['screenshot']
                record['screenshot_sha256'] = hashlib.sha256(screenshot).hexdigest()
                record['actions'] = eligible_actions(state['prompt'], outputs)
                order = list(range(5))
                random.Random(stable_seed(*identity, 'permutation')).shuffle(order)
                record['permutation'] = order
                record['url'] = state['observation'].get('active_tab_url', '')
                record['history'] = state['history']
                payload = dict(mode='selection', task=state['task'], url=record['url'],
                    history=[split_response(r) for r in state['history']],
                    candidates=[split_response(outputs[i][0]) for i in order],
                    screenshot=base64.b64encode(screenshot).decode())
                label_start = time.monotonic()
                record['selector_requests'] = 1
                result = await self.request(self.config['selector_endpoint'], payload,
                    self.config['label_timeout_seconds'], None)
                record['selector_seconds'] = time.monotonic()-label_start
                record['raw_label'] = result.get('raw', '')
                record['selected_index'] = order[parse_selection(record['raw_label'], 5)]
                record.update(eligible=True, reason='admitted',
                    unit_bonus=float(record['selected_index'] == 0)-.2)
        except asyncio.CancelledError:
            record['reason'] = 'cancelled'
            raise
        except Exception as exc:
            record.update(reason='label_unavailable', error_type=type(exc).__name__, error=str(exc)[:160])
        finally:
            for worker in workers:
                if not worker.done():
                    worker.cancel()
            if workers:
                await asyncio.gather(*workers, return_exceptions=True)
            record['label_pipeline_seconds'] = time.monotonic()-started

    async def finish(self, cancelled=False):
        tasks = tuple(self.pending)
        if cancelled:
            for task in tasks:
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        if cancelled:
            for record in self.records.values():
                record.setdefault('reason', 'cancelled')


def state(args):
    global _STATE
    if _STATE is not None:
        return _STATE
    config = json.loads(Path(os.environ['OPENWEBRL_ARM_TURN_BONUS_CONFIG']).read_text())
    from openwebrl.arm_turn_bonus_runtime import validate_config
    validate_config(config)
    if args.judge_api_model != 'gpt-4.1' or args.judge_prompt_variant != 'action_history':
        raise ValueError('Keep the native training judge')
    if args.advantage_estimator != 'grpo' or not args.grpo_std_normalization:
        raise ValueError('Expected the native normalized GRPO recipe')
    if any(getattr(args,k,False) for k in ('normalize_advantages','use_opd','use_kl_loss','use_critic')):
        raise ValueError('Keep native GRPO without additional whitening, OPD, critic or KL loss')
    if getattr(args,'kl_coef',0) != 0 or getattr(args,'entropy_coef',0) != 0:
        raise ValueError('Keep native zero KL/entropy coefficients')
    evaluation = [json.loads(x) for x in Path(config['evaluation_tasks']).read_text().splitlines() if x.strip()]
    excluded_ids, excluded_intents = exclusion_sets(evaluation)
    _STATE = dict(config=config, started=time.monotonic(), records=[],
        excluded_ids=excluded_ids, excluded_intents=excluded_intents)
    log_progress(_STATE, force=True)
    return _STATE


async def generate(args, sample, sampling_params, evaluation=False):
    from openwebrl.generate_browser import generate_turn_sample
    if evaluation:
        raise ValueError('Calibration must not use evaluation tasks')
    current = state(args)
    meta = sample.metadata or {}
    task_id, intent = task_identity(meta)
    excluded = (bool(task_id) and task_id in current['excluded_ids']) or (bool(intent) and intent in current['excluded_intents'])
    local_args = copy(args)
    local_args.browser_action_selector = None
    if excluded:
        current['excluded_trajectories'] = current.get('excluded_trajectories',0)+1
        log_progress(current)
        turns = await generate_turn_sample(local_args, sample, sampling_params)
        for turn in turns:
            if turn.metadata is None:
                turn.metadata = {}
            turn.metadata['arm_calibration_excluded'] = True
        return turns
    selector = ShadowSelector(dict(current['config'], excluded_intents=list(current['excluded_intents'])), sample.index)
    local_args.browser_action_selector = selector
    cancelled = True
    try:
        turns = await generate_turn_sample(local_args, sample, sampling_params)
        await selector.finish()
        cancelled = False
        for turn in turns:
            metadata = turn.metadata or {}
            turn_index = metadata.get('turn_index')
            if turn_index is None:
                # Native initialization failures can return the original ABORTED
                # prompt even without remove_sample: a bare TimeoutError has no
                # text for the native removal heuristic to match. Preserve this
                # no-response sentinel; generate_and_rm skips ABORTED samples and
                # the native group filter drops their missing rewards.
                aborted_prompt = (
                    turn is sample
                    and getattr(getattr(turn, 'status', None), 'value', None) == 'aborted'
                    and getattr(turn, 'response_length', None) == 0
                    and not getattr(turn, 'response', '')
                )
                if turn.remove_sample or aborted_prompt:
                    current['native_failure_sentinels'] = current.get('native_failure_sentinels', 0) + 1
                    continue
                raise ValueError('Trainable browser Sample has no turn_index')
            record = selector.records.get(turn_index)
            if record is not None:
                if record['reason'] == 'evaluation_overlap':
                    turn.metadata['arm_calibration_excluded'] = True
                expected = record['context']['parent_sample_index']
                if expected != turn.index:
                    raise ValueError('ARM label attached to the wrong executed Sample')
                turn.metadata['arm_turn_bonus'] = {k: record[k] for k in (
                    'policy_id', 'eligible', 'sampled', 'reason', 'executed_index')}
                for key in ('unit_bonus', 'selected_index'):
                    if key in record:
                        turn.metadata['arm_turn_bonus'][key] = record[key]
                turn.metadata['arm_turn_bonus']['record_file'] = str(
                    Path(current['config']['output']) / 'labels' / f'{sample.index}.json')
        return turns
    finally:
        if cancelled:
            await selector.finish(cancelled=True)
        records = list(selector.records.values())
        current['records'].extend(records)
        # Store labels on shared durable storage; executed tensors live in recovery.
        write_json(Path(current['config']['output']) / 'labels' / f'{sample.index}.json',
                   dict(cancelled=cancelled, records=records))
        log_progress(current)


def unit_bonus(sample, policy_id):
    label = (sample.metadata or {}).get('arm_turn_bonus', {})
    if sample.remove_sample or not label.get('eligible'):
        return 0.
    if label.get('policy_id') != policy_id or label.get('executed_index') != 0:
        raise ValueError('Incorrect ARM policy/executed-candidate provenance')
    selected = label.get('selected_index')
    if type(selected) is not int or not 0 <= selected < 5:
        raise ValueError('Invalid selected candidate')
    value = float(selected == 0) - .2
    if label.get('unit_bonus') != value:
        raise ValueError('Inconsistent stored ARM label')
    return value


def apply_bonus(samples, normalized_rewards, policy_id, beta):
    """One scalar per executed turn; native GRPO broadcasts it to response tokens."""
    if len(samples) != len(normalized_rewards) or not math.isfinite(beta) or beta < 0:
        raise ValueError('Invalid bonus inputs')
    if beta == 0:
        return normalized_rewards
    return [a + beta*unit_bonus(s,policy_id)
            if not (s.metadata or {}).get('arm_calibration_excluded')
            and (s.loss_mask is None or any(s.loss_mask)) else a
            for s,a in zip(samples,normalized_rewards)]


def panel(samples, normalized_rewards, policy_id, betas=(0., .2, .5, 1.)):
    if len(samples) != len(normalized_rewards):
        raise ValueError('Reward/Sample count mismatch')
    rows = [(s, a) for s, a in zip(samples, normalized_rewards)
            if not s.remove_sample and not (s.metadata or {}).get('arm_calibration_excluded')
            and (s.loss_mask is None or any(s.loss_mask))]
    if not rows or any(not math.isfinite(a) for _, a in rows):
        raise ValueError('No finite trainable outcome rows')
    n = len(rows)
    units = [unit_bonus(s, policy_id) for s, _ in rows]
    scale = math.sqrt(sum(a*a for _, a in rows)/n)
    unit_rms = math.sqrt(sum(u*u for u in units)/n)
    admitted = sum(u != 0 for u in units)
    return dict(rows=n, admitted=admitted, effective_scored_fraction=admitted/n,
        admitted_distinct_tasks=len({str(s.metadata.get('task_id'))
                                    for (s, _), u in zip(rows, units) if u}),
        admitted_per_256_rows=256*admitted/n,
        selected_original_rate=sum(u > 0 for u in units)/admitted if admitted else None,
        outcome_rms=scale, unit_bonus_rms=unit_rms,
        beta_for_7_percent_rms=.07*scale/unit_rms if unit_rms else None,
        # This is an advantage-scale diagnostic, never a gradient-norm claim.
        variants={str(beta): dict(bonus_rms=beta*unit_rms,
            bonus_to_outcome_rms=beta*unit_rms/scale if scale else None,
            bonus_mean=beta*sum(units)/n,
            sign_flips=sum(a*(a+beta*u) < 0 for (_, a), u in zip(rows, units)))
            for beta in betas},
        task_counts=dict(Counter(str(s.metadata.get('task_id')) for s, _ in rows)),
        label_reasons=dict(Counter((s.metadata or {}).get('arm_turn_bonus', {}).get('reason',
            'unscored_or_excluded') for s, _ in rows)),
        limitation='Before native epoch trimming/shuffle; RMS is not gradient strength; one frozen collection.')


def post_process_rewards(args, samples, raw_rewards, normalized_rewards):
    current = state(args)
    config = current['config']
    report = panel(samples, normalized_rewards, config['policy_id'])
    records = current['records']
    report.update(shadow_only=True, optimizer_updates=0, batch_rows=len(samples), policy_id=config['policy_id'],
        checkpoint=config['checkpoint'], config=config,
        elapsed_collection_seconds=time.monotonic()-current['started'],
        collection_records=len(records),
        collection_label_reasons=dict(Counter(x.get('reason', 'pending') for x in records)),
        requests={key: sum(x.get(key, 0) for x in records) for key in (
            'candidate_requests', 'candidate_response_tokens', 'selector_requests', 'executed_response_tokens')},
        overhead_caveat='Parallel request durations do not add to wall time. A matched q=0 collection is needed for causal wall-time overhead.')
    from openwebrl.arm_turn_bonus_runtime import calibration_decision
    report['decision'] = calibration_decision(report)
    training_requested = not config.get('shadow_only',True)
    report['applied_beta'] = config['beta'] if training_requested and report['decision']['passed'] else 0.
    report['outcome_breakdown'] = {
        name: dict(rows=len(indices),admitted=sum(unit_bonus(samples[i],config['policy_id']) != 0 for i in indices))
        for name,indices in ((name,[i for i,s in enumerate(samples)
            if not s.remove_sample and not (s.metadata or {}).get('arm_calibration_excluded')
            and (s.loss_mask is None or any(s.loss_mask)) and predicate(raw_rewards[i])])
            for name,predicate in [('success',lambda r:r==1),('failure',lambda r:r!=1)])}
    write_json(Path(config['output']) / 'calibration.json', report)
    log_progress(current, force=True)
    try:
        import wandb
        if wandb.run is not None:
            wandb.log({'arm_calibration/'+k:v for k,v in report.items() if isinstance(v,(int,float))})
    except Exception as exc:
        print('[ARM final telemetry unavailable] '+type(exc).__name__, flush=True)
    if config.get('additive_failure_groups'):
        from openwebrl.arm_failure_additive import write_auxiliary
        write_auxiliary(args,samples,current,report)
    # Native outcomes remain unchanged. Beta=0/shadow preserves object identity.
    return raw_rewards, apply_bonus(samples,normalized_rewards,config['policy_id'],report['applied_beta'])


def verify_shadow_complete(args, rollout_id):
    config = json.loads(Path(os.environ['OPENWEBRL_ARM_TURN_BONUS_CONFIG']).read_text())
    report = json.loads((Path(config['output']) / 'calibration.json').read_text())
    if not report['shadow_only'] or report['policy_id'] != config['policy_id']:
        raise ValueError('Missing matching shadow calibration')
    recovery = Path(args.save_debug_rollout_data.format(rollout_id=rollout_id))
    cursor = Path(args.save) / 'rollout' / f'global_dataset_state_dict_{rollout_id}.pt'
    if not recovery.is_file() or not cursor.is_file():
        raise ValueError('Preserve the native batch and consumed dataset cursor before stopping')
    from openwebrl.arm_turn_bonus_runtime import verify_torch_archive
    verify_torch_archive(recovery)
    verify_torch_archive(cursor)
    write_json(Path(config['output']) / 'collection_complete.json', dict(
        rollout_id=rollout_id, checkpoint=config['checkpoint'], optimizer_updates=0,
        recovery_file=str(recovery), dataset_cursor=str(cursor), calibration_rows=report['rows']))
