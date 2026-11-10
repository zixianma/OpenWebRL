"""ARM-guided in-group rescue for outcome-only GRPO (approved 2026-11-03).

When a completed group has five valid zero-reward trajectories, one extra SelectionARM-guided
attempt runs on the same task with the current policy (K=5 candidates per turn; ARM picks which one
executes). If the GPT-4.1 training judge scores it a success, the guided trajectory replaces one of
the five failures (chosen by hash) and the group joins the batch as an ordinary mixed group
[0,0,0,0,1]; GRPO normalizes it like any other group. Otherwise the group stays rejected.
Rescued groups are added to the 48 native mixed groups; they never consume native quota.

Advantages are outcome-only. With telemetry_labels true (v1), ARM labels (q=0.2) are still collected by the
frozen bonus hook for telemetry and their bonus is discarded. With telemetry_labels false (v2, 2026-11-04) no
labels are requested, so the selector serves only rescue attempts; smoke 418271 showed label traffic starving
them (723 label timeouts, 36/36 rescues invalid at step 1). Either way the ARM calibration gate does not apply:
before_optimizer keeps the frozen collection checks and the time budget only.

v3 (2026-11-06): smoke 420434 ran end to end (27/29 valid attempts, 317 selections) but rescued 0. A sixth plain
sample on an all-failure group succeeds ~6.4% of the time (beta-binomial over 324 earlier groups), so one guided
attempt per group cannot tell whether ARM guidance helps. ``guided_attempts`` runs several guided attempts per group
(the first success, in attempt order, is inserted); ``control_attempts`` runs paired unguided attempts (one policy
sample per turn through the same path) that are telemetry only and never enter the batch. Every attempt records its
terminal status, terminate reason, judge text and executed action names.

Configuration: OPENWEBRL_ARM_RESCUE_CONFIG (JSON) with keys
  output, selector_endpoint, policy_id, seed, max_rescues_per_rollout, wait_seconds, max_steps,
  inference_timeout, telemetry_labels (default true), guided_attempts (default 1), control_attempts (default 0).
"""
import asyncio
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import time

NATIVE_FILTER = 'slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonempty_nonzero_std'
_ORIGINALS = {}
# Sample-index offsets (original.index*10 + offset); 7 is the v1/v2 single guided attempt.
GUIDED_SLOTS = (7, 8, 9)
CONTROL_SLOTS = (1, 2, 3)
_ROUND = None


def config():
    c = json.loads(Path(os.environ['OPENWEBRL_ARM_RESCUE_CONFIG']).read_text())
    for k in ('output', 'selector_endpoint', 'policy_id', 'seed', 'max_rescues_per_rollout', 'wait_seconds'):
        if k not in c:
            raise ValueError(f'Rescue config missing {k}')
    if not 0 < int(c['max_rescues_per_rollout']) <= 64 or not 0 < float(c['wait_seconds']) <= 3600:
        raise ValueError('Rescue budget out of range')
    if not 1 <= int(c.get('guided_attempts', 1)) <= len(GUIDED_SLOTS) or not 0 <= int(c.get('control_attempts', 0)) <= len(CONTROL_SLOTS):
        raise ValueError('Rescue attempts out of range')
    live = os.environ.get('OPENWEBRL_ARM_TURN_BONUS_CONFIG')
    if live and Path(live).is_file():
        # The controller leases ports at launch; use the selector the label hook actually talks to.
        c['selector_endpoint'] = json.loads(Path(live).read_text())['selector_endpoint']
    return c


def _round(rollout_id):
    global _ROUND
    if _ROUND is None or _ROUND['rollout_id'] != rollout_id:
        _ROUND = dict(rollout_id=rollout_id, tasks=set(), rescued=[], records=[], started=time.monotonic())
    return _ROUND


def _trajectories(group):
    return [t if isinstance(t, list) else [t] for t in group]


def _terminal(turns):
    return max(turns, key=lambda t: (t.metadata or {}).get('turn_index', 0))


def all_valid_failures(args, group):
    """Five valid trajectories, every terminal reward exactly 0, none removed/aborted/excluded."""
    from slime.utils.types import Sample
    trajs = _trajectories(group)
    if len(trajs) != args.n_samples_per_prompt:
        return False
    for turns in trajs:
        if not turns or any(t.remove_sample or t.status == Sample.Status.ABORTED for t in turns):
            return False
        if any((t.metadata or {}).get('arm_calibration_excluded') for t in turns):
            return False
        try:
            reward = _terminal(turns).get_reward_value(args)
        except Exception:
            return False
        if reward != 0:
            return False
    return True


def drop_index(policy_id, task_id, n):
    return int(hashlib.sha256(f'{policy_id}:{task_id}:drop'.encode()).hexdigest(), 16) % n


async def generate(args, sample, sampling_params, evaluation=False):
    """Frozen ARM label hook, plus capture of each group's initial prompt sample for rescue."""
    from openwebrl import arm_turn_bonus
    if not evaluation:
        _ORIGINALS.setdefault(sample.group_index, (deepcopy(sample), dict(sampling_params)))
    if config().get('telemetry_labels', True):
        return await arm_turn_bonus.generate(args, sample, sampling_params, evaluation=evaluation)
    if evaluation:
        raise ValueError('Calibration must not use evaluation tasks')
    from copy import copy
    from openwebrl.generate_browser import generate_turn_sample
    current = arm_turn_bonus.state(args)  # same recipe/judge validation; the calibration report has zero labels
    task_id, intent = arm_turn_bonus.task_identity(sample.metadata or {})
    excluded = (bool(task_id) and task_id in current['excluded_ids']) or (bool(intent) and intent in current['excluded_intents'])
    local = copy(args)
    local.browser_action_selector = None
    turns = await generate_turn_sample(local, sample, sampling_params)
    if excluded:  # evaluation-overlap tasks keep the v1 marking, so they are never rescued
        for turn in turns:
            turn.metadata = dict(turn.metadata or {}, arm_calibration_excluded=True)
    return turns


async def guided_trajectory(args, original, params, c, rollout_id, mode='selection', slot=GUIDED_SLOTS[0]):
    """One trajectory where SelectionARM picks which of five policy candidates executes at every turn
    (mode 'baseline': one policy sample per turn, no selector call; the unguided control)."""
    from copy import copy
    from openwebrl.arm_inference import ActionSelector
    from openwebrl.generate_browser import generate_turn_sample
    from openwebrl.reward_browser import reward_func
    from slime.utils.rollout_transport import file_back_completed_group
    task_id = str(original.metadata['task_id'])
    # Slot 7 keeps the v1/v2 seed; other slots draw independent seeds.
    key = f"{c['seed']}:{c['policy_id']}:{rollout_id}:{task_id}" + ('' if slot == GUIDED_SLOTS[0] else f':{slot}')
    seed = int(hashlib.sha256(key.encode()).hexdigest(), 16) % 2**31
    folder = 'selections' if mode == 'selection' else 'controls'
    selector = ActionSelector(mode, c['selector_endpoint'],
                              Path(c['output'])/f'rescue-{rollout_id:04d}'/folder/str(slot), seed=seed, candidates=5,
                              timeout=int(c.get('selector_timeout', 120)))
    local, initial = copy(args), deepcopy(original)
    initial.index = original.index*10 + slot  # disjoint from native sample indices
    initial.session_id = None
    local.browser_action_selector = selector
    turns = await generate_turn_sample(local, initial, dict(params, sampling_seed=seed))
    if turns and not any(t.remove_sample for t in turns):
        rewards = await reward_func(local, turns)
        for t, reward in zip(turns, rewards, strict=True):
            t.reward = reward
    await asyncio.to_thread(file_back_completed_group, turns)
    return turns


def outcome(turns):
    """Terminal diagnostics of one attempt (no screenshots or tokens)."""
    import re
    last = _terminal(turns)
    meta = last.metadata or {}
    names = []
    for t in sorted(turns, key=lambda t: (t.metadata or {}).get('turn_index', 0)):
        m = re.search(r'"name"\s*:\s*"(\w+)"', str(t.response or ''))
        names.append(m.group(1) if m else None)
    judge = (meta.get('reward') or {}) if isinstance(meta.get('reward'), dict) else {}
    return dict(status=str(last.status), terminate_reason=str(meta.get('terminate_reason')),
                judge_text=str(judge.get('judge_text', ''))[:400], actions=names)


async def _attempt(args, original, params, c, rollout_id, mode, slot):
    from slime.utils.types import Sample
    record = dict(arm='guided' if mode == 'selection' else 'control', slot=slot, started=time.time())
    turns = None
    try:
        turns = await guided_trajectory(args, original, params, c, rollout_id, mode=mode, slot=slot)
        valid = bool(turns) and not any(t.remove_sample or t.status == Sample.Status.ABORTED for t in turns)
        reward = _terminal(turns).get_reward_value(args) if valid else None
        record.update(valid=valid, reward=reward, steps=len(turns), success=valid and reward == 1)
        if turns:
            record.update(outcome(turns))
    except asyncio.CancelledError:
        record.update(success=False, cancelled=True)
        raise
    except Exception as exc:
        record.update(success=False, error_type=type(exc).__name__, error=str(exc)[:200])
    finally:
        record['seconds'] = time.time()-record['started']
    return record, turns


async def _rescue(args, group, rollout_id):
    """Guided (and optional control) attempts on one all-failure group; insert the first guided success."""
    c, r = config(), _round(rollout_id)
    trajs = _trajectories(group)
    gid = trajs[0][0].group_index
    task_id = str(trajs[0][0].metadata['task_id'])
    original, params = _ORIGINALS[gid]
    plan = [('selection', s) for s in GUIDED_SLOTS[:int(c.get('guided_attempts', 1))]]
    plan += [('baseline', s) for s in CONTROL_SLOTS[:int(c.get('control_attempts', 0))]]
    record = dict(rollout_id=rollout_id, group_index=gid, task_id=task_id, started=time.time(), rescued=False)
    jobs = [asyncio.ensure_future(_attempt(args, original, params, c, rollout_id, m, s)) for m, s in plan]
    try:
        results = await asyncio.gather(*jobs)
        record['attempts'] = [a for a, _ in results]
        guided = [(a, t) for a, t in results if a['arm'] == 'guided']
        first = guided[0][0]  # v1/v2 summary fields describe the first guided attempt
        record.update({k: first.get(k) for k in ('valid', 'reward', 'steps', 'error_type', 'error') if k in first})
        win = next(((a, t) for a, t in guided if a.get('success')), None)
        if win is not None:
            turns = win[1]
            drop = drop_index(c['policy_id'], task_id, len(trajs))
            dropped_tid = (trajs[drop][0].metadata or {}).get('trajectory_id')
            for t in turns:
                t.group_index = gid
                t.metadata = dict(t.metadata or {}, arm_rescue=dict(policy_id=c['policy_id'], rollout_id=rollout_id,
                                                                     replaced_trajectory_id=dropped_tid,
                                                                     slot=win[0]['slot']))
            rescued = [x for i, x in enumerate(group) if i != drop]
            rescued.insert(drop, turns if isinstance(group[drop], list) else turns[-1])
            r['rescued'].append(rescued)
            record.update(rescued=True, replaced_index=drop, rescued_slot=win[0]['slot'])
    except asyncio.CancelledError:
        for j in jobs:
            j.cancel()
        record.update(cancelled=True)
        raise
    finally:
        record['seconds'] = time.time()-record['started']
        r['records'].append(record)


def filter_groups(args, samples, **kwargs):
    """Native mixed-group filter; valid all-failure groups additionally launch one guided rescue."""
    from slime.rollout.filter_hub.dynamic_sampling_filters import check_reward_nonempty_nonzero_std
    native = check_reward_nonempty_nonzero_std(args, samples, **kwargs)
    if native.keep or not all_valid_failures(args, samples):
        return native
    c = config()
    rid = int(getattr(args, '_arm_rescue_rollout_id', -1))
    r = _round(rid)
    gid = _trajectories(samples)[0][0].group_index
    if len(r['tasks']) >= int(c['max_rescues_per_rollout']) or gid not in _ORIGINALS:
        r.setdefault('skipped', 0)
        r['skipped'] += 1
        return native
    task = asyncio.get_running_loop().create_task(_rescue(args, samples, rid))
    r['tasks'].add(task)
    return native


async def collect(args, rollout_id):
    """Await in-flight rescues (bounded), cancel the rest, persist a report, return rescued groups."""
    c, r = config(), _round(rollout_id)
    pending = set(r['tasks'])
    if pending:
        done, late = await asyncio.wait(pending, timeout=float(c['wait_seconds']))
        for t in late:
            t.cancel()
        if late:
            await asyncio.gather(*late, return_exceptions=True)
    groups = list(r['rescued'])
    arms = {}
    for rec in r['records']:
        for a in rec.get('attempts', []):
            v = arms.setdefault(a['arm'], dict(attempts=0, valid=0, successes=0, groups_with_success=set()))
            v['attempts'] += 1
            v['valid'] += bool(a.get('valid'))
            v['successes'] += bool(a.get('success'))
            if a.get('success'):
                v['groups_with_success'].add(rec['group_index'])
    for v in arms.values():
        v['groups_with_success'] = len(v['groups_with_success'])
    report = dict(rollout_id=rollout_id, attempted=len(r['tasks']), rescued=len(groups), arms=arms,
                  skipped_cap=r.get('skipped', 0), records=r['records'],
                  collection_seconds=time.monotonic()-r['started'])
    out = Path(c['output'])/'rescue-reports'
    out.mkdir(parents=True, exist_ok=True)
    tmp = out/f'{rollout_id:04d}.json.partial'
    tmp.write_text(json.dumps(report, indent=1, default=str))
    tmp.replace(out/f'{rollout_id:04d}.json')
    print(f'[ARMRescue] rollout={rollout_id} attempted={report["attempted"]} rescued={len(groups)}', flush=True)
    try:
        import wandb
        if wandb.run is not None:
            wandb.log({'arm_rescue/attempted': report['attempted'], 'arm_rescue/rescued': len(groups),
                       'arm_rescue/yield': len(groups)/max(1, report['attempted'])})
    except Exception:
        pass
    _ORIGINALS.clear()
    return groups


def before_optimizer(args, rollout_id):
    """Rescue stage boundary (replaces the ARM bonus calibration gate): return True to stop.

    ARM labels never reach the rescue reward, so the label/task/coverage/scale gate does not apply. Keep the frozen
    collection checks (matching report, durable batch and cursor) and the time-budget check."""
    from openwebrl.arm_turn_bonus import verify_shadow_complete, write_json
    cfg = json.loads(Path(os.environ['OPENWEBRL_ARM_TURN_BONUS_CONFIG']).read_text())
    verify_shadow_complete(args, rollout_id)
    report = json.loads((Path(cfg['output'])/'calibration.json').read_text())
    updates = (report['batch_rows']//args.global_batch_size)*args.ppo_epochs
    required = updates*cfg.get('seconds_per_optimizer_update', 180)+600
    deadline = cfg.get('deadline_epoch_seconds', 0)
    ready = updates > 0 and time.time()+required < deadline
    write_json(Path(cfg['output'])/'training_gate.json', dict(ready=ready, rescue=True,
        calibration=dict(applies=False, reason='ARM labels are telemetry only in the rescue run'),
        expected_optimizer_updates=updates, estimated_required_seconds=required,
        seconds_remaining=max(0, deadline-time.time()), beta=0., rollout_id=rollout_id))
    return not ready


def post_process_rewards(args, samples, raw_rewards, normalized_rewards):
    """Run the frozen label calibration for telemetry, then return outcome-only advantages."""
    from openwebrl import arm_turn_bonus as parent
    if getattr(args, 'dynamic_sampling_filter_path', '') != 'openwebrl.arm_rescue_rl.filter_groups':
        raise ValueError('Rescue filter required')
    cfg = parent.state(args)['config']
    if cfg.get('additive_failure_groups') or cfg.get('admit_all_failure_groups') or os.environ.get('OPENWEBRL_ARM_FAILURE_AUX_MANIFEST'):
        raise ValueError('No failure auxiliary signal in the rescue run')
    raw, _ = parent.post_process_rewards(args, samples, raw_rewards, normalized_rewards)
    return raw, normalized_rewards
