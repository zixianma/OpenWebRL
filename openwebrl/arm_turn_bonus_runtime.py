"""Durability and stage gates for the first executed-turn ARM training batch."""
import hashlib
import json
import math
import os
from pathlib import Path
import time
import zipfile
from copy import copy


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def validate_config(config):
    from openwebrl.arm_failure_recipe import validate_recipe
    validate_recipe(config)
    from openwebrl.arm_turn_bonus import candidate_minimum, credit_rule
    candidate_minimum(config)
    credit_rule(config)
    for key in ('beta', 'scored_fraction', 'label_timeout_seconds'):
        if not math.isfinite(config[key]):
            raise ValueError(f'Nonfinite ARM setting: {key}')
    if config['k'] != 5 or config['beta'] not in (0., .5):
        raise ValueError('First pilot supports K=5 and beta=0 or 0.5')
    if not 0 <= config['scored_fraction'] <= 1:
        raise ValueError('Invalid sampling fraction')
    if config['label_timeout_seconds'] <= 0 or config['max_pending_per_trajectory'] < 1:
        raise ValueError('Invalid label timeout/queue bound')
    if not config['shadow_only'] and not config.get('train_after_calibration'):
        raise ValueError('Training requires the explicit calibration-to-training gate')
    if not config['shadow_only'] and config['scored_fraction'] != .2:
        raise ValueError('Keep the predeclared q=20% in the first training comparison')


def calibration_decision(report):
    if report.get('all_failure_experiment'):
        from openwebrl.arm_failure_bonus import calibration_decision as failure_decision
        return failure_decision(report)
    ratio = report['variants']['0.5']['bonus_to_outcome_rms']
    proposed = report['beta_for_7_percent_rms']
    checks = dict(labels=report['admitted'] >= 100,
        tasks=report['admitted_distinct_tasks'] >= 20,
        coverage=report['effective_scored_fraction'] >= .05,
        scale=ratio is not None and math.isfinite(ratio) and .04 <= ratio <= .10)
    return dict(passed=all(checks.values()), checks=checks,
        beta=.5, q=.2,
        suggested_beta=max(.2,min(1.,proposed)) if proposed is not None and math.isfinite(proposed) else None,
        rule='Beta stays 0.5 for ARM or 0 for the matched control; any failed gate stops training for review.')


def verify_torch_archive(path):
    path = Path(path)
    with zipfile.ZipFile(path) as archive:
        if not any(x.endswith('/data.pkl') for x in archive.namelist()):
            raise ValueError(f'Incomplete torch archive: {path}')
    return dict(path=str(path), bytes=path.stat().st_size)


def record_completed_group(args, rollout_id, group, filter_output, accepted_position):
    """Save outcome-bearing groups before a later browser failure can lose them.

    Called after the native filter, without altering its inputs/results. Images
    have already been file-backed by the native collector. These are recovery
    artifacts, not automatically a complete or safely resumable dataset cursor.
    """
    import torch
    from openwebrl.arm_turn_bonus import state, write_json
    current = state(args)
    root = Path(current['config']['output'])/'groups'/str(rollout_id)
    rows = [s for trajectory in group for s in (trajectory if isinstance(trajectory,list) else [trajectory])]
    ids = {s.group_index for s in rows}
    if len(ids) != 1:
        raise ValueError('Completed group has inconsistent query-group identity')
    key = str(next(iter(ids)))
    if not key.isdecimal():
        raise ValueError('Expected numeric query-group identity')
    path = root/f'{key}.pt'
    root.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise ValueError('Completed group would overwrite an existing journal entry')
    kept = filter_output.samples if filter_output.samples is not None else group
    accepted = bool(filter_output.keep) and bool(kept) and accepted_position is not None
    kept_ids = [s.index for trajectory in kept for s in (trajectory if isinstance(trajectory,list) else [trajectory])] if accepted else []
    temporary = path.with_suffix('.pt.partial')
    torch.save(dict(samples=[s.to_dict() for s in rows],accepted=accepted,
        accepted_sample_indices=kept_ids,accepted_position=accepted_position,
        policy_id=current['config']['policy_id'],checkpoint=current['config']['checkpoint'],
        rollout_id=rollout_id,filter_reason=filter_output.reason),temporary)
    with temporary.open('rb') as handle:
        os.fsync(handle.fileno())
    temporary.replace(path)
    write_json(path.with_suffix('.json'),dict(accepted=accepted,accepted_position=accepted_position,
        policy_id=current['config']['policy_id'],rollout_id=rollout_id,group_id=key,
        filter_reason=filter_output.reason,accepted_sample_indices=kept_ids,
        rows=[dict(index=s.index,trajectory_id=(s.metadata or {}).get('trajectory_id',s.index),
            task_id=(s.metadata or {}).get('task_id'),reward=s.reward,status=s.status.value,
            remove_sample=s.remove_sample,response_length=s.response_length,
            trainable=not s.remove_sample and s.response_length>0 and (s.loss_mask is None or any(s.loss_mask)),
            arm_turn_bonus=(s.metadata or {}).get('arm_turn_bonus',{}),
            arm_all_failure_group=(s.metadata or {}).get('arm_all_failure_group',False),
            terminate_reason=(s.metadata or {}).get('terminate_reason'),
            reward_metadata=(s.metadata or {}).get('reward',{})) for s in rows]))


def before_optimizer(args, rollout_id):
    """Durable stage boundary: return True to stop, False to train one batch."""
    from openwebrl.arm_turn_bonus import verify_shadow_complete, write_json
    config = json.loads(Path(os.environ['OPENWEBRL_ARM_TURN_BONUS_CONFIG']).read_text())
    verify_shadow_complete(args,rollout_id)
    if config['shadow_only']:
        return True
    report = json.loads((Path(config['output'])/'calibration.json').read_text())
    decision = calibration_decision(report)
    if decision['passed'] and report.get('applied_beta') != config['beta']:
        raise ValueError('Training bonus was not applied at the reward boundary')
    updates = (report['batch_rows']//args.global_batch_size)*args.ppo_epochs
    required = updates*config.get('seconds_per_optimizer_update',180)+600
    deadline = config.get('deadline_epoch_seconds',0)
    ready = decision['passed'] and time.time()+required < deadline and updates > 0
    write_json(Path(config['output'])/'training_gate.json',dict(ready=ready,
        calibration=decision,expected_optimizer_updates=updates,
        estimated_required_seconds=required,seconds_remaining=max(0,deadline-time.time()),
        beta=config['beta'],rollout_id=rollout_id))
    return not ready


def load_replay_cursor(data_source, rollout_id):
    """Restore the exact consumed cursor; never guess the number of submissions."""
    config=json.loads(Path(os.environ['OPENWEBRL_ARM_TURN_BONUS_CONFIG']).read_text())
    origin=config['replay_origin']
    path=Path(os.environ['OPENWEBRL_ARM_REPLAY_CURSOR']).resolve()
    if path!=Path(origin['dataset_cursor']).resolve() or path.name!=f'global_dataset_state_dict_{rollout_id}.pt':
        raise ValueError('Replay cursor provenance mismatch')
    if file_sha256(path)!=origin['cursor_sha256']:
        raise ValueError('Replay cursor changed since validation')
    verify_torch_archive(path)
    original_args=data_source.args
    try:
        data_source.args=copy(original_args)
        data_source.args.load=str(path.parent.parent)
        data_source.load(rollout_id)
    finally:
        data_source.args=original_args
