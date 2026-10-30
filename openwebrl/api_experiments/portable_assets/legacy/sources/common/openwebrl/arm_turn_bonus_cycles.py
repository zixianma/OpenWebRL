"""Collection boundaries and teacher lifetime for repeated native ARM RL cycles."""
import json
import os
from pathlib import Path
import time

from openwebrl.arm_turn_bonus import write_json


def config():
    return json.loads(Path(os.environ['OPENWEBRL_ARM_TURN_BONUS_CONFIG']).read_text())


def teacher_phase(current, phase):
    root = Path(current['run_output'])
    request = dict(rollout_id=current['rollout_id'], phase=phase)
    write_json(root/'teacher-request.json', request)
    limit = min(time.time()+600, current['deadline_epoch_seconds'])
    while time.time() < limit:
        ack = root/'teacher-ready.json'
        if ack.exists() and json.loads(ack.read_text()) == request:
            return
        failure = root/'status.json'
        if failure.exists() and json.loads(failure.read_text()).get('failed'):
            raise RuntimeError('ARM controller failed while changing teacher phase')
        time.sleep(1)
    raise TimeoutError('Teacher phase acknowledgement timed out')


def before_collection(args, rollout_id):
    current = config()
    root = Path(current['run_output'])
    # Reserve a complete collection, native PPO, checkpoint save and shutdown.
    # The first run has four hours; pilot evidence was ~49 minutes per cycle.
    if time.time()+current.get('minimum_cycle_seconds', 3300) >= current['deadline_epoch_seconds']:
        write_json(root/'budget-stop.json', dict(next_rollout_id=rollout_id,
            reason='Preserve the last checkpoint; insufficient time for another complete cycle'))
        return True
    checkpoint = current['initial_checkpoint'] if rollout_id == 0 else str(Path(args.save)/f'iter_{rollout_id-1:07d}')
    if rollout_id and not (Path(checkpoint)/'common.pt').is_file():
        raise ValueError('Next collection requires the preceding saved actor checkpoint')
    if rollout_id == 0 and args.start_rollout_id != 0:
        raise ValueError('Fresh ARM comparison must start at RL iteration zero')
    current.update(rollout_id=rollout_id, checkpoint=checkpoint,
        policy_id=f"{current['run_id']}:rollout{rollout_id}",
        sampling_policy_id=f'executed-turn-from-zero-v1:rollout{rollout_id}',
        output=str(root/'iterations'/f'{rollout_id:04d}'))
    Path(current['output']).mkdir(parents=True, exist_ok=False)
    write_json(Path(os.environ['OPENWEBRL_ARM_TURN_BONUS_CONFIG']), current)
    write_json(Path(current['output'])/'arm-config.json', current)
    teacher_phase(current, 'collection')
    return False


def reset_collection(args, rollout_id):
    from openwebrl import arm_turn_bonus
    current = config()
    if current['rollout_id'] != rollout_id:
        raise ValueError('Driver/rollout worker ARM policy version mismatch')
    arm_turn_bonus._STATE = None
    return arm_turn_bonus.state(args)


def before_training():
    teacher_phase(config(), 'training')


def after_checkpoint(args, rollout_id):
    current = config()
    if current['rollout_id'] != rollout_id:
        raise ValueError('Checkpoint/label iteration mismatch')
    write_json(Path(current['output'])/'checkpoint-saved.json', dict(
        rollout_id=rollout_id, checkpoint=str(Path(args.save)/f'iter_{rollout_id:07d}'),
        saved_at_epoch_seconds=time.time()))
    # Stop before costly actor-to-rollout weight transfer at the final boundary.
    return (rollout_id+1 >= args.num_rollout
        or time.time()+current.get('minimum_cycle_seconds',3300) >= current['deadline_epoch_seconds'])
