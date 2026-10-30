"""Export recovery candidates and value targets, including dynamically rejected groups.

Attach export_rollouts to --rollout-all-samples-process-path. Outputs are training
preparation artifacts; a live browser cannot be restored from a textual prefix alone.
"""
import json
import uuid
from pathlib import Path
from openwebrl.rl_recipe import observation_memory, valid_trajectory


def build_records(args, groups):
    values, recovery, memories = [], [], []
    for group in groups:
        trajectories = [x if isinstance(x, list) else [x] for x in group]
        valid = [ts for ts in trajectories if valid_trajectory(ts)]
        all_fail = bool(valid) and all(ts[-1].get_reward_value(args) <= 0 for ts in valid)
        for turns in trajectories:
            terminal = turns[-1]
            meta = terminal.metadata
            key = {'task_id': meta.get('task_id'), 'group_index': terminal.group_index,
                   'trajectory_id': meta.get('trajectory_id', terminal.index)}
            if not valid_trajectory(turns):
                recovery.append({**key, 'route': 'infrastructure_review', 'reason': meta.get('terminate_reason'),
                                 'judge': meta.get('reward', {})})
                continue
            reward = terminal.get_reward_value(args)
            for i, s in enumerate(turns):
                if 'turn_index' not in s.metadata:
                    continue
                state = {**key, 'turn_index': s.metadata['turn_index'], 'prompt': s.prompt,
                         'images': (s.multimodal_inputs or {}).get('images', [])}
                if reward in (0, 1):
                    values.append({**state, 'target_return': reward})
                memories.append({**state, 'observed_memory': observation_memory(turns[:i])})
            if reward <= 0:
                reason = meta.get('terminate_reason', '')
                route = ('longer_horizon_candidate' if reason == 'max_steps_exhausted' else
                         'protocol_recovery' if reason == 'format_error_failed' else 'teacher_review')
                recovery.append({**key, 'route': route, 'reason': reason, 'all_valid_attempts_failed': all_fail,
                                 'prefix_prompt': terminal.prompt,
                                 'prefix_images': (terminal.multimodal_inputs or {}).get('images', []),
                                 'failed_response': terminal.response,
                                 'requires_live_state_verification': True})
    return {'value_targets': values, 'recovery_candidates': recovery, 'memory_targets': memories}


def export_rollouts(args, groups, data_source):
    root = getattr(args, 'browser_recipe_export_dir', None)
    if not root:
        raise ValueError('Set browser_recipe_export_dir in the browser training YAML')
    directory = Path(root) / f'batch-{uuid.uuid4().hex}'
    directory.mkdir(parents=True, exist_ok=False)
    records = build_records(args, groups)
    for kind, rows in records.items():
        path = directory / f'{kind}.jsonl'
        with path.open('x') as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + '\n')
    (directory / 'manifest.json').write_text(json.dumps({
        'schema_version': 1, 'counts': {k: len(v) for k, v in records.items()},
        'split_by': 'task_id', 'note': 'Split tasks before expanding turns; fit values on prior rollouts or disjoint folds.'
    }, indent=2) + '\n')


def main():
    """Process JSON containing nested groups of serialized turn Samples on CPU."""
    import argparse
    from types import SimpleNamespace
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument('--input', required=True, help='JSON: groups -> trajectories -> Sample dictionaries')
    parser.add_argument('--output', required=True)
    parser.add_argument('--reward-key', default=None)
    options = parser.parse_args()
    args = SimpleNamespace(browser_recipe_export_dir=options.output)
    raw_groups = json.loads(Path(options.input).read_text())
    groups = []
    for raw_group in raw_groups:
        group = []
        for raw_turns in raw_group:
            turns = []
            for row in raw_turns:
                reward = row.get('reward')
                if isinstance(reward, dict):
                    if not options.reward_key:
                        raise ValueError('--reward-key is required for dictionary rewards')
                    reward = reward[options.reward_key]
                s = SimpleNamespace(metadata=row.get('metadata', {}), reward=reward,
                    remove_sample=row.get('remove_sample', False), index=row.get('index'),
                    group_index=row.get('group_index'), prompt=row.get('prompt', ''),
                    response=row.get('response', ''), multimodal_inputs=row.get('multimodal_inputs'))
                s.get_reward_value = lambda args, value=reward: value
                turns.append(s)
            if not turns:
                raise ValueError('Empty trajectory in input')
            group.append(turns)
        groups.append(group)
    export_rollouts(args, groups, None)


if __name__ == '__main__':
    main()
