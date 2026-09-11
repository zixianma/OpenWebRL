#!/usr/bin/env python3
"""Queue checkpoints entering the top five rewards within an explicit budget."""
import argparse
import fcntl
import json
import math
from pathlib import Path
import re
import subprocess


def number(row, key):
    value = row.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        raise ValueError(f'Missing or nonfinite {key}')
    return value


def initialize(rows, source):
    rows = [x for x in rows if x.get('train/reward') is not None and x.get('train/reward_iteration') is not None]
    if not rows:
        raise ValueError('Need existing verified reward history')
    for row in rows:
        for key in ['train/reward', 'train/reward_iteration', '_step']:
            number(row, key)
    best = max(rows, key=lambda x: x['train/reward'])
    by_iteration = {}
    for row in rows:
        key = str(int(row['train/reward_iteration']))
        if key not in by_iteration or row['train/reward'] > by_iteration[key]['train/reward']:
            by_iteration[key] = row
    return {'version': 1, 'training_run': 'qcq7i4ug', 'metric': 'train/reward',
            'trigger': 'top_five_distinct_generating_checkpoints', 'reward_history': by_iteration,
            'record_reward': best['train/reward'], 'record_reward_iteration': int(best['train/reward_iteration']),
            'last_seen_wandb_history_step': max(x['_step'] for x in rows),
            'source': str(source), 'browser_env': 'local_process',
            'submission_policy': 'Submission requires --submit-approved and the recorded bounded approval.',
            'requested_profile': {'h200_gpus': 2, 'hours': 2, 'cpus': 16, 'memory_gib': 480,
                                  'estimated_gpu_cost_usd': 3.60, 'judge_api_additional': True},
            'entries': []}


def observe(state, audit, inventory):
    if not audit.get('matched'):
        raise ValueError('Reward must already be verified against archive and W&B')
    row = audit['wandb']
    reward, iteration, step = (number(row, k) for k in ['train/reward', 'train/reward_iteration', '_step'])
    if step <= state['last_seen_wandb_history_step']:
        return None
    entry = None
    before = top_five(state)
    cutoff = before[-1]['train/reward'] if len(before) == 5 else None
    candidate_history = dict(state['reward_history'])
    key = str(int(iteration))
    if key not in candidate_history or reward > candidate_history[key]['train/reward']:
        candidate_history[key] = row
    ranked = sorted(candidate_history.values(), key=lambda x: (-x['train/reward'], x['_step']))[:5]
    rank = next((i+1 for i,x in enumerate(ranked) if x['train/reward_iteration'] == iteration), None)
    if rank is not None and (cutoff is None or reward >= ranked[-1]['train/reward']):
        # Collection N uses the policy saved after training N-1 (directory N-2).
        candidates = [c for c in inventory['checkpoints'] if c['after_training_iteration'] == iteration-1]
        if len(candidates) != 1:
            raise ValueError('Need exactly one verified checkpoint that produced this reward')
        checkpoint = candidates[0]['checkpoint']
        if (checkpoint not in state.get('completed_evaluations', [])
                and not any(e['checkpoint'] == checkpoint for e in state['entries'])):
            entry = {'checkpoint': checkpoint, 'completed_training_iterations': int(iteration)-1,
                     'trigger_reward_iteration': int(iteration), 'trigger_reward': reward,
                     'trigger_rank': rank, 'previous_fifth_reward': cutoff, 'wandb_history_step': step,
                     'source': state['source'], 'browser_env': 'local_process',
                     'tasks': 300, 'status': 'AWAITING_COMPUTE_APPROVAL'}
            state['entries'].append(entry)
    if reward > state['record_reward']:
        state.update(record_reward=reward, record_reward_iteration=int(iteration))
    state['reward_history'] = candidate_history
    state['last_seen_wandb_history_step'] = step
    return entry


def top_five(state):
    return sorted(state['reward_history'].values(), key=lambda x: (-x['train/reward'], x['_step']))[:5]


def reserved_jobs(state):
    return sum(bool(e.get('job_id')) or e.get('status') in
               {'SUBMITTING', 'SUBMITTED', 'SUBMISSION_UNCERTAIN', 'SUBMISSION_FAILED'} for e in state['entries'])


def submission_command(state, entry, approval, template):
    if (approval.get('training_run') != state['training_run']
            or approval.get('browser_env') != 'local_process'
            or approval.get('max_jobs') != 4
            or approval.get('gpus_per_job') != 2
            or approval.get('max_hours_per_job') != 2
            or not approval.get('user_approval_text')):
        raise ValueError('Missing the recorded four-job, two-GPU, two-hour approval')
    reserved = reserved_jobs(state)
    if reserved >= approval['max_jobs']:
        raise ValueError('Approved evaluation job cap reached')
    if entry['status'] != 'AWAITING_COMPUTE_APPROVAL':
        raise ValueError('This checkpoint already has a submission attempt')
    text = template.read_text()
    for option, value in {'gpus':'h200:2', 'time':'02:00:00', 'cpus-per-task':'16', 'mem':'480G', 'nodes':'1'}.items():
        values = re.findall(r'^#SBATCH --'+re.escape(option)+r'=(\S+)$', text, re.M)
        if values != [value]:
            raise ValueError('Batch resources do not match the approved profile')
    exports = {'RAY_ADDRESS':'local', 'OPENWEBRL_RECORD_EVAL_SOURCE':entry['source'],
               'OPENWEBRL_RECORD_EVAL_CHECKPOINT':entry['checkpoint'],
               'OPENWEBRL_RECORD_EVAL_ITERATION':str(entry['completed_training_iterations'])}
    if any(',' in v or '\n' in v for v in exports.values()):
        raise ValueError('Invalid Slurm export value')
    return ['sbatch', '--parsable', '--export=ALL,'+','.join(f'{k}={v}' for k,v in exports.items()), str(template)]


def submit_pending(state, approval, state_path):
    from evaluate_baseline_checkpoint import build_plan, RUNTIME
    template = Path(__file__).resolve().parent / 'evaluate_record_checkpoint_2gpu.sbatch'
    submitted = []
    for entry in state['entries']:
        if entry['status'] != 'AWAITING_COMPUTE_APPROVAL':
            continue
        reserved = reserved_jobs(state)
        if reserved >= approval.get('max_jobs', 0):
            entry['status'] = 'HELD_BUDGET_CAP'
            state_path.write_text(json.dumps(state, indent=2)+'\n')
            continue
        command = submission_command(state, entry, approval, template)
        # Validate checkpoint identity and frozen source before any paid action.
        build_plan(entry['source'], entry['checkpoint'], RUNTIME/'evaluations'/f"record-plan-{entry['wandb_history_step']}",
                   'APPROVED_JOB', gpus=2)
        entry.update(status='SUBMITTING', submission_command=command)
        state_path.write_text(json.dumps(state, indent=2)+'\n')
        try:
            result = subprocess.run(command, text=True, capture_output=True, timeout=60)
        except Exception:
            entry['status'] = 'SUBMISSION_UNCERTAIN'
            state_path.write_text(json.dumps(state, indent=2)+'\n')
            raise
        ids = re.findall(r'^(\d+)(?:;[^\n]+)?$', result.stdout, re.M)
        entry['submission_stdout'] = result.stdout
        entry['submission_stderr'] = result.stderr
        if result.returncode or len(ids) != 1:
            entry['status'] = 'SUBMISSION_FAILED' if result.returncode else 'SUBMISSION_UNCERTAIN'
            state_path.write_text(json.dumps(state, indent=2)+'\n')
            raise RuntimeError('Inspect submission result before any retry')
        entry.update(status='SUBMITTED', job_id=ids[0], approval_text=approval['user_approval_text'])
        state_path.write_text(json.dumps(state, indent=2)+'\n')
        submitted.append(ids[0])
    return submitted


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--state', type=Path, required=True)
    p.add_argument('--initialize-history', type=Path)
    p.add_argument('--source', type=Path)
    p.add_argument('--reward-audit', type=Path)
    p.add_argument('--inventory', type=Path)
    p.add_argument('--submit-approved', action='store_true')
    p.add_argument('--approval', type=Path)
    args = p.parse_args()
    with args.state.with_suffix('.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        run(args, p)


def run(args, parser):
    if args.initialize_history:
        if args.state.exists() or not args.source:
            parser.error('Initialization requires a new state path and source')
        state = initialize(json.loads(args.initialize_history.read_text()), args.source)
        entry = None
    else:
        state = json.loads(args.state.read_text())
        entry = None
        if args.reward_audit:
            if not args.inventory:
                parser.error('Observation requires a checkpoint inventory')
            entry = observe(state, json.loads(args.reward_audit.read_text()), json.loads(args.inventory.read_text()))
    args.state.write_text(json.dumps(state, indent=2)+'\n')
    jobs = []
    if args.submit_approved:
        if not args.approval:
            parser.error('Submission requires the explicit approval record')
        jobs = submit_pending(state, json.loads(args.approval.read_text()), args.state)
    print(json.dumps({'top_five': top_five(state),
                      'new_queue_entry': entry, 'paid_jobs_submitted': jobs}, indent=2))


if __name__ == '__main__':
    main()
