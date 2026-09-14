#!/usr/bin/env python3
"""Prepare the frozen 90x15 -> 50x30 continuation; never allocate compute."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil

from prepare_resume_pending_eval import prepare as prepare_pending
from resume_baseline import validate_source, write_json

RUNTIME = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime')
STATE = RUNTIME / 'current_baseline.json'
PLAN = RUNTIME / 'baseline_stage2_plan.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def upgrade_launcher(text):
    replacements = {
        "parser.add_argument('--steps', type=int, default=15)":
            "parser.add_argument('--steps', type=int, default=30)",
        "args.rollouts = (90 if paper else 30) if args.rollouts is None else args.rollouts":
            "args.rollouts = 140 if args.rollouts is None else args.rollouts\n"
            "    if not reference or not args.resume_from or args.steps != 30 or args.rollouts != 140:\n"
            "        parser.error('Stage 2 requires reference resume, 30 steps and cumulative target 140')",
        "    if args.resume_from:\n        marker=":
            "    # Update the horizon, retaining loaded optimizer and scheduler counters.\n"
            "    env['OVERRIDE_OPT_PARAM_SCHEDULER'] = '1'\n"
            "    env['WANDB_PROJECT'] = 'openwebrl'\n"
            "    env['JUDGE_MODEL'] = 'gpt-4.1'\n"
            "    if args.resume_from:\n        marker=",
        "        env['SLIME_LOAD_CHECKPOINT'] = str(args.resume_from.resolve())":
            "        if not 89 <= int(marker.read_text()) < 139:\n"
            "            raise SystemExit('Stage 2 requires a checkpoint after 90 through 139 completed iterations')\n"
            "        env['SLIME_LOAD_CHECKPOINT'] = str(args.resume_from.resolve())",
        "    manifest['allocation_resources'] = resources":
            "    manifest['training_stage'] = 2\n"
            "    manifest['stage2'] = {'start_completed_iterations': 90, 'target_completed_iterations': 140,\n"
            "        'scheduler_horizon_override': True, 'optimizer_reset': False, 'data_cursor_reset': False}\n"
            "    manifest['allocation_resources'] = resources",
    }
    for old, new in replacements.items():
        if text.count(old) != 1:
            raise ValueError(f'Unexpected frozen launcher anchor: {old!r}')
        text = text.replace(old, new, 1)
    ast.parse(text)
    return text


def prepare(source, destination, state_path=STATE, plan_path=PLAN):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    pending = destination.with_name(destination.name + '-pending-eval')
    if destination.exists() or pending.exists() or Path(plan_path).exists():
        raise ValueError('Stage-2 outputs already exist; inspect the existing plan')
    validate_source(source)
    state = json.loads(Path(state_path).read_text())
    if (state.get('completed_training_iterations') != 90
            or state.get('pending_evaluation_iteration_one_based') is not None
            or state.get('last_completed_evaluation_iteration_one_based') != 90
            or state.get('pending_replay_batch') is not None
            or Path(state['source_directory']).resolve() != source):
        raise ValueError('Require completed stage 1, its exact source, and no pending work')
    checkpoint = Path(state['last_valid_checkpoint'])
    if checkpoint.name != 'iter_0000089' or int((checkpoint.parent / 'latest_checkpointed_iteration.txt').read_text()) != 89:
        raise ValueError('Checkpoint marker must be after iteration 90')
    text = upgrade_launcher((source / 'scripts/run_small_baseline.py').read_text())
    shutil.copytree(source, destination, symlinks=True,
                    ignore=shutil.ignore_patterns('__pycache__', '.git', '.env*', 'wandb', 'checkpoints', 'outputs'))
    (destination / 'scripts/run_small_baseline.py').write_text(text)
    manifest = json.loads((destination / 'reference_manifest.json').read_text())
    manifest['stage2_preparation'] = {
        'source': str(source), 'source_launcher_sha256': sha(source / 'scripts/run_small_baseline.py'),
        'initial_checkpoint': str(checkpoint), 'target_completed_iterations': 140, 'max_steps': 30,
        'recipe_files_unchanged': True, 'scheduler_horizon_override': True,
        'launcher_sha256': sha(destination / 'scripts/run_small_baseline.py'),
    }
    # Bind executable launch and optimization code in addition to original recipe files.
    for name in ['scripts/run_small_baseline.py', 'scripts/run_h200_browser.sh',
                 'slime/backends/megatron_utils/model.py', 'slime/utils/wandb_utils.py']:
        manifest['recipe_files_sha256'][name] = sha(destination / name)
    write_json(destination / 'reference_manifest.json', manifest)
    validate_source(destination)
    prepare_pending(destination, pending)
    snapshot = RUNTIME / 'baseline_stage1_completed90_before_stage2.json'
    if snapshot.exists():
        raise ValueError('Stage-1 snapshot already exists; inspect it before preparing again')
    shutil.copy2(state_path, snapshot)
    state.update(source_directory=str(destination), pending_evaluation_resume_source=str(pending),
                 next_prepared_source_directory=str(destination), training_stage=2,
                 target_completed_iterations=140,
                 status_note='Stage 2 prepared; no new allocation authorized or launched.')
    prepared_state = RUNTIME / 'baseline_stage2_prepared_state.json'
    write_json(prepared_state, state)
    plan = dict(status='PREPARED_AWAITING_EXPLICIT_BUDGET_APPROVAL', source=str(destination),
                pending_source=str(pending), prepared_state=str(prepared_state), prepared_state_sha256=sha(prepared_state),
                stage1_snapshot=str(snapshot), state=str(state_path), expected_state_sha256=sha(state_path),
                initial_checkpoint=str(checkpoint), wandb_run_id=state['wandb_run_id'],
                target_completed_iterations=140, max_steps=30,
                requested_resources=dict(h200_gpus=4, cpus=32, memory_gib=480, hours=8, maximum_gpu_hours=32),
                gpu_restore_verified=False)
    write_json(plan_path, plan)
    return plan


def activate(plan_path, job_id):
    """Called by the approved batch after its successful GPU restore verification."""
    if os.environ.get('SLURM_JOB_ID') != job_id:
        raise ValueError('Activation must execute inside the approved allocation')
    plan = json.loads(Path(plan_path).read_text())
    receipt = json.loads((RUNTIME / 'logs' / f'resume-{job_id}-4gpu-verification.json').read_text())
    if sha(plan['state']) != plan['expected_state_sha256']:
        raise ValueError('Baseline pointer changed after preparation; review before activating')
    if sha(plan['prepared_state']) != plan['prepared_state_sha256']:
        raise ValueError('Prepared state changed; inspect before activation')
    state = json.loads(Path(plan['prepared_state']).read_text())
    expected = dict(source=plan['source'], launcher_sha256=sha(Path(plan['source']) / 'scripts/run_small_baseline.py'),
                    resume_from=str(Path(plan['initial_checkpoint']).parent), checkpoint=plan['initial_checkpoint'],
                    optimizer_updates=state['durable_optimizer_updates'], gpus=4, job_id=job_id)
    if receipt['identity'] != expected or receipt['report'].get('full_model_and_optimizer_load') != 'passed':
        raise ValueError('GPU verification does not match the prepared continuation')
    validate_source(Path(plan['source']))
    # The resume driver revalidates the full receipt before training starts.
    state['status_note'] = f'Stage 2 activated in approved job {job_id}; training launch follows verified restoration.'
    state['training_stage'] = 2
    state['queued_continuation'] = dict(job_id=job_id, state='RUNNING', training_stage=2,
        requested_resources=plan['requested_resources'], target_completed_iterations=140,
        browser_concurrency=32, wandb_run_id=plan['wandb_run_id'], target_fulfilled=False)
    write_json(plan['state'], state)
    plan.update(status='GPU_VERIFIED_LAUNCHING', job_id=job_id, gpu_restore_verified=True,
                verification_run=receipt['verification_run'])
    write_json(plan_path, plan)
    return plan


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--plan', type=Path, default=PLAN)
    parser.add_argument('--activate', action='store_true')
    parser.add_argument('--job-id')
    args = parser.parse_args()
    if args.activate:
        result = activate(args.plan, args.job_id)
    else:
        if not args.source or not args.output:
            parser.error('Preparation requires --source and --output')
        result = prepare(args.source, args.output, plan_path=args.plan)
    print(json.dumps(result, indent=2))
