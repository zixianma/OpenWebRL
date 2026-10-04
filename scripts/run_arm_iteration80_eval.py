#!/usr/bin/env python3
"""Prepare or run a full-300 ARM checkpoint evaluation (default: iteration 80)."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

import evaluate_baseline_checkpoint as evaluation
from prepare_arm_turn_bonus import copy_plain
from resume_baseline import validate_source, write_json
from runtime_ports import patch_rollout_ports

REPO, RUNTIME = evaluation.REPO, evaluation.RUNTIME
CONTROL = RUNTIME/'arm-turn-bonus-preparation/iteration80-evals-20260919'
VARIANTS = {
    'original': ('reference-arm-turn-bonus-cycles-20260913-v3', 'arm-turn-bonus-fresh-303573'),
    'additive': ('reference-arm-failure-additive-20260914-v2', 'arm-failure-additive-303574'),
    'allfailure': ('reference-arm-failure-bonus-20260913-v2', 'arm-turn-bonus-fresh-allfailure-309490'),
}


def source_for(variant):
    return RUNTIME/f'reference-arm-eval80-{variant}-20260920-v2'


def prepare_source(variant):
    parent = RUNTIME/VARIANTS[variant][0]
    source = source_for(variant)
    validate_source(parent)
    relative = 'openwebrl/eval_monitor.py'
    patched = (REPO/relative).read_bytes()
    if not source.exists():
        shutil.copytree(parent, source, symlinks=True, copy_function=copy_plain,
            ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.git', '.browser_use_sessions'))
        (source/relative).write_bytes(patched)
        port_relative = 'slime/ray/rollout.py'
        (source/port_relative).write_text(patch_rollout_ports((source/port_relative).read_text()))
        manifest = json.loads((source/'reference_manifest.json').read_text())
        manifest['recipe_files_sha256'][relative] = hashlib.sha256(patched).hexdigest()
        manifest['recipe_files_sha256'][port_relative] = hashlib.sha256((source/port_relative).read_bytes()).hexdigest()
        manifest['runtime_port_leases'] = True
        manifest['task_addressable_evaluation'] = dict(version=2, parent=str(parent),
            changed_files=[relative, port_relative], lossless_task_archives=True, per_task_verdict_json=True)
        write_json(source/'reference_manifest.json', manifest)
    validate_source(source)
    if (source/relative).read_bytes() != patched:
        raise ValueError('Frozen evaluation saver differs from the tested version')
    parent_hashes = json.loads((parent/'reference_manifest.json').read_text())['recipe_files_sha256']
    hashes = json.loads((source/'reference_manifest.json').read_text())['recipe_files_sha256']
    if any(hashes.get(k) != v for k, v in parent_hashes.items() if k not in (relative, 'slime/ray/rollout.py')):
        raise ValueError('Evaluation source changed outside the persistence hook')
    if (source/'slime/ray/rollout.py').read_text() != patch_rollout_ports((parent/'slime/ray/rollout.py').read_text()):
        raise ValueError('Evaluation port isolation differs from the tested change')
    return source


def checkpoint_ready(variant, completed_iteration=80, training_root=None):
    if completed_iteration < 1:
        raise ValueError('Completed iteration must be positive')
    root = (Path(training_root) if training_root is not None
            else RUNTIME/'evaluations'/VARIANTS[variant][1]).resolve()
    if not root.is_relative_to(RUNTIME/'evaluations'):
        raise ValueError('Training root must be under runtime evaluations')
    index = completed_iteration - 1
    checkpoint = root/f'runtime/iter_{index:07d}'
    report = root/f'iterations/{index:04d}/checkpoint-validation.json'
    saved = root/f'iterations/{index:04d}/checkpoint-saved.json'
    if not report.is_file() or not saved.is_file():
        return None
    record = json.loads(report.read_text())
    saved_record = json.loads(saved.read_text())
    if (record['iteration'] != index or Path(record['checkpoint']).resolve() != checkpoint.resolve()
            or saved_record['rollout_id'] != index or record['completed_optimizer_updates'] <= 0
            or record['scheduler_minus_optimizer_updates'] != 0):
        raise ValueError(f'Iteration-{completed_iteration} checkpoint validation disagrees with expected identity')
    for name, expected_size in record['shard_files'].items():
        if (checkpoint/name).stat().st_size != expected_size:
            raise ValueError('Checkpoint shard size changed after validation')
    for path in [checkpoint/'common.pt', checkpoint/'.metadata',
                 root/f'runtime/rollout/global_dataset_state_dict_{index}.pt']:
        if not path.is_file():
            raise ValueError('Missing completed checkpoint component')
    return checkpoint


def plan(variant, job, completed_iteration=80, training_root=None):
    checkpoint = checkpoint_ready(variant, completed_iteration, training_root)
    if checkpoint is None:
        raise ValueError(f'Iteration-{completed_iteration} checkpoint is not ready')
    source = prepare_source(variant)
    output = RUNTIME/f'evaluations/arm-{variant}-iter{completed_iteration}-{job}'
    p = evaluation.build_plan(source, checkpoint, output, job, gpus=2)
    # Pin the full frozen monitor cohort, irrespective of inherited subset env.
    p['command'][p['command'].index('--eval-config')+1] = str(source/'openwebrl/online_mind2web_monitor.yaml')
    tasks = [json.loads(line) for line in (source/'online_mind2web_monitor.jsonl').read_text().splitlines()]
    ids = [str(task['metadata']['task_id']) for task in tasks]
    if len(ids) != 300 or len(set(ids)) != 300:
        raise ValueError('The frozen monitor cohort must contain 300 distinct tasks')
    p.update(expected_task_count=300, expected_rollout_task_ids=ids, require_task_rollouts=True,
             arm_variant=variant, parent_training_run=checkpoint.parent.parent.name)
    p['command'][p['command'].index('--wandb-group')+1] = f'arm-{variant}-iter{completed_iteration}-full300'
    p['environment'].update(OPENWEBRL_EXPECTED_SCHEDULER_OFFSET='0',
                            OPENWEBRL_RECORD_EVAL_EXPECTED_TASK_COUNT='300',WANDB_RESUME='never')
    evaluation.configure_evaluation_tracking(p, 'openwebrl-evals')
    return p


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=VARIANTS, required=True)
    parser.add_argument('--job-id', default='PREPARE')
    parser.add_argument('--completed-iteration', type=int, default=80)
    parser.add_argument('--training-root', type=Path)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if args.prepare:
        source = prepare_source(args.variant)
        control = (CONTROL if args.completed_iteration == 80 else
                   CONTROL.parent/f'iteration{args.completed_iteration}-evals')
        control.mkdir(parents=True, exist_ok=True)
        checkpoint = checkpoint_ready(args.variant, args.completed_iteration, args.training_root)
        if checkpoint:
            write_json(control/f'{args.variant}-plan.json', plan(
                args.variant, args.job_id, args.completed_iteration, args.training_root))
        print(json.dumps(dict(variant=args.variant, source=str(source), checkpoint=str(checkpoint))))
        return
    p = plan(args.variant, args.job_id, args.completed_iteration, args.training_root)
    if args.execute:
        os.environ['OPENWEBRL_EXPECTED_SCHEDULER_OFFSET'] = '0'
        os.environ['OPENWEBRL_RECORD_EVAL_EXPECTED_TASK_COUNT'] = '300'
        evaluation.run(p, REPO/'.env')
    else:
        print(json.dumps(p, indent=2))


if __name__ == '__main__':
    main()
