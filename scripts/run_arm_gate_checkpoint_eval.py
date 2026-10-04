#!/usr/bin/env python3
"""Evaluate B/C with the validated additive actor-only full-300 protocol."""
import argparse
import json
import os
from pathlib import Path

import run_arm_iteration80_eval as base

RUN_IDS = {'B': 'arm-gate-b-309053', 'C': 'arm-gate-c-309054'}
CREDIT = {'B': 'response_index', 'C': 'action_class'}


def checkpoint_ready(variant, training_root, completed_iteration=20):
    root = Path(training_root).resolve()
    manifest_path = root/'launch_manifest.json'
    if not manifest_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get('gate_ablation') != variant
            or manifest.get('wandb_run_id') != RUN_IDS[variant]
            or manifest['arm_config'].get('candidate_gate') != 'min2'
            or manifest['arm_config'].get('credit_assignment') != CREDIT[variant]):
        raise ValueError('B/C evaluation training lineage or credit rule differs')
    return base.checkpoint_ready('additive', completed_iteration, root)


def plan(variant, training_root, job, completed_iteration=20):
    if checkpoint_ready(variant, training_root, completed_iteration) is None:
        raise ValueError('Requested B/C checkpoint is not durable yet')
    p = base.plan('additive', job, completed_iteration, training_root)
    old = p['output']
    output = str(base.RUNTIME/f'evaluations/arm-gate-{variant.lower()}-iter{completed_iteration}-{job}')
    p['command'] = [x.replace(old, output) for x in p['command']]
    p['environment'] = {k: v.replace(old, output) for k, v in p['environment'].items()}
    run_id = f'arm-gate-{variant.lower()}-iter{completed_iteration}-{job}'
    p.update(output=output, arm_variant=variant, parent_training_run=RUN_IDS[variant],
             wandb_run_id=run_id)
    p['command'][p['command'].index('--wandb-group')+1] = f'ARM-{variant}-iter{completed_iteration}-full300'
    p['environment']['WANDB_RUN_ID'] = run_id
    return base.evaluation.configure_evaluation_tracking(p, 'openwebrl-evals')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=RUN_IDS, required=True)
    parser.add_argument('--training-root', type=Path, required=True)
    parser.add_argument('--completed-iteration', type=int, default=20)
    parser.add_argument('--job-id', default='PREPARE')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    p = plan(args.variant, args.training_root, args.job_id, args.completed_iteration)
    if args.execute:
        os.environ['OPENWEBRL_EXPECTED_SCHEDULER_OFFSET'] = '0'
        os.environ['OPENWEBRL_RECORD_EVAL_EXPECTED_TASK_COUNT'] = '300'
        base.evaluation.run(p, base.REPO/'.env')
    else:
        print(json.dumps(p, indent=2))


if __name__ == '__main__':
    main()
