#!/usr/bin/env python3
"""Await explicitly queued checkpoint evaluations inside their owning allocation.

This helper never submits, extends, requeues, or cancels a Slurm job.
"""
import json
import os
from pathlib import Path
import time

from resume_baseline import REPO, RUNTIME, write_json

CONTROL = RUNTIME/'arm-turn-bonus-preparation/milestone-evaluations'


def run_due_evaluations(variant, training_root, target, job, *, skip_unavailable=False):
    request_path = CONTROL/f'{job}.json'
    if not request_path.exists():
        return
    request = json.loads(request_path.read_text())
    if (os.getenv('SLURM_JOB_ID') != job or request.get('job_id') != job
            or request.get('variant') != variant
            or request.get('use_existing_allocation') is not True):
        raise ValueError('Milestone evaluation must use its approved owning allocation')
    if variant in ('B', 'C'):
        from run_arm_gate_checkpoint_eval import checkpoint_ready
        from prepare_arm_gate_to60 import evaluation_plan
    elif variant in ('weight', 'coverage'):
        from run_arm_iteration80_eval import checkpoint_ready as arm_checkpoint_ready
        from prepare_arm_failure_ablations import evaluation_plan as ablation_plan
        manifest = json.loads((Path(training_root)/'launch_manifest.json').read_text())
        if (manifest.get('failure_ablation_training') != variant or
                manifest.get('wandb_run_id') != f'arm-failure-{variant}-fromzero-{job}'):
            raise ValueError('Ablation evaluation training lineage changed')
        checkpoint_ready = lambda method, root, iteration: arm_checkpoint_ready('additive', iteration, root)
        evaluation_plan = lambda method, root, iteration, allocation: ablation_plan(method, root, allocation, iteration)
    else:
        raise ValueError('Unknown evaluation lineage')
    from run_stage1_to100 import audit_rollouts
    import evaluate_baseline_checkpoint as evaluator

    root = Path(training_root).resolve()
    for item in sorted(request['evaluations'], key=lambda value: value['iteration']):
        iteration = item['iteration']
        if (Path(item['training_root']).resolve() != root or iteration >= target):
            continue
        if iteration <= 0 or iteration % 10:
            raise ValueError('Expected a positive tenth training iteration')
        if checkpoint_ready(variant, root, iteration) is None:
            if skip_unavailable:
                item.update(state='waiting_for_durable_checkpoint')
                write_json(request_path, request)
                continue
            raise ValueError(f'Required iteration {iteration} checkpoint is not durable')
        plan = evaluation_plan(variant, root, iteration, job)
        output = Path(plan['output'])
        receipt = CONTROL/f'{job}-iteration{iteration}-audit.json'
        # A receipt is trusted only after rechecking all saved task artifacts.
        if receipt.exists():
            audit_rollouts(output/'rollouts', Path(plan['source']))
            continue
        if output.exists():
            # Reuse a finished run whose controller stopped before its receipt.
            # An incomplete run needs explicit recovery, never a blind rerun.
            audit = audit_rollouts(output/'rollouts', Path(plan['source']))
        else:
            write_json(CONTROL/f'{job}-active.json', dict(stage='evaluation',
                iteration=iteration, variant=variant, output=str(output),
                updated_epoch=time.time()))
            write_json(CONTROL/f'{job}-iteration{iteration}-plan.json', plan)
            os.environ.update(OPENWEBRL_EXPECTED_SCHEDULER_OFFSET='0',
                              OPENWEBRL_RECORD_EVAL_EXPECTED_TASK_COUNT='300')
            try:
                evaluator.run(plan, REPO/'.env')
                audit = audit_rollouts(output/'rollouts', Path(plan['source']))
            except BaseException as exc:
                write_json(CONTROL/f'{job}-active.json', dict(stage='failed',
                    iteration=iteration, output=str(output),
                    error_type=type(exc).__name__, updated_epoch=time.time()))
                raise
        write_json(receipt, audit)
        item.update(state='complete', audit=str(receipt))
        write_json(request_path, request)
        write_json(CONTROL/f'{job}-active.json', dict(stage='complete',
            iteration=iteration, output=str(output), updated_epoch=time.time()))
