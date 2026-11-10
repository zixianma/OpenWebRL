#!/usr/bin/env python3
"""Prepare the unchanged expanded-data continuation; never submit or train.

The existing recovery and its original budget remain untouched. Native launcher
dry runs validate the proposed continuation before additional budget approval.
"""
import json
from pathlib import Path
import shlex
import statistics
import subprocess
import time

import prepare_expanded_baseline as original
from resume_baseline import write_json

CONTROL = original.CONTROL / 'to90-20261005'
TARGET = 90
MILESTONES = tuple(range(30, TARGET + 1, 10))


def training_plan(target):
    if target not in MILESTONES:
        raise ValueError('Unknown continuation milestone')
    plan = original.training_command(60, resume=True)
    plan['target_iteration'] = target
    plan['environment'].update(
        NUM_ROLLOUT=str(TARGET),
        OPENWEBRL_STOP_AFTER_SAVED_ROLLOUT=str(target - 1),
        WANDB_RESUME='must',
    )
    # --use-checkpoint-opt-param-scheduler restores the original constant-LR
    # scheduler and its counter; do not reset it when extending the loop bound.
    return plan


def verify_native(plan):
    env = original.clean_env()
    env.update(plan['environment'], DRY_RUN='1')
    output = subprocess.check_output(plan['command'], env=env,
                                     cwd=original.SOURCE, text=True)
    argv = shlex.split(output)

    def value(key):
        if argv.count(key) != 1:
            raise ValueError(f'Expected one {key}')
        return argv[argv.index(key) + 1]

    expected = {
        '--num-rollout': '90', '--tensor-model-parallel-size': '2',
        '--actor-num-gpus-per-node': '8', '--global-batch-size': '256',
        '--micro-batch-size': '1', '--ppo-epochs': '2', '--lr': '1e-6',
        '--lr-decay-style': 'constant', '--load': str(original.RUN),
        '--prompt-data': str(original.CONTROL / 'combined-tasks.parquet'),
        '--judge-prompt-variant': 'action_history',
        '--custom-rm-path': 'openwebrl.reward_browser.reward_func',
        '--custom-generate-function-path': 'openwebrl.generate_browser.generate_turn_sample',
    }
    for key, expected_value in expected.items():
        if value(key) != expected_value:
            raise ValueError(f'Scientific setting changed: {key}')
    if '--use-checkpoint-opt-param-scheduler' not in argv:
        raise ValueError('Must restore the existing optimizer/scheduler')
    forbidden = ('--finetune', '--no-load-optim', '--override-opt-param-scheduler')
    if any(flag in argv for flag in forbidden) or any('arm_' in arg for arg in argv):
        raise ValueError('Unexpected scientific/reset flag')
    if not value('--dynamic-sampling-filter-path').endswith('.check_reward_nonempty_nonzero_std'):
        raise ValueError('Outcome-only group filter changed')
    if plan['environment']['OPENWEBRL_STOP_AFTER_SAVED_ROLLOUT'] != str(plan['target_iteration'] - 1):
        raise ValueError('Wrong checkpoint stop boundary')
    return argv


def prepare():
    CONTROL.mkdir(exist_ok=True)
    timings = []
    for iteration in range(11, 21):
        report = json.loads((original.CONTROL / f'iteration{iteration}-throughput-review.json').read_text())
        elapsed = report.get('full_iteration_seconds', report.get('generate_to_checkpoint_complete_seconds'))
        timings.append(elapsed / 60)
    judge = json.loads((original.CONTROL / 'judge-budget/ledger.json').read_text())
    completed = int((original.RUN / 'latest_checkpointed_iteration.txt').read_text()) + 1
    plans = [training_plan(i) for i in MILESTONES]
    native = [verify_native(p) for p in plans]
    proposal = dict(
        prepared_epoch=time.time(), approved=False, submitted=False,
        target_iteration=TARGET, latest_durable_iteration=completed,
        full300_evaluations=list(MILESTONES), already_verified_evaluations=[10, 20],
        resources=dict(gpus=8, gpu_type='H200', cpus=64, memory_gib=960,
                       additional_hours=96, additional_gpu_hours=768,
                       max_hours_per_allocation=24, blocks_hours=[24, 24, 24, 24]),
        additional_judge_cap_usd=400, proposed_lifetime_judge_cap_usd=600,
        original_judge_cap_usd=200, prior_judge_charged_or_reserved_usd=judge['charged_or_reserved_usd'],
        original_approval_unchanged=True, includes_all_retries_and_evaluations=True,
        release_unused_time_after_verified_endpoint=True,
        scientific_change='Only extend the stopping iteration; unchanged expanded4102 outcome-only recipe.',
        optimizer_scheduler_cursor='Restore unchanged, including any verified pending-batch cursor advance.',
        wandb_run_id=original.RUN_ID,
        data_sha256=original.digest(original.CONTROL / 'combined-tasks.parquet'),
        current_recovery_job='344274',
        launch_order='Current recovery owns the lineage first; successors must verify it has exited and hold the same writer lock.',
        independent_work='Preparation and independent experiments need not wait for this recovery.',
        estimate=dict(
            observed_iterations=list(range(11, 21)),
            observed_minutes_per_iteration=timings,
            observed_mean_minutes=statistics.mean(timings),
            observed_median_minutes=statistics.median(timings),
            remaining_training_hours_at_observed_rate=(TARGET - completed)*statistics.mean(timings)/60,
            evaluation_and_handoff_hours_allowance=7,
            faster_node_runtime_hours_inference=[50, 65],
            caveat='Observed timings are from thermally throttled g011; healthy-node speedup is not yet measured for this lineage.',
            exclude_nodes=['g011'],
        ),
        validation=dict(native_launcher_dry_runs_passed=True,
                        prepared_stage_count=len(plans), gpu_resume_validation_pending=True,
                        allocation_controller_extension_pending=True,
                        judge_ledger_extension_pending=True),
    )
    write_json(CONTROL / 'proposal.json', proposal)
    write_json(CONTROL / 'training-plans.json', dict(plans=plans, native_argv=native))
    print(json.dumps({k: proposal[k] for k in ('target_iteration', 'latest_durable_iteration',
          'resources', 'additional_judge_cap_usd', 'estimate', 'validation')}, indent=2))


if __name__ == '__main__':
    prepare()
