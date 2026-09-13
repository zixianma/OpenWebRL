#!/usr/bin/env python3
"""Publish durable Sol evaluation results to W&B without modifying the worker."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

from monitor_arm_turn_bonus import REPO, RUNTIME, TERMINAL, read, scheduler


def snapshot(root):
    rows = {}
    for path in sorted(root.glob('*/results/*.json')):
        row = read(path)
        if not row:
            continue
        task = row['task_id']
        if task in rows:
            raise ValueError('Duplicate task across evaluation segments')
        rows[task] = row
    valid = [r for r in rows.values() if r.get('valid')]
    successes = sum(r.get('reward') == 1 for r in valid)
    status = read(root / 'status.json')
    usage = read(root / 'sol-api/usage.json')
    metrics = {
        'progress/completed_tasks': len(rows),
        'progress/scheduled_tasks': 300,
        'progress/stage': status.get('stage', 'waiting'),
        'eval/valid_tasks': len(valid),
        'eval/invalid_tasks': len(rows) - len(valid),
        'eval/successes': successes,
        'eval/success_rate_all_scheduled': successes / 300,
        'eval/complete': bool(status.get('complete')),
        'eval/failed': bool(status.get('failed')),
    }
    if rows:
        metrics['eval/success_rate_completed'] = successes / len(rows)
    if valid:
        metrics['eval/success_rate_valid'] = successes / len(valid)
    for key in ('requests', 'accounted_cost_usd', 'consecutive_failures'):
        if key in usage:
            metrics['selector/' + key] = usage[key]
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job-id', required=True)
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    if not args.job_id.isdigit():
        raise ValueError('Expected numeric Slurm job ID')
    root = RUNTIME / f'evaluations/sol-selection300-{args.job_id}'
    # The controller owns this directory; the logger must not pre-create it.
    while not (root / 'launch_manifest.json').exists():
        if not args.watch or scheduler(args.job_id)['state'].split()[0].rstrip('+') in TERMINAL:
            raise RuntimeError('Evaluation did not create a launch manifest')
        time.sleep(30)
    from dotenv import dotenv_values
    for key, value in dotenv_values(REPO / '.env').items():
        if value and key.startswith('WANDB_'):
            os.environ[key] = value
    os.environ['WANDB_PROJECT'] = 'openwebrl-evals'
    for key, dirname in (('WANDB_CACHE_DIR', 'cache'), ('WANDB_CONFIG_DIR', 'config'), ('WANDB_DATA_DIR', 'data')):
        os.environ[key] = str(root / 'wandb-sync' / dirname)
    import wandb
    manifest = read(root / 'launch_manifest.json')
    run = wandb.init(
        entity=os.environ.get('WANDB_ENTITY', 'zixianma'), project='openwebrl-evals',
        id=f'sol-selection300-{args.job_id}', name=f'Sol best-of-five · 300 tasks · {args.job_id}',
        group='sol-selection300', job_type='evaluation', resume='allow', dir=str(root),
        settings=wandb.Settings(x_disable_stats=True, disable_code=True, disable_git=True),
        config={k: manifest[k] for k in ('job_id', 'selector', 'selector_reasoning_effort', 'protocol', 'requested_resources')},
    )
    run.define_metric('progress/completed_tasks')
    run.define_metric('eval/*', step_metric='progress/completed_tasks')
    receipt = root / 'wandb_sync.json'
    last = None
    try:
        while True:
            metrics = snapshot(root)
            slurm = scheduler(args.job_id)
            metrics['progress/slurm_state'] = slurm['state']
            terminal = slurm['state'].split()[0].rstrip('+') in TERMINAL
            fingerprint = hashlib.sha256(json.dumps(metrics, sort_keys=True).encode()).hexdigest()
            if fingerprint != last:
                run.log(metrics)
                run.summary.update(metrics)
                receipt.write_text(json.dumps(dict(url=run.url, updated_at=time.time(), metrics=metrics), indent=2))
                print(json.dumps(metrics), flush=True)
                last = fingerprint
            if terminal or metrics['eval/complete'] or metrics['eval/failed'] or not args.watch:
                break
            time.sleep(30)
        run.finish(exit_code=1 if metrics['eval/failed'] or (terminal and not metrics['eval/complete']) else 0)
    except BaseException:
        run.finish(exit_code=1)
        raise


if __name__ == '__main__':
    main()
