#!/usr/bin/env python3
"""Backfill collected progress for the moved calibration control run."""
import argparse
import json
import os
from pathlib import Path
import re


def progress_rows(text):
    expression = re.compile(r'\[RolloutProgress\] event=\w+ rollout_id=(\d+) accepted_groups=(\d+)/(\d+) completed_groups=(\d+) pending_groups=(\d+) remaining_batch_size=\d+ samples=(\d+) elapsed_secs=([\d.]+)')
    rows = []
    for match in expression.finditer(text):
        iteration, accepted, target, completed, pending, trajectories, elapsed = match.groups()
        rows.append({'arm_collection/elapsed_seconds':float(elapsed),
            'arm_collection/accepted_groups':int(accepted), 'arm_collection/target_groups':int(target),
            'arm_collection/completed_groups':int(completed), 'arm_collection/pending_groups':int(pending),
            'arm_collection/accepted_trajectories':int(trajectories),
            'arm_collection/admitted_turns':0, 'arm_collection/selector_requests':0,
            'arm_collection/optimizer_updates':0})
    return rows


def main():
    from dotenv import dotenv_values
    import wandb
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-root',type=Path,required=True)
    p.add_argument('--run-id',required=True)
    args = p.parse_args()
    os.environ['WANDB_API_KEY'] = dotenv_values(Path(__file__).resolve().parents[1]/'.env')['WANDB_API_KEY']
    os.environ['WANDB_CACHE_DIR'] = '/gpfs/scrubbed/zixianma/openwebrl-runtime/wandb-cache'
    root = args.run_root/'wandb-repair'
    root.mkdir(exist_ok=True)
    rows = progress_rows((args.run_root/'collection.log').read_text(errors='replace'))
    if not rows:
        raise ValueError('No durable collection progress to backfill')
    with wandb.init(entity='zixianma',project='openwebrl',id=args.run_id,resume='must',
        name='ARM calibration control (filter bug) | after70 | job291983',
        group='executed-turn-bonus-calibration',dir=str(root),
        settings=wandb.Settings(mode='online',init_timeout=60)) as run:
        run.define_metric('arm_collection/elapsed_seconds')
        run.define_metric('arm_collection/*',step_metric='arm_collection/elapsed_seconds')
        run.config.update({'actual_stage':'q=0 control due empty-intent exclusion bug',
            'actual_arm_scored_turns':0,'optimizer_updates':0,'repaired_attempt':'arm-turn-bonus-calibration-after70-291983-r2'},allow_val_change=True)
        for row in rows:
            run.log(row)
        run.summary.update({'instrumentation_bug':'Empty evaluation intent incorrectly matched missing training intent',
            'calibration_valid':False,'native_batch_preserved':True,'backfilled_progress_rows':len(rows),
            'corrected_run_url':'https://wandb.ai/zixianma/openwebrl/runs/arm-turn-bonus-calibration-after70-291983-r2'})
        receipt={'url':run.url,'backfilled_rows':len(rows),'last_progress':rows[-1]}
        (args.run_root/'wandb-repair-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
        print(json.dumps(receipt),flush=True)


if __name__ == '__main__':
    main()
