"""Mirror measured rewards into paper-scale charts in an existing W&B run."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

from slime.utils.paper_metrics import PAPER_AXIS, PAPER_METRICS, define_paper_metrics, pending_paper_metrics


def main():
    raise SystemExit('Retired: use train/reward with train/reward_iteration; duplicate paper/* logging is disabled.')
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True)
    p.add_argument('--run-directory', required=True, type=Path)
    p.add_argument('--env-file', required=True, type=Path)
    p.add_argument('--deadline', required=True)
    args = p.parse_args()
    deadline = datetime.fromisoformat(args.deadline).timestamp()
    from dotenv import dotenv_values
    for key, value in dotenv_values(args.env_file).items():
        if key.startswith('WANDB') and value:
            os.environ[key] = value
    import wandb
    api = wandb.Api(timeout=20)
    remote = api.run(args.run)
    if remote.config.get('wandb_always_use_train_step', False):
        raise ValueError('This bridge requires zero-based rollout/step; refusing to guess the axis.')
    entity, project, run_id = args.run.split('/')
    run = wandb.init(entity=entity, project=project, id=run_id, dir=str(args.run_directory),
        settings=wandb.Settings(mode='shared', x_primary=False, x_update_finish_state=False,
                               x_label='paper-reward-comparison', x_disable_stats=True))
    define_paper_metrics(run)
    published = set()
    try:
        with (args.run_directory / 'paper_reward_sync.jsonl').open('a', buffering=1) as audit:
            while time.time() < deadline:
                try:
                    history = list(remote.scan_history(page_size=100))
                    for point in pending_paper_metrics(history, published):
                        run.log(point)
                        published.update((point[PAPER_AXIS], key) for key in PAPER_METRICS if key in point)
                        record = {'utc': datetime.now(timezone.utc).isoformat(), 'metrics': point}
                        audit.write(json.dumps(record) + '\n')
                        print(json.dumps(record), flush=True)
                    if (args.run_directory / 'exit_status.json').exists():
                        break
                except Exception as exc:
                    print(f'Paper reward sync retry: {type(exc).__name__}: {exc}', flush=True)
                time.sleep(min(30, max(0, deadline - time.time())))
    finally:
        run.finish()


if __name__ == '__main__':
    main()
