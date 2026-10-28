#!/usr/bin/env python3
"""Execute one assigned Online-Mind2Web Table 5-style cell inside its approved allocation.

The owning controller registers the cell with the shared broker and awaits this
process. A cell with a durable result is never rerun (resume returns it). No model
server, allocation or retry is created here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from openwebrl.webprm_lifecycle import EPISODE_WATCHDOG
from openwebrl.webprm_om2w_table5 import load_tasks, run_episode
from openwebrl.webprm_webarena import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--task-id', required=True)
    parser.add_argument('--actor', required=True)
    parser.add_argument('--selector', required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    if args.actor != plan.get('actor_model') or args.selector != plan.get('selector') or args.task_id != plan.get('task_id'):
        parser.error('Assigned cell differs from its frozen worker plan')
    record = load_tasks(plan['task_file'])[args.task_id]
    if digest(record) != plan['task_config_sha256']:
        raise ValueError('Cell task configuration differs from the fixed cohort')
    from scripts.check_webprm_reddit_reset import safe_path, write_new
    output = safe_path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    identity = digest(plan)
    manifest = dict(identity=plan, identity_sha256=identity)
    path = output / 'manifest.json'
    if path.exists():
        if json.loads(path.read_text()) != manifest:
            raise ValueError('Output has a different frozen plan')
    else:
        write_new(path, manifest)
    config = dict(plan, output=str(output), identity_sha256=identity, episode_watchdog=dict(EPISODE_WATCHDOG))
    result = run_episode(record, args.selector, config)
    print(json.dumps(dict(cell=result.get('arm'), task_id=result.get('task_id'), valid=result.get('valid'),
                          success=result.get('success'), error_type=result.get('error_type'))))


if __name__ == '__main__':
    main()
