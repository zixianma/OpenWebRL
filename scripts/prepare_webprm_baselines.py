#!/usr/bin/env python3
"""Prepare private OM2W WebPRM pilot manifest without network or paid calls."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from openwebrl.jev_eval import load_tasks, write_json
from openwebrl.webprm_baselines import prepare_plan


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task-file', type=Path, default=REPO/'openwebrl/data/eval/online-mind2web.jsonl')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--pilot-tasks', type=int, default=12)
    args = p.parse_args()
    output = args.output.resolve()
    if output.is_relative_to(REPO):
        raise ValueError('Keep task IDs and task payloads outside the repository')
    tasks = load_tasks(args.task_file)
    plan = prepare_plan(tasks, hashlib.sha256(args.task_file.read_bytes()).hexdigest(), args.pilot_tasks)
    output.mkdir(parents=True, exist_ok=True)
    target = output/'plan.json'
    if target.exists() and json.loads(target.read_text()) != plan:
        raise ValueError('Existing plan differs; use a new output directory')
    write_json(target, plan)
    write_json(output/'tasks.json', tasks)
    print(json.dumps(dict(plan=str(target), status=plan['status'], tasks=plan['task_count'],
                         pilot_episodes=plan['pilot_episodes'], proposal=plan['proposal']), indent=2))


if __name__ == '__main__':
    main()
