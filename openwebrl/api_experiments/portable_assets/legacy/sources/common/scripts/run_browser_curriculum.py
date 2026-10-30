#!/usr/bin/env python3
"""Print the 90x15 -> 50x30 curriculum; --execute runs the GPU launcher."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess


def stages(output):
    root = Path(output).resolve()
    return [
        {'BROWSER_MAX_STEPS': '15', 'NUM_ROLLOUT': '90', 'SLIME_SAVE_DIR': str(root / 'stage1'),
         'SLIME_LOAD_CHECKPOINT': '', 'SLIME_CKPT_STEP': ''},
        {'BROWSER_MAX_STEPS': '30', 'NUM_ROLLOUT': '140', 'SLIME_SAVE_DIR': str(root / 'stage2'),
         'SLIME_LOAD_CHECKPOINT': str(root / 'stage1'), 'SLIME_CKPT_STEP': '',
         'OVERRIDE_OPT_PARAM_SCHEDULER': '1'},
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--launcher', default='scripts/run_browser_Qwen3VL_4B_Instruct.sh')
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    launcher = repo / args.launcher
    plan = stages(args.output)
    for stage in plan:
        print(shlex.join(['env', *[f'{k}={v}' for k, v in stage.items()], 'bash', str(launcher)]), flush=True)
    if not args.execute:
        return
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if any(Path(stage['SLIME_SAVE_DIR']).exists() for stage in plan):
        raise SystemExit('Choose a fresh output directory; an existing curriculum stage would be overwritten.')
    (root / 'curriculum.json').write_text(json.dumps(plan, indent=2) + '\n')
    for i, stage in enumerate(plan):
        if i:
            checkpoint = Path(stage['SLIME_LOAD_CHECKPOINT']) / 'latest_checkpointed_iteration.txt'
            if not checkpoint.is_file():
                raise SystemExit(f'Stage 1 did not produce {checkpoint}')
            expected = int(plan[0]['NUM_ROLLOUT']) - 1
            try:
                actual = int(checkpoint.read_text().strip())
            except ValueError:
                raise SystemExit(f'Invalid checkpoint marker: {checkpoint}')
            if actual != expected or not (checkpoint.parent / f'iter_{actual:07d}').is_dir():
                raise SystemExit(f'Stage 1 must finish and save rollout {expected}; found marker {actual}.')
        subprocess.run(['bash', str(launcher)], cwd=repo, env={**os.environ, **stage}, check=True)


if __name__ == '__main__':
    main()
