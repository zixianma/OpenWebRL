"""Launch a bounded, W&B-logged real-web baseline in an existing Slurm allocation.

This never submits or extends an allocation. Credentials are read from .env into
the child environment, never passed as CLI arguments or saved in run manifests.
"""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import time

from dotenv import dotenv_values

REPO = Path(__file__).resolve().parents[1]
RUNTIME = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--env-file', type=Path, default=REPO / '.env')
    parser.add_argument('--rollouts', type=int, default=30)
    parser.add_argument('--steps', type=int, default=15)
    args = parser.parse_args()
    if args.rollouts < 1 or args.steps < 1:
        parser.error('--rollouts and --steps must be positive')
    env = os.environ.copy()
    if args.env_file.exists():
        for key, value in dotenv_values(args.env_file).items():
            if value and key.startswith(('JUDGE_', 'WANDB_', 'OPENAI_', 'AZURE_')):
                env.setdefault(key, value)
    missing = []
    mode = env.get('JUDGE_API_MODE', 'served')
    needed = {'served': ['JUDGE_API_BASE'], 'token': ['AZURE_RESOURCE_NAME', 'AZURE_TOKEN_PATH'],
              'api_key': ['OPENAI_API_BASE', 'OPENAI_API_KEY']}
    if mode not in needed:
        raise SystemExit(f'Unsupported judge mode: {mode}')
    missing.extend(k for k in needed[mode] if not env.get(k))
    if not env.get('WANDB_API_KEY'):
        missing.append('WANDB_API_KEY (or a saved W&B login)')
    job = env.get('SLURM_JOB_ID')
    if not args.dry_run and not job:
        raise SystemExit('An existing authorized Slurm allocation is required.')
    if not args.dry_run and missing:
        # A saved W&B login is accepted; do not search credential stores directly.
        if missing == ['WANDB_API_KEY (or a saved W&B login)']:
            import wandb
            try:
                if wandb.Api(timeout=10).viewer:
                    missing.clear()
            except Exception:
                pass
        if missing:
            raise SystemExit('Configure before launch: ' + ', '.join(missing))
    # Default seven hours, additionally bounded by the allocation's actual end.
    seconds = 7 * 3600
    if not args.dry_run:
        info = subprocess.check_output(['scontrol', 'show', 'job', job, '-o'], text=True)
        match = re.search(r'EndTime=(\S+)', info)
        if not match:
            raise SystemExit('Cannot determine allocation end time.')
        end = time.mktime(time.strptime(match.group(1), '%Y-%m-%dT%H:%M:%S'))
        seconds = min(seconds, int(end - time.time()) - 300)
        if seconds < 600:
            raise SystemExit('Less than ten minutes remain after the shutdown margin.')
    name = f'openwebrl-4b-small-{job or "dryrun"}-{time.strftime("%Y%m%dT%H%M%S", time.gmtime())}'
    out = RUNTIME / 'runs' / name
    env.update(NUM_GPUS='2', TP_SIZE='2', NUM_ROLLOUT=str(args.rollouts),
               BROWSER_MAX_STEPS=str(args.steps), ROLLOUT_BATCH_SIZE='4', N_SAMPLES='5',
               GLOBAL_BATCH_SIZE='16', CONTEXT_LEN='16384', RESPONSE_LEN='1024',
               BROWSER_CONCURRENCY='4', SAVE_INTERVAL='5', SAVE_DIR=str(out),
               WANDB_MODE='online')
    command = ['timeout', '--signal=INT', '--kill-after=120', str(seconds),
               'bash', str(REPO / 'scripts/run_h200_browser.sh'),
               '--use-wandb', '--wandb-mode', 'online', '--wandb-project', env.get('WANDB_PROJECT', 'openwebrl'),
               '--wandb-group', name, '--disable-wandb-random-suffix', '--wandb-dir', str(out / 'wandb'),
               '--sglang-disable-cuda-graph', '--eval-interval', '10',
               '--eval-prompt-data', 'webvoyager-smoke', str(REPO / 'openwebrl/data/webvoyager_val.parquet') + '@[:8]',
               '--n-samples-per-eval-prompt', '1', '--eval-temperature', '0', '--eval-max-response-len', '1024']
    if env.get('WANDB_ENTITY'):
        command += ['--wandb-team', env['WANDB_ENTITY']]
    manifest = {'kind': 'real-web small GRPO baseline', 'job_id': job, 'gpus': 2,
                'maximum_seconds': seconds, 'rollouts': args.rollouts, 'max_browser_steps': args.steps,
                'prompt_groups': 4, 'trajectories_per_group': 5, 'global_batch_size': 16,
                'context_tokens': 16384, 'ppo_epochs': 2, 'save_interval': 5,
                'wandb_project': env.get('WANDB_PROJECT', 'openwebrl'), 'run_name': name,
                'output': str(out), 'missing_configuration': missing}
    print(json.dumps(manifest, indent=2), flush=True)
    if args.dry_run:
        return
    out.mkdir(parents=True, exist_ok=False)
    (out / 'launch_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    with (out / 'training.log').open('w') as log:
        result = subprocess.run(command, env=env, cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
    (out / 'exit_status.json').write_text(json.dumps({'exit_code': result.returncode}) + '\n')
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
