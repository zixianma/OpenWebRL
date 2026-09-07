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
    parser.add_argument('--profile', choices=['small', 'paper', 'reference'], default='small')
    parser.add_argument('--rollouts', type=int)
    parser.add_argument('--steps', type=int, default=15)
    parser.add_argument('--wandb-mode', choices=['online', 'offline'], default='online')
    args = parser.parse_args()
    paper = args.profile in {'paper', 'reference'}
    reference = args.profile == 'reference'
    if reference and not (REPO / 'reference_manifest.json').is_file():
        raise SystemExit('Use scripts/prepare_reference_baseline.py to isolate the reference recipe first.')
    args.rollouts = (90 if paper else 30) if args.rollouts is None else args.rollouts
    groups, batch, context, concurrency = (48, 256, 32768, 16) if paper else (4, 16, 16384, 4)
    if args.rollouts < 1 or args.steps < 1:
        parser.error('--rollouts and --steps must be positive')
    env = os.environ.copy()
    if args.env_file.exists():
        for key, value in dotenv_values(args.env_file).items():
            if value and key.startswith(('JUDGE_', 'WANDB_', 'OPENAI_', 'AZURE_')):
                env.setdefault(key, value)
    missing = []
    if env.get('OPENAI_API_KEY') and not any(env.get(k) for k in ['JUDGE_API_MODE', 'JUDGE_API_BASE', 'OPENAI_API_BASE', 'AZURE_RESOURCE_NAME']):
        env['JUDGE_API_MODE'] = 'served'
        env['JUDGE_API_BASE'] = 'https://api.openai.com/v1'
    mode = env.get('JUDGE_API_MODE', 'served')
    needed = {'served': ['JUDGE_API_BASE'], 'token': ['AZURE_RESOURCE_NAME', 'AZURE_TOKEN_PATH'],
              'api_key': ['OPENAI_API_BASE', 'OPENAI_API_KEY']}
    if mode not in needed:
        raise SystemExit(f'Unsupported judge mode: {mode}')
    missing.extend(k for k in needed[mode] if not env.get(k))
    if args.wandb_mode == 'online' and not env.get('WANDB_API_KEY'):
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
    name = f'openwebrl-4b-{args.profile}-{job or "dryrun"}-{time.strftime("%Y%m%dT%H%M%S", time.gmtime())}'
    out = RUNTIME / 'runs' / name
    env.update(NUM_GPUS='2', TP_SIZE='2', NUM_ROLLOUT=str(args.rollouts),
               BROWSER_MAX_STEPS=str(args.steps), ROLLOUT_BATCH_SIZE=str(groups), N_SAMPLES='5',
               GLOBAL_BATCH_SIZE=str(batch), CONTEXT_LEN=str(context), RESPONSE_LEN='1024',
               BROWSER_CONCURRENCY=str(concurrency), SGLANG_CONCURRENCY=str(48 if paper else 4),
               LEARNING_RATE='1e-6' if paper else '5e-7', RECOMPUTE_ACTIVATIONS='1' if paper else '0', SAVE_INTERVAL='1' if paper else '5', SAVE_DIR=str(out),
               WANDB_MODE=args.wandb_mode)
    # A baseline profile always starts from SFT; never inherit pilot resume settings.
    for key in ['SLIME_LOAD_CHECKPOINT', 'SLIME_CKPT_STEP', 'OVERRIDE_OPT_PARAM_SCHEDULER']:
        env.pop(key, None)
    command = ['timeout', '--signal=INT', '--kill-after=120', str(seconds),
               'bash', str(REPO / 'scripts/run_h200_browser.sh'),
               '--use-wandb', '--wandb-mode', args.wandb_mode, '--wandb-project', env.get('WANDB_PROJECT', 'openwebrl'),
               '--wandb-group', name, '--disable-wandb-random-suffix', '--wandb-dir', str(out / 'wandb'),
               '--sglang-disable-cuda-graph', '--eval-interval', '5' if paper else '10',
               '--eval-prompt-data', 'webvoyager-val' if paper else 'webvoyager-smoke',
               str(REPO / 'openwebrl/data/webvoyager_val.parquet') + ('' if paper else '@[:8]'),
               '--n-samples-per-eval-prompt', '1', '--eval-temperature', '0', '--eval-top-p', '1', '--eval-top-k', '1', '--eval-max-response-len', '1024']
    if reference:
        # Replace the small-run eval block with full Online-Mind2Web monitoring.
        command = command[:command.index('--eval-interval')]
        command += ['--eval-interval', '10', '--eval-config', str(REPO / 'openwebrl/online_mind2web_monitor.yaml'),
                    '--rollout-health-check-first-wait', '180', '--use-fault-tolerance']
        env['SLIME_ADAPTIVE_QUERY_BLACKLIST_PATH'] = str(REPO / 'reference_empty_blacklist.txt')
        env['SLIME_BROWSER_QUERY_BLACKLIST_PATH'] = env['SLIME_ADAPTIVE_QUERY_BLACKLIST_PATH']
    if env.get('WANDB_ENTITY'):
        command += ['--wandb-team', env['WANDB_ENTITY']]
    manifest = {'kind': f'real-web {args.profile} GRPO baseline', 'job_id': job, 'gpus': 2,
                'maximum_seconds': seconds, 'rollouts': args.rollouts, 'max_browser_steps': args.steps,
                'prompt_groups': groups, 'trajectories_per_group': 5, 'global_batch_size': batch,
                'context_tokens': context, 'ppo_epochs': 2, 'save_interval': 1 if paper else 5,
                'wandb_project': env.get('WANDB_PROJECT', 'openwebrl'), 'wandb_mode': args.wandb_mode,
                'judge_model': env.get('JUDGE_MODEL', 'gpt-4.1'), 'run_name': name,
                'output': str(out), 'missing_configuration': missing,
                'learning_rate': env['LEARNING_RATE'], 'browser_concurrency': concurrency,
                'fresh_sft_start': True, 'paper_reference': 'https://arxiv.org/pdf/2606.02031',
                'paper_differences': ['TP2 H200 rather than TP4 B200',
                    'Local browsers, concurrency 16 rather than Kubernetes sandboxes',
                    'Activation recomputation; own runtime; strict invalid-judge masking',
                    'Checkpoint every iteration to preserve progress within the current allocation',
                    'Online validation uses released 70-task WebVoyager split at training step limit; not official benchmark evaluation',
                    'Only stage 1 (90 x 15) requested here; stage 2 is 50 x 30 after completion'] if paper else []}
    if reference:
        manifest['recipe'] = json.loads((REPO / 'reference_manifest.json').read_text())
        manifest['evaluation'] = {'dataset': 'Online-Mind2Web', 'tasks': 300,
            'interval': 10, 'initial': True, 'max_steps': 30, 'max_response_tokens': 4096,
            'temperature': 0, 'judge': 'gpt-4.1', 'official_score': False}
        manifest['paper_differences'] = [
            'Two H200 GPUs, TP2, 16 local browsers; paper TP4 and 80-100 training sandboxes',
            'Full activation recomputation and SDPA vision attention in our own runtime',
            'Standard GPU optimizer rather than precision-aware CPU-offloaded launcher optimizer',
            'SGLang CUDA graphs disabled, smaller KV allocation and prefill chunks',
            'Synchronous checkpoint every iteration; bounded by existing allocation',
            'Local browser request/exit behavior differs from Kubernetes',
            'Online-Mind2Web deterministic GPT-4.1 monitoring, not official o4-mini score',
            'Live websites and unpinned GPT-4.1 endpoint may differ from paper dates']
    print(json.dumps(manifest, indent=2), flush=True)
    if args.dry_run:
        return
    out.mkdir(parents=True, exist_ok=False)
    (out / 'launch_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    import hashlib
    import shutil
    snapshot = out / 'run_config' / 'source'
    source_manifest = {}
    for folder in ['scripts', 'slime', 'slime_plugins', 'openwebrl']:
        for source in (REPO / folder).rglob('*'):
            if source.is_file() and source.suffix in {'.py', '.sh', '.yaml'} and not source.is_symlink():
                relative = source.relative_to(REPO)
                if any(part in {'data', 'outputs', '__pycache__', '.env', 'checkpoints'} for part in relative.parts):
                    continue
                target = snapshot / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
                source_manifest[str(relative)] = hashlib.sha256(source.read_bytes()).hexdigest()
    shutil.copy2(REPO / 'train.py', snapshot / 'train.py')
    source_manifest['train.py'] = hashlib.sha256((REPO / 'train.py').read_bytes()).hexdigest()
    (out / 'run_config' / 'source_manifest.json').write_text(json.dumps(source_manifest, indent=2) + '\n')
    with (out / 'training.log').open('w') as log:
        result = subprocess.run(command, env=env, cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
    (out / 'exit_status.json').write_text(json.dumps({'exit_code': result.returncode}) + '\n')
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
