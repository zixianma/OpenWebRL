"""Prepare and run the frozen Sol 6.1 coordinate-browser API studies.

No Slurm controller, local model server, or model weights are used. Preparation
restores exact source and public task IDs. Execution keeps the original browser,
policy/history, judge and disposition loop with caller-supplied spending caps.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from tasks import dataset_rows, normalize

ASSETS = Path(__file__).resolve().parents[1] / 'portable_assets'
LEGACY = ASSETS / 'legacy'
CAMPAIGNS = tuple(sorted(p.name for p in LEGACY.iterdir() if (p / 'plan.json').is_file()))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def read_spec(campaign):
    if campaign not in CAMPAIGNS:
        raise ValueError('Choose one of: ' + ', '.join(CAMPAIGNS))
    return json.loads((LEGACY / campaign / 'plan.json').read_text())


def validate_tokenizer_source(path):
    manifest = json.loads((LEGACY / 'tokenizer-source-manifest.json').read_text())
    root = Path(path).resolve()
    for name, expected in manifest['files'].items():
        source = root / name
        if not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest() != expected:
            raise ValueError('Pinned tokenizer/processor/chat-template identity differs: ' + name)
    return manifest


def restore_source(campaign, output):
    manifest = json.loads((LEGACY / campaign / 'source-manifest.json').read_text())
    checked = []
    for filename, entry in manifest['files'].items():
        target = output / filename
        source = (ASSETS / entry['published_path']).resolve()
        if not source.is_relative_to(ASSETS.resolve()) or not target.resolve().is_relative_to(output.resolve()):
            raise ValueError('Source manifest path escapes its root')
        data = source.read_bytes()
        if hashlib.sha256(data).hexdigest() != entry['sha256']:
            raise ValueError('Frozen source hash mismatch: ' + filename)
        checked.append((target, data))
    for target, data in checked:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    return manifest


def prepare(campaign, output, *, tokenizer_source=None, dependencies=None):
    """Restore a new local workspace without model calls or browser launches."""
    spec = read_spec(campaign)
    dependency_record = json.loads(Path(dependencies).read_text()) if dependencies else {}
    selected_tokenizer = str(Path(tokenizer_source).resolve()) if tokenizer_source else dependency_record.get('tokenizer_source')
    tokenizer_manifest = validate_tokenizer_source(selected_tokenizer) if selected_tokenizer else json.loads((LEGACY / 'tokenizer-source-manifest.json').read_text())
    output = Path(output).resolve()
    if output.is_relative_to(Path('/gpfs/home/zixianma')):
        raise ValueError('Choose a project or temporary output directory')
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    manifest = restore_source(campaign, output / 'source')
    roster = json.loads((LEGACY / campaign / 'task-roster.json').read_text())
    datasets = {}
    rows = []
    for identity in roster:
        bench = identity['benchmark']
        if bench not in datasets:
            datasets[bench] = dataset_rows(bench)
        ident = identity['task_id']
        if ident not in datasets[bench]:
            raise ValueError('Original cohort ID missing from public source: ' + ident)
        task = normalize(datasets[bench][ident], bench)
        rows.append({'id': ident, 'metadata': {'task_id': ident, 'intent': task['intent'], 'start_url': task['start_url']}})
    (output / 'public-tasks.jsonl').write_text(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in rows))
    (output / 'scientific-plan.json').write_text(json.dumps(spec, indent=2) + '\n')
    # Task injection changes only the local task-file reference, never prompts/tools.
    import yaml
    env_path = output / 'source/openwebrl/env/config.yaml'
    env = yaml.safe_load(env_path.read_text())
    env['path_to_task_file'] = str(output / 'public-tasks.jsonl')
    env_path.write_text(yaml.safe_dump(env, sort_keys=False))
    record = {'schema': 'portable-coordinate-runtime-v1', 'campaign': campaign,
              'scientific_plan_sha256': digest(spec), 'source_manifest': manifest,
              'tokenizer_source': selected_tokenizer,
              'tokenizer_files_sha256': tokenizer_manifest['files'],
              'browser_cache': dependency_record.get('browser_cache'),
              'overlays': ['Local task-file path in env/config.yaml', 'GenerateState-only import instead of GPU-serving imports',
                           'Run-level resource/cap validation replaces original Slurm approval checks']}
    (output / 'portable-runtime.json').write_text(json.dumps(record, indent=2) + '\n')
    return record


def execute(args):
    root = args.runtime.resolve()
    record = json.loads((root / 'portable-runtime.json').read_text())
    spec = read_spec(record['campaign'])
    if digest(spec) != record['scientific_plan_sha256']:
        raise ValueError('Scientific plan changed')
    count = min(args.limit, len(spec['task_ids']))
    modes = sorted({mode for row in spec['schedule'] for mode in row['modes']})
    if args.arm and args.arm not in modes:
        raise ValueError('Arm must be one of: ' + ', '.join(modes))
    summary = {'campaign': record['campaign'], 'tasks': count, 'arms': [args.arm] if args.arm else modes,
               'actor': spec['protocol']['actor'], 'judge': spec['protocol']['judge'], 'execute': args.execute,
               'model_usd': args.model_usd, 'judge_usd': args.judge_usd}
    if record.get('tokenizer_source'):
        validate_tokenizer_source(record['tokenizer_source'])
    if not args.execute:
        return summary
    if args.model_usd is None or args.judge_usd is None or args.model_usd <= 0 or args.judge_usd <= 0:
        raise ValueError('Execution requires positive --model-usd and --judge-usd caps')
    if not os.environ.get('OPENAI_API_KEY'):
        raise ValueError('OPENAI_API_KEY is required')
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    output = args.output.resolve()
    if output.is_relative_to(Path('/gpfs/home/zixianma')):
        raise ValueError('Choose a project or temporary output directory')
    output.mkdir(parents=True)
    config = dict(summary, runtime=str(root), output=str(output), limit=count, arm=args.arm,
                  model_calls=args.model_calls, judge_calls=args.judge_calls, browser_starts=args.browser_starts or count * len(summary['arms']),
                  tokenizer_source=record['tokenizer_source'], browser_port=args.browser_port)
    path = output / 'portable-execution.json'
    path.write_text(json.dumps(config, indent=2) + '\n')
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=str(root / 'source'), WANDB_PROJECT='openwebrl-evals',
               WANDB_MODE=args.wandb_mode, WANDB_DIR=str(output), WANDB_CACHE_DIR=str(output / 'wandb-cache'),
               WANDB_CONFIG_DIR=str(output / 'wandb-config'), XDG_CACHE_HOME=str(output / 'cache'),
               HF_HOME=str(output / 'huggingface-cache'), TOKENIZERS_PARALLELISM='false', CUDA_VISIBLE_DEVICES='')
    if record.get('browser_cache'):
        env['PLAYWRIGHT_BROWSERS_PATH'] = record['browser_cache']
    subprocess.run([sys.executable, str(Path(__file__).with_name('legacy_runtime.py')), '--config', str(path)], env=env,
                   cwd=root / 'source', check=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--campaign', required=True, choices=CAMPAIGNS)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--dependencies', type=Path, help='Installer dependencies.json, supplies the installed Playwright browser cache')
    p.add_argument('--tokenizer-source', type=Path, help='Optional pinned local Qwen tokenizer/processor metadata directory')
    p = sub.add_parser('verify')
    p.add_argument('--runtime', required=True, type=Path)
    p.add_argument('--browser-preflight', action='store_true', help='Zero-model local browser screenshot/action/cleanup check')
    p = sub.add_parser('run')
    p.add_argument('--runtime', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--limit', type=int, default=1)
    p.add_argument('--arm', help='Default: both policies in the frozen within-task order')
    p.add_argument('--execute', action='store_true')
    p.add_argument('--model-usd', type=float)
    p.add_argument('--judge-usd', type=float)
    p.add_argument('--model-calls', type=int, default=100000)
    p.add_argument('--judge-calls', type=int, default=1000)
    p.add_argument('--browser-starts', type=int)
    p.add_argument('--browser-port', type=int, default=23000)
    p.add_argument('--wandb-mode', choices=['offline', 'online', 'disabled'], default='offline')
    args = parser.parse_args()
    try:
        if args.command == 'prepare':
            result = prepare(args.campaign, args.output, tokenizer_source=args.tokenizer_source, dependencies=args.dependencies)
            print(json.dumps({'campaign': result['campaign'], 'runtime': str(args.output), 'model_calls': 0}))
        elif args.command == 'verify':
            import tempfile
            runtime = args.runtime.resolve()
            record = json.loads((runtime / 'portable-runtime.json').read_text())
            with tempfile.TemporaryDirectory(prefix='api-legacy-verify-') as temporary:
                cfg = Path(temporary) / 'config.json'
                cfg.write_text(json.dumps({'runtime': str(runtime)}))
                command = [sys.executable, str(Path(__file__).with_name('legacy_runtime.py')), '--config', str(cfg), '--verify-imports']
                if args.browser_preflight:
                    command.append('--browser-preflight')
                env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=str(runtime / 'source'), XDG_CACHE_HOME=temporary, HF_HOME=temporary)
                if record.get('browser_cache'):
                    env['PLAYWRIGHT_BROWSERS_PATH'] = record['browser_cache']
                subprocess.run(command, env=env, cwd=runtime / 'source', check=True)
        else:
            if args.limit < 1 or args.model_calls < 1 or args.judge_calls < 1:
                raise ValueError('Task and call caps must be positive')
            print(json.dumps(execute(args)))
    except (ValueError, FileExistsError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
