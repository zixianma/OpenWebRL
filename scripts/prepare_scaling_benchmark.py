#!/usr/bin/env python3
"""Prepare isolated eight-H200 benchmarking support without submitting compute."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil

from resume_baseline import RUNTIME, validate_source, write_json

BROWSERS = (64, 96, 128, 192, 256)
TOPOLOGIES = (4, 2, 8, 1)


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f'Unsupported source: expected one occurrence of {old!r}')
    return text.replace(old, new, 1)


def launcher(text):
    changes = {
        "choices=[2, 4], default=2)": "choices=[2, 4, 8], default=2)\n    parser.add_argument('--tensor-parallel', type=int, choices=[1, 2, 4, 8], default=4)\n    parser.add_argument('--browser-concurrency', type=int, choices=[64, 96, 128, 192, 256], default=64)",
        'args = parser.parse_args()': "args = parser.parse_args()\n    if args.gpus != 8 or args.gpus % args.tensor_parallel:\n        parser.error('Benchmark requires eight GPUs and a tensor parallel divisor of eight')",
        '(48, 256, 32768, 8 * args.gpus)': '(48, 256, 32768, args.browser_concurrency)',
        'TP_SIZE=str(args.gpus)': 'TP_SIZE=str(args.tensor_parallel)',
        "'tensor_parallel_size': args.gpus, 'data_parallel_size': 1": "'tensor_parallel_size': args.tensor_parallel, 'data_parallel_size': args.gpus // args.tensor_parallel",
        'f\'{args.gpus} H200 GPUs, TP{args.gpus},': 'f\'{args.gpus} H200 GPUs, TP{args.tensor_parallel},',
        "out = RUNTIME / 'runs' / name": "out = Path(os.environ.get('OPENWEBRL_BENCHMARK_OUTPUT', str(RUNTIME / 'runs' / name)))",
    }
    for old, new in changes.items():
        text = replace_once(text, old, new)
    # Both gate and pool must change; inherited training settings must not select a different case.
    text = replace_once(text, "env = os.environ.copy()", "env = os.environ.copy()\n    env['BROWSER_TRAIN_CONFIG'] = str(REPO / 'benchmark_configs' / f'browsers-{args.browser_concurrency}.yaml')")
    ast.parse(text)
    return text


def prepare(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists() or not output.is_relative_to(RUNTIME) or output.is_relative_to(source):
        raise ValueError('Use a new directory under runtime storage')
    validate_source(source)
    changed = launcher((source / 'scripts/run_small_baseline.py').read_text())
    wrapper = (source / 'scripts/resume_baseline.py').read_text()
    wrapper = replace_once(wrapper, 'requested_gpus not in (2, 4)', 'requested_gpus not in (2, 4, 8)')
    wrapper = wrapper.replace('Supported profiles use 2 or 4 H200 GPUs.', 'Benchmark supports 2, 4 or 8 H200 GPUs.')
    shell = (source / 'scripts/run_h200_browser.sh').read_text()
    shell = replace_once(shell, '--tensor-model-parallel-size "$TP_SIZE" --sequence-parallel', '--tensor-model-parallel-size "$TP_SIZE"')
    shell = replace_once(shell, 'if [[ "${RECOMPUTE_ACTIVATIONS:-0}" == 1 ]]', 'if (( TP_SIZE > 1 )); then ARGS+=(--sequence-parallel); fi\nif [[ "${RECOMPUTE_ACTIVATIONS:-0}" == 1 ]]')
    config = (source / 'openwebrl/browser_training_config.yaml').read_text()
    if config.count('browser_rollout_concurrency: 32') != 1:
        raise ValueError('Expected the validated 32-browser source')
    shutil.copytree(source, output, symlinks=True,
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.git', '.env*', 'wandb', 'checkpoints', 'outputs'))
    (output / 'scripts/run_small_baseline.py').write_text(changed)
    (output / 'scripts/resume_baseline.py').write_text(wrapper)
    (output / 'scripts/run_h200_browser.sh').write_text(shell)
    (output / 'benchmark_configs').mkdir()
    for count in BROWSERS:
        (output / f'benchmark_configs/browsers-{count}.yaml').write_text(config.replace('browser_rollout_concurrency: 32', f'browser_rollout_concurrency: {count}', 1))
    manifest = json.loads((output / 'reference_manifest.json').read_text())
    manifest['scaling_benchmark'] = {'parent': str(source), 'gpus': 8, 'cpus': 64, 'memory_gib': 960,
        'tensor_parallel_candidates': list(TOPOLOGIES), 'browser_candidates': list(BROWSERS),
        'training_recipe_files_unchanged': True,
        'runtime_changes': ['GPU/TP/DP selection', 'browser task gate and pool', 'sequence parallel disabled for TP1'],
        'benchmark_only': True, 'gpu_verified': False,
        'changed_scripts_sha256': {p: hashlib.sha256((output / p).read_bytes()).hexdigest() for p in
            ['scripts/run_small_baseline.py', 'scripts/resume_baseline.py', 'scripts/run_h200_browser.sh']},
        'browser_configs_sha256': {f'benchmark_configs/browsers-{n}.yaml':
            hashlib.sha256((output / f'benchmark_configs/browsers-{n}.yaml').read_bytes()).hexdigest()
            for n in BROWSERS}}
    write_json(output / 'reference_manifest.json', manifest)
    validate_source(output)
    return manifest['scaling_benchmark']


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(prepare(a.source, a.output), indent=2))
