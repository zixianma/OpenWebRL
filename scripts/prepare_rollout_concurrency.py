#!/usr/bin/env python3
"""Prepare a preserved four-GPU baseline with a larger browser task gate."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

from resume_baseline import RUNTIME, validate_source, write_json


def prepare(source, output, concurrency=32):
    source, output = Path(source).resolve(), Path(output).resolve()
    if concurrency not in (24, 32):
        raise ValueError('Use 24 or 32 browsers with the four-GPU pool of 32')
    if output.exists() or not output.is_relative_to(RUNTIME) or output.is_relative_to(source):
        raise ValueError('Use a new source directory under runtime storage')
    validate_source(source)
    name = 'openwebrl/browser_training_config.yaml'
    original = (source / name).read_text()
    old = 'browser_rollout_concurrency: 16'
    if original.count(old) != 1:
        raise ValueError('Expected the preserved 16-browser baseline')
    shutil.copytree(source, output, symlinks=True,
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.git', '.env*', 'wandb', 'checkpoints', 'outputs'))
    (output / name).write_text(original.replace(old, f'browser_rollout_concurrency: {concurrency}', 1))
    # The source-local inner launcher imports allocation() from this module.
    shutil.copyfile(Path(__file__).with_name('resume_baseline.py'), output / 'scripts/resume_baseline.py')
    path = output / 'reference_manifest.json'
    manifest = json.loads(path.read_text())
    before = dict(manifest['recipe_files_sha256'])
    manifest['recipe_files_sha256'][name] = hashlib.sha256((output / name).read_bytes()).hexdigest()
    manifest['rollout_concurrency_preparation'] = {
        'source': str(source), 'required_gpus': 4, 'recommended_cpus': 32,
        'browser_task_concurrency': concurrency, 'browser_pool_capacity': 32,
        'previous_browser_task_concurrency': 16,
        'applies_to': 'training and scheduled local-browser evaluation',
        'changed_recipe_files': [name], 'parent_recipe_files_sha256': before,
        'resume_wrapper_sha256': hashlib.sha256((output / 'scripts/resume_baseline.py').read_bytes()).hexdigest(),
        'unchanged': '48 accepted groups x5 attempts, rewards, GRPO, optimizer, batch size, prompts, decoding, step/time limits, checkpoint/evaluation schedules',
        'caveat': 'Concurrent live-web completion order can change; no claim of bitwise-equivalent trajectories.',
        'gpu_runtime_verified': False,
    }
    write_json(path, manifest)
    validate_source(output)
    return manifest['rollout_concurrency_preparation']


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--concurrency', type=int, default=32)
    a = p.parse_args()
    print(json.dumps(prepare(a.source, a.output, a.concurrency), indent=2))
