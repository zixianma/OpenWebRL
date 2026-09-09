#!/usr/bin/env python3
"""Clone a preserved baseline and enable TP2/TP4 launch profiles, without compute."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil

from resume_baseline import validate_source


def upgrade_launcher(text):
    if 'OPENWEBRL_RESUME_TOPOLOGY_V1' in text:
        raise ValueError('Launcher already supports topology selection.')
    replacements = {
        "from dotenv import dotenv_values": "from dotenv import dotenv_values\nfrom resume_baseline import allocation\n\nOPENWEBRL_RESUME_TOPOLOGY_V1 = True",
        "    parser.add_argument('--wandb-mode', choices=['online', 'offline'], default='online')": "    parser.add_argument('--wandb-mode', choices=['online', 'offline'], default='online')\n    parser.add_argument('--gpus', type=int, choices=[2, 4], default=2)",
        "    groups, batch, context, concurrency = (48, 256, 32768, 16) if paper else (4, 16, 16384, 4)": "    groups, batch, context, concurrency = (48, 256, 32768, 8 * args.gpus) if paper else (4, 16, 16384, 4)",
        "    seconds = 8 * 3600": "    seconds = 15 * 60 if args.verify_resume_only else 8 * 3600\n    resources = {'cpus': 4 * args.gpus, 'memory_gib': 120 * args.gpus, 'gpus': args.gpus}",
        "        info = subprocess.check_output(['scontrol', 'show', 'job', job, '-o'], text=True)": "        info = subprocess.check_output(['scontrol', 'show', 'job', job, '-o'], text=True)\n        assigned = allocation(info, job, requested_gpus=args.gpus)\n        resources = {'cpus': int(env.get('SLURM_CPUS_PER_TASK', assigned['cpus'])), 'memory_gib': assigned['allocated_memory_gib'], 'gpus': args.gpus}",
        "env.update(NUM_GPUS='2', TP_SIZE='2',": "env.update(NUM_GPUS=str(args.gpus), TP_SIZE=str(args.gpus),",
        "'job_id': job, 'gpus': 2,": "'job_id': job, 'gpus': args.gpus, 'tensor_parallel_size': args.gpus, 'data_parallel_size': 1,",
        "'Two H200 GPUs, TP2, 16 local browsers; paper TP4 and 80-100 training sandboxes'": "f'{args.gpus} H200 GPUs, TP{args.gpus}, {concurrency} local browsers; paper TP4 and 80-100 training sandboxes'",
        "manifest['allocation_resources'] = {'cpus': 8, 'memory_gib': 240, 'gpus': 2}": "manifest['allocation_resources'] = resources\n    manifest['allocation_resources_source'] = 'profile minimum (dry run)' if args.dry_run else 'existing allocation and Slurm step'",
    }
    for old, new in replacements.items():
        if text.count(old) != 1:
            raise ValueError(f'Unsupported preserved launcher; expected one occurrence of {old!r}')
        text = text.replace(old, new, 1)
    ast.parse(text)
    return text


def prepare(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.exists() or destination.is_relative_to(source):
        raise ValueError('Use a new destination outside the existing source tree.')
    validate_source(source)
    upgraded = upgrade_launcher((source / 'scripts/run_small_baseline.py').read_text())
    shutil.copytree(source, destination, symlinks=True,
                    ignore=shutil.ignore_patterns('__pycache__', '.git', '.env*', 'wandb', 'checkpoints', 'outputs'))
    (destination / 'scripts/run_small_baseline.py').write_text(upgraded)
    shutil.copy2(Path(__file__).with_name('resume_baseline.py'), destination / 'scripts/resume_baseline.py')
    manifest_path = destination / 'reference_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['topology_preparation'] = {
        'source': str(source), 'profiles': {'2': {'tp': 2, 'browsers': 16}, '4': {'tp': 4, 'browsers': 32}},
        'launcher_sha256': hashlib.sha256(upgraded.encode()).hexdigest(),
        'resume_wrapper_sha256': hashlib.sha256((destination / 'scripts/resume_baseline.py').read_bytes()).hexdigest(),
        'recipe_files_unchanged': True, 'four_gpu_restore_verified': False,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    validate_source(destination)
    return manifest['topology_preparation']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.output), indent=2))
