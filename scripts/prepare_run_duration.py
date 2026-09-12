#!/usr/bin/env python3
"""Copy a preserved baseline with a new launcher time cap; never allocate compute."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

from resume_baseline import RUNTIME, validate_source, write_json


def duration_launcher(text, hours):
    if type(hours) is not int or not 1 <= hours <= 24:
        raise ValueError('Runtime cap must be an integer between 1 and 24 hours')
    old = 'seconds = 15 * 60 if args.verify_resume_only else 8 * 3600'
    deadline = 'seconds = min(seconds, int(end - time.time()) - (30 if args.verify_resume_only else 180))'
    if text.count(old) != 1 or text.count(deadline) != 1:
        raise ValueError('Expected the preserved eight-hour launcher with allocation deadline guard')
    return text.replace(old, f'seconds = 15 * 60 if args.verify_resume_only else {hours} * 3600', 1)


def prepare(source, output, hours):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists() or not output.is_relative_to(RUNTIME) or output.is_relative_to(source):
        raise ValueError('Use a new directory under runtime storage')
    validate_source(source)
    relative = 'scripts/run_small_baseline.py'
    original = (source / relative).read_text()
    changed = duration_launcher(original, hours)
    manifest = json.loads((source / 'reference_manifest.json').read_text())
    if relative in manifest['recipe_files_sha256']:
        raise ValueError('Launcher unexpectedly belongs to protected recipe files')
    shutil.copytree(source, output, symlinks=True,
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.git', '.env*',
                                                 'wandb', 'checkpoints', 'outputs'))
    (output / relative).write_text(changed)
    manifest['duration_preparation'] = {
        'source': str(source), 'maximum_training_hours': hours,
        'verification_cap_seconds': 900, 'training_shutdown_margin_seconds': 180,
        'allocation_deadline_still_enforced': True, 'changed_files': [relative],
        'parent_launcher_sha256': hashlib.sha256(original.encode()).hexdigest(),
        'launcher_sha256': hashlib.sha256(changed.encode()).hexdigest(),
        'recipe_files_changed': [], 'creates_or_extends_allocation': False,
    }
    write_json(output / 'reference_manifest.json', manifest)
    validate_source(output)
    return manifest['duration_preparation']


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--hours', type=int, required=True)
    a = p.parse_args()
    print(json.dumps(prepare(a.source, a.output, a.hours), indent=2))
