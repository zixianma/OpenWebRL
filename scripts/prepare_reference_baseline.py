"""Create an isolated RL reference source tree without changing experimental code.

Launch its run_small_baseline.py --profile reference --env-file <private .env>.
No GPU work or allocation submission happens here.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

REPO = Path(__file__).resolve().parents[1]
REFERENCE = '9a12094'
RECIPE_FILES = [
    'openwebrl/reward_browser.py',
    'slime/rollout/filter_hub/dynamic_sampling_filters.py',
    'slime/rollout/data_source.py',
]


def prepare(destination):
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    for folder in ['scripts', 'slime', 'slime_plugins', 'openwebrl']:
        for source in (REPO / folder).rglob('*'):
            if source.is_file() and not source.is_symlink() and source.suffix in {'.py', '.sh', '.yaml', '.yml', '.json', '.jinja', '.jinja2'}:
                relative = source.relative_to(REPO)
                if any(p in {'data', 'outputs', 'checkpoints', '__pycache__', 'wandb'} for p in relative.parts):
                    continue
                target = destination / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
    shutil.copy2(REPO / 'train.py', destination / 'train.py')
    (destination / 'openwebrl/data').symlink_to(REPO / 'openwebrl/data', target_is_directory=True)
    hashes = {}
    for name in RECIPE_FILES:
        data = subprocess.check_output(['git', 'show', f'{REFERENCE}:{name}'], cwd=REPO)
        (destination / name).write_bytes(data)
        hashes[name] = hashlib.sha256(data).hexdigest()
    # HEAD has original normalization plus committed telemetry; the experimental
    # invalid-trajectory normalization patch remains only in the working tree.
    name = 'slime/ray/rollout.py'
    data = subprocess.check_output(['git', 'show', f'HEAD:{name}'], cwd=REPO)
    assert b'invalid_ids =' not in data
    (destination / name).write_bytes(data)
    hashes[name] = hashlib.sha256(data).hexdigest()
    # Restore generation semantics, retain only runtime robustness fixes:
    # corrected exception logging and an empty-trajectory return guard.
    name = 'openwebrl/generate_browser.py'
    data = subprocess.check_output(['git', 'show', f'{REFERENCE}:{name}'], cwd=REPO).decode()
    data = data.replace('logger.opt(depth=0).warning("Task {}: generate_turn_sample failed: {}", task_id, e, exc_info=True)',
                        'logger.opt(depth=0, exception=True).warning("Task {}: generate_turn_sample failed: {}", task_id, e)')
    current = (REPO / name).read_text()
    start = current.index('        if not turn_samples:\n            # A prompt rejected before inference')
    end = current.index('        if not terminated:', start)
    guard = current[start:end]
    anchor = '        # --- Finalization ---\n'
    prefix, turn_impl = data.split('async def _generate_turn_sample_impl(', 1)
    assert turn_impl.count(anchor) == 1
    data = prefix + 'async def _generate_turn_sample_impl(' + turn_impl.replace(anchor, anchor + guard)
    (destination / name).write_text(data)
    hashes[name] = hashlib.sha256(data.encode()).hexdigest()
    (destination / 'reference_empty_blacklist.txt').write_text('# Reference baseline: no added host exclusions.\n')
    manifest = {'reference_commit': subprocess.check_output(['git', 'rev-parse', REFERENCE], cwd=REPO,text=True).strip(),
        'recipe_files_sha256': hashes,
        'normalization_source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'],cwd=REPO,text=True).strip(),
        'task_sampling': 'All released 2102 tasks, shuffled; no added host blacklist',
        'reward': 'Released judge prompt/parser and dynamic filter; original group normalization',
        'retained_generation_fixes': ['exception logging', 'empty trajectory guard'],
        'experimental_features': False}
    (destination / 'reference_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(destination)
    return manifest


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args=parser.parse_args()
    prepare(args.output)
