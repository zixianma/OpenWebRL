#!/usr/bin/env python3
"""Prepare an isolated baseline source that finishes a due eval before resuming."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil

from resume_baseline import validate_source


HELPER = '''
def _finish_pending_evaluation(args, rollout_manager):
    # OPENWEBRL_RESUME_PENDING_EVAL_V1: explicit recovery, not a new eval schedule.
    value = os.environ.get("OPENWEBRL_PENDING_EVAL_ITERATION")
    if not value:
        return
    iteration = int(value)
    if (iteration <= 0 or iteration != args.start_rollout_id
            or args.eval_interval is None or iteration % args.eval_interval):
        raise ValueError("Pending evaluation does not match the restored checkpoint boundary")
    append_progress_log(args, f"[ResumePendingEvaluation] iteration={iteration} status=started")
    _ray_get_with_actor_retry(
        rollout_manager.eval.remote(iteration - 1),
        label=f"rollout_manager.eval({iteration - 1}) pending from previous allocation",
    )
    if args.save_debug_rollout_data:
        eval_path = args.save_debug_rollout_data.format(rollout_id=f"eval_{iteration - 1}")
        evict_file_cache(eval_path, sync=True)
    evict_file_backed_cache()
    append_progress_log(args, f"[ResumePendingEvaluation] iteration={iteration} status=completed")

'''


def upgrade_train(text):
    if 'OPENWEBRL_RESUME_PENDING_EVAL_V1' in text:
        raise ValueError('Source already supports pending evaluation recovery')
    anchor = '    # train loop.\n'
    invocation = '    try:\n        for rollout_id in range(args.start_rollout_id, args.num_rollout):'
    if text.count(anchor) != 1 or text.count(invocation) != 1:
        raise ValueError('Unexpected preserved training loop')
    text = text.replace('logger = logging.getLogger(__name__)\n',
                        'logger = logging.getLogger(__name__)\n' + HELPER, 1)
    text = text.replace(invocation,
                        '    try:\n        _finish_pending_evaluation(args, rollout_manager)\n'
                        '        for rollout_id in range(args.start_rollout_id, args.num_rollout):', 1)
    ast.parse(text)
    return text


def prepare(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.exists() or destination.is_relative_to(source):
        raise ValueError('Use a new destination outside the existing source tree')
    validate_source(source)
    upgraded = upgrade_train((source / 'train.py').read_text())
    shutil.copytree(source, destination, symlinks=True,
                    ignore=shutil.ignore_patterns('__pycache__', '.git', '.env*', 'wandb', 'checkpoints', 'outputs'))
    (destination / 'train.py').write_text(upgraded)
    manifest_path = destination / 'reference_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['pending_evaluation_recovery'] = {
        'source': str(source), 'train_sha256': hashlib.sha256(upgraded.encode()).hexdigest(),
        'recipe_files_unchanged': True, 'gpu_recovery_verified': False,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    validate_source(destination)
    return manifest['pending_evaluation_recovery']


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(prepare(args.source, args.output), indent=2))
