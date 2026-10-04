#!/usr/bin/env python3
"""Freeze the count-only continuation fix; no submission or GPU work."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

from prepare_arm_turn_bonus import copy_plain
from resume_baseline import validate_source, write_json
from run_arm_turn_bonus_cycles import REPO, RUNTIME

PARENT = RUNTIME/'reference-arm-failure-bonus-20260913-v2'
SOURCE = RUNTIME/'reference-arm-failure-bonus-20260919-countguard-v1'
PREPARATION = RUNTIME/'arm-turn-bonus-preparation/all-failure-to100-20260919'
CHANGED = ('openwebrl/arm_failure_bonus.py', 'openwebrl/arm_continuation_guard.py')


def expected_files():
    old = (PARENT/CHANGED[0]).read_text()
    if old.count('    return decision\n') != 1:
        raise ValueError('Unexpected parent calibration implementation')
    patched = old.replace('    return decision\n',
        '    from openwebrl.arm_continuation_guard import apply_adjacent_count_support\n'
        '    return apply_adjacent_count_support(report, decision)\n')
    return {CHANGED[0]: patched.encode(), CHANGED[1]: (REPO/CHANGED[1]).read_bytes()}


def validate_frozen_source():
    validate_source(PARENT)
    validate_source(SOURCE)
    parent = json.loads((PARENT/'reference_manifest.json').read_text())['recipe_files_sha256']
    frozen = json.loads((SOURCE/'reference_manifest.json').read_text())['recipe_files_sha256']
    if set(frozen) != set(parent) | set(CHANGED):
        raise ValueError('Unexpected continuation source inventory')
    for relative, digest in parent.items():
        if relative not in CHANGED and frozen[relative] != digest:
            raise ValueError('Continuation changed a file outside the count guard')
    for relative, data in expected_files().items():
        if (SOURCE/relative).read_bytes() != data:
            raise ValueError('Continuation count guard changed after preparation')


def prepare():
    validate_source(PARENT)
    if SOURCE.exists():
        validate_frozen_source()
        return
    expected = expected_files()
    shutil.copytree(PARENT, SOURCE, symlinks=True, copy_function=copy_plain,
        ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.git', '.browser_use_sessions'))
    for relative, data in expected.items():
        (SOURCE/relative).write_bytes(data)
    manifest = json.loads((SOURCE/'reference_manifest.json').read_text())
    hashes = {relative: hashlib.sha256(data).hexdigest() for relative, data in expected.items()}
    manifest['recipe_files_sha256'].update(hashes)
    manifest['continuation_label_guard'] = dict(parent_source=str(PARENT), changed_files_sha256=hashes,
        mode='adjacent_batch_v1', gpu_validated=False, reward_recipe_unchanged=True)
    write_json(SOURCE/'reference_manifest.json', manifest)
    validate_frozen_source()


if __name__ == '__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    prepare()
    print(SOURCE)
