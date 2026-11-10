#!/usr/bin/env python3
"""Freeze operational continuation fixes without touching the queued recovery."""
import json
from pathlib import Path
import shutil
import time

import prepare_expanded_baseline as original
from prepare_arm_turn_bonus import copy_plain
from resume_baseline import validate_source, write_json
from resume_expanded_baseline_to90 import CONTROL, SOURCE

GUARD = '''\n\ndef _expanded_budget_boundary(environ, now):
    deadline = environ.get("OPENWEBRL_TRAINING_DEADLINE_EPOCH")
    if deadline is None:
        return False
    return now + float(environ.get("OPENWEBRL_TRAINING_MIN_CYCLE_SECONDS", "4500")) >= float(deadline)
'''


def prepare():
    validate_source(original.SOURCE)
    if SOURCE.exists():
        validate_source(SOURCE)
        return json.loads((CONTROL/'source-preparation.json').read_text())
    shutil.copytree(original.SOURCE, SOURCE, symlinks=True, copy_function=copy_plain,
        ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.git', '.browser_use_sessions'))
    budget='openwebrl/expanded_baseline_budget.py'
    copy_plain(original.REPO/'openwebrl/expanded_baseline_to90_budget.py', SOURCE/budget)
    p=SOURCE/'train.py';text=p.read_text()
    marker='logger = logging.getLogger(__name__)\n'
    assert text.count(marker)==1
    text=text.replace(marker,marker+GUARD,1)
    marker='        for rollout_id in range(args.start_rollout_id, args.num_rollout):\n'
    assert text.count(marker)==1
    text=text.replace(marker,marker+
        '            if _expanded_budget_boundary(os.environ, time.time()):\n'
        '                _log_train_progress(rollout_id, args.num_rollout, "allocation_boundary_before_collection")\n'
        '                break\n',1)
    p.write_text(text)
    changed=[]
    for p in SOURCE.rglob('*'):
        if not p.is_file() or p.is_symlink(): continue
        name=str(p.relative_to(SOURCE))
        if name=='reference_manifest.json': continue
        if original.digest(p)!=original.digest(original.SOURCE/name): changed.append(name)
        if p.suffix=='.py': compile(p.read_bytes(),str(p),'exec')
    if set(changed)!={budget,'train.py'}:
        raise ValueError('Unexpected changes to the preserved scientific source')
    manifest=json.loads((SOURCE/'reference_manifest.json').read_text())
    manifest['recipe_files_sha256'].update({name:original.digest(SOURCE/name) for name in changed})
    manifest['expanded4102_to90']=dict(parent=str(original.SOURCE),changed_files=changed,
        changes=['Cumulative judge-budget extension; requests unchanged',
                 'Stop at a durable boundary before collecting another batch near allocation end'],
        objective_unchanged=True)
    write_json(SOURCE/'reference_manifest.json',manifest)
    validate_source(SOURCE)
    report=dict(source=str(SOURCE),parent=str(original.SOURCE),changed_files=changed,
                other_files_identical=True,compilation_passed=True,epoch=time.time())
    write_json(CONTROL/'source-preparation.json',report)
    return report


if __name__=='__main__': print(json.dumps(prepare(),indent=2))
