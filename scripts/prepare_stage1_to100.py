#!/usr/bin/env python3
"""Freeze runtime-only continuation changes; preserve both training recipes."""
import hashlib
import json
from pathlib import Path
import shutil

from prepare_arm_turn_bonus import copy_plain, replace_once
from resume_baseline import RUNTIME, REPO, validate_source, write_json
from runtime_ports import patch_rollout_ports

BASELINE_PARENT=RUNTIME/'reference-stage1-browsers32-20260911'
BASELINE_SOURCE=RUNTIME/'reference-stage1-to100-20260922-24h-v3'
BASELINE_PENDING=RUNTIME/'reference-stage1-to100-pending-eval-20260922-24h-v3'
ADDITIVE_PARENT=RUNTIME/'reference-arm-failure-additive-20260914-v3'
ADDITIVE_SOURCE=RUNTIME/'reference-arm-additive-to100-20260921-v1'
CONTROL=RUNTIME/'arm-turn-bonus-preparation/stage1-to100-20260921'


def expected_files(parent, baseline):
    files={'slime/ray/rollout.py':patch_rollout_ports((parent/'slime/ray/rollout.py').read_text()).encode()}
    if baseline:
        text=(parent/'scripts/run_small_baseline.py').read_text()
        text=replace_once(text,'args.rollouts = (90 if paper else 30)', 'args.rollouts = (100 if paper else 30)')
        text=replace_once(text,'seconds = 15 * 60 if args.verify_resume_only else 8 * 3600',
                         'seconds = 15 * 60 if args.verify_resume_only else 24 * 3600')
        # Extending the collection target changes Megatron's inferred schedule
        # length. Resume the saved scheduler; never reset or override it.
        text=replace_once(text,"    command += ['--router-balance-abs-threshold', '2']",
            "    if args.resume_from:\n"
            "        command += ['--use-checkpoint-opt-param-scheduler']\n"
            "    command += ['--router-balance-abs-threshold', '2']")
        text=replace_once(text,"env['OPENWEBRL_MULTIMODAL_STORAGE_DIR'] = f'/tmp/openwebrl-{job}-multimodal'",
            "env['OPENWEBRL_MULTIMODAL_STORAGE_DIR'] = str(RUNTIME/'multimodal-scratch'/f'baseline-to100-{job}')\n"
            "    env['OPENWEBRL_EVAL_ROLLOUT_DIR'] = str(out/'evaluation/after100/rollouts')")
        files['scripts/run_small_baseline.py']=text.encode()
        files['openwebrl/eval_monitor.py']=(RUNTIME/'reference-arm-eval80-additive-20260920-v2/openwebrl/eval_monitor.py').read_bytes()
    return files


def freeze(parent,source,baseline):
    validate_source(parent);files=expected_files(parent,baseline)
    if not source.exists():
        shutil.copytree(parent,source,symlinks=True,copy_function=copy_plain,
            ignore=shutil.ignore_patterns('__pycache__','*.pyc','.git','.browser_use_sessions'))
        for name,data in files.items():(source/name).write_bytes(data)
        manifest=json.loads((source/'reference_manifest.json').read_text())
        manifest['recipe_files_sha256'].update({k:hashlib.sha256(v).hexdigest() for k,v in files.items()})
        manifest['runtime_port_leases']=True
        manifest['stage1_to100']=dict(parent=str(parent),changed_files=list(files),training_objective_unchanged=True)
        write_json(source/'reference_manifest.json',manifest)
    validate_source(source)
    hashes=json.loads((source/'reference_manifest.json').read_text())['recipe_files_sha256']
    old=json.loads((parent/'reference_manifest.json').read_text())['recipe_files_sha256']
    if any(hashes.get(k)!=v for k,v in old.items() if k not in files):
        raise ValueError('Unplanned training source change')
    if any((source/k).read_bytes()!=v for k,v in files.items()):
        raise ValueError('Frozen continuation source changed')


def prepare_sources():
    freeze(BASELINE_PARENT,BASELINE_SOURCE,True)
    freeze(ADDITIVE_PARENT,ADDITIVE_SOURCE,False)
    if not BASELINE_PENDING.exists():
        from prepare_resume_pending_eval import prepare
        prepare(BASELINE_SOURCE,BASELINE_PENDING)
    validate_source(BASELINE_PENDING)
    CONTROL.mkdir(parents=True,exist_ok=True)
    write_json(CONTROL/'sources.json',dict(baseline=str(BASELINE_SOURCE),baseline_pending_eval=str(BASELINE_PENDING),
        additive=str(ADDITIVE_SOURCE),changes='Target/runtime/storage/ports/eval persistence only; objectives and optimizer hyperparameters unchanged'))


if __name__=='__main__':
    prepare_sources();print(CONTROL/'sources.json')
