#!/usr/bin/env python3
"""Full saved-batch/32K/save/reload test inside each requeued B/C allocation."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from resume_baseline import REPO,RUNTIME,source_command,clean_environment,validate_source,write_json
from prepare_arm_gate_dp import SOURCES,prepare_sources,replace_once

VALIDATION_SOURCES={v:RUNTIME/f'reference-arm-gate-{v.lower()}-dp-validation-20260922-v1' for v in SOURCES}


def prepare_source(parent,source):
    from prepare_arm_turn_bonus import copy_plain
    changed={}
    for name in ('train.py','slime/ray/rollout.py'):
        s=(parent/name).read_text()
        s=s.replace('from openwebrl.arm_turn_bonus_cycles import before_collection, before_training, after_checkpoint',
                    'from openwebrl.arm_dp_validation import before_collection, before_training, after_checkpoint')
        s=s.replace('from openwebrl.arm_turn_bonus_runtime import before_optimizer',
                    'from openwebrl.arm_dp_validation import before_optimizer')
        s=s.replace('from openwebrl.arm_turn_bonus_cycles import reset_collection',
                    'from openwebrl.arm_dp_validation import reset_collection')
        s=s.replace('from openwebrl.arm_turn_bonus import post_process_rewards',
                    'from openwebrl.arm_dp_validation import post_process_rewards')
        if name=='slime/ray/rollout.py':
            s=replace_once(s,'        return train_data\n',
                '        from openwebrl.arm_dp_validation import trim_train_data\n        return trim_train_data(train_data)\n')
        changed[name]=s
    name='openwebrl/arm_failure_aux.py';s=(parent/name).read_text()
    s=replace_once(s,'    return [AuxiliaryIterator(mixed, micro_batch_size=1)], new_counts',
        '    from openwebrl.arm_dp_validation import add_stress_padding\n'
        '    mixed,new_counts=add_stress_padding(mixed,new_counts,aux)\n'
        '    return [AuxiliaryIterator(mixed, micro_batch_size=1)], new_counts')
    changed[name]=s
    name='slime/backends/megatron_utils/model.py'
    changed[name]=(parent/name).read_text()+'\nfrom openwebrl.arm_dp_validation import measure_step\ntrain_one_step=measure_step(train_one_step)\n'
    for name in ('openwebrl/arm_dp_validation.py','openwebrl/arm_tpdp_diagnostic.py'):
        changed[name]=(REPO/name).read_text()
    if not source.exists():
        shutil.copytree(parent,source,symlinks=True,copy_function=copy_plain,
            ignore=shutil.ignore_patterns('__pycache__','*.pyc','.git','.browser_use_sessions'))
        manifest=json.loads((parent/'reference_manifest.json').read_text())
        for name,s in changed.items():
            compile(s,name,'exec');(source/name).write_text(s)
            manifest['recipe_files_sha256'][name]=hashlib.sha256(s.encode()).hexdigest()
        manifest['production']=False
        write_json(source/'reference_manifest.json',manifest)
    validate_source(source)
    if any((source/k).read_text()!=v for k,v in changed.items()):raise ValueError('Validation source changed')


def prepare():
    prepare_sources()
    for variant,parent in SOURCES.items():
        prepare_source(parent,VALIDATION_SOURCES[variant])


def execute(job,variant):
    import prepare_arm_tpdp_test as prior
    from inspect_training_checkpoint import inspect_checkpoint
    prepare();source=VALIDATION_SOURCES[variant]
    out=RUNTIME/f'benchmarks/arm-gate-{variant.lower()}-tp2-full-{job}'
    out.mkdir(parents=True,exist_ok=False)
    # Reuse the pinned B23 checkpoint and complete saved B24 collection.
    prior.SOURCE=source
    p=prior.case_plan(2,out)
    p['environment']['WANDB_RUN_ID']=f'arm-{variant.lower()}-tp2-full-{job}'
    p['command'][p['command'].index('--wandb-group')+1]='ARM full-batch TP2 validation with zero-loss32K probe'
    current=prior.ORIGIN/'iterations/0023';fixture=out/'fixture';fixture.mkdir()
    manifest=json.loads((current/'failure_auxiliary.json').read_text())
    (fixture/manifest['tensor_file']).symlink_to(current/manifest['tensor_file'])
    write_json(fixture/'failure_auxiliary.json',manifest)
    config=json.loads((current/'arm-config.json').read_text())
    config.update(output=str(out),run_output=str(out),diagnostic_only=True,
        diagnostic_saved_applied_beta=json.loads((current/'calibration.json').read_text())['applied_beta'])
    write_json(out/'arm-config.json',config)
    expected_steps=json.loads((current/'training_gate.json').read_text())['expected_optimizer_updates']
    p.update(production=False,optimizer_updates=expected_steps,auxiliary_rows=len(manifest['records']),
        expected_final_adam_updates=322+expected_steps,validation_stress_context=32768)
    write_json(out/'plan.json',p)
    env=dict(clean_environment(),**p['environment'])
    from dotenv import dotenv_values
    for k,v in dotenv_values(REPO/'.env').items():
        if v and k.startswith('WANDB_'):env.setdefault(k,v)
    start=time.monotonic()
    with (out/'train.log').open('w') as log:
        subprocess.run(source_command(source,['timeout','--signal=TERM','--kill-after=30','1500',*p['command']]),
            cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    report=inspect_checkpoint(out/'runtime',expected_updates=p['expected_final_adam_updates'])
    write_json(out/'checkpoint-validation.json',report)
    restore=out/'restore';restore.mkdir()
    argv,restore_env=prior.restore_command(p,env,restore)
    with (out/'restore.log').open('w') as log:
        subprocess.run(source_command(source,['timeout','--signal=TERM','--kill-after=20','300',*argv]),
            cwd=source,env=restore_env,stdout=log,stderr=subprocess.STDOUT,check=True)
    receipt=json.loads((restore/'resume_verification.json').read_text())
    if receipt['full_model_and_optimizer_load']!='passed' or receipt['loaded_iteration']!=23 or receipt['optimizer_updates_executed']:
        raise ValueError('Full DP validation reload failed')
    rows=[]
    for rank in range(8):
        updates=[json.loads(x) for x in (out/f'updates-rank{rank}.jsonl').read_text().splitlines()]
        if len(updates)!=expected_steps:raise ValueError('Wrong full-batch update count')
        rows.extend(updates)
    result=dict(passed=True,production_source=str(SOURCES[variant]),batch=json.loads((out/'batch.json').read_text()),
        optimizer_updates=expected_steps,restore=receipt,wall_seconds=time.monotonic()-start,
        peak_allocated_gib=max(r['peak_allocated_bytes'] for r in rows)/1024**3,
        seconds_per_update_rank0=[json.loads(x)['seconds'] for x in (out/'updates-rank0.jsonl').read_text().splitlines()],
        stress='One zero-loss32K auxiliary row per DP rank in the first update of each PPO epoch',
        training_lineage_modified=False)
    write_json(out/'result.json',result)
    print(json.dumps(result),flush=True)
