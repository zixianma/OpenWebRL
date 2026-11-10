#!/usr/bin/env python3
"""Stage B/C calibration recovery from their preserved, unoptimized first batch.

Never submits Slurm jobs. A replacement allocation and the guard change need
explicit approval; --check is bounded CPU preparation, not GPU validation.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile
from unittest.mock import patch

import run_arm_gate_ablation as gate
from prepare_arm_turn_bonus import copy_plain, replace_once
from resume_baseline import validate_source, write_json

REPO, RUNTIME = gate.REPO, gate.RUNTIME
SOURCE = RUNTIME/'reference-arm-gate-recovery-20260920-v3'
PREPARATION = RUNTIME/'arm-turn-bonus-preparation/gate-recovery-json-20260920'
ORIGINS = {'B': '309053', 'C': '309054'}


def sha(path):
    path = Path(path)
    if path.stat().st_size > 8*1024**2:
        raise ValueError('CPU preparation only hashes small metadata files')
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def prepare():
    validate_source(gate.SOURCE)
    if SOURCE.exists():
        validate_source(SOURCE)
        return
    shutil.copytree(gate.SOURCE, SOURCE, symlinks=True, copy_function=copy_plain,
        ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.git', '.browser_use_sessions'))
    changed = ['openwebrl/arm_gate_recovery.py', 'openwebrl/arm_turn_bonus_runtime.py',
               'openwebrl/arm_failure_additive.py', 'scripts/run_arm_turn_bonus_cycles.py',
               'openwebrl/arm_failure_aux.py', 'slime/ray/rollout.py']
    copy_plain(REPO/changed[0], SOURCE/changed[0])
    path = SOURCE/changed[1]
    text = replace_once(path.read_text(), 'def calibration_decision(report):',
                        'def _original_calibration_decision(report):')
    text += '\n\ndef calibration_decision(report):\n    from openwebrl.arm_gate_recovery import calibration_guard\n    return calibration_guard(report, _original_calibration_decision(report))\n'
    path.write_text(text)
    path = SOURCE/changed[2]
    path.write_text(replace_once(path.read_text(), 'def write_auxiliary(args,samples,current,report):\n',
        'def write_auxiliary(args,samples,current,report):\n    from openwebrl.arm_gate_recovery import replay_auxiliary\n    if replay_auxiliary(args,samples,current,report):\n        return\n'))
    text = replace_once(path.read_text(), "    if path.stat().st_size>2*1024**3:raise ValueError('Auxiliary payload exceeds 2GiB')\n",
        '    from openwebrl.arm_gate_recovery import auxiliary_payload_limit\n    limit=auxiliary_payload_limit(config)\n')
    text = replace_once(text, '        expected_windows=len(samples)//args.global_batch_size,\n',
        '        expected_windows=len(samples)//args.global_batch_size,payload_limit_bytes=limit,\n')
    text += "\n    if path.stat().st_size>limit:raise ValueError(f'Auxiliary payload exceeds {limit//1024**3}GiB')\n"
    path.write_text(text)
    path = SOURCE/'openwebrl/arm_failure_aux.py'
    path.write_text(gate.replace_definitions(path.read_text(),(REPO/'openwebrl/arm_failure_aux.py').read_text(),['prepare_auxiliary']))
    path = SOURCE/'slime/ray/rollout.py'
    text = replace_once(path.read_text(), '''        data = self._convert_samples_to_train_data(data)
        if os.environ.get("OPENWEBRL_ARM_TURN_BONUS_CONFIG"):
            self.data_source.save(rollout_id)
''', '''        if os.environ.get("OPENWEBRL_ARM_TURN_BONUS_CONFIG"):
            self.data_source.save(rollout_id)
        data = self._convert_samples_to_train_data(data)
''')
    path.write_text(text)
    # Separate controller: no running/pending launcher's fingerprint is changed.
    controller = (REPO/changed[3]).read_text()
    controller = replace_once(controller, "    if p.get('gate_ablation'):\n",
        "    if p.get('gate_recovery'):\n        from recover_arm_gate_ablation import validate_plan\n        readiness=validate_plan(p)\n    elif p.get('gate_ablation'):\n")
    (SOURCE/changed[3]).write_text(controller)
    manifest = read(SOURCE/'reference_manifest.json')
    manifest['recipe_files_sha256'].update({f:sha(SOURCE/f) for f in changed})
    manifest['arm_gate_recovery'] = dict(parent=str(gate.SOURCE), changed_files=changed,
        method='fixed beta/q; coverage-audit scale guard; exact replay; min2 4GiB auxiliary cap; early durable cursor',
        gpu_validated=False)
    write_json(SOURCE/'reference_manifest.json', manifest)
    validate_source(SOURCE)


def provenance(variant):
    root = RUNTIME/f'evaluations/arm-failure-additive-{ORIGINS[variant]}'
    status = read(root/'status.json')
    if (not status.get('failed') or not status.get('no_optimizer_updates')
            or status.get('completed_iterations') != 0
            or (root/'runtime/latest_checkpointed_iteration.txt').exists()):
        raise ValueError('Recovery is only for failed zero-update first batches')
    folder = root/'iterations/0000'
    report = read(folder/'calibration.json')
    if report['decision']['checks'] != dict(labels=True, tasks=True, coverage=True, scale=False):
        raise ValueError('Expected only the old scale check to fail')
    config = report['config']
    if (config['candidate_gate'] != 'min2' or config['credit_assignment'] != gate.RULES[variant]
            or config['beta'] != .5 or config['scored_fraction'] != .2
            or report['applied_beta'] != 0 or report['optimizer_updates'] != 0):
        raise ValueError('Unexpected saved recipe or previously optimized batch')
    complete = read(folder/'collection_complete.json')
    if (complete['rollout_id'] != 0 or complete['optimizer_updates'] != 0
            or complete['checkpoint'] != str(gate.training.INITIAL)
            or complete['calibration_rows'] != report['batch_rows']):
        raise ValueError('Saved batch provenance differs from original SFT')
    batch = root/'runtime/rollout_recovery/0.pt'
    cursor = root/'runtime/rollout/global_dataset_state_dict_0.pt'
    if complete['recovery_file'] != str(batch) or complete['dataset_cursor'] != str(cursor):
        raise ValueError('Saved batch or dataset cursor path changed')
    # Read only ZIP central directories, never deserialize the large batch here.
    for path in (batch,cursor):
        with zipfile.ZipFile(path) as archive:
            if not any(n.endswith('/data.pkl') for n in archive.namelist()):
                raise ValueError('Incomplete torch archive')
    auxiliary = folder/'failure_auxiliary.json'
    manifest = read(auxiliary)
    for key in ('policy_id','checkpoint','candidate_gate','credit_assignment'):
        if manifest[key] != config[key]:raise ValueError('Auxiliary provenance changed')
    tensor = folder/'failure_auxiliary.pt'
    return dict(source=str(root), batch=str(batch), batch_bytes=batch.stat().st_size,
        batch_mtime_ns=batch.stat().st_mtime_ns, dataset_cursor=str(cursor), cursor_sha256=sha(cursor),
        auxiliary_manifest_sha256=sha(auxiliary), auxiliary_bytes=tensor.stat().st_size,
        calibration_sha256=sha(folder/'calibration.json'), policy_id=config['policy_id'],
        checkpoint=complete['checkpoint'], run_id=config['run_id'])


def plan(job, variant):
    validate_source(SOURCE)
    origin = provenance(variant)
    p = gate.plan(job, variant)
    p['command'] = [x.replace(str(gate.SOURCE),str(SOURCE)) for x in p['command']]
    p['environment'] = {k:v.replace(str(gate.SOURCE),str(SOURCE)) for k,v in p['environment'].items()}
    p['arm_config'] = {k:v.replace(str(gate.SOURCE),str(SOURCE)) if isinstance(v,str) else v
                       for k,v in p['arm_config'].items()}
    p.update(source=str(SOURCE), gate_recovery=variant, compute_approved=False,
        wandb_run_id=origin['run_id'], recovery_origin=origin)
    p['requested_resources'].update(hours=7, gpu_hours=28)
    p['environment'].update(WANDB_RUN_ID=origin['run_id'], WANDB_RESUME='must',
        OPENWEBRL_REPLAY_FIRST_BATCH=origin['batch'], OPENWEBRL_REPLAY_ROLLOUT_ID='0',
        OPENWEBRL_ARM_REPLAY_CURSOR=origin['dataset_cursor'])
    directory=RUNTIME/'multimodal-scratch'/f'arm-variant-{job}'
    p['environment']['OPENWEBRL_MULTIMODAL_STORAGE_DIR']=str(directory)
    p['multimodal_storage']=dict(mode='shared',directory=str(directory),
        minimum_free_bytes=2*1024**4,retention='preserve; no automatic deletion')
    p['arm_config'].update(run_id=origin['run_id'], policy_id=origin['run_id']+':uninitialized',
        gate_scale_guard='pretraining_coverage_audit_v1', gate_replay_origin=origin,
        replay_origin=origin)
    p['budget_note'] = 'Proposed replacement: 4 H200 x 7h, 32 CPU, 480 GiB; exact approval required before sbatch.'
    p['comparison']['recovery'] = 'First saved batch, exact consumed cursor and auxiliary population; no recollection; fresh Adam at SFT'
    return p


def fingerprint():
    files = [SOURCE/'reference_manifest.json', REPO/'scripts/recover_arm_gate_ablation.py',
        REPO/'scripts/recover_arm_gate_ablation_4gpu.sbatch', REPO/'openwebrl/arm_gate_recovery.py',
        REPO/'scripts/resume_arm_failure_variants.py',
        gate.training.INITIAL/'config.json', gate.training.INITIAL/'tokenizer_config.json']
    return {str(path):sha(path) for path in files}


def validate_plan(p):
    if p != plan(p['job_id'],p['gate_recovery']):raise ValueError('Recovery plan changed')
    receipt = read(PREPARATION/'readiness.json')
    if (not receipt['cpu_passed'] or not receipt['native_parse_passed']
            or receipt['fingerprint'] != fingerprint()):raise ValueError('Recovery readiness stale')
    if p['gate_recovery'] not in receipt['variants']:raise ValueError('Variant not preflighted')
    prepared = read(PREPARATION/p['gate_recovery']/'plan.json')
    if p['recovery_origin'] != prepared['recovery_origin']:
        raise ValueError('Saved batch/cursor/auxiliary metadata changed since preflight')
    return receipt


def check(variants):
    prepare()
    PREPARATION.mkdir(parents=True,exist_ok=True)
    env = dict(os.environ, PYTHONPATH=str(SOURCE), OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    code = "import sys,unittest;sys.path.insert(0,%r);sys.path.insert(1,%r);r=unittest.TextTestRunner(verbosity=2).run(unittest.TestLoader().loadTestsFromNames(['test_arm_gate_recovery','test_arm_min2_gate','test_arm_failure_additive']));raise SystemExit(not r.wasSuccessful())" % (str(REPO/'tests'),str(REPO/'scripts'))
    with (PREPARATION/'frozen-tests.log').open('w') as log:
        subprocess.run([sys.executable,'-c',code],cwd=SOURCE,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=180)
    subprocess.run(['bash','-n',str(REPO/'scripts/recover_arm_gate_ablation_4gpu.sbatch')],check=True)
    import run_arm_failure_additive as additive
    for variant in variants:
        p = plan('PREPARE-'+variant,variant)
        out = PREPARATION/variant;out.mkdir(exist_ok=True)
        write_json(out/'plan.json',p)
        with patch.object(additive,'SOURCE',SOURCE):additive.native_check(p,out)
        from openwebrl.arm_gate_recovery import calibration_guard
        report = read(Path(p['recovery_origin']['source'])/'iterations/0000/calibration.json')
        report['config'].update(gate_scale_guard='pretraining_coverage_audit_v1')
        decision = calibration_guard(report,report['decision'])
        if not decision['passed']:raise ValueError('Proposed guard does not permit the saved batch')
        write_json(out/'replay-calibration-preview.json',decision)
    write_json(PREPARATION/'readiness.json',dict(cpu_passed=True,native_parse_passed=True,
        gpu_validated=False,compute_approved=False,variants=variants,fingerprint=fingerprint()))
    for variant in variants:validate_plan(plan('PREPARE-'+variant,variant))
    return dict(preparation=str(PREPARATION),variants=variants,submitted=False)


def execute(p):
    validate_plan(p)
    if os.environ.get('SLURM_JOB_ID') != p['job_id']:raise ValueError('Use the authorized allocation')
    if os.environ.get('ARM_GATE_RECOVERY_APPROVED') != 'coverage_guard_4gpu_7h':
        raise ValueError('Explicit methodological and exact replacement-budget approval required')
    from resume_arm_failure_variants import check_multimodal_storage
    write_json(RUNTIME/f'logs/arm-variant-storage-{p["job_id"]}.json',check_multimodal_storage(p))
    # Import the isolated copy so pending/running controllers remain untouched.
    spec = importlib.util.spec_from_file_location('gate_recovery_controller',SOURCE/'scripts/run_arm_turn_bonus_cycles.py')
    controller = importlib.util.module_from_spec(spec);spec.loader.exec_module(controller)
    write_json(RUNTIME/f'logs/arm-gate-recovery-{p["job_id"]}.json',p)
    with (RUNTIME/f'logs/arm-gate-recovery-monitor-{p["job_id"]}.log').open('w') as log:
        monitor = subprocess.Popen([sys.executable,str(REPO/'scripts/monitor_arm_turn_bonus.py'),
            '--job-id',p['job_id'],'--watch','--interval','900','--hours','7'],stdout=log,stderr=subprocess.STDOUT)
        try:controller.execute(p)
        finally:
            monitor.terminate()
            try:monitor.wait(timeout=30)
            except subprocess.TimeoutExpired:monitor.kill();monitor.wait()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true')
    parser.add_argument('--variant',choices=ORIGINS,default='B')
    parser.add_argument('--job-id',default='UNAPPROVED')
    parser.add_argument('--execute',action='store_true')
    args = parser.parse_args()
    if args.check:print(json.dumps(check([args.variant]),indent=2))
    else:
        p=plan(args.job_id,args.variant)
        if args.execute:execute(p)
        else:print(json.dumps(p,indent=2))
