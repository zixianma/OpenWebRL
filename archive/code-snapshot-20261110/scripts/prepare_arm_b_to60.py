#!/usr/bin/env python3
"""Prepare B49 -> B60 with exact saved-batch replay; never submit Slurm jobs."""
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

from resume_baseline import REPO, RUNTIME, source_command, validate_source, write_json, allocation

ORIGIN = RUNTIME/'evaluations/arm-failure-additive-318934-tp2-iter60'
PARENT = RUNTIME/'reference-arm-gate-b-tp2-20260922-v1'
SOURCE = RUNTIME/'reference-arm-gate-b-resume49-20260924-v1'
CONTROL = RUNTIME/'arm-turn-bonus-preparation/gate-b-to60-20260924'
RESOURCES = dict(gpus=8, hours=8, gpu_hours=64, cpus=64, memory_gib=960, browsers=64)
PYTHON = RUNTIME/'venv/bin/python'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def changed_files():
    from prepare_arm_gate_dp import replace_once
    changes = {}
    name = 'openwebrl/arm_gate_recovery.py'
    text = (PARENT/name).read_text()
    text = replace_once(text, "    if not origin or config['rollout_id'] != 0:\n",
                        "    if not origin or config['rollout_id'] != origin.get('rollout_id', 0):\n")
    text = replace_once(text, "    folder = Path(origin['source'])/'iterations/0000'\n",
                        "    folder = Path(origin['source'])/'iterations'/f\"{config['rollout_id']:04d}\"\n"
                        "    if origin.get('calibration_sha256') and hashlib.sha256((folder/'calibration.json').read_bytes()).hexdigest() != origin['calibration_sha256']:\n"
                        "        raise ValueError('Saved calibration changed')\n")
    text = replace_once(text, "manifest['rollout_id'] != 0", "manifest['rollout_id'] != config['rollout_id']")
    changes[name] = text.encode()
    name = 'openwebrl/arm_turn_bonus_cycles.py'
    text = (PARENT/name).read_text()
    anchor = "    current.update(rollout_id=rollout_id, checkpoint=checkpoint,\n"
    patch = """    replay = current.get('gate_replay_origin')
    if replay and rollout_id == replay.get('rollout_id', 0):
        if Path(checkpoint).resolve() != Path(replay['checkpoint']).resolve():
            raise ValueError('Replay actor differs from the saved batch actor')
        checkpoint = replay['checkpoint']
"""
    text = replace_once(text, anchor, patch+anchor)
    changes[name] = text.encode()
    # Isolated controllers avoid changing live C/beta launch fingerprints.
    name = 'scripts/run_arm_turn_bonus_cycles.py'
    text = (REPO/name).read_text()
    text = replace_once(text, "    if p.get('failure_ablation_training'):\n",
                        "    if p.get('saved_batch_continuation'):\n"
                        "        from prepare_arm_b_to60 import validate_plan\n"
                        "        readiness=validate_plan(p)\n"
                        "    elif p.get('failure_ablation_training'):\n")
    text = replace_once(text, "if reserve not in (0,3600):", "if reserve not in (0,3600,7200):")
    changes[name] = text.encode()
    name = 'scripts/check_arm_gpu_checkpoint_restore.py'
    text = (REPO/name).read_text()
    text = replace_once(text, "        if continuation.get('failure_ablation_training'):\n",
                        "        if continuation.get('saved_batch_continuation'):\n"
                        "            from prepare_arm_b_to60 import validate_plan\n"
                        "            validate_plan(continuation)\n"
                        "            source=Path(continuation['source'])\n"
                        "            validate_source(source)\n"
                        "        elif continuation.get('failure_ablation_training'):\n")
    changes[name] = text.encode()
    return changes


def prepare_source():
    from prepare_arm_turn_bonus import copy_plain
    validate_source(PARENT)
    changes = changed_files()
    for name, data in changes.items(): compile(data, name, 'exec')
    if not SOURCE.exists():
        shutil.copytree(PARENT, SOURCE, symlinks=True, copy_function=copy_plain,
            ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.git', '.browser_use_sessions'))
        manifest = read(PARENT/'reference_manifest.json')
        for name, data in changes.items():
            (SOURCE/name).write_bytes(data)
            manifest['recipe_files_sha256'][name] = hashlib.sha256(data).hexdigest()
        manifest['saved_batch_continuation'] = dict(parent=str(PARENT), rollout_id=49,
            changed_files=list(changes), objective_unchanged=True,
            method='Exact mixed+auxiliary replay and cursor restoration; preserve optimizer state')
        write_json(SOURCE/'reference_manifest.json', manifest)
    validate_source(SOURCE)
    old = read(PARENT/'reference_manifest.json')['recipe_files_sha256']
    new = read(SOURCE/'reference_manifest.json')['recipe_files_sha256']
    if (set(new) != set(old)|set(changes)
            or any(new[k] != v for k,v in old.items() if k not in changes)
            or any((SOURCE/k).read_bytes() != v for k,v in changes.items())):
        raise ValueError('Unexpected frozen continuation change')


def replay_provenance():
    folder = ORIGIN/'iterations/0049'
    complete, report = read(folder/'collection_complete.json'), read(folder/'calibration.json')
    gate, auxiliary = read(folder/'training_gate.json'), read(folder/'failure_auxiliary.json')
    state = read(ORIGIN/'status.json')
    checkpoint = ORIGIN/'runtime/iter_0000048'
    batch, cursor = ORIGIN/'runtime/rollout_recovery/49.pt', ORIGIN/'runtime/rollout/global_dataset_state_dict_49.pt'
    if (state['stage'] != 'complete' or state['completed_optimizer_updates'] != 662
            or Path(state['last_valid_checkpoint']) != checkpoint
            or (ORIGIN/'runtime/iter_0000049').exists()
            or gate['ready'] or not gate['calibration']['passed']
            or gate['expected_optimizer_updates'] != 14
            or complete['optimizer_updates'] != 0 or complete['rollout_id'] != 49
            or complete['checkpoint'] != str(checkpoint)
            or complete['recovery_file'] != str(batch) or complete['dataset_cursor'] != str(cursor)):
        raise ValueError('B49 or saved unoptimized batch provenance changed')
    if (auxiliary['policy_id'] != 'arm-gate-b-309053:rollout49'
            or auxiliary['checkpoint'] != str(checkpoint) or auxiliary['rollout_id'] != 49
            or auxiliary['candidate_gate'] != 'min2' or auxiliary['credit_assignment'] != 'response_index'
            or auxiliary['beta'] != .5 or auxiliary['q'] != .2 or auxiliary['mixed_groups'] != 48
            or auxiliary['tensor_file'] != 'failure_auxiliary.pt' or auxiliary['expected_windows'] != 7
            or report['batch_rows'] != complete['calibration_rows']):
        raise ValueError('Saved B recipe or batch size changed')
    archives = {}
    for path in (batch, cursor, folder/'failure_auxiliary.pt'):
        s = path.stat()
        with zipfile.ZipFile(path) as z:
            name = next(n for n in z.namelist() if n.endswith('/data.pkl'))
            metadata_hash = hashlib.sha256(z.read(name)).hexdigest()
            entries = [[i.filename, i.file_size, i.CRC] for i in z.infolist()]
        archives[str(path)] = dict(bytes=s.st_size, inode=s.st_ino, mtime_ns=s.st_mtime_ns,
            metadata_sha256=metadata_hash, inventory_sha256=hashlib.sha256(json.dumps(entries).encode()).hexdigest())
    return dict(source=str(ORIGIN), rollout_id=49, checkpoint=str(checkpoint),
        batch=str(batch), batch_bytes=batch.stat().st_size, dataset_cursor=str(cursor),
        cursor_sha256=sha(cursor), auxiliary_manifest_sha256=sha(folder/'failure_auxiliary.json'),
        auxiliary_bytes=(folder/'failure_auxiliary.pt').stat().st_size,
        calibration_sha256=sha(folder/'calibration.json'), archives=archives)


def plan(job='PREPARE'):
    from prepare_arm_gate_dp import configure
    from resume_arm_failure_variants import plan as resume_plan
    configure('B', 60)
    p = resume_plan(job, ORIGIN, RESOURCES['hours'], gpus=8)
    # Keep the existing staged-controller discovery layout; TP2 is explicit in
    # the environment and controller metadata, independent of directory names.
    old_output = p['output']
    p = json.loads(json.dumps(p).replace(old_output, old_output.replace('-tp2-iter60', '-iter60')))
    original_source = p['source']
    p['command'] = [x.replace(original_source, str(SOURCE)) for x in p['command']]
    p['environment'] = {k:v.replace(original_source, str(SOURCE)) for k,v in p['environment'].items()}
    origin = replay_provenance()
    p.update(source=str(SOURCE), saved_batch_continuation=True,
        evaluation_reserve_seconds=7200, evaluation_in_allocation=True,
        requested_resources=dict(RESOURCES), replay_provenance=origin)
    p['environment'].update(OPENWEBRL_REPLAY_FIRST_BATCH=origin['batch'], OPENWEBRL_REPLAY_ROLLOUT_ID='49',
        OPENWEBRL_ARM_REPLAY_CURSOR=origin['dataset_cursor'], WANDB_PROJECT='openwebrl')
    p['arm_config'].update(gate_replay_origin=origin, replay_origin=origin,
        minimum_cycle_seconds=2700, seconds_per_optimizer_update=90)
    return p


def fingerprints():
    paths = [Path(__file__).resolve(), REPO/'scripts/resume_arm_b_to60_8gpu.sbatch',
        SOURCE/'reference_manifest.json', REPO/'tests/test_arm_b_resume.py']
    return {str(p):sha(p) for p in paths}


def validate_plan(p):
    from resume_arm_turn_bonus import validate_resume_plan
    prepare_source(); validate_resume_plan(p)
    if p != plan(p['job_id']): raise ValueError('Prepared B continuation plan changed')
    ready = read(CONTROL/'readiness.json')
    if (not ready.get('cpu_passed') or not ready.get('native_parse_passed')
            or ready['fingerprints'] != fingerprints()
            or ready['replay_provenance'] != p['replay_provenance']):
        raise ValueError('Missing or stale B continuation readiness')
    return dict(passed=True, starts_from_iteration=49, initial_adam_updates=662,
        replay_rollout_id=49, unchanged_recipe=True, native_gpu_restore_required=True)


def train(job):
    from resume_arm_failure_variants import check_multimodal_storage
    from resume_arm_turn_bonus import validate_resume_plan
    p = plan(job); validate_plan(p)
    write_json(CONTROL/f'{job}-storage.json', check_multimodal_storage(p))
    manifest = CONTROL/f'{job}-launch.json';write_json(manifest,p)
    # Use current orchestration helpers while restoring the frozen actor recipe.
    probe = (f"import sys,runpy;sys.path.insert(0,{str(REPO/'scripts')!r});"
             f"runpy.run_path({str(SOURCE/'scripts/check_arm_gpu_checkpoint_restore.py')!r},run_name='__main__')")
    subprocess.run(source_command(SOURCE, [str(PYTHON), '-c', probe,
        '--training-root', str(ORIGIN), '--job-id', job, '--gpus', '8', '--output', p['gpu_restore_output'],
        '--continuation-plan', str(manifest), '--execute']), check=True)
    validate_resume_plan(p, require_gpu_restore=True)
    spec = importlib.util.spec_from_file_location('b_saved_batch_controller', SOURCE/'scripts/run_arm_turn_bonus_cycles.py')
    controller = importlib.util.module_from_spec(spec);spec.loader.exec_module(controller)
    controller.execute(p)


def execute(job):
    if os.getenv('SLURM_JOB_ID') != job: raise ValueError('Run inside the approved allocation')
    if os.getenv('ARM_B_CONTINUATION_APPROVED') != '8gpu_8h_to60':
        raise ValueError('Explicit 8 H200 x8h approval is required')
    allocation(subprocess.check_output(['scontrol','show','job',job,'-o'],text=True), job,
        requested_gpus=8, maximum_hours=8)
    p = plan(job);validate_plan(p)
    root = RUNTIME/f'evaluations/arm-gate-b-to60-{job}';root.mkdir(exist_ok=False)
    write_json(root/'controller-plan.json', dict(job_id=job, variant='B', target=60,
        training_root=p['output'], resume_from=str(ORIGIN), evaluation_in_allocation=True,
        evaluations=[50,60], topology='TP2/DP4/microbatch1'))
    step=['srun',f'--jobid={job}','--nodes=1','--ntasks=1','--cpus-per-task=64','--gres=gpu:h200:8',
          '--exact','--cpu-bind=none',str(PYTHON),str(Path(__file__).resolve()),'--job-id',job]
    try:
        write_json(root/'status.json',dict(stage='training',complete=False,current_target=60))
        subprocess.run([*step,'--worker','train'],check=True)
        from run_arm_gate_checkpoint_eval import checkpoint_ready
        for iteration in (50,60):
            if checkpoint_ready('B',Path(p['output']),iteration) is None:
                write_json(root/'status.json',dict(stage='budget-stop',complete=False,missing_iteration=iteration))
                return
            write_json(root/'status.json',dict(stage='evaluation',complete=False,iteration=iteration))
            subprocess.run([*step,'--worker','eval','--iteration',str(iteration)],check=True)
        write_json(root/'status.json',dict(stage='complete',complete=True,evaluated_iterations=[50,60]))
    except BaseException as exc:
        write_json(root/'status.json',dict(stage='failed',complete=False,error_type=type(exc).__name__,error=str(exc)[:300]))
        raise


def evaluate(job, iteration):
    from prepare_arm_gate_to60 import evaluation_plan
    from run_stage1_to100 import audit_rollouts
    import evaluate_baseline_checkpoint as evaluator
    p = plan(job)
    e = evaluation_plan('B',Path(p['output']),iteration,job)
    os.environ.update(OPENWEBRL_EXPECTED_SCHEDULER_OFFSET='0',OPENWEBRL_RECORD_EVAL_EXPECTED_TASK_COUNT='300')
    active = RUNTIME/f'arm-turn-bonus-preparation/milestone-evaluations/{job}-active.json'
    item = dict(variant='B', iteration=iteration, output=e['output'])
    write_json(active, dict(item,stage='evaluation'))
    try:
        evaluator.run(e,REPO/'.env')
        write_json(CONTROL/f'{job}-evaluation{iteration}-audit.json',audit_rollouts(Path(e['output'])/'rollouts',Path(e['source'])))
        write_json(active,dict(item,stage='complete'))
    except BaseException:
        write_json(active,dict(item,stage='failed'))
        raise


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job-id',default='PREPARE');parser.add_argument('--worker',choices=('train','eval'))
    parser.add_argument('--iteration',type=int,choices=(50,60));parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    if args.worker:
        if os.getenv('SLURM_JOB_ID')!=args.job_id:raise ValueError('Wrong worker allocation')
        if args.worker=='train':train(args.job_id)
        else:evaluate(args.job_id,args.iteration)
    elif args.execute:execute(args.job_id)
    else:
        prepare_source();CONTROL.mkdir(exist_ok=True,parents=True)
        write_json(CONTROL/'plan.json',plan());print(CONTROL/'plan.json')
