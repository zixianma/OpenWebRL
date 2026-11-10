#!/usr/bin/env python3
"""Prepare bounded unchanged beta/coverage continuations; never submit jobs."""
import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import zipfile

from resume_baseline import REPO, RUNTIME, allocation, source_command, validate_source, write_json

CONTROL = REPO/'logs/arm-ablation-to60-20260926'
PYTHON = RUNTIME/'venv/bin/python'
ORIGINS = {
    'weight': RUNTIME/'evaluations/arm-failure-additive-318949-tp2-iter20',
    'coverage': RUNTIME/'evaluations/arm-failure-additive-329911-tp2-iter20',
}
BETA_PARENT = RUNTIME/'reference-arm-failure-ablations-tp2-20260922-v1'
SOURCES = {
    'weight': RUNTIME/'reference-arm-failure-weight-replay20-20260926-v1',
    'coverage': RUNTIME/'reference-arm-failure-coverage-identity-20260924-v1',
}
RUN_IDS = {'weight': 'arm-failure-weight-fromzero-318949',
           'coverage': 'arm-failure-coverage-fromzero-318950'}
FAILED_BETA = RUNTIME/'evaluations/arm-failure-additive-330278-tp2-iter30'
PROFILES = {
    'weight-recovery': dict(variant='weight', start=20, end=40, hours=13.5),
    'weight-to60': dict(variant='weight', start=40, end=60, hours=14),
    'coverage-to40': dict(variant='coverage', start=20, end=40, hours=18),
    'coverage-to60': dict(variant='coverage', start=40, end=60, hours=18),
}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resources(profile):
    hours = PROFILES[profile]['hours']
    return dict(gpus=8, hours=hours, gpu_hours=8*hours, cpus=64, memory_gib=960, browsers=64)


def beta_replay_changes():
    from prepare_arm_gate_dp import replace_once
    changes = {}
    name = 'openwebrl/arm_gate_recovery.py'
    text = (BETA_PARENT/name).read_text()
    text = replace_once(text, "config['rollout_id'] != 0:",
                        "config['rollout_id'] != origin['rollout_id']:")
    text = replace_once(text, "    folder = Path(origin['source'])/'iterations/0000'\n",
        "    folder = Path(origin['source'])/'iterations'/f\"{config['rollout_id']:04d}\"\n"
        "    if hashlib.sha256((folder/'calibration.json').read_bytes()).hexdigest() != origin['calibration_sha256']:\n"
        "        raise ValueError('Saved calibration changed')\n")
    text = replace_once(text, "manifest['rollout_id'] != 0 or manifest['beta'] != .5 or manifest['q'] != .2",
        "manifest['rollout_id'] != config['rollout_id'] or manifest['beta'] != config['failure_beta'] or manifest['q'] != config['failure_scored_fraction']")
    changes[name] = text.encode()
    name = 'openwebrl/arm_turn_bonus_cycles.py'
    text = (BETA_PARENT/name).read_text()
    anchor = '    current.update(rollout_id=rollout_id, checkpoint=checkpoint,\n'
    text = replace_once(text, anchor,
        "    replay = current.get('gate_replay_origin')\n"
        "    if replay and rollout_id == replay['rollout_id']:\n"
        "        if Path(checkpoint).resolve() != Path(replay['checkpoint']).resolve():\n"
        "            raise ValueError('Replay actor differs from saved actor')\n"
        "        checkpoint = replay['checkpoint']\n"+anchor)
    changes[name] = text.encode()
    name = 'openwebrl/arm_failure_additive.py'
    text = (BETA_PARENT/name).read_text()
    anchor = "    config=current['config']\n"
    text = replace_once(text, anchor, anchor+
        "    from openwebrl.arm_gate_recovery import replay_auxiliary\n"
        "    if replay_auxiliary(args,samples,current,report):\n"
        "        return\n")
    changes[name] = text.encode()
    return changes


def prepare_source():
    from prepare_arm_turn_bonus import copy_plain
    validate_source(BETA_PARENT)
    changes = beta_replay_changes()
    for name, data in changes.items():
        compile(data, name, 'exec')
    source = SOURCES['weight']
    if not source.exists():
        shutil.copytree(BETA_PARENT, source, symlinks=True, copy_function=copy_plain,
            ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.git', '.browser_use_sessions'))
        manifest = read(BETA_PARENT/'reference_manifest.json')
        for name, data in changes.items():
            (source/name).write_bytes(data)
            manifest['recipe_files_sha256'][name] = hashlib.sha256(data).hexdigest()
        manifest['beta_saved_batch_recovery'] = dict(parent=str(BETA_PARENT), rollout_id=20,
            changed_files=list(changes), objective_unchanged=True)
        write_json(source/'reference_manifest.json', manifest)
    validate_source(source)
    old = read(BETA_PARENT/'reference_manifest.json')['recipe_files_sha256']
    new = read(source/'reference_manifest.json')['recipe_files_sha256']
    if (set(new) != set(old)|set(changes) or
            any(new[k] != v for k,v in old.items() if k not in changes) or
            any((source/k).read_bytes() != v for k,v in changes.items())):
        raise ValueError('Unexpected frozen source change')
    validate_source(SOURCES['coverage'])


def cached_origin(root, variant):
    root = Path(root).resolve()
    manifest, state = read(root/'launch_manifest.json'), read(root/'status.json')
    if state['stage'] != 'complete' or not state.get('has_saved_checkpoints'):
        raise ValueError('Require completed inactive training stage')
    allowed = {str(SOURCES[variant])}
    if variant == 'weight':
        allowed.add(str(BETA_PARENT))
    if manifest['wandb_run_id'] != RUN_IDS[variant] or manifest['source'] not in allowed:
        raise ValueError('Training source or lineage mismatch')
    index = int((root/'runtime/latest_checkpointed_iteration.txt').read_text())
    report = read(root/f'iterations/{index:04d}/checkpoint-validation.json')
    checkpoint = root/f'runtime/iter_{index:07d}'
    if (report['checkpoint'] != str(checkpoint) or report['iteration'] != index or
        report['completed_optimizer_updates'] != state['completed_optimizer_updates'] or
        report['scheduler_minus_optimizer_updates'] != 0):
        raise ValueError('Checkpoint identity or counters changed')
    for name, size in report['shard_files'].items():
        if (checkpoint/name).stat().st_size != size:
            raise ValueError('Checkpoint shard changed')
    paths = [checkpoint/'common.pt', checkpoint/'.metadata', Path(report['dataset_cursor']),
             root/'runtime/latest_checkpointed_iteration.txt']
    return dict(root=str(root), manifest=manifest, checkpoint_report=report,
                identity_sha256={str(p):sha(p) for p in paths})


def replay_provenance():
    folder = FAILED_BETA/'iterations/0020'
    complete = read(folder/'collection_complete.json')
    auxiliary, gate = read(folder/'failure_auxiliary.json'), read(folder/'training_gate.json')
    checkpoint = FAILED_BETA/'runtime/iter_0000019'
    batch = FAILED_BETA/'runtime/rollout_recovery/20.pt'
    cursor = FAILED_BETA/'runtime/rollout/global_dataset_state_dict_20.pt'
    if (checkpoint.resolve() != ORIGINS['weight']/'runtime/iter_0000019' or
        complete['rollout_id'] != 20 or complete['checkpoint'] != str(checkpoint) or
        complete['recovery_file'] != str(batch) or complete['dataset_cursor'] != str(cursor) or
        not gate['calibration']['passed'] or gate['expected_optimizer_updates'] != 14 or
        (FAILED_BETA/'runtime/iter_0000020').exists()):
        raise ValueError('Saved beta iteration21 provenance changed')
    if (auxiliary['beta'], auxiliary['q'], auxiliary['candidate_gate'], auxiliary['credit_assignment'],
        auxiliary['rollout_id'], auxiliary['expected_windows']) != (1., .2, 'distinct5', 'response_index', 20, 7):
        raise ValueError('Saved beta auxiliary recipe changed')
    archives = {}
    for path in (batch, cursor, folder/'failure_auxiliary.pt'):
        stat = path.stat()
        with zipfile.ZipFile(path) as zipped:
            if not any(n.endswith('/data.pkl') for n in zipped.namelist()):
                raise ValueError('Incomplete saved archive')
        archives[str(path)] = dict(bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
    return dict(source=str(FAILED_BETA), rollout_id=20, checkpoint=str(checkpoint),
        batch=str(batch), batch_bytes=batch.stat().st_size, dataset_cursor=str(cursor),
        cursor_sha256=sha(cursor), auxiliary_manifest_sha256=sha(folder/'failure_auxiliary.json'),
        auxiliary_bytes=(folder/'failure_auxiliary.pt').stat().st_size,
        calibration_sha256=sha(folder/'calibration.json'), archives=archives)


def plan(job, profile, root, target):
    spec = PROFILES[profile]; variant = spec['variant']
    if target not in range(spec['start']+10, spec['end']+1, 10):
        raise ValueError('Stage outside approved profile')
    origin = cached_origin(root, variant)
    prior, report = origin['manifest'], origin['checkpoint_report']
    start = report['iteration']+1
    if start != target-10:
        raise ValueError('Cannot skip a ten-iteration stage')
    if start == 20 and Path(root).resolve() != ORIGINS[variant]:
        raise ValueError('Unexpected initial continuation root')
    source = SOURCES[variant]; validate_source(source)
    out = RUNTIME/f'evaluations/arm-failure-additive-{job}-tp2-iter{target}'
    old = prior['output']; old_source = prior['source']
    p = copy.deepcopy(prior)
    for key in ['command', 'environment', 'arm_config']:
        if isinstance(p[key], list):
            p[key] = [x.replace(old, str(out)).replace(old_source, str(source)) for x in p[key]]
        else:
            p[key] = {k:v.replace(old,str(out)).replace(old_source,str(source)) if isinstance(v,str) else v
                      for k,v in p[key].items()}
    for key in ('deadline_epoch_seconds', 'gate_replay_origin', 'replay_origin'):
        p['arm_config'].pop(key, None)
    for key in ('OPENWEBRL_REPLAY_FIRST_BATCH', 'OPENWEBRL_REPLAY_ROLLOUT_ID', 'OPENWEBRL_ARM_REPLAY_CURSOR'):
        p['environment'].pop(key, None)
    p.pop('replay_provenance', None)
    p.update(job_id=str(job), output=str(out), source=str(source), resume_from=str(root), resume_origin=origin,
        variant_continuation=True, ablation_to60_profile=profile, fresh_optimizer=False,
        initial_optimizer_updates=report['completed_optimizer_updates'], start_rollout_id=start,
        requested_iterations=10, target_completed_iterations=target, checkpoint=report['checkpoint'],
        requested_resources=resources(profile), evaluation_in_allocation=True,
        evaluation_reserve_seconds=3600)
    scratch = RUNTIME/f'multimodal-scratch/arm-variant-{job}'
    p['multimodal_storage'] = dict(mode='shared', directory=str(scratch), minimum_free_bytes=2*1024**4,
                                   retention='preserve; no automatic deletion')
    p['environment'].update(NUM_GPUS='8', TP_SIZE='2', NUM_ROLLOUT=str(target), WANDB_RESUME='must',
        OPENWEBRL_CUDA_CACHE_LIMIT_GIB='48', WANDB_RUN_ID=RUN_IDS[variant], WANDB_PROJECT='openwebrl',
        SLIME_LOAD_CHECKPOINT=str(Path(root)/'runtime'), BROWSER_CONCURRENCY='64', SGLANG_CONCURRENCY='64',
        OPENWEBRL_MULTIMODAL_STORAGE_DIR=str(scratch))
    p['browser_config']['browser_rollout_concurrency'] = 64
    if '--use-checkpoint-opt-param-scheduler' not in p['command']:
        p['command'].append('--use-checkpoint-opt-param-scheduler')
    p['selector_port'] = 31000+int(job)%20000 if str(job).isdigit() else 27511
    p['arm_config'].update(checkpoint=report['checkpoint'], policy_id=RUN_IDS[variant]+':uninitialized',
        selector_endpoint=f"http://127.0.0.1:{p['selector_port']}", minimum_cycle_seconds=3600,
        seconds_per_optimizer_update=90)
    p['gpu_restore_output'] = str(Path(root)/f'gpu-restore-check-{job}-{profile}-to{target}')
    p['resume_restore_receipt'] = str(Path(p['gpu_restore_output'])/'result.json')
    p['comparison'].update(endpoint=spec['end'], continuation_from_iteration=spec['start'], automatic_extension=False,
        fresh_optimizer_scheduler_cursor=False, fresh_wandb_lineage=False,
        limit='Continue existing experiment; exact milestones and full300 evaluations before next stage')
    if profile == 'weight-recovery' and start == 20:
        replay = replay_provenance()
        p['replay_provenance'] = replay
        p['environment'].update(OPENWEBRL_REPLAY_FIRST_BATCH=replay['batch'], OPENWEBRL_REPLAY_ROLLOUT_ID='20',
                                OPENWEBRL_ARM_REPLAY_CURSOR=replay['dataset_cursor'])
        p['arm_config'].update(gate_replay_origin=replay, replay_origin=replay,
                               candidate_gate='distinct5', credit_assignment='response_index')
    return p


def fingerprints():
    paths = [Path(__file__).resolve(), CONTROL/'training-controller.py', CONTROL/'restore-controller.py',
             REPO/'scripts/resume_arm_turn_bonus.py', REPO/'scripts/recover_arm_unfinished.py',
             *[s/'reference_manifest.json' for s in SOURCES.values()]]
    return {str(p):sha(p) for p in paths}


def validate_plan(p):
    if read(CONTROL/'readiness.json')['fingerprints'] != fingerprints():
        raise ValueError('Continuation readiness changed')
    profile = p['ablation_to60_profile']
    if p != plan(p['job_id'], profile, Path(p['resume_from']), p['target_completed_iterations']):
        raise ValueError('Continuation plan changed')
    c = p['arm_config']; variant = PROFILES[profile]['variant']
    expected = (1., .2) if variant == 'weight' else (.5, .4)
    if (c['beta'], c['scored_fraction'], c['failure_beta'], c['failure_scored_fraction'], c['failure_group_cap']) != (.5,.2,*expected,8):
        raise ValueError('Scientific recipe changed')
    return dict(passed=True, objective_unchanged=True, restore_optimizer_scheduler_cursor=True,
                native_gpu_restore_required=True, maximum_iteration=PROFILES[profile]['end'])


def check_recovery_budget(accounting):
    rows = [r.split('|') for r in accounting.splitlines() if r.strip()]
    matching = [r for r in rows if r[0] == '330278']
    if len(matching) != 1 or matching[0][2] != 'FAILED':
        raise ValueError('Failed beta allocation must be accounted for')
    used = int(matching[0][1]); requested = int(PROFILES['weight-recovery']['hours']*3600)
    if used+requested > 14*3600:
        raise ValueError('Recovery exceeds previous14h approval')
    return dict(previous_job='330278', previous_used_seconds=used, requested_seconds=requested,
                original_approved_seconds=14*3600)


def prepare():
    CONTROL.mkdir(parents=True, exist_ok=True)
    prepare_source()
    for old,new in [('run_arm_turn_bonus_cycles.py','training-controller.py'),
                    ('check_arm_gpu_checkpoint_restore.py','restore-controller.py')]:
        text = (REPO/'scripts'/old).read_text()
        before = 'from prepare_arm_failure_ablations import validate_plan'
        if text.count(before) != 1:
            raise ValueError('Controller validation boundary changed')
        text = text.replace(before, 'from prepare_arm_ablation_to60 import validate_plan')
        compile(text, new, 'exec')
        path = CONTROL/new
        if path.exists() and path.read_text() != text:
            raise ValueError('Frozen controller changed')
        path.write_text(text)
    write_json(CONTROL/'readiness.json', dict(fingerprints=fingerprints(), gpu_restore_pending=True))
    for profile in ('weight-recovery', 'coverage-to40'):
        p = plan('PREPARE', profile, ORIGINS[PROFILES[profile]['variant']], 30)
        validate_plan(p); write_json(CONTROL/f'{profile}-plan.json', p)
    write_json(CONTROL/'proposal.json', dict(profiles=PROFILES, resources={p:resources(p) for p in PROFILES},
        priority='Gate B331778 first for storage; register ablations after B',
        additional_allocation_approval_pending=['weight-to60','coverage-to40','coverage-to60'],
        new_gpu_hours=400, recovery_gpu_hours=108, eval_iterations=[30,40,50,60],
        no_automatic_extension=True, retain_existing_checkpoints_optimizer_cursor_wandb=True))


def train(job, profile, root, target):
    from resume_arm_failure_variants import check_multimodal_storage
    from resume_arm_turn_bonus import validate_resume_plan
    p = plan(job,profile,root,target); validate_plan(p); check_multimodal_storage(p)
    manifest = CONTROL/f'{job}-to{target}-launch.json'; write_json(manifest,p)
    probe = f"import sys,runpy;sys.path.insert(0,{str(REPO/'scripts')!r});runpy.run_path({str(CONTROL/'restore-controller.py')!r},run_name='__main__')"
    subprocess.run(source_command(Path(p['source']),[str(PYTHON),'-c',probe,'--training-root',str(root),
        '--job-id',job,'--gpus','8','--output',p['gpu_restore_output'],'--continuation-plan',str(manifest),'--execute']),check=True)
    validate_resume_plan(p,require_gpu_restore=True)
    spec = importlib.util.spec_from_file_location('ablation60_training',CONTROL/'training-controller.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    with (CONTROL/f'{job}-monitor.log').open('a') as log:
        monitor = subprocess.Popen([str(PYTHON),str(REPO/'scripts/monitor_arm_turn_bonus.py'),'--job-id',job,
            '--watch','--interval','900','--hours',str(PROFILES[profile]['hours'])],stdout=log,stderr=subprocess.STDOUT)
        try:
            module.execute(p)
        finally:
            monitor.terminate()
            try: monitor.wait(timeout=30)
            except subprocess.TimeoutExpired: monitor.kill(); monitor.wait()


def evaluate(job, profile, root, target):
    from recover_arm_unfinished import evaluation_plan
    from run_stage1_to100 import audit_rollouts
    import evaluate_baseline_checkpoint as evaluator
    p = evaluation_plan(PROFILES[profile]['variant'],job,target,root,8)
    os.environ.update(OPENWEBRL_EXPECTED_SCHEDULER_OFFSET='0',OPENWEBRL_RECORD_EVAL_EXPECTED_TASK_COUNT='300')
    evaluator.run(p,REPO/'.env')
    audit = audit_rollouts(Path(p['output'])/'rollouts',Path(p['source']))
    write_json(CONTROL/f'{job}-eval{target}-audit.json',audit)


def execute(job, profile, root):
    spec = PROFILES[profile]
    approval = read(CONTROL/f'{profile}-approval.json')
    if not approval.get('approved') or approval['resources'] != resources(profile) or os.getenv('SLURM_JOB_ID') != job:
        raise ValueError('Exact resource approval and matching allocation required')
    allocation(subprocess.check_output(['scontrol','show','job',job,'-o'],text=True),job,
               requested_gpus=8,maximum_hours=spec['hours'])
    if profile == 'weight-recovery':
        accounting = subprocess.check_output(['sacct','-X','-n','-P','-j','330278','--format=JobID,ElapsedRaw,State'],text=True)
        check_recovery_budget(accounting)
    controller = RUNTIME/f'evaluations/arm-ablation-{profile}-{job}'
    controller.mkdir(exist_ok=False)
    step = ['srun',f'--jobid={job}','--nodes=1','--ntasks=1','--cpus-per-task=64','--gres=gpu:h200:8',
            '--exact','--cpu-bind=none',str(PYTHON),str(Path(__file__).resolve()),'--job-id',job,'--profile',profile]
    completed = []
    try:
        for target in range(spec['start']+10,spec['end']+1,10):
            p = plan(job,profile,root,target); validate_plan(p)
            write_json(controller/'controller-plan.json',dict(job_id=job,variant=spec['variant'],profile=profile,
                target=spec['end'],current_target=target,training_root=p['output'],resume_from=str(root)))
            write_json(controller/'status.json',dict(stage='training',complete=False,target=target,completed_stages=completed))
            subprocess.run([*step,'--worker','train','--root',str(root),'--target',str(target)],check=True)
            root = Path(p['output'])
            # Exit nonzero if a budget stop leaves a partial stage: afterok successors
            # must never allocate GPUs expecting a nonexistent milestone.
            saved = cached_origin(root,spec['variant'])['checkpoint_report']['iteration']+1
            if saved != target:
                raise RuntimeError(f'Budget stop at durable iteration{saved}; target{target} not ready')
            write_json(controller/'status.json',dict(stage='evaluation',complete=False,iteration=target,completed_stages=completed))
            subprocess.run([*step,'--worker','eval','--root',str(root),'--target',str(target)],check=True)
            completed.append(target)
        write_json(controller/'status.json',dict(stage='complete',complete=True,hard_stop_iteration=spec['end'],evaluated_iterations=completed))
    except BaseException as exc:
        write_json(controller/'status.json',dict(stage='failed',complete=False,error_type=type(exc).__name__,
            error=str(exc)[:400],completed_stages=completed)); raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare',action='store_true'); parser.add_argument('--job-id',default='PREPARE')
    parser.add_argument('--profile',choices=PROFILES); parser.add_argument('--root',type=Path)
    parser.add_argument('--worker',choices=('train','eval')); parser.add_argument('--target',type=int)
    args = parser.parse_args()
    if args.prepare: prepare()
    elif args.worker:
        if os.getenv('SLURM_JOB_ID') != args.job_id: raise ValueError('Wrong worker allocation')
        (train if args.worker == 'train' else evaluate)(args.job_id,args.profile,args.root,args.target)
    else:
        dispatch=CONTROL/'dispatch'/f'{args.job_id}.json'
        if dispatch.exists():
            request=read(dispatch)
            if (request['module']!='recover_arm_beta_memory' or args.profile!='weight-to60'
                    or request['job_id']!=args.job_id or request['profile']!=args.profile
                    or request['previous_resume_from']!=str(args.root)):
                raise ValueError('Unexpected bounded beta continuation dispatch')
            import recover_arm_beta_memory
            recover_arm_beta_memory.execute(args.job_id,args.profile,Path(request['resume_from']))
        else: execute(args.job_id,args.profile,args.root)
