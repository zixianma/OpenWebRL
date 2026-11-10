#!/usr/bin/env python3
"""Recover Gate B69->70->80->90 within unused approved time; never submit Slurm jobs."""
import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess

from resume_baseline import REPO,RUNTIME,allocation,source_command,validate_source,write_json

CONTROL=REPO/'logs/arm-b-recovery-20260926'
ORIGIN=RUNTIME/'evaluations/arm-failure-additive-330304-tp2-iter70'
SOURCE=RUNTIME/'reference-arm-gate-b-resume49-20260924-v1'
PYTHON=RUNTIME/'venv/bin/python'
RUN_ID='arm-gate-b-309053'
SECONDS=19*3600+18*60
RESOURCES=dict(gpus=8,hours=SECONDS/3600,gpu_hours=8*SECONDS/3600,cpus=64,memory_gib=960,browsers=64)
PREVIOUS_ATTEMPTS=('330304','331767')


def read(path):return json.loads(Path(path).read_text())


def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''):value.update(block)
    return value.hexdigest()


def cached_origin(root):
    """Check saved validation and file identities without loading tensors on login.

    Native counters, tensor extents and full GPU restoration are rechecked inside
    the allocated worker before any collection or optimizer update.
    """
    root=Path(root).resolve();manifest=read(root/'launch_manifest.json');state=read(root/'status.json')
    if not (state['stage']=='complete' or state.get('failed')) or not state.get('last_valid_checkpoint'):
        raise ValueError('Requires a completed inactive training stage')
    if manifest['wandb_run_id']!=RUN_ID or manifest['source']!=str(SOURCE):
        raise ValueError('B training lineage or source changed')
    index=int((root/'runtime/latest_checkpointed_iteration.txt').read_text())
    report=read(root/f'iterations/{index:04d}/checkpoint-validation.json')
    checkpoint=root/f'runtime/iter_{index:07d}'
    if (report['checkpoint']!=str(checkpoint) or report['iteration']!=index
            or report['completed_optimizer_updates']!=state['completed_optimizer_updates']
            or report['scheduler_minus_optimizer_updates']!=0):
        raise ValueError('Saved checkpoint counters or identity disagree')
    for name,size in report['shard_files'].items():
        if (checkpoint/name).stat().st_size!=size:raise ValueError('Checkpoint shard changed')
    paths=[checkpoint/'common.pt',checkpoint/'.metadata',Path(report['dataset_cursor']),
        root/'runtime/latest_checkpointed_iteration.txt']
    return dict(root=str(root),manifest=manifest,checkpoint_report=report,
        identity_sha256={str(p):sha(p) for p in paths})


def plan(job,root=ORIGIN,target=70):
    if target not in (70,80,90):raise ValueError('B continuation is capped at90')
    origin=cached_origin(root);prior=origin['manifest'];report=origin['checkpoint_report']
    start=report['iteration']+1
    if start!=(69 if target==70 else target-10):raise ValueError('Only69->70,70->80 and80->90 stages are permitted')
    expected_root=ORIGIN if target==70 else RUNTIME/f'evaluations/arm-failure-additive-{job}-tp2-iter{target-10}'
    if Path(root).resolve()!=expected_root.resolve():raise ValueError('Unexpected B continuation parent')
    validate_source(SOURCE)
    out=RUNTIME/f'evaluations/arm-failure-additive-{job}-tp2-iter{target}'
    old=prior['output'];p=copy.deepcopy(prior)
    for key in ['command','environment','arm_config']:
        if isinstance(p[key],list):p[key]=[x.replace(old,str(out)) for x in p[key]]
        else:p[key]={k:v.replace(old,str(out)) if isinstance(v,str) else v for k,v in p[key].items()}
    for key in ['deadline_epoch_seconds','b_to90_continuation']:p['arm_config'].pop(key,None)
    p.update(job_id=str(job),output=str(out),resume_from=str(root),resume_origin=origin,
        variant_continuation=True,b_to90_continuation=True,fresh_optimizer=False,
        initial_optimizer_updates=report['completed_optimizer_updates'],start_rollout_id=start,
        requested_iterations=target-start,target_completed_iterations=target,checkpoint=report['checkpoint'],
        requested_resources=dict(RESOURCES),evaluation_in_allocation=True,
        evaluation_reserve_seconds=(100-target)//10*3600)
    # The inherited B49 recovery batch has already been consumed by B60.
    for key in ('saved_batch_continuation','replay_provenance'):p.pop(key,None)
    for key in ('OPENWEBRL_REPLAY_FIRST_BATCH','OPENWEBRL_REPLAY_ROLLOUT_ID','OPENWEBRL_ARM_REPLAY_CURSOR'):p['environment'].pop(key,None)
    for key in ('gate_replay_origin','replay_origin'):p['arm_config'].pop(key,None)
    scratch=RUNTIME/f'multimodal-scratch/arm-variant-{job}'
    p['multimodal_storage']=dict(mode='shared',directory=str(scratch),minimum_free_bytes=2*1024**4,
        retention='preserve; no automatic deletion')
    p['environment'].update(NUM_GPUS='8',TP_SIZE='2',NUM_ROLLOUT=str(target),WANDB_RESUME='must',
        OPENWEBRL_CUDA_CACHE_LIMIT_GIB='48',
        WANDB_RUN_ID=RUN_ID,WANDB_PROJECT='openwebrl',SLIME_LOAD_CHECKPOINT=str(Path(root)/'runtime'),
        BROWSER_CONCURRENCY='64',SGLANG_CONCURRENCY='64',OPENWEBRL_MULTIMODAL_STORAGE_DIR=str(scratch))
    p['browser_config']['browser_rollout_concurrency']=64
    if '--use-checkpoint-opt-param-scheduler' not in p['command']:p['command'].append('--use-checkpoint-opt-param-scheduler')
    p['selector_port']=31000+int(job)%20000 if str(job).isdigit() else 27511
    p['arm_config'].update(checkpoint=report['checkpoint'],policy_id=RUN_ID+':uninitialized',
        selector_endpoint=f"http://127.0.0.1:{p['selector_port']}",minimum_cycle_seconds=2700,
        seconds_per_optimizer_update=90)
    p['gpu_restore_output']=str(Path(root)/f'gpu-restore-check-{job}-b-to{target}')
    p['resume_restore_receipt']=str(Path(p['gpu_restore_output'])/'result.json')
    p['comparison'].update(endpoint=90,continuation_from_iteration=69,automatic_extension=False)
    if target==70:
        replay=replay_provenance()
        p['replay_provenance']=replay
        p['environment'].update(OPENWEBRL_REPLAY_FIRST_BATCH=replay['batch'],
            OPENWEBRL_REPLAY_ROLLOUT_ID='69',OPENWEBRL_ARM_REPLAY_CURSOR=replay['dataset_cursor'])
        p['arm_config'].update(gate_replay_origin=replay,replay_origin=replay)
    p['recovery_note']='Restore checkpoint69; replay full saved iteration70, discarding partial unsaved updates. Release unused CUDA cache at48GiB rather than100GiB. No objective/batch/topology change.'
    return p


def fingerprints():
    paths=[Path(__file__).resolve(),CONTROL/'training-controller.py',CONTROL/'restore-controller.py',
        SOURCE/'reference_manifest.json',REPO/'scripts/resume_arm_turn_bonus.py']
    return {str(p):sha(p) for p in paths}


def validate_plan(p):
    if read(CONTROL/'readiness.json')['fingerprints']!=fingerprints():
        raise ValueError('B90 readiness changed')
    if p!=plan(p['job_id'],Path(p['resume_from']),p['target_completed_iterations']):
        raise ValueError('Bounded continuation plan changed')
    c=p['arm_config']
    if (c['beta'],c['scored_fraction'],c['failure_group_cap'],c['candidate_gate'],c['credit_assignment'])!=(.5,.2,8,'min2','response_index'):
        raise ValueError('Gate B objective changed')
    return dict(passed=True,target=p['target_completed_iterations'],maximum_iteration=90,
        objective_unchanged=True,restore_optimizer_scheduler_cursor=True,native_gpu_restore_required=True)


def prepare():
    CONTROL.mkdir(parents=True,exist_ok=True)
    validate_source(SOURCE)
    for old,new,count in [('run_arm_turn_bonus_cycles.py','training-controller.py',1),
                           ('check_arm_gpu_checkpoint_restore.py','restore-controller.py',2)]:
        text=(REPO/'scripts'/old).read_text()
        before='from resume_arm_failure_variants import validate_plan'
        if text.count(before)!=count:raise ValueError('Controller validation boundary changed')
        text=text.replace(before,'from recover_arm_b69 import validate_plan')
        text=text.replace('if reserve not in (0,3600):','if reserve not in (0,3600,7200,10800):')
        compile(text,new,'exec')
        path=CONTROL/new
        if path.exists() and path.read_text()!=text:raise ValueError('Frozen controller changed')
        path.write_text(text)
    write_json(CONTROL/'readiness.json',dict(fingerprints=fingerprints(),gpu_restore_pending=True,
        checkpoint60_pending=not (ORIGIN/'status.json').exists(),
        validation='Bind the actual B60 native validation receipt at handoff; recheck native counters and restore model/optimizer on GPUs'))
    previous=RUNTIME/'evaluations/arm-failure-additive-318934-tp2-iter60'
    rows=[read(f) for f in sorted((previous/'iterations').glob('*/checkpoint-saved.json'))]
    intervals=[(b['saved_at_epoch_seconds']-a['saved_at_epoch_seconds'])/60
        for a,b in zip(rows,rows[1:]) if b['rollout_id']==a['rollout_id']+1]
    import statistics
    median=statistics.median(intervals[-8:])
    proposal=dict(submitted=False,resource_approval_pending=False,resources=RESOURCES,
        approval='Unused portion of approved330304:24h minus4:41:49; capped19:18:00',
        dependency_job=None,origin=str(ORIGIN),initial_completed_iteration=69,hard_stop_iteration=90,
        stages=['restore69','replay70','full300-eval70','restore70','train71-80','full300-eval80',
                'restore80','train81-90','full300-eval90'],
        recent_median_minutes_per_iteration=median,estimated_training_hours=21*median/60,
        expected_total_hours='About11-14; remaining19:18 allocation cap',no_automatic_extension=True)
    path=CONTROL/'proposal.json'
    if not path.exists():write_json(path,proposal)


def replay_provenance():
    import zipfile
    folder=ORIGIN/'iterations/0069'
    report=read(folder/'calibration.json');complete=read(folder/'collection_complete.json')
    auxiliary=read(folder/'failure_auxiliary.json');gate=read(folder/'training_gate.json')
    origin=cached_origin(ORIGIN);checkpoint=origin['checkpoint_report']['checkpoint']
    batch=ORIGIN/'runtime/rollout_recovery/69.pt';cursor=ORIGIN/'runtime/rollout/global_dataset_state_dict_69.pt'
    if (complete['rollout_id']!=69 or complete['checkpoint']!=checkpoint or
        complete['recovery_file']!=str(batch) or complete['dataset_cursor']!=str(cursor) or
        not gate['calibration']['passed'] or gate['expected_optimizer_updates']!=12 or
        (ORIGIN/'runtime/iter_0000069').exists()):raise ValueError('Saved iteration70 identity changed')
    if (auxiliary['candidate_gate'],auxiliary['credit_assignment'],auxiliary['beta'],auxiliary['q'])!=('min2','response_index',.5,.2):raise ValueError('Recipe changed')
    archives={}
    for path in (batch,cursor,folder/'failure_auxiliary.pt'):
        stat=path.stat()
        with zipfile.ZipFile(path) as z:
            if not any(n.endswith('/data.pkl') for n in z.namelist()):raise ValueError('Incomplete saved archive')
        archives[str(path)]=dict(bytes=stat.st_size,mtime_ns=stat.st_mtime_ns)
    return dict(source=str(ORIGIN),rollout_id=69,checkpoint=checkpoint,
        batch=str(batch),batch_bytes=batch.stat().st_size,dataset_cursor=str(cursor),
        cursor_sha256=sha(cursor),auxiliary_manifest_sha256=sha(folder/'failure_auxiliary.json'),
        auxiliary_bytes=(folder/'failure_auxiliary.pt').stat().st_size,
        calibration_sha256=sha(folder/'calibration.json'),archives=archives)


def prerequisite_ready():
    origin=cached_origin(ORIGIN)
    if origin['checkpoint_report']['iteration']!=68 or origin['checkpoint_report']['completed_optimizer_updates']!=902:
        raise ValueError('Expected durable iteration69 /902 Adam updates')
    replay_provenance()
    return True


def train(job,root,target):
    from resume_arm_failure_variants import check_multimodal_storage
    from resume_arm_turn_bonus import validate_resume_plan
    p=plan(job,root,target);validate_plan(p);check_multimodal_storage(p)
    manifest=CONTROL/f'{job}-to{target}-launch.json';write_json(manifest,p)
    # The copied restore controller imports this prepared validator through the
    # current scripts path, while loading the unchanged frozen training source.
    probe=f"import sys,runpy;sys.path.insert(0,{str(REPO/'scripts')!r});runpy.run_path({str(CONTROL/'restore-controller.py')!r},run_name='__main__')"
    subprocess.run(source_command(SOURCE,[str(PYTHON),'-c',probe,'--training-root',str(root),
        '--job-id',job,'--gpus','8','--output',p['gpu_restore_output'],'--continuation-plan',str(manifest),'--execute']),check=True)
    validate_resume_plan(p,require_gpu_restore=True)
    spec=importlib.util.spec_from_file_location('b90_training',CONTROL/'training-controller.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    with (CONTROL/f'{job}-monitor.log').open('a') as log:
        monitor=subprocess.Popen([str(PYTHON),str(REPO/'scripts/monitor_arm_turn_bonus.py'),'--job-id',job,
            '--watch','--interval','900','--hours','24'],stdout=log,stderr=subprocess.STDOUT)
        try:module.execute(p)
        finally:
            monitor.terminate()
            try:monitor.wait(timeout=30)
            except subprocess.TimeoutExpired:monitor.kill();monitor.wait()


def evaluate(job,root,target):
    from prepare_arm_gate_to60 import evaluation_plan
    import evaluate_baseline_checkpoint as evaluator
    from run_stage1_to100 import audit_rollouts
    p=evaluation_plan('B',root,target,job)
    os.environ.update(OPENWEBRL_EXPECTED_SCHEDULER_OFFSET='0',OPENWEBRL_RECORD_EVAL_EXPECTED_TASK_COUNT='300')
    evaluator.run(p,REPO/'.env')
    write_json(CONTROL/f'{job}-eval{target}-audit.json',audit_rollouts(Path(p['output'])/'rollouts',Path(p['source'])))


def check_remaining_budget(accounting):
    rows={row[0]:row for row in (line.split('|') for line in accounting.splitlines()) if len(row)>=3}
    if any(job not in rows or rows[job][2]!='FAILED' for job in PREVIOUS_ATTEMPTS):
        raise ValueError('Every prior recovery attempt must be terminal and accounted for')
    used=sum(int(rows[job][1]) for job in PREVIOUS_ATTEMPTS)
    if used+SECONDS>24*3600:
        raise ValueError('Recovery exceeds cumulative unused approved budget')
    return dict(prior_attempts=list(PREVIOUS_ATTEMPTS),used_seconds=used,
                requested_seconds=SECONDS,approved_seconds=24*3600)


def execute(job):
    if os.getenv('ARM_B90_APPROVED')!='8gpu_24h' or os.getenv('SLURM_JOB_ID')!=job:
        raise ValueError('Exact8-H200 x24h approval and allocation required')
    allocation(subprocess.check_output(['scontrol','show','job',job,'-o'],text=True),job,
        requested_gpus=8,maximum_hours=SECONDS/3600)
    used=subprocess.check_output(['sacct','-X','-n','-P','-j',','.join(PREVIOUS_ATTEMPTS),'--format=JobID,ElapsedRaw,State'],text=True)
    budget=check_remaining_budget(used)
    if not prerequisite_ready():raise ValueError('B60 and preceding evaluations must finish before allocation release')
    controller=RUNTIME/f'evaluations/arm-gate-b-recovery-{job}';controller.mkdir(exist_ok=False)
    write_json(controller/'budget.json',budget)
    root=ORIGIN
    step=['srun',f'--jobid={job}','--nodes=1','--ntasks=1','--cpus-per-task=64','--gres=gpu:h200:8',
        '--exact','--cpu-bind=none',str(PYTHON),str(Path(__file__).resolve()),'--job-id',job]
    try:
        for target in (70,80,90):
            p=plan(job,root,target);validate_plan(p)
            write_json(controller/'controller-plan.json',dict(job_id=job,variant='B',target=90,
                current_target=target,training_root=p['output'],resume_from=str(root),hard_stop=90))
            write_json(controller/'status.json',dict(stage='training',complete=False,target=target))
            subprocess.run([*step,'--worker','train','--root',str(root),'--target',str(target)],check=True)
            training=Path(p['output'])
            from run_arm_gate_checkpoint_eval import checkpoint_ready
            if checkpoint_ready('B',training,target) is None:
                write_json(controller/'status.json',dict(stage='budget-stop',complete=False,missing_target=target));return
            write_json(controller/'status.json',dict(stage='evaluation',complete=False,iteration=target))
            subprocess.run([*step,'--worker','eval','--root',str(training),'--target',str(target)],check=True)
            root=training
        write_json(controller/'status.json',dict(stage='complete',complete=True,hard_stop_iteration=90))
    except BaseException as exc:
        write_json(controller/'status.json',dict(stage='failed',complete=False,error_type=type(exc).__name__,error=str(exc)[:400]));raise


if __name__=='__main__':
    # Same-ID recovery uses only the charged attempt's remaining approved time.
    # Its independent controller preserves the interrupted stage and checkpoints.
    if os.getenv('SLURM_JOB_ID') == '331778' and int(os.getenv('SLURM_RESTART_COUNT','0')):
        from recover_arm_browser_slots import execute as recover_browser_slots
        recover_browser_slots('331778')
        raise SystemExit(0)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare',action='store_true');parser.add_argument('--job-id',default='PREPARE')
    parser.add_argument('--worker',choices=('train','eval'));parser.add_argument('--root',type=Path)
    parser.add_argument('--target',type=int,choices=(70,80,90))
    args=parser.parse_args()
    if args.prepare:prepare()
    elif args.worker:
        if os.getenv('SLURM_JOB_ID')!=args.job_id:raise ValueError('Wrong worker allocation')
        (train if args.worker=='train' else evaluate)(args.job_id,args.root,args.target)
    else:execute(args.job_id)
