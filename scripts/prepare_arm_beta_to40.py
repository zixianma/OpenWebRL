#!/usr/bin/env python3
"""Prepare beta1's bounded20->30->40 continuation; never submit Slurm jobs."""
import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess

from resume_baseline import REPO,RUNTIME,allocation,source_command,validate_source,write_json

CONTROL=REPO/'logs/arm-beta-to40-20260925'
ORIGIN=RUNTIME/'evaluations/arm-failure-additive-318949-tp2-iter20'
SOURCE=RUNTIME/'reference-arm-failure-ablations-tp2-20260922-v1'
PYTHON=RUNTIME/'venv/bin/python'
RUN_ID='arm-failure-weight-fromzero-318949'
RESOURCES=dict(gpus=8,hours=14,gpu_hours=112,cpus=64,memory_gib=960,browsers=64)


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
    if state['stage']!='complete' or not state.get('has_saved_checkpoints'):
        raise ValueError('Requires a completed inactive training stage')
    if manifest['wandb_run_id']!=RUN_ID or manifest['source']!=str(SOURCE):
        raise ValueError('Beta training lineage or source changed')
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


def plan(job,root=ORIGIN,target=30):
    if target not in (30,40):raise ValueError('Beta continuation is capped at40')
    origin=cached_origin(root);prior=origin['manifest'];report=origin['checkpoint_report']
    start=report['iteration']+1
    if start!=target-10:raise ValueError('Only20->30 and30->40 stages are permitted')
    validate_source(SOURCE)
    out=RUNTIME/f'evaluations/arm-failure-additive-{job}-tp2-iter{target}'
    old=prior['output'];p=copy.deepcopy(prior)
    for key in ['command','environment','arm_config']:
        if isinstance(p[key],list):p[key]=[x.replace(old,str(out)) for x in p[key]]
        else:p[key]={k:v.replace(old,str(out)) if isinstance(v,str) else v for k,v in p[key].items()}
    for key in ['deadline_epoch_seconds','beta_to40_continuation']:p['arm_config'].pop(key,None)
    p.update(job_id=str(job),output=str(out),resume_from=str(root),resume_origin=origin,
        variant_continuation=True,beta_to40_continuation=True,fresh_optimizer=False,
        initial_optimizer_updates=report['completed_optimizer_updates'],start_rollout_id=start,
        requested_iterations=10,target_completed_iterations=target,checkpoint=report['checkpoint'],
        requested_resources=dict(RESOURCES),evaluation_in_allocation=True,
        evaluation_reserve_seconds=7200 if target==30 else 3600)
    scratch=RUNTIME/f'multimodal-scratch/arm-variant-{job}'
    p['multimodal_storage']=dict(mode='shared',directory=str(scratch),minimum_free_bytes=2*1024**4,
        retention='preserve; no automatic deletion')
    p['environment'].update(NUM_GPUS='8',TP_SIZE='2',NUM_ROLLOUT=str(target),WANDB_RESUME='must',
        WANDB_RUN_ID=RUN_ID,WANDB_PROJECT='openwebrl',SLIME_LOAD_CHECKPOINT=str(Path(root)/'runtime'),
        BROWSER_CONCURRENCY='64',SGLANG_CONCURRENCY='64',OPENWEBRL_MULTIMODAL_STORAGE_DIR=str(scratch))
    p['browser_config']['browser_rollout_concurrency']=64
    if '--use-checkpoint-opt-param-scheduler' not in p['command']:p['command'].append('--use-checkpoint-opt-param-scheduler')
    p['selector_port']=31000+int(job)%20000 if str(job).isdigit() else 27511
    p['arm_config'].update(checkpoint=report['checkpoint'],policy_id=RUN_ID+':uninitialized',
        selector_endpoint=f"http://127.0.0.1:{p['selector_port']}",minimum_cycle_seconds=2700,
        seconds_per_optimizer_update=90)
    p['gpu_restore_output']=str(Path(root)/f'gpu-restore-check-{job}-beta-to{target}')
    p['resume_restore_receipt']=str(Path(p['gpu_restore_output'])/'result.json')
    p['comparison'].update(endpoint=40,continuation_from_iteration=20,automatic_extension=False)
    return p


def fingerprints():
    paths=[Path(__file__).resolve(),CONTROL/'training-controller.py',CONTROL/'restore-controller.py',
        SOURCE/'reference_manifest.json',REPO/'scripts/resume_arm_turn_bonus.py']
    return {str(p):sha(p) for p in paths}


def validate_plan(p):
    if read(CONTROL/'readiness.json')['fingerprints']!=fingerprints():
        raise ValueError('Beta40 readiness changed')
    if p!=plan(p['job_id'],Path(p['resume_from']),p['target_completed_iterations']):
        raise ValueError('Bounded continuation plan changed')
    c=p['arm_config']
    if (c['beta'],c['scored_fraction'],c['failure_beta'],c['failure_scored_fraction'],c['failure_group_cap'])!=(.5,.2,1.,.2,8):
        raise ValueError('Beta objective changed')
    return dict(passed=True,target=p['target_completed_iterations'],maximum_iteration=40,
        objective_unchanged=True,restore_optimizer_scheduler_cursor=True,native_gpu_restore_required=True)


def prepare():
    CONTROL.mkdir(parents=True,exist_ok=True)
    for old,new in [('run_arm_turn_bonus_cycles.py','training-controller.py'),
                    ('check_arm_gpu_checkpoint_restore.py','restore-controller.py')]:
        text=(REPO/'scripts'/old).read_text()
        before='from prepare_arm_failure_ablations import validate_plan'
        if text.count(before)!=1:raise ValueError('Controller validation boundary changed')
        text=text.replace(before,'from prepare_arm_beta_to40 import validate_plan')
        text=text.replace('if reserve not in (0,3600):','if reserve not in (0,3600,7200):')
        compile(text,new,'exec')
        path=CONTROL/new
        if path.exists() and path.read_text()!=text:raise ValueError('Frozen controller changed')
        path.write_text(text)
    p=plan('PREPARE')
    write_json(CONTROL/'readiness.json',dict(fingerprints=fingerprints(),gpu_restore_pending=True,
        validation='Saved native validation receipt, shard sizes and metadata/cursor identities; no tensor loading on login'))
    validate_plan(p);write_json(CONTROL/'training-plan.json',p)
    intervals=[]
    files=sorted((ORIGIN/'iterations').glob('*/checkpoint-saved.json'))
    rows=[read(f) for f in files]
    for a,b in zip(rows,rows[1:]):
        if b['rollout_id']==a['rollout_id']+1:intervals.append((b['saved_at_epoch_seconds']-a['saved_at_epoch_seconds'])/60)
    import statistics
    median=statistics.median(intervals[-10:])
    write_json(CONTROL/'proposal.json',dict(submitted=False,resource_approval_pending=True,resources=RESOURCES,
        origin=str(ORIGIN),initial_completed_iteration=20,initial_adam_updates=298,hard_stop_iteration=40,
        stages=['restore20','train21-30','full300-eval30','restore30','train31-40','full300-eval40'],
        recent_median_minutes_per_iteration=median,estimated_training_hours=20*median/60,
        no_automatic_extension=True,decision='Review completed20/30/40 task success; stop at40 without further approval'))


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
    spec=importlib.util.spec_from_file_location('beta40_training',CONTROL/'training-controller.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    with (CONTROL/f'{job}-monitor.log').open('a') as log:
        monitor=subprocess.Popen([str(PYTHON),str(REPO/'scripts/monitor_arm_turn_bonus.py'),'--job-id',job,
            '--watch','--interval','900','--hours','14'],stdout=log,stderr=subprocess.STDOUT)
        try:module.execute(p)
        finally:
            monitor.terminate()
            try:monitor.wait(timeout=30)
            except subprocess.TimeoutExpired:monitor.kill();monitor.wait()


def evaluate(job,root,target):
    from recover_arm_unfinished import evaluation_plan
    import evaluate_baseline_checkpoint as evaluator
    from run_stage1_to100 import audit_rollouts
    p=evaluation_plan('weight',job,target,root,8)
    os.environ.update(OPENWEBRL_EXPECTED_SCHEDULER_OFFSET='0',OPENWEBRL_RECORD_EVAL_EXPECTED_TASK_COUNT='300')
    evaluator.run(p,REPO/'.env')
    write_json(CONTROL/f'{job}-eval{target}-audit.json',audit_rollouts(Path(p['output'])/'rollouts',Path(p['source'])))


def execute(job):
    if os.getenv('ARM_BETA40_APPROVED')!='8gpu_14h' or os.getenv('SLURM_JOB_ID')!=job:
        raise ValueError('Exact8-H200 x14h approval and allocation required')
    allocation(subprocess.check_output(['scontrol','show','job',job,'-o'],text=True),job,
        requested_gpus=8,maximum_hours=14)
    controller=RUNTIME/f'evaluations/arm-beta-to40-{job}';controller.mkdir(exist_ok=False)
    root=ORIGIN
    step=['srun',f'--jobid={job}','--nodes=1','--ntasks=1','--cpus-per-task=64','--gres=gpu:h200:8',
        '--exact','--cpu-bind=none',str(PYTHON),str(Path(__file__).resolve()),'--job-id',job]
    try:
        for target in (30,40):
            p=plan(job,root,target);validate_plan(p)
            write_json(controller/'controller-plan.json',dict(job_id=job,variant='weight',target=40,
                current_target=target,training_root=p['output'],resume_from=str(root),hard_stop=40))
            write_json(controller/'status.json',dict(stage='training',complete=False,target=target))
            subprocess.run([*step,'--worker','train','--root',str(root),'--target',str(target)],check=True)
            training=Path(p['output'])
            from run_arm_iteration80_eval import checkpoint_ready
            if checkpoint_ready('additive',target,training) is None:
                write_json(controller/'status.json',dict(stage='budget-stop',complete=False,missing_target=target));return
            write_json(controller/'status.json',dict(stage='evaluation',complete=False,iteration=target))
            subprocess.run([*step,'--worker','eval','--root',str(training),'--target',str(target)],check=True)
            root=training
        write_json(controller/'status.json',dict(stage='complete',complete=True,hard_stop_iteration=40))
    except BaseException as exc:
        write_json(controller/'status.json',dict(stage='failed',complete=False,error_type=type(exc).__name__,error=str(exc)[:400]));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare',action='store_true');parser.add_argument('--job-id',default='PREPARE')
    parser.add_argument('--worker',choices=('train','eval'));parser.add_argument('--root',type=Path)
    parser.add_argument('--target',type=int,choices=(30,40))
    args=parser.parse_args()
    if args.prepare:prepare()
    elif args.worker:
        if os.getenv('SLURM_JOB_ID')!=args.job_id:raise ValueError('Wrong worker allocation')
        (train if args.worker=='train' else evaluate)(args.job_id,args.root,args.target)
    else:execute(args.job_id)
