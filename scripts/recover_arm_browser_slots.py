#!/usr/bin/env python3
"""Recover B77 within the same job's unused budget; no submissions or deletions."""
import sys
sys.dont_write_bytecode=True
import argparse,copy,hashlib,importlib.util,json,os,shutil,subprocess,time
from pathlib import Path
from resume_baseline import REPO,RUNTIME,allocation,validate_source,source_command,write_json
from prepare_arm_turn_bonus import copy_plain
from resume_arm_turn_bonus import continuation_identity

CONTROL=RUNTIME/'arm-turn-bonus-preparation/browser-slot-recovery-20260927'
PARENT=RUNTIME/'reference-arm-gate-b-resume49-20260924-v1'
SOURCE=RUNTIME/'reference-arm-gate-b-browser-cleanup-20260927-v1'
ORIGIN=RUNTIME/'evaluations/arm-failure-additive-331778-tp2-iter80'
RUN_ID='arm-gate-b-309053'
PYTHON=RUNTIME/'venv/bin/python'


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def prepare_source():
    validate_source(PARENT)
    name='openwebrl/env/local_process_env.py';data=(REPO/name).read_bytes()
    if not SOURCE.exists():
        shutil.copytree(PARENT,SOURCE,symlinks=True,copy_function=copy_plain,
            ignore=shutil.ignore_patterns('__pycache__','*.pyc','.git','.browser_use_sessions'))
        (SOURCE/name).write_bytes(data)
        m=read(PARENT/'reference_manifest.json');m['recipe_files_sha256'][name]=hashlib.sha256(data).hexdigest()
        m['browser_cleanup_recovery']=dict(parent=str(PARENT),changed_files=[name],objective_unchanged=True)
        write_json(SOURCE/'reference_manifest.json',m)
    validate_source(SOURCE)
    old=read(PARENT/'reference_manifest.json')['recipe_files_sha256'];new=read(SOURCE/'reference_manifest.json')['recipe_files_sha256']
    if (SOURCE/name).read_bytes()!=data or any(new.get(k)!=v for k,v in old.items() if k!=name):
        raise ValueError('Source differs outside the tested browser cleanup')


def origin(root):
    root=Path(root).resolve();m=read(root/'launch_manifest.json');s=read(root/'status.json')
    if m['wandb_run_id']!=RUN_ID or Path(m['source']) not in (PARENT,SOURCE):raise ValueError('Wrong lineage/source')
    if not (s.get('failed') or s.get('stage')=='complete'):raise ValueError('Resume requires an inactive stage')
    validate_source(Path(m['source']))
    idx=int((root/'runtime/latest_checkpointed_iteration.txt').read_text());r=read(root/f'iterations/{idx:04d}/checkpoint-validation.json')
    cp=root/f'runtime/iter_{idx:07d}'
    if r['checkpoint']!=str(cp) or r['completed_optimizer_updates']!=s['completed_optimizer_updates'] or r['scheduler_minus_optimizer_updates']!=0:raise ValueError('Counters disagree')
    for f,size in r['shard_files'].items():
        if (cp/f).stat().st_size!=size:raise ValueError('Shard changed')
    files=[cp/'common.pt',cp/'.metadata',Path(r['dataset_cursor']),root/'runtime/latest_checkpointed_iteration.txt']
    return dict(root=str(root),manifest=m,checkpoint_report=r,identity_sha256={str(f):sha(f) for f in files})


def plan(job,root,target):
    if job!='331778' or target not in (80,90):raise ValueError('Only approved B continuation to80/90')
    prepare_source();o=origin(root);prior=o['manifest'];r=o['checkpoint_report'];start=r['iteration']+1
    if start!=(77 if target==80 else 80):raise ValueError('Unexpected resume iteration')
    expected=ORIGIN if target==80 else RUNTIME/f'evaluations/arm-failure-additive-{job}-slots-r1-iter80'
    if Path(root).resolve()!=expected:raise ValueError('Unexpected parent')
    budget=read(CONTROL/f'{job}-requeue-budget.json');out=RUNTIME/f'evaluations/arm-failure-additive-{job}-slots-r1-iter{target}'
    p=copy.deepcopy(prior);old=prior['output'];old_source=prior['source']
    def replace(v):return v.replace(old,str(out)).replace(old_source,str(SOURCE)) if isinstance(v,str) else v
    p['command']=[replace(v) for v in p['command']]
    for key in ('environment','arm_config'):p[key]={k:replace(v) for k,v in p[key].items()}
    p['arm_config'].pop('deadline_epoch_seconds',None)
    p.update(job_id=job,source=str(SOURCE),output=str(out),resume_from=str(root),resume_origin=o,
        start_rollout_id=start,requested_iterations=target-start,target_completed_iterations=target,
        initial_optimizer_updates=r['completed_optimizer_updates'],checkpoint=r['checkpoint'],
        fresh_optimizer=False,variant_continuation=True,browser_slot_recovery=True,evaluation_in_allocation=True,
        evaluation_reserve_seconds=3600,gpu_restore_output=str(Path(root)/f'gpu-restore-slots-{job}-to{target}'))
    p['resume_restore_receipt']=str(Path(p['gpu_restore_output'])/'result.json')
    p['requested_resources'].update(gpus=8,cpus=64,memory_gib=960,browsers=64,hours=budget['requested_seconds']/3600,gpu_hours=8*budget['requested_seconds']/3600)
    p['environment'].update(NUM_GPUS='8',TP_SIZE='2',NUM_ROLLOUT=str(target),
        SLIME_LOAD_CHECKPOINT=str(Path(root)/'runtime'),PYTHONDONTWRITEBYTECODE='1',WANDB_RESUME='must')
    p['arm_config'].update(checkpoint=r['checkpoint'],policy_id=RUN_ID+':uninitialized',minimum_cycle_seconds=2700)
    p['recovery_note']='Browser cleanup ownership fix; B77/988 updates to80/90 within unused approved time; partial78 artifacts preserved.'
    return p


def fingerprints():
    paths=[Path(__file__),SOURCE/'reference_manifest.json',CONTROL/'training-controller.py',CONTROL/'restore-controller.py',REPO/'tests/test_local_browser_cleanup.py']
    return {str(p.resolve()):sha(p) for p in paths}


def validate_plan(p):
    if read(CONTROL/'readiness.json')['fingerprints']!=fingerprints():raise ValueError('Recovery implementation changed')
    expected=plan(p['job_id'],Path(p['resume_from']),p['target_completed_iterations'])
    if expected!=p:raise ValueError('Continuation plan changed')
    c=p['arm_config']
    if (c['beta'],c['scored_fraction'],c['failure_group_cap'],c['candidate_gate'],c['credit_assignment'])!=(.5,.2,8,'min2','response_index'):raise ValueError('Objective changed')
    if p['environment']['TP_SIZE']!='2' or p['browser_config']['browser_rollout_concurrency']!=64:raise ValueError('Topology/concurrency changed')
    return dict(passed=True,browser_cleanup_only=True,restore_optimizer_scheduler_cursor=True)


def validate_resume_plan(p,require_gpu_restore=False):
    current=origin(p['resume_from'])
    if current!=p['resume_origin']:raise ValueError('Resume identity changed')
    r=current['checkpoint_report']
    if p['initial_optimizer_updates']!=r['completed_optimizer_updates'] or p['start_rollout_id']!=r['iteration']+1:raise ValueError('Wrong resume counters')
    if require_gpu_restore:
        receipt=read(p['resume_restore_receipt'])
        expected=dict(passed=True,job_id=p['job_id'],source=p['source'],source_checkpoint_root=str(Path(p['resume_from'])/'runtime'),
            checkpoint_completed_optimizer_updates=p['initial_optimizer_updates'],loaded_iteration=p['start_rollout_id']-1,
            next_rollout_id=p['start_rollout_id'],gpus=8,optimizer_updates_executed=0,browser_collections_executed=0,
            continuation_identity_sha256=continuation_identity(p))
        if any(receipt.get(k)!=v for k,v in expected.items()):raise ValueError('Missing exact native GPU restore evidence')


def prepare():
    prepare_source()
    for name in ['training-controller.py','restore-controller.py']:
        text=(REPO/'logs/arm-b-recovery-20260926'/name).read_text()
        text=text.replace('from recover_arm_b69 import validate_plan','from recover_arm_browser_slots import validate_plan')
        text=text.replace('from resume_arm_turn_bonus import validate_resume_plan','from recover_arm_browser_slots import validate_resume_plan')
        (CONTROL/name).write_text(text)
        compile(text,name,'exec')
    write_json(CONTROL/'readiness.json',dict(fingerprints=fingerprints(),cleanup_tests_passed=3,source_change_only='openwebrl/env/local_process_env.py'))
    p=plan('331778',ORIGIN,80);validate_plan(p);write_json(CONTROL/'B77-to80-plan.json',p)
    print(json.dumps(dict(prepared=True,checkpoint=p['checkpoint'],target=80,remaining_seconds=15060)))


def train(job,root,target):
    p=plan(job,root,target);validate_plan(p)
    from resume_arm_failure_variants import check_multimodal_storage
    check_multimodal_storage(p)
    manifest=CONTROL/f'{job}-to{target}-launch.json';write_json(manifest,p)
    probe=f"import sys,runpy;sys.path.insert(0,{str(REPO/'scripts')!r});runpy.run_path({str(CONTROL/'restore-controller.py')!r},run_name='__main__')"
    subprocess.run(source_command(SOURCE,[str(PYTHON),'-B','-c',probe,'--training-root',str(root),'--job-id',job,
        '--gpus','8','--output',p['gpu_restore_output'],'--continuation-plan',str(manifest),'--execute']),check=True)
    validate_resume_plan(p,require_gpu_restore=True)
    spec=importlib.util.spec_from_file_location('slot_training',CONTROL/'training-controller.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.execute(p)


def evaluate(job,root,target):
    from prepare_arm_gate_to60 import evaluation_plan
    import evaluate_baseline_checkpoint as e
    from run_stage1_to100 import audit_rollouts
    p=evaluation_plan('B',root,target,job);p['environment']['PYTHONDONTWRITEBYTECODE']='1'
    os.environ.update(OPENWEBRL_EXPECTED_SCHEDULER_OFFSET='0',OPENWEBRL_RECORD_EVAL_EXPECTED_TASK_COUNT='300')
    e.run(p,REPO/'.env');write_json(CONTROL/f'{job}-eval{target}-audit.json',audit_rollouts(Path(p['output'])/'rollouts',Path(p['source'])))


def execute(job):
    budget=read(CONTROL/f'{job}-requeue-budget.json')
    if job!='331778' or os.environ.get('SLURM_JOB_ID')!=job or int(os.getenv('SLURM_RESTART_COUNT','0'))!=budget['attempt']:raise ValueError('Wrong recovery attempt')
    if budget['previous_elapsed_seconds']+budget['requested_seconds']>budget['approved_this_job_seconds']:raise ValueError('Budget extension prohibited')
    allocation(subprocess.check_output(['scontrol','show','job',job,'-o'],text=True),job,requested_gpus=8,maximum_hours=budget['requested_seconds']/3600)
    out=RUNTIME/f'evaluations/arm-gate-b-slot-recovery-{job}-r1';out.mkdir(exist_ok=False)
    write_json(out/'budget.json',budget);root=ORIGIN
    step=['srun',f'--jobid={job}','--nodes=1','--ntasks=1','--cpus-per-task=64','--gres=gpu:h200:8','--exact','--cpu-bind=none',str(PYTHON),'-B',str(Path(__file__).resolve()),'--job-id',job]
    try:
        for target in (80,90):
            p=plan(job,root,target);validate_plan(p)
            write_json(out/'controller-plan.json',dict(job_id=job,variant='B',target=90,current_target=target,training_root=p['output'],resume_from=str(root),hard_stop=90))
            write_json(out/'status.json',dict(stage='training',target=target,complete=False))
            subprocess.run([*step,'--worker','train','--root',str(root),'--target',str(target)],check=True)
            training=Path(p['output'])
            from run_arm_gate_checkpoint_eval import checkpoint_ready
            if checkpoint_ready('B',training,target) is None:
                write_json(out/'status.json',dict(stage='budget-stop',complete=False,missing_target=target));return
            write_json(out/'status.json',dict(stage='evaluation',iteration=target,complete=False))
            subprocess.run([*step,'--worker','eval','--root',str(training),'--target',str(target)],check=True)
            root=training
        write_json(out/'status.json',dict(stage='complete',complete=True))
    except BaseException as exc:
        write_json(out/'status.json',dict(stage='failed',complete=False,error_type=type(exc).__name__,error=str(exc)[:400]));raise


if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--prepare',action='store_true');a.add_argument('--job-id',default='331778');a.add_argument('--worker',choices=['train','eval']);a.add_argument('--root',type=Path);a.add_argument('--target',type=int,choices=[80,90]);o=a.parse_args()
    if o.prepare:prepare()
    elif o.worker:
        if os.getenv('SLURM_JOB_ID')!=o.job_id:raise ValueError('Wrong worker allocation')
        (train if o.worker=='train' else evaluate)(o.job_id,o.root,o.target)
    else:execute(o.job_id)
