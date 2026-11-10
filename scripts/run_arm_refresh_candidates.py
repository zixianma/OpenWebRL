#!/usr/bin/env python3
"""Own the approved two-H200, two-hour baseline40 candidate-generation stage."""
import argparse,hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path
import evaluate_baseline_checkpoint as evaluation
from prepare_arm_turn_bonus import copy_plain
from resume_baseline import allocation,source_command,validate_source,write_json
from run_arm_quality_audit import environment,stop
from runtime_ports import PORT_ENV,lease_ports

REPO,R=evaluation.REPO,evaluation.RUNTIME
C=R/'arm-turn-bonus-preparation/arm-refresh-20260924'
PARENT=R/'reference-arm-quality-audit-20260921-v2'
SOURCE=R/'reference-arm-refresh-baseline40-20260925-v1'
CHANGED=['openwebrl/arm_refresh_generation.py','openwebrl/arm_refresh_format.py']

def prepare_source():
    validate_source(PARENT)
    if not SOURCE.exists():
        shutil.copytree(PARENT,SOURCE,symlinks=True,copy_function=copy_plain,
            ignore=shutil.ignore_patterns('__pycache__','*.pyc','.git','.browser_use_sessions'))
        for p in CHANGED:copy_plain(REPO/p,SOURCE/p)
        m=json.loads((SOURCE/'reference_manifest.json').read_text())
        m['recipe_files_sha256'].update({p:hashlib.sha256((SOURCE/p).read_bytes()).hexdigest() for p in CHANGED})
        m['arm_refresh_candidates']=dict(parent=str(PARENT),optimizer_updates=0,browsers=0)
        write_json(SOURCE/'reference_manifest.json',m)
    validate_source(SOURCE)
    if any((REPO/p).read_bytes()!=(SOURCE/p).read_bytes() for p in CHANGED):raise ValueError('Frozen source changed')

def plan(job):
    approval=json.loads((C/'baseline40-approval.json').read_text())
    if not approval['approved'] or approval['gpu_hours_cap']!=8:raise ValueError('Missing approved budget')
    prepare_source();manifest=json.loads((C/'baseline40/states-manifest.json').read_text())
    from openwebrl.arm_refresh_generation import validate_inputs
    validate_inputs(manifest)
    root=R/f'evaluations/arm-refresh-candidates-{job}'
    p=evaluation.build_plan(SOURCE,manifest['actor_checkpoint'],root/'baseline40',job,gpus=2)
    cmd=p['command'];i=cmd.index('--save-debug-rollout-data');del cmd[i:i+2]
    cmd+=['--eval-function-path','openwebrl.arm_refresh_generation.generate_rollout']
    cmd[cmd.index('--wandb-group')+1]='ARM-refresh-baseline40-candidates'
    p['environment'].pop('OPENWEBRL_EVAL_ROLLOUT_DIR',None)
    p['environment'].update(RESPONSE_LEN='4096',SGLANG_CONCURRENCY='48',BROWSER_CONCURRENCY='1',
        WANDB_RUN_ID=f'arm-refresh-candidates-{job}',WANDB_RESUME='never',
        OPENWEBRL_ARM_REFRESH_CONFIG=str(root/'baseline40/generation-config.json'))
    p['wandb_run_id']=p['environment']['WANDB_RUN_ID']
    p['protocol']='offline baseline40 fresh candidates; no terminal evaluation or training'
    p['generation_config']={k:manifest[k] for k in ('states','states_sha256','states_count','targets')}
    p['generation_config'].update(checkpoint=p['checkpoint'],output=p['output'])
    return dict(job_id=str(job),output=str(root),stage=evaluation.configure_evaluation_tracking(p,'openwebrl-evals'),
        approval=str(C/'baseline40-approval.json'),resources=dict(gpus=2,hours=2,cpus=16,memory_gib=480))

def worker(p):
    job=p['job_id']
    if os.getenv('SLURM_JOB_ID')!=job or f'/job_{job}/' not in Path('/proc/self/cgroup').read_text():
        raise ValueError('Dedicated approved GPU job required')
    stage=p['stage'];root=Path(stage['output']);root.mkdir(parents=True,exist_ok=False)
    write_json(root/'evaluation_manifest.json',stage);write_json(root/'generation-config.json',stage['generation_config'])
    env=environment();env.update(stage['environment'])
    if len(env.get('CUDA_VISIBLE_DEVICES','').split(','))!=2:raise ValueError('Expected two assigned GPUs')
    ckpt=Path(stage['checkpoint']);view=root/'checkpoint-view';view.mkdir()
    (view/ckpt.name).symlink_to(ckpt,target_is_directory=True)
    (view/'rollout').symlink_to(ckpt.parent/'rollout',target_is_directory=True)
    (view/'latest_checkpointed_iteration.txt').write_text('39\n')
    subprocess.run(source_command(SOURCE,[sys.executable,str(REPO/'scripts/inspect_training_checkpoint.py'),str(view),
        '--expected-scheduler-offset-updates','1','--report',str(root/'checkpoint_validation.json')]),env=env,check=True,stdout=subprocess.DEVNULL)
    lease,base=lease_ports(job);env[PORT_ENV]=str(base);child=None
    try:
        with (root/'evaluation.log').open('w') as log:
            child=subprocess.Popen(source_command(SOURCE,stage['command']),env=env,cwd=SOURCE,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            while child.poll() is None:
                evaluation.record_restore_evidence(stage);time.sleep(10)
            evaluation.record_restore_evidence(stage)
            if child.returncode:raise RuntimeError(f'Native worker exited {child.returncode}')
        if not (root/'checkpoint_restore_evidence.json').exists():raise ValueError('Missing GPU restoration evidence')
        result=json.loads((root/'collection-complete.json').read_text())
        if not result['complete']:raise ValueError('Candidate quota incomplete')
        write_json(root/'status.json',result)
    finally:stop(child);lease.close()

def execute(p):
    job=p['job_id']
    if os.getenv('SLURM_JOB_ID')!=job:raise ValueError('Only execute inside approved allocation')
    a=allocation(subprocess.check_output(['scontrol','show','job',job,'-o'],text=True),job,requested_gpus=2,maximum_hours=2)
    if a['cpus']<16 or a['allocated_memory_gib']<480:raise ValueError('Resource mismatch')
    root=Path(p['output']);root.mkdir(exist_ok=False);write_json(root/'launch-plan.json',p)
    child=None
    try:
        child=subprocess.Popen(['srun',f'--jobid={job}','--nodes=1','--ntasks=1','--cpus-per-task=16',
            '--gres=gpu:h200:2','--exact','--cpu-bind=none',sys.executable,str(Path(__file__).resolve()),
            '--job-id',job,'--worker'],start_new_session=True)
        code=child.wait()
        if code:raise RuntimeError(f'Candidate worker failed: {code}')
        result=json.loads((Path(p['stage']['output'])/'collection-complete.json').read_text())
        write_json(root/'status.json',dict(result,stage='complete'))
    except BaseException as exc:
        write_json(root/'status.json',dict(complete=False,stage='failed',error_type=type(exc).__name__,error=str(exc)[:300]));raise
    finally:stop(child)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--job-id',default='PREPARE')
    parser.add_argument('--execute',action='store_true');parser.add_argument('--worker',action='store_true');a=parser.parse_args()
    if a.worker:worker(json.loads((R/f'evaluations/arm-refresh-candidates-{a.job_id}/launch-plan.json').read_text()))
    else:
        p=plan(a.job_id)
        if a.execute:execute(p)
        else:
            write_json(C/'baseline40/candidate-launch-plan.json',p)
            print(json.dumps(dict(resources=p['resources'],states=p['stage']['generation_config']['states_count'],source=str(SOURCE)),indent=2))
