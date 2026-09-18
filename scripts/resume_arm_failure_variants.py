#!/usr/bin/env python3
"""Continue a frozen failure variant inside an authorized job."""
import argparse,copy,json,os,subprocess,sys
from pathlib import Path
from resume_baseline import write_json,source_command,validate_source
import resume_arm_turn_bonus as resume
import run_arm_turn_bonus_cycles as training
RUNTIME=training.RUNTIME

def plan(job,root,hours):
    initial=resume.origin(root);prior=initial['manifest'];report=initial['checkpoint_report']
    variant=prior.get('experiment')
    if variant not in ('arm-all-failure-bonus','additive-all-failure'):raise ValueError('Not a failure-variant checkpoint')
    if not 0<hours<=30:raise ValueError('Budget exceeds approved range')
    next_id=report['iteration']+1
    target=int(os.environ.get('ARM_VARIANT_TARGET_ITERATION','20'))
    if target <= next_id:raise ValueError('Checkpoint already at/past target')
    if target > 90:raise ValueError('Continuation target exceeds safety cap (90)')
    prefix='arm-turn-bonus-fresh-allfailure' if variant=='arm-all-failure-bonus' else 'arm-failure-additive'
    output=RUNTIME/f'evaluations/{prefix}-{job}';old=prior['output'];p=copy.deepcopy(prior)
    p['command']=[x.replace(old,str(output)) for x in prior['command']]
    p['environment']={k:v.replace(old,str(output)) for k,v in prior['environment'].items()}
    p['arm_config']={k:(v.replace(old,str(output)) if isinstance(v,str) else v) for k,v in prior['arm_config'].items()}
    p['arm_config'].pop('deadline_epoch_seconds',None)
    p.update(job_id=str(job),output=str(output),resume_from=initial['root'],resume_origin=initial,
        variant_continuation=True,initial_optimizer_updates=report['completed_optimizer_updates'],
        start_rollout_id=next_id,requested_iterations=target-next_id,checkpoint=report['checkpoint'],fresh_optimizer=False,
        target_completed_iterations=target)
    p['requested_resources'].update(gpus=4,hours=hours,gpu_hours=4*hours,cpus=32,memory_gib=480,browsers=32)
    p['environment'].update(NUM_GPUS='4',TP_SIZE='4',NUM_ROLLOUT=str(target),WANDB_RESUME='must',
        SLIME_LOAD_CHECKPOINT=str(Path(initial['root'])/'runtime'),BROWSER_CONCURRENCY='32',SGLANG_CONCURRENCY='32',
        OPENWEBRL_MULTIMODAL_STORAGE_DIR=f'/tmp/arm-variant-{job}-multimodal')
    p['browser_config']['browser_rollout_concurrency']=32
    if '--use-checkpoint-opt-param-scheduler' not in p['command']:p['command'].append('--use-checkpoint-opt-param-scheduler')
    p['selector_port']=31000+int(job)%20000 if str(job).isdigit() else 27511
    p['arm_config'].update(selector_endpoint=f"http://127.0.0.1:{p['selector_port']}",checkpoint=report['checkpoint'],policy_id=p['wandb_run_id']+':uninitialized')
    p['gpu_restore_output']=str(Path(initial['root'])/f'gpu-restore-check-{job}-4gpu')
    p['resume_restore_receipt']=str(Path(p['gpu_restore_output'])/'result.json')
    return p

def validate_plan(p):
    resume.validate_resume_plan(p)
    if p!=plan(p['job_id'],p['resume_from'],p['requested_resources']['hours']):raise ValueError('Continuation configuration changed')
    if int(p['environment']['NUM_ROLLOUT'])!=p['target_completed_iterations']:raise ValueError('Target iteration changed')
    return dict(passed=True,source_unchanged=True,checkpoint_counters_verified=True)

def execute(p):
    validate_plan(p)
    if os.environ.get('SLURM_JOB_ID')!=p['job_id']:raise ValueError('Authorized allocation required')
    manifest=RUNTIME/f'logs/arm-variant-resume-{p["job_id"]}.json';write_json(manifest,p)
    subprocess.run(source_command(Path(p['source']),[sys.executable,str(training.REPO/'scripts/check_arm_gpu_checkpoint_restore.py'),
        '--training-root',p['resume_from'],'--job-id',p['job_id'],'--gpus','4','--output',p['gpu_restore_output'],
        '--continuation-plan',str(manifest),'--execute']),check=True)
    resume.validate_resume_plan(p,require_gpu_restore=True)
    with (RUNTIME/f'logs/arm-variant-monitor-{p["job_id"]}.log').open('w') as log:
        monitor=subprocess.Popen([sys.executable,str(training.REPO/'scripts/monitor_arm_turn_bonus.py'),'--job-id',p['job_id'],
            '--watch','--interval','900','--hours',str(p['requested_resources']['hours'])],stdout=log,stderr=subprocess.STDOUT)
        try:training.execute(p)
        finally:
            monitor.terminate()
            try:monitor.wait(timeout=30)
            except subprocess.TimeoutExpired:monitor.kill();monitor.wait()

def main():
    a=argparse.ArgumentParser();a.add_argument('--job-id',default='PREPARE');a.add_argument('--resume-from',type=Path,required=True);a.add_argument('--hours',type=float,required=True);a.add_argument('--execute',action='store_true');args=a.parse_args()
    p=plan(args.job_id,args.resume_from,args.hours)
    if args.execute:execute(p)
    else:print(json.dumps(p,indent=2))
if __name__=='__main__':main()
