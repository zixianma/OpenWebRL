#!/usr/bin/env python3
"""Continue a frozen failure variant inside an authorized job."""
import argparse,copy,json,os,shutil,subprocess,sys
from pathlib import Path
from resume_baseline import write_json,source_command,validate_source
import resume_arm_turn_bonus as resume
import run_arm_turn_bonus_cycles as training
RUNTIME=training.RUNTIME

def configure_multimodal_storage(p):
    """Opt in to shared scratch without changing frozen training code."""
    mode=os.environ.get('ARM_VARIANT_MULTIMODAL_STORAGE','local')
    if mode not in ('local','shared'):
        raise ValueError('Unknown ARM continuation multimodal storage mode')
    if mode == 'shared':
        directory=RUNTIME/'multimodal-scratch'/f'arm-variant-{p["job_id"]}'
        p['environment']['OPENWEBRL_MULTIMODAL_STORAGE_DIR']=str(directory)
        p['multimodal_storage']=dict(mode=mode,directory=str(directory),
            minimum_free_bytes=2*1024**4,retention='preserve; no automatic deletion')

def check_multimodal_storage(p):
    """Fail before GPU restoration if shared scratch cannot accept the run."""
    spec=p.get('multimodal_storage')
    if not spec:
        return None
    from check_storage_quota import check_storage_quota
    quota=check_storage_quota(spec['minimum_free_bytes'])
    directory=Path(spec['directory'])
    expected=RUNTIME/'multimodal-scratch'/f'arm-variant-{p["job_id"]}'
    if (directory != expected or spec['mode'] != 'shared'
            or p['environment']['OPENWEBRL_MULTIMODAL_STORAGE_DIR'] != str(directory)):
        raise ValueError('Shared scratch identity changed')
    directory.mkdir(parents=True,exist_ok=True)
    if directory.resolve() != expected.absolute():
        raise ValueError('Shared scratch cannot redirect through symlinks')
    usage=shutil.disk_usage(directory)
    if usage.free < spec['minimum_free_bytes']:
        raise OSError('Insufficient shared scratch space before GPU restoration')
    probe=directory/'.write-probe'
    payload=b'OpenWebRL shared scratch write/read probe\n'*100
    with probe.open('wb') as handle:
        handle.write(payload);handle.flush();os.fsync(handle.fileno())
    if probe.read_bytes()!=payload:
        raise OSError('Shared scratch write/read probe failed')
    return dict(passed=True,directory=str(directory),free_bytes=usage.free,
        minimum_free_bytes=spec['minimum_free_bytes'],probe_bytes=len(payload),
        filesystem_free_is_not_user_quota=True,personal_quota=quota)

def plan(job,root,hours,gpus=4):
    initial=resume.origin(root);prior=initial['manifest'];report=initial['checkpoint_report']
    variant=prior.get('experiment')
    if variant not in ('arm-all-failure-bonus','additive-all-failure'):raise ValueError('Not a failure-variant checkpoint')
    if not 0<hours<=24:raise ValueError('Budget exceeds the normal-QoS 24-hour limit')
    if gpus not in (4,8):raise ValueError('Continuation supports four or eight GPUs')
    next_id=report['iteration']+1
    target=int(os.environ.get('ARM_VARIANT_TARGET_ITERATION','20'))
    if target <= next_id:raise ValueError('Checkpoint already at/past target')
    if target > 100:raise ValueError('Continuation target exceeds safety cap (100)')
    prefix='arm-turn-bonus-fresh-allfailure' if variant=='arm-all-failure-bonus' else 'arm-failure-additive'
    tag=os.environ.get('ARM_VARIANT_OUTPUT_STAGE','')
    ablation=prior.get('failure_ablation_training')
    gate_stage=prior.get('gate_ablation') in ('B','C') and target in (40,60)
    ablation_stage=ablation in ('weight','coverage') and target==20
    if tag and (tag!=f'iter{target}' or not (gate_stage or ablation_stage)):
        raise ValueError('Unsupported controller stage suffix')
    tp_override=os.environ.get('ARM_VARIANT_TP_SIZE')
    if tp_override and (tp_override!='2' or gpus!=8 or not (gate_stage or ablation_stage) or not tag):
        raise ValueError('TP override requires an eight-GPU staged ARM TP2 continuation')
    output=RUNTIME/f'evaluations/{prefix}-{job}{"-tp2" if tp_override else ""}{"-"+tag if tag else ""}';old=prior['output'];p=copy.deepcopy(prior)
    p['command']=[x.replace(old,str(output)) for x in prior['command']]
    p['environment']={k:v.replace(old,str(output)) for k,v in prior['environment'].items()}
    p['arm_config']={k:(v.replace(old,str(output)) if isinstance(v,str) else v) for k,v in prior['arm_config'].items()}
    p['arm_config'].pop('deadline_epoch_seconds',None)
    p.update(job_id=str(job),output=str(output),resume_from=initial['root'],resume_origin=initial,
        variant_continuation=True,initial_optimizer_updates=report['completed_optimizer_updates'],
        start_rollout_id=next_id,requested_iterations=target-next_id,checkpoint=report['checkpoint'],fresh_optimizer=False,
        target_completed_iterations=target)
    p['requested_resources'].update(gpus=gpus,hours=hours,gpu_hours=gpus*hours,cpus=8*gpus,memory_gib=120*gpus,browsers=8*gpus)
    p['environment'].update(NUM_GPUS=str(gpus),TP_SIZE=str(gpus),NUM_ROLLOUT=str(target),WANDB_RESUME='must',
        SLIME_LOAD_CHECKPOINT=str(Path(initial['root'])/'runtime'),BROWSER_CONCURRENCY=str(8*gpus),SGLANG_CONCURRENCY=str(8*gpus),
        OPENWEBRL_MULTIMODAL_STORAGE_DIR=f'/tmp/arm-variant-{job}-multimodal')
    configure_multimodal_storage(p)
    if os.environ.get('ARM_VARIANT_PORT_SOURCE'):
        if p.get('gate_ablation') in ('B','C'):
            from prepare_arm_gate_to60 import SOURCES,prepare_sources
            destination=SOURCES[p['gate_ablation']]
        else:
            from prepare_stage1_to100 import ADDITIVE_SOURCE, prepare_sources
            destination=ADDITIVE_SOURCE
        prepare_sources()
        if os.environ['ARM_VARIANT_PORT_SOURCE'] != str(destination) or variant != 'additive-all-failure':
            raise ValueError('Unrecognized port-isolated continuation source')
        old_source=p['source'];p['source']=str(destination)
        p['command']=[x.replace(old_source,str(destination)) for x in p['command']]
        p['environment']={k:v.replace(old_source,str(destination)) for k,v in p['environment'].items()}
    if tp_override:
        if ablation_stage:
            from prepare_arm_ablation_dp import SOURCE,prepare_source
            prepare_source();destination=SOURCE
            p['ablation_dp']=True
        else:
            from prepare_arm_gate_dp import SOURCES,prepare_sources
            prepare_sources()
            destination=SOURCES[p['gate_ablation']]
        old_source=p['source'];p['source']=str(destination)
        p['command']=[x.replace(old_source,str(destination)) for x in p['command']]
        p['environment']={k:v.replace(old_source,str(destination)) for k,v in p['environment'].items()}
        p['environment']['TP_SIZE']='2'
        p['topology_continuation']='TP2/DP4/microbatch1'
    reserve=int(os.environ.get('ARM_VARIANT_EVAL_RESERVE_SECONDS','0'))
    if reserve not in (0,3600):raise ValueError('Unsupported evaluation reserve')
    if reserve:p['evaluation_reserve_seconds']=reserve
    if p.get('gate_ablation') in ('B', 'C'):
        credit_name=('response-index' if p['gate_ablation']=='B' else 'duplicate-aware')
        display_name=f'ARM-{p["gate_ablation"]} | relaxed gate | {credit_name} credit'
        p['command'][p['command'].index('--wandb-group')+1]=display_name
    p['browser_config']['browser_rollout_concurrency']=8*gpus
    if '--use-checkpoint-opt-param-scheduler' not in p['command']:p['command'].append('--use-checkpoint-opt-param-scheduler')
    p['selector_port']=31000+int(job)%20000 if str(job).isdigit() else 27511
    p['arm_config'].update(selector_endpoint=f"http://127.0.0.1:{p['selector_port']}",checkpoint=report['checkpoint'],policy_id=p['wandb_run_id']+':uninitialized')
    p['gpu_restore_output']=str(Path(initial['root'])/f'gpu-restore-check-{job}-{gpus}gpu{"-tp2" if tp_override else ""}')
    p['resume_restore_receipt']=str(Path(p['gpu_restore_output'])/'result.json')
    if os.environ.get('ARM_CONTINUATION_LABEL_GUARD') == 'adjacent_batch_v1':
        from prepare_arm_failure_continuation import SOURCE, validate_frozen_source
        if variant != 'arm-all-failure-bonus':raise ValueError('Count guard is only prepared for all-failure')
        validate_frozen_source()
        old_source=p['source']
        p['source']=str(SOURCE)
        p['command']=[x.replace(old_source,str(SOURCE)) for x in p['command']]
        p['environment']={k:v.replace(old_source,str(SOURCE)) for k,v in p['environment'].items()}
        p['continuation_label_guard']='adjacent_batch_v1'
        p['arm_config'].update(label_count_guard='adjacent_batch_v1',
            label_guard_start_rollout_id=next_id,label_guard_resume_from=initial['root'])
    elif os.environ.get('ARM_CONTINUATION_LABEL_GUARD'):
        raise ValueError('Unknown continuation label-count guard')
    return p

def validate_plan(p):
    resume.validate_resume_plan(p)
    if p!=plan(p['job_id'],p['resume_from'],p['requested_resources']['hours'],p['requested_resources']['gpus']):raise ValueError('Continuation configuration changed')
    if int(p['environment']['NUM_ROLLOUT'])!=p['target_completed_iterations']:raise ValueError('Target iteration changed')
    return dict(passed=True,source_unchanged=not p.get('continuation_label_guard'),
        count_guard=p.get('continuation_label_guard'),checkpoint_counters_verified=True)

def execute(p):
    validate_plan(p)
    if os.environ.get('SLURM_JOB_ID')!=p['job_id']:raise ValueError('Authorized allocation required')
    storage=check_multimodal_storage(p)
    if storage:
        write_json(RUNTIME/f'logs/arm-variant-storage-{p["job_id"]}.json',storage)
    manifest=RUNTIME/f'logs/arm-variant-resume-{p["job_id"]}.json';write_json(manifest,p)
    subprocess.run(source_command(Path(p['source']),[sys.executable,str(training.REPO/'scripts/check_arm_gpu_checkpoint_restore.py'),
        '--training-root',p['resume_from'],'--job-id',p['job_id'],'--gpus',str(p['requested_resources']['gpus']),'--output',p['gpu_restore_output'],
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
