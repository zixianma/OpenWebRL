#!/usr/bin/env python3
"""Recover the authorized September25 unfinished jobs; never submit compute."""
import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

from resume_baseline import REPO, RUNTIME, allocation, validate_source, write_json

CONTROL = REPO/'logs/arm-unfinished-20260925'
COVERAGE_OLD = RUNTIME/'evaluations/arm-failure-additive-318950-tp2-iter20'
COVERAGE_SOURCE = RUNTIME/'reference-arm-failure-coverage-identity-20260924-v1'
BETA_ROOT = RUNTIME/'evaluations/arm-failure-additive-318949-tp2-iter20'
BETA_OLD = RUNTIME/'evaluations/arm-additive-iter10-318949'
PYTHON = RUNTIME/'venv/bin/python'
COVERAGE_SECONDS = 23*3600+25*60


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def records(root):
    result = {}
    for path in sorted((Path(root)/'rollouts').glob('*.json')):
        record = read(path)
        task = record['task_id']
        archive = path.parent/record['rollout_file']
        if task in result or not archive.is_file() or not archive.stat().st_size:
            raise ValueError('Duplicate task or missing saved rollout: '+task)
        with zipfile.ZipFile(archive) as zipped:
            if not zipped.namelist():
                raise ValueError('Empty rollout archive: '+task)
        if (record['judge_model'],record['judge_prompt_variant']) != ('gpt-4.1','action_history'):
            raise ValueError('Saved judge protocol differs')
        result[task] = (record,path)
    return result


def coverage_plan(job):
    p = read(COVERAGE_OLD/'launch_manifest.json')
    old_source, old_output = p['source'],p['output']
    output = str(RUNTIME/f'evaluations/arm-failure-additive-{job}-tp2-iter20')
    # Preserve the original W&B experiment identity and the scientific recipe.
    def rewrite(value):
        if isinstance(value,str):
            return value.replace(old_source,str(COVERAGE_SOURCE)).replace(old_output,output).replace(
                '/multimodal-scratch/arm-variant-318950',f'/multimodal-scratch/arm-variant-{job}')
        if isinstance(value,list): return [rewrite(v) for v in value]
        if isinstance(value,dict): return {k:rewrite(v) for k,v in value.items()}
        return value
    p = rewrite(p)
    p['job_id'] = job
    p['requested_resources'].update(hours=COVERAGE_SECONDS/3600,gpu_hours=8*COVERAGE_SECONDS/3600)
    p['arm_config'].pop('deadline_epoch_seconds',None)
    p['arm_config']['policy_id'] = p['wandb_run_id']+':uninitialized'
    p['environment']['WANDB_RESUME'] = 'allow'
    p['evaluation_reserve_seconds'] = 7200
    p['coverage_identity_recovery'] = True
    return p


def validate_coverage_plan(p):
    from prepare_arm_failure_ablations import validate_initialization
    ready = read(CONTROL/'readiness.json')
    if ready['fingerprints'] != fingerprints():
        raise ValueError('Recovery readiness changed')
    validate_source(COVERAGE_SOURCE)
    expected = coverage_plan(p['job_id'])
    if p != expected: raise ValueError('Recovery recipe differs from its pinned plan')
    # W&B resume is independent of fresh model/optimizer/task-cursor initialization.
    check = copy.deepcopy(p);check['environment']['WANDB_RESUME'] = 'never'
    validate_initialization(check)
    if (p['environment']['TP_SIZE'],p['environment']['NUM_GPUS']) != ('2','8'):
        raise ValueError('Expected TP2/DP4')
    c = p['arm_config']
    if (c['beta'],c['scored_fraction'],c['failure_beta'],c['failure_scored_fraction'],c['failure_group_cap']) != (.5,.2,.5,.4,8):
        raise ValueError('Coverage knobs changed')
    if not read(RUNTIME/'benchmarks/arm-ablation-tp2-full-318950/result.json')['passed']:
        raise ValueError('Prior full-batch32K/save/reload GPU validation missing')
    return dict(passed=True,starts_at_iteration_zero=True,fresh_optimizer_scheduler_cursor=True,
        prior_topology_gpu_validation=True,identity_fix_live_validation_pending=True,
        source=str(COVERAGE_SOURCE))


def fingerprints():
    paths = [Path(__file__).resolve(),CONTROL/'coverage-controller.py',
        COVERAGE_SOURCE/'reference_manifest.json',COVERAGE_OLD/'launch_manifest.json',
        REPO/'scripts/evaluate_baseline_checkpoint.py',REPO/'scripts/run_arm_iteration80_eval.py']
    return {str(p):sha(p) for p in paths}


def prepare():
    CONTROL.mkdir(parents=True,exist_ok=True)
    validate_source(COVERAGE_SOURCE)
    text = (REPO/'scripts/run_arm_turn_bonus_cycles.py').read_text()
    old = '        from prepare_arm_failure_ablations import validate_plan\n'
    if text.count(old) != 1: raise ValueError('Training validation boundary changed')
    text = text.replace(old,'        from recover_arm_unfinished import validate_coverage_plan as validate_plan\n')
    text = text.replace('if reserve not in (0,3600):','if reserve not in (0,3600,7200):')
    compile(text,'coverage-controller.py','exec')
    path = CONTROL/'coverage-controller.py'
    if path.exists() and path.read_text() != text: raise ValueError('Frozen recovery controller changed')
    path.write_text(text)
    source = RUNTIME/'reference-arm-eval80-additive-20260920-v2'
    manifest = read(BETA_OLD/'evaluation_manifest.json')
    saved = records(BETA_OLD)
    expected = manifest['expected_rollout_task_ids']
    if len(expected)!=300 or len(set(expected))!=300 or not set(saved)<=set(expected):
        raise ValueError('Original beta evaluation cohort differs')
    missing = [task for task in expected if task not in saved]
    import yaml
    config = yaml.safe_load((source/'openwebrl/online_mind2web_monitor.yaml').read_text())
    all_tasks = [json.loads(line) for line in (source/'online_mind2web_monitor.jsonl').read_text().splitlines()]
    subset = [t for t in all_tasks if t['metadata']['task_id'] in missing]
    if {t['metadata']['task_id'] for t in subset} != set(missing):
        raise ValueError('Missing-task extraction mismatch')
    dataset = CONTROL/'beta10-missing.jsonl'
    dataset.write_text(''.join(json.dumps(t)+'\n' for t in subset))
    config['eval']['datasets'][0]['path'] = str(dataset)
    (CONTROL/'beta10-missing.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
    write_json(CONTROL/'beta10-recovery.json',dict(source=str(source),checkpoint=manifest['checkpoint'],
        prior=str(BETA_OLD),expected_ids=expected,missing_ids=missing,saved_tasks=len(saved),
        verdict_sha256={task:sha(path) for task,(_,path) in saved.items()},
        prior_manifest_sha256=sha(BETA_OLD/'evaluation_manifest.json')))
    write_json(CONTROL/'readiness.json',dict(fingerprints=fingerprints(),cpu_prepared=True,
        coverage_seconds=COVERAGE_SECONDS,beta_saved=len(saved),beta_missing=len(missing)))
    validate_coverage_plan(coverage_plan('PREPARE'))


def evaluation_plan(variant,job,iteration,training_root,gpus):
    import run_arm_iteration80_eval as module
    p = module.plan('additive',job,iteration,training_root)
    old = p['output'];out = str(RUNTIME/f'evaluations/arm-failure-{variant}-iter{iteration}-{job}')
    p = json.loads(json.dumps(p).replace(old,out))
    run = f'arm-failure-{variant}-iter{iteration}-{job}'
    p.update(gpus=gpus,wandb_run_id=run,arm_variant=f'failure-{variant}',
        parent_training_run=f'arm-failure-{variant}-fromzero-'+('318949' if variant=='weight' else '318950'))
    p['environment'].update(NUM_GPUS=str(gpus),TP_SIZE=str(gpus),WANDB_RUN_ID=run,WANDB_RESUME='never')
    p['command'][p['command'].index('--wandb-group')+1] = f'ARM failure {variant} | iter{iteration} recovery'
    if variant=='weight' and iteration==10:
        recovery = read(CONTROL/'beta10-recovery.json')
        if p['checkpoint'] != recovery['checkpoint']: raise ValueError('Beta10 checkpoint changed')
        current = records(BETA_OLD)
        if {task:sha(path) for task,(_,path) in current.items()} != recovery['verdict_sha256']:
            raise ValueError('Original beta results changed after preparation')
        p['command'][p['command'].index('--eval-config')+1] = str(CONTROL/'beta10-missing.yaml')
        p.update(expected_task_count=len(recovery['missing_ids']),expected_rollout_task_ids=recovery['missing_ids'],
            recovery_of=str(BETA_OLD),protocol='existing deterministic GPT-4.1 monitor; missing-task recovery only')
        p['environment']['OPENWEBRL_RECORD_EVAL_EXPECTED_TASK_COUNT']=str(len(recovery['missing_ids']))
    return module.evaluation.configure_evaluation_tracking(p,'openwebrl-evals')


def merge_beta10(job,plan):
    from run_stage1_to100 import audit_rollouts
    recovery = read(CONTROL/'beta10-recovery.json')
    original,retry = records(BETA_OLD),records(Path(plan['output']))
    if set(original)&set(retry) or set(retry)!=set(recovery['missing_ids']):
        raise ValueError('Recovery duplicated or omitted tasks')
    out = RUNTIME/f'evaluations/arm-failure-weight-iter10-{job}-merged'
    archive = out/'rollouts';archive.mkdir(parents=True,exist_ok=False)
    for record,path in [*original.values(),*retry.values()]:
        for target in (path,path.parent/record['rollout_file']):
            (archive/target.name).symlink_to(target)
    audit = audit_rollouts(archive,Path(plan['source']))
    write_json(out/'recovery-provenance.json',dict(original=str(BETA_OLD),missing_task_retry=plan['output'],
        checkpoint=plan['checkpoint'],mixed_evaluation_dates=True,original_results_unchanged=True))
    write_json(out/'metrics.json',audit)
    write_json(out/'status.json',dict(complete=True,expected_tasks=300,saved_rollouts=300,aggregate=audit))
    return audit


def worker(kind,job,stage):
    if os.getenv('SLURM_JOB_ID') != job: raise ValueError('Wrong worker allocation')
    from check_storage_quota import check_storage_quota
    if stage=='train':
        from resume_arm_failure_variants import check_multimodal_storage
        p=coverage_plan(job);validate_coverage_plan(p);check_multimodal_storage(p)
        spec=importlib.util.spec_from_file_location('coverage_recovery_controller',CONTROL/'coverage-controller.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        with (CONTROL/f'training-monitor-{job}.log').open('a') as log:
            monitor=subprocess.Popen([str(PYTHON),str(REPO/'scripts/monitor_arm_turn_bonus.py'),
                '--job-id',job,'--watch','--interval','900','--hours',str(COVERAGE_SECONDS/3600)],
                stdout=log,stderr=subprocess.STDOUT)
            try:return module.execute(p)
            finally:
                monitor.terminate()
                try:monitor.wait(timeout=30)
                except subprocess.TimeoutExpired:monitor.kill();monitor.wait()
    import evaluate_baseline_checkpoint as evaluator
    iteration=int(stage)
    root=BETA_ROOT if kind=='beta' else Path(coverage_plan(job)['output'])
    p=evaluation_plan('weight' if kind=='beta' else 'coverage',job,iteration,root,2 if kind=='beta' else 8)
    os.environ.update(OPENWEBRL_EXPECTED_SCHEDULER_OFFSET='0',OPENWEBRL_RECORD_EVAL_EXPECTED_TASK_COUNT=str(p['expected_task_count']))
    check_storage_quota(256*1024**3)
    evaluator.run(p,REPO/'.env')
    if kind=='beta' and iteration==10: merge_beta10(job,p)


def execute(kind,job):
    if os.getenv('SLURM_JOB_ID')!=job: raise ValueError('Run in authorized allocation')
    gpus,cpus,seconds=(2,16,7200) if kind=='beta' else (8,64,COVERAGE_SECONDS)
    allocation(subprocess.check_output(['scontrol','show','job',job,'-o'],text=True),job,
        requested_gpus=gpus,maximum_hours=seconds/3600)
    root=RUNTIME/f'evaluations/arm-{kind}-recovery-{job}';root.mkdir(exist_ok=False)
    stages=['10','20'] if kind=='beta' else ['train','10','20']
    write_json(root/'controller-plan.json',dict(job_id=job,kind=kind,stages=stages,
        training_root=str(BETA_ROOT) if kind=='beta' else coverage_plan(job)['output'],
        gpus=gpus,maximum_seconds=seconds))
    step=['srun',f'--jobid={job}','--nodes=1','--ntasks=1',f'--cpus-per-task={cpus}',
        f'--gres=gpu:h200:{gpus}','--exact','--cpu-bind=none',str(PYTHON),str(Path(__file__).resolve()),
        '--kind',kind,'--job-id',job,'--worker']
    try:
        for stage in stages:
            write_json(root/'status.json',dict(stage=stage,complete=False))
            subprocess.run([*step,stage],check=True)
        write_json(root/'status.json',dict(stage='complete',complete=True))
    except BaseException as exc:
        write_json(root/'status.json',dict(stage='failed',complete=False,error_type=type(exc).__name__,error=str(exc)[:400]))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--kind',choices=('coverage','beta'))
    parser.add_argument('--job-id',default='PREPARE')
    parser.add_argument('--worker',choices=('train','10','20'))
    args=parser.parse_args()
    if args.prepare: prepare()
    elif args.worker: worker(args.kind,args.job_id,args.worker)
    else: execute(args.kind,args.job_id)
