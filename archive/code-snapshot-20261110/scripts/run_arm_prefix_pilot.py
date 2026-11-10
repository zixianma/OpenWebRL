#!/usr/bin/env python3
"""Prepare/own the ARM-prefix pilot. Preparation is CPU-only; never submits jobs."""
import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import socket
import subprocess
import sys
import time

import run_arm_quality_audit as quality
from prepare_arm_turn_bonus import copy_plain
from resume_baseline import allocation,clean_environment,source_command,validate_source,write_json
from check_storage_quota import check_storage_quota

REPO,RUNTIME = quality.REPO,quality.RUNTIME
PARENT = RUNTIME/'reference-arm-refresh-browser-20260926-v1'
SOURCE = RUNTIME/'reference-arm-prefix-pilot-20260926-v1'
CONTROL = RUNTIME/'arm-turn-bonus-preparation/prefix-pilot-20260926'
INITIAL = Path('/gpfs/scrubbed/zixianma/checkpoints/web/OpenWebRL-4B-SFT')
TRAIN_TASKS = REPO/'openwebrl/data/webgym_filtered_popular_2102_cleaned.parquet'
FILES = ['openwebrl/arm_prefix_pilot.py']
RESOURCES = dict(gpus=2,gpu_type='H200',hours=2,gpu_hours=4,cpus=16,memory_gib=480)


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def normalized(text):return ' '.join(str(text).casefold().split())


def prepare():
    import pandas as pd
    from openwebrl.arm_rescue_yield import stable_seed
    validate_source(PARENT)
    CONTROL.mkdir(parents=True,exist_ok=True)
    evaluation = [json.loads(line) for line in (REPO/'openwebrl/data/eval/online-mind2web.jsonl').read_text().splitlines()]
    excluded_ids = {str(r.get('task_id',r.get('id'))).rsplit('/',1)[-1] for r in evaluation}
    excluded_intents = {normalized(r.get('confirmed_task',r.get('intent',r.get('task','')))) for r in evaluation}
    pool = []
    for r in pd.read_parquet(TRAIN_TASKS,columns=['prompt','metadata']).to_dict('records'):
        meta = dict(r['metadata'])
        if (str(meta['task_id']).rsplit('/',1)[-1] in excluded_ids or
            normalized(meta.get('task',meta.get('intent',''))) in excluded_intents):
            continue
        pool.append(dict(prompt=list(r['prompt']),metadata=meta))
    pool.sort(key=lambda r:stable_seed(20260926,str(r['metadata']['task_id']),'prefix-pilot'))
    rows = pool[:48]
    ids = [str(r['metadata']['task_id']) for r in rows]
    if len(set(ids)) != 48:
        raise ValueError('Need 48 distinct training tasks')
    payload = ''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows)
    config = {'eval':{'defaults':dict(n_samples_per_eval_prompt=1,temperature=.8,top_p=1.,
        top_k=-1,max_response_len=1024),'datasets':[dict(name='arm-prefix-pilot',
        path=str(SOURCE/'prefix_tasks.jsonl'),input_key='prompt',metadata_key='metadata',
        custom_generate_function_path='openwebrl.arm_prefix_pilot.generate')]}}
    if not SOURCE.exists():
        shutil.copytree(PARENT,SOURCE,symlinks=True,copy_function=copy_plain,
            ignore=shutil.ignore_patterns('__pycache__','*.pyc','.git','.browser_use_sessions'))
        for name in FILES:copy_plain(REPO/name,SOURCE/name)
        (SOURCE/'prefix_tasks.jsonl').write_text(payload)
        write_json(SOURCE/'prefix_eval.json',config)
        import yaml
        browser = yaml.safe_load((SOURCE/'openwebrl/browser_training_config.yaml').read_text())
        browser['browser_rollout_concurrency'] = 16
        write_json(SOURCE/'prefix_browser.json',browser)
        manifest = read(SOURCE/'reference_manifest.json')
        changed = FILES+['prefix_tasks.jsonl','prefix_eval.json','prefix_browser.json']
        manifest['recipe_files_sha256'].update({name:sha(SOURCE/name) for name in changed})
        manifest['prefix_pilot'] = dict(parent=str(PARENT),changed_files=changed,optimizer_updates=0,
            actor_initialization='Original OpenWebRL-4B-SFT; no RL checkpoint')
        write_json(SOURCE/'reference_manifest.json',manifest)
    validate_source(SOURCE)
    if (SOURCE/'prefix_tasks.jsonl').read_text() != payload or read(SOURCE/'prefix_eval.json') != config:
        raise ValueError('Fixed cohort/protocol changed')
    if any(sha(REPO/name) != sha(SOURCE/name) for name in FILES):
        raise ValueError('Freeze a new source version after editing implementation')
    if not (INITIAL/'config.json').is_file():raise ValueError('SFT checkpoint missing')
    write_json(CONTROL/'cohort-audit.json',dict(eligible_training_tasks=len(pool),selected_tasks=48,
        seed=20260926,selection='Outcome-blind hash sample; no pass@5 or validity screening',
        task_ids=ids,train_file_sha256=sha(TRAIN_TASKS),cohort_sha256=sha(SOURCE/'prefix_tasks.jsonl'),
        online_mind2web_ids_and_intents_excluded=True))
    return ids


def plan(job='PREPARE'):
    ids = prepare()
    original = read(RUNTIME/'evaluations/arm-quality-audit-313774/audit-plan.json')
    template = next(s for s in original['stages'] if s['actor']=='sft')
    root = RUNTIME/f'evaluations/arm-prefix-pilot-{job}'
    output = root/'sft'
    def rewrite(value):
        if isinstance(value,str):
            return value.replace(template['source'],str(SOURCE)).replace(template['output'],str(output)).replace('313774',str(job))
        if isinstance(value,list):return [rewrite(v) for v in value]
        if isinstance(value,dict):return {k:rewrite(v) for k,v in value.items()}
        return value
    stage = rewrite(copy.deepcopy(template))
    stage.update(actor='sft',source=str(SOURCE),output=str(output),checkpoint=str(INITIAL),job_id=str(job),
        wandb_run_id=f'arm-prefix-pilot-{job}',protocol='Training-task ARM prefix k0/2/4; SFT actor; two repeats; no updates')
    command = stage['command']
    for name,value in {'--eval-config':str(SOURCE/'prefix_eval.json'),
        '--eval-function-path':'openwebrl.arm_prefix_pilot.generate_rollout',
        '--wandb-group':'ARM-prefix-curriculum-pilot'}.items():
        command[command.index(name)+1] = value
    stage['environment'].update(WANDB_RUN_ID=stage['wandb_run_id'],WANDB_RESUME='never',
        BROWSER_CONCURRENCY='16',SGLANG_CONCURRENCY='16',BROWSER_TRAIN_CONFIG=str(SOURCE/'prefix_browser.json'),
        HF_CHECKPOINT=str(INITIAL),SLIME_LOAD_CHECKPOINT=str(INITIAL),
        FLASHINFER_WORKSPACE_BASE=str(RUNTIME),
        OPENWEBRL_MULTIMODAL_STORAGE_DIR=str(RUNTIME/f'multimodal-scratch/arm-prefix-pilot-{job}'))
    stage['quality_config'] = dict(output=str(output),actor='sft',checkpoint=str(INITIAL),
        policy_id='OpenWebRL-4B-SFT:iteration0',tasks=str(SOURCE/'prefix_tasks.jsonl'),task_order=ids,
        seed=20260926,max_steps=15,budget=str(root/'terminal-budget'),judge_max_calls=900,judge_max_usd=20.)
    # Use the frozen released ARM from the already validated torch2.8 serving environment.
    model = RUNTIME/'arm-turn-bonus-preparation/arm-refresh-20260924/baseline40/model-compat'
    stage['selector'] = dict(python=str(RUNTIME/'arm-reproduction/venv-refresh/bin/python'),
        model=str(model),base=str(model),environment={
        'TRITON_CACHE_DIR':str(output/'triton-cache'),
        'LD_LIBRARY_PATH':str(RUNTIME/'arm-reproduction/refresh-torch28/torch/lib'),'OMP_NUM_THREADS':'1'})
    stage = quality.evaluation.configure_evaluation_tracking(stage,'openwebrl-evals')
    return dict(job_id=str(job),output=str(root),stages=[stage],requested_resources=RESOURCES,
        expected_collection=dict(complete=True,tasks=48,trajectories=288,smoke_trajectories=12,optimizer_updates=0),
        judge_max_usd=20.,judge_max_calls=900,storage_reserve_bytes=150*1024**3,
        approval_status=('Approved: two H200s x2 hours and $20 terminal judge cap'
            if (CONTROL/'approval.json').exists() and read(CONTROL/'approval.json').get('approved')
            else 'Experiment requested; exact allocation and judge budget pending approval'),
        artifacts='All rollouts/verdicts, full guided candidate traces, actor suffixes, handoff records; no cleanup')


def validate_plan(p):
    s = p['stages'][0]
    if (s['checkpoint'] != str(INITIAL) or s['environment']['NUM_ROLLOUT'] != '0' or
        s['environment']['WANDB_PROJECT'] != 'openwebrl-evals' or
        s['environment']['SLIME_LOAD_CHECKPOINT'] != str(INITIAL) or
        'adapter' in s['selector'] or '--use-checkpoint-opt-param-scheduler' in s['command']):
        raise ValueError('Actor initialization, frozen ARM, or evaluation-only recipe changed')
    for path in [Path(s['selector']['python']),Path(s['selector']['model'])/'config.json']:
        if not path.is_file():raise ValueError('Missing model/serving path: '+str(path))
    if p['requested_resources'] != RESOURCES:raise ValueError('Resource request changed')
    return dict(cpu_recipe_passed=True,gpu_startup_pending=True)


def native_check(p):
    stage = p['stages'][0]
    env = dict(clean_environment(),**stage['environment'])
    env.update(DRY_RUN='1',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    argv = shlex.split(subprocess.check_output(stage['command'],env=env,text=True))
    env.pop('DRY_RUN')
    code = """from unittest.mock import patch
from slime.utils.arguments import parse_args
with patch('megatron.training.arguments.get_device_arch_version',return_value=9): a=parse_args()
expected={'num_rollout':0,'tensor_model_parallel_size':2,'browser_rollout_concurrency':16,'judge_api_model':'gpt-4.1','judge_prompt_variant':'action_history','eval_function_path':'openwebrl.arm_prefix_pilot.generate_rollout','use_checkpoint_opt_param_scheduler':False}
assert all(getattr(a,k)==v for k,v in expected.items()),{k:getattr(a,k) for k in expected}
assert a.load.endswith('/OpenWebRL-4B-SFT'),a.load
assert len(a.eval_datasets)==1
assert a.eval_datasets[0].n_samples_per_eval_prompt==1
assert a.eval_datasets[0].max_response_len==1024
print('PREFIX_PILOT_NATIVE_PARSE_OK')
"""
    with (CONTROL/'native-parse.log').open('w') as output:
        subprocess.run(source_command(SOURCE,[str(RUNTIME/'venv/bin/python'),'-c',code,*argv[2:]]),
            cwd=SOURCE,env=env,stdout=output,stderr=subprocess.STDOUT,check=True,timeout=120)


def execute(p):
    validate_plan(p)
    readiness = read(CONTROL/'readiness.json')
    if (not readiness.get('native_parse_passed') or
        readiness.get('source_manifest_sha256') != sha(SOURCE/'reference_manifest.json')):
        raise ValueError('Missing or stale native CPU preflight')
    job = p['job_id']
    if os.getenv('SLURM_JOB_ID') != job:raise ValueError('Use the approved dedicated allocation')
    approval,submission = read(CONTROL/'approval.json'),read(CONTROL/'submission.json')
    if (not approval.get('approved') or approval.get('resources') != RESOURCES or
        approval.get('judge_max_usd') != 20. or submission['job_id'] != job):
        raise ValueError('Missing matching exact resource/budget approval')
    res = allocation(subprocess.check_output(['scontrol','show','job',job,'-o'],text=True),
        job,requested_gpus=2,maximum_hours=2)
    if res['cpus'] < 16 or res['allocated_memory_gib'] < 480:raise ValueError('Wrong resource profile')
    check_storage_quota(p['storage_reserve_bytes'])
    root = Path(p['output']);root.mkdir(exist_ok=False)
    write_json(root/'audit-plan.json',p)
    Path(p['stages'][0]['environment']['OPENWEBRL_MULTIMODAL_STORAGE_DIR']).mkdir(parents=True,exist_ok=False)
    worker = None
    deadline = time.monotonic()+res['maximum_seconds']-120
    try:
        worker = subprocess.Popen(['srun',f'--jobid={job}','--nodes=1','--ntasks=1','--cpus-per-task=16',
            '--gres=gpu:h200:2','--exact','--cpu-bind=none',sys.executable,str(Path(__file__).resolve()),
            '--job-id',job,'--worker'],start_new_session=True)
        while worker.poll() is None:
            if time.monotonic()>deadline:raise TimeoutError('Leave time to preserve artifacts')
            out = Path(p['stages'][0]['output'])
            for halt in (root/'terminal-budget/halt.json',out/'selector-failure.json'):
                if halt.exists():raise RuntimeError('Selector/judge failure; stop escalation')
            write_json(root/'status.json',dict(complete=False,updated_epoch=time.time(),
                stage='compare' if (out/'smoke-gate.json').exists() else 'startup_or_smoke',
                primary_attempts=len(list((out/'compare_records').glob('*.json'))),
                smoke_attempts=len(list((out/'smoke_records').glob('*.json')))))
            time.sleep(10)
        if worker.returncode:raise RuntimeError('Prefix pilot worker failed; artifacts preserved')
        summary = read(root/'sft/prefix-summary.json')
        if not summary.get('complete') or summary['trajectories'] != 288:raise ValueError('Incomplete pilot')
        write_json(root/'status.json',dict(complete=True,stage='complete',updated_epoch=time.time(),summary=summary))
    except BaseException as exc:
        write_json(root/'status.json',dict(complete=False,failed=True,updated_epoch=time.time(),
            error_type=type(exc).__name__,error=str(exc)[:500]))
        raise
    finally:quality.stop(worker)


def watch():
    """Persistent observation; no automatic submission or unvalidated bug repair."""
    with (CONTROL/'watch.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        write_json(CONTROL/'watch-process.json',dict(pid=os.getpid(),hostname=socket.gethostname(),
            started_epoch=time.time(),automatic_bug_repair=False))
        end = time.monotonic()+48*3600
        while time.monotonic()<end:
            job = read(CONTROL/'submission.json')['job_id']
            root = RUNTIME/f'evaluations/arm-prefix-pilot-{job}'
            state = read(root/'status.json') if (root/'status.json').exists() else {}
            lookup = subprocess.run(['squeue','-h','-j',job,'-o','%i|%T|%M|%R'],capture_output=True,text=True,timeout=30)
            if lookup.returncode and 'Invalid job id' not in lookup.stderr:
                raise RuntimeError('Queue lookup failed: '+lookup.stderr[:200])
            q = lookup.stdout.strip()
            result = dict(job_id=job,queue=q,controller=state,updated_epoch=time.time(),automatic_bug_repair=False)
            if state.get('complete'):
                from openwebrl.arm_prefix_pilot import summarize,verify_archives
                records = [read(f) for f in (root/'sft/compare_records').glob('*.json')]
                cfg = read(root/'sft/quality-config.json')
                result['verified_results'] = summarize(records,cfg['task_order'],cfg['policy_id'])
                verify_archives(records)
            result['needs_attention'] = bool(state.get('failed') or not q and not state.get('complete'))
            write_json(CONTROL/'watch-status.json',result)
            if state.get('complete') or result['needs_attention']:return
            for _ in range(10 if state.get('stage')=='compare' or not state else 1):time.sleep(60)
        write_json(CONTROL/'watch-expired.json',dict(updated_epoch=time.time()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job-id',default='PREPARE')
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--watch',action='store_true')
    args = parser.parse_args()
    if args.watch:watch();return
    if args.worker:
        p = read(RUNTIME/f'evaluations/arm-prefix-pilot-{args.job_id}/audit-plan.json')
        validate_source(SOURCE);quality.SOURCE=SOURCE;quality.worker(p,'sft');return
    p = plan(args.job_id);checks = validate_plan(p)
    if args.execute:execute(p)
    else:
        native_check(p)
        write_json(CONTROL/'launch-plan.json',p)
        write_json(CONTROL/'readiness.json',dict(**checks,native_parse_passed=True,
            source_manifest_sha256=sha(SOURCE/'reference_manifest.json')))
        print(json.dumps(dict(resources=p['requested_resources'],collection=p['expected_collection'],
            judge_max_usd=p['judge_max_usd'],approval=p['approval_status'],**checks),indent=2))


if __name__=='__main__':main()
