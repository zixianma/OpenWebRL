#!/usr/bin/env python3
"""Prepare/own the requested 2K actor screen. Never submits an allocation."""
import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time

from prepare_arm_turn_bonus import copy_plain
from rebuild_arm_task_pool import load_upstream
from resume_baseline import allocation, clean_environment, source_command, validate_source, write_json
from evaluate_baseline_checkpoint import configure_evaluation_tracking
from run_arm_quality_audit import environment, stop
from runtime_ports import PORT_ENV, lease_ports
from check_storage_quota import check_storage_quota

REPO = Path(__file__).resolve().parents[1]
RUNTIME = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime')
POOL = RUNTIME/'arm-turn-bonus-preparation/task-pool-expansion-20260922/curation-v3-20260929'
COHORT = POOL/'cohorts-20260930/comparison-min5-weighted-2000'
CONTROL = POOL/'actor-screen-min5-2000'
SOURCE = RUNTIME/'reference-arm-task-screen-20260930-v2'
PARENT = RUNTIME/'reference-arm-prefix-pilot-20260926-v1'
INITIAL = Path('/gpfs/scrubbed/zixianma/checkpoints/web/OpenWebRL-4B-SFT')
TASK_SHA = '4f8f5b0c8b128934727d49fcdac32ed665ac2c64d562562236d7247f4060c6a6'
RESOURCES = dict(gpus=4, gpu_type='H200', hours=16, gpu_hours=64, cpus=32, memory_gib=480)


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def freeze(path, text):
    if path.exists() and path.read_text() != text:
        raise ValueError('Refuse to replace frozen input: '+str(path))
    path.write_text(text)


def prepare():
    if sha(COHORT/'weighted-tasks.jsonl') != TASK_SHA:
        raise ValueError('Chosen native cohort changed')
    selected = read(POOL/'cohorts-20260930/approved-selection.json')
    if selected['selection'] != COHORT.name:
        raise ValueError('Screen only the user-selected cohort')
    records = [json.loads(x) for x in (COHORT/'weighted-tasks.jsonl').read_text().splitlines()]
    if len(records) != 2000 or len({str(r['task_id']) for r in records}) != 2000 or min(r['difficulty'] for r in records) < 5:
        raise ValueError('Wrong selected cohort')
    upstream, _ = load_upstream(POOL/'upstream')
    # Hash-shuffle execution order without changing website quotas/selection.
    records.sort(key=lambda r: hashlib.sha256(f'20260930/{r["task_id"]}'.encode()).hexdigest())
    rows = [upstream.record_to_row(r, 'webvoyager') for r in records]
    ids = [r['metadata']['task_id'] for r in rows]
    payload = ''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows)
    CONTROL.mkdir(parents=True, exist_ok=True)
    validate_source(PARENT)
    if not SOURCE.exists():
        shutil.copytree(PARENT, SOURCE, symlinks=True, copy_function=copy_plain,
            ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.git', '.browser_use_sessions'))
    name = 'openwebrl/arm_task_screen.py'
    freeze(SOURCE/name, (REPO/name).read_text())
    freeze(SOURCE/'screen-tasks.jsonl', payload)
    config = {'eval': {'defaults': dict(n_samples_per_eval_prompt=5, temperature=.8,
        top_p=1., top_k=-1, max_response_len=1024), 'datasets': [dict(name='actor-screen',
        path=str(SOURCE/'screen-tasks.jsonl'), input_key='prompt', metadata_key='metadata',
        custom_generate_function_path='openwebrl.arm_task_screen.generate')]}}
    freeze(SOURCE/'screen-eval.json', json.dumps(config, indent=2)+'\n')
    import yaml
    browser = yaml.safe_load((SOURCE/'openwebrl/browser_training_config.yaml').read_text())
    browser['browser_rollout_concurrency'] = 32
    freeze(SOURCE/'screen-browser.json', json.dumps(browser, indent=2)+'\n')
    manifest = read(SOURCE/'reference_manifest.json')
    names = [name, 'screen-tasks.jsonl', 'screen-eval.json', 'screen-browser.json']
    manifest['recipe_files_sha256'].update({n:sha(SOURCE/n) for n in names})
    manifest['task_screen'] = dict(parent=str(PARENT), changed_files=names,
        actor='original SFT iteration0', optimizer_updates=0, selected_native_sha256=TASK_SHA)
    write_json(SOURCE/'reference_manifest.json', manifest)
    validate_source(SOURCE)
    actor = read(INITIAL/'download_manifest.json')
    if actor['repo_id'] != 'OpenWebRL/OpenWebRL-4B-SFT' or not (INITIAL/'model.safetensors').is_file():
        raise ValueError('Missing/wrong initial actor')
    return ids, actor


def plan(job='PREPARE'):
    ids, actor = prepare()
    template = read(RUNTIME/'evaluations/arm-prefix-pilot-331932/audit-plan.json')['stages'][0]
    output = CONTROL/'run'
    attempt = output/'workers'/str(job)
    def rewrite(v):
        if isinstance(v, str):
            return v.replace(template['source'],str(SOURCE)).replace(template['output'],str(attempt)).replace('331932',str(job))
        if isinstance(v, list): return [rewrite(x) for x in v]
        if isinstance(v, dict): return {k:rewrite(x) for k,x in v.items()}
        return v
    stage = rewrite(copy.deepcopy(template))
    stage.pop('selector', None)
    stage.pop('quality_config', None)
    stage.update(job_id=str(job), output=str(attempt), source=str(SOURCE), checkpoint=str(INITIAL),
        wandb_run_id=f'arm-task-screen-{job}', protocol='Native SFT actor, five attempts, rubric>=5 weighted 2K',
        gpus=4, optimizer_updates_requested=0)
    for flag,value in {'--eval-config':str(SOURCE/'screen-eval.json'),
        '--eval-function-path':'openwebrl.arm_task_screen.generate_rollout',
        '--wandb-group':'ARM-task-selection-min5-weighted-2000'}.items():
        stage['command'][stage['command'].index(flag)+1] = value
    env = stage['environment']
    for k in list(env):
        if k.startswith('OPENWEBRL_ARM_'):
            del env[k]
    env.update(NUM_GPUS='4', TP_SIZE='2', BROWSER_CONCURRENCY='32', SGLANG_CONCURRENCY='32',
        OPENWEBRL_CUDA_CACHE_LIMIT_GIB='48',
        BROWSER_TRAIN_CONFIG=str(SOURCE/'screen-browser.json'),
        OPENWEBRL_TASK_SCREEN_CONFIG=str(attempt/'screen-config.json'),
        OPENWEBRL_MULTIMODAL_STORAGE_DIR=str(output/'multimodal'/str(job)),
        WANDB_RUN_ID=stage['wandb_run_id'],WANDB_RESUME='allow')
    stage = configure_evaluation_tracking(stage,'openwebrl-evals')
    cfg = dict(output=str(output),job_id=str(job), tasks=str(SOURCE/'screen-tasks.jsonl'),
        task_order=ids, policy_id='OpenWebRL-4B-SFT:iteration0', seed=20260930,
        checkpoint=str(INITIAL), max_steps=15, inference_timeout=180,
        attempt_output=str(output/'attempts'/str(job)/str(time.time_ns())))
    return dict(job_id=str(job), source=str(SOURCE), output=str(output), stage=stage,
        screen_config=cfg, resources=RESOURCES, actor_manifest=actor,
        selected_native_sha256=TASK_SHA, judge_max_usd=150., judge_max_calls=30000,
        judge_budget_partitions=dict(count=5,tasks_each=400,usd_each=30.),
        primary_tasks=2000, primary_attempts=10000, browser_concurrency=32,
        storage_reserve_bytes=5*1024**4, optimizer_updates=0,
        status='prepared_pending_exact_allocation_and_judge_budget_approval')


def native_check(p):
    s = p['stage']
    env = dict(clean_environment(), **s['environment'])
    env.update(DRY_RUN='1', CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    argv = shlex.split(subprocess.check_output(s['command'],env=env,text=True))
    env.pop('DRY_RUN')
    code = """from unittest.mock import patch
from slime.utils.arguments import parse_args
with patch('megatron.training.arguments.get_device_arch_version',return_value=9): a=parse_args()
expected={'num_rollout':0,'tensor_model_parallel_size':2,'actor_num_gpus_per_node':4,
'browser_rollout_concurrency':32,'judge_api_model':'gpt-4.1','judge_prompt_variant':'action_history',
'eval_function_path':'openwebrl.arm_task_screen.generate_rollout','use_checkpoint_opt_param_scheduler':False,
'max_steps':15,'context_num_screenshots':1,'turn_history_reasoning_mode':'full'}
assert all(getattr(a,k)==v for k,v in expected.items()),{k:getattr(a,k) for k in expected}
assert a.load.endswith('/OpenWebRL-4B-SFT')
d=a.eval_datasets[0]
assert (d.n_samples_per_eval_prompt,d.temperature,d.top_p,d.top_k,d.max_response_len)==(5,.8,1.,-1,1024)
print('TASK_SCREEN_NATIVE_PARSE_OK')
"""
    with (CONTROL/'native-parse.log').open('w') as out:
        subprocess.run(source_command(SOURCE,[str(RUNTIME/'venv/bin/python'),'-c',code,*argv[2:]]),
            env=env,cwd=SOURCE,stdout=out,stderr=subprocess.STDOUT,check=True,timeout=120)
    write_json(CONTROL/'readiness.json',dict(native_parse_passed=True,gpu_validation_pending=True,
        source_manifest_sha256=sha(SOURCE/'reference_manifest.json'),
        selected_native_sha256=TASK_SHA, no_actor_collection_launched=True))


def worker(p):
    s = p['stage']; root = Path(s['output']); root.mkdir(parents=True,exist_ok=True)
    env = environment(); env.update(s['environment'])
    if len(env.get('CUDA_VISIBLE_DEVICES','').split(',')) != 4:
        raise ValueError('Worker requires four assigned GPUs')
    write_json(root/'screen-config.json',p['screen_config'])
    write_json(root/'evaluation_manifest.json',s)
    Path(env['OPENWEBRL_MULTIMODAL_STORAGE_DIR']).mkdir(parents=True,exist_ok=True)
    lease, base = lease_ports(p['job_id']); env[PORT_ENV] = str(base)
    child = None
    try:
        with (root/'evaluation.log').open('a') as log:
            child = subprocess.Popen(source_command(SOURCE,s['command']),env=env,cwd=SOURCE,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            if child.wait(): raise RuntimeError('Actor screen failed; inspect worker log before retry')
    finally:
        stop(child); lease.close()


def execute(p):
    job = p['job_id']
    if os.environ.get('SLURM_JOB_ID') != job:
        raise ValueError('Run inside the approved allocation')
    approval = read(CONTROL/'approval.json')
    if (not approval.get('approved') or approval.get('resources') != RESOURCES or
            approval.get('judge_max_usd') != 150. or approval.get('judge_max_calls') != 30000):
        raise ValueError('Exact allocation and judge budget approval required')
    prior = approval.get('attempts',[])
    current = [r for r in prior if r['job_id'] == job]
    if len(current) != 1 or sum(r['consumed_seconds'] for r in prior if r['job_id'] != job)+current[0]['time_limit_seconds'] > 16*3600:
        raise ValueError('Attempt IDs/time must fit the original total approval')
    res = allocation(subprocess.check_output(['scontrol','show','job',job,'-o'],text=True),
        job,requested_gpus=4,maximum_hours=current[0]['time_limit_seconds']/3600)
    if res['cpus'] < 32 or res['allocated_memory_gib'] < 480:
        raise ValueError('Wrong CPU/memory profile')
    ready = read(CONTROL/'readiness.json')
    if not ready['native_parse_passed'] or ready['source_manifest_sha256'] != sha(SOURCE/'reference_manifest.json'):
        raise ValueError('Run native preflight after any source changes')
    check_storage_quota(p['storage_reserve_bytes'])
    root = Path(p['output']);root.mkdir(parents=True,exist_ok=True)
    with (root/'owner.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        write_json(root/f'plan-{job}.json',p)
        child = None
        deadline = time.monotonic()+res['maximum_seconds']-120
        try:
            child = subprocess.Popen(['srun',f'--jobid={job}','--nodes=1','--ntasks=1',
                '--cpus-per-task=32','--gres=gpu:h200:4','--mem=480G','--exact','--cpu-bind=none',
                sys.executable,str(Path(__file__).resolve()),'--job-id',job,'--worker'],start_new_session=True)
            while child.poll() is None:
                if time.monotonic() > deadline: raise TimeoutError('Allocation ending; preserve durable primary records')
                write_json(root/'status.json',dict(job_id=job,complete=False,stage='running',
                    epoch=time.time(),primary_attempts=len(list((root/'records').glob('*.json')))))
                time.sleep(30)
            if child.returncode: raise RuntimeError('Screen worker failed; needs diagnosis')
            from openwebrl.arm_task_screen import summarize
            records = [read(p) for p in (root/'records').glob('*.json')]
            result = summarize(records,p['screen_config']['task_order'],require_complete=True)
            write_json(root/'verification.json',dict(complete=True,archives_and_verdicts_checked=True,
                primary_attempts=result['primary_attempts'],selected_native_sha256=TASK_SHA))
            write_json(root/'status.json',dict(job_id=job,complete=True,stage='verified_complete',epoch=time.time()))
        except BaseException as e:
            write_json(root/'status.json',dict(job_id=job,complete=False,failed=True,epoch=time.time(),
                error_type=type(e).__name__,error=str(e)[:500]))
            raise
        finally:
            stop(child)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job-id',default='PREPARE')
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--worker',action='store_true')
    args = parser.parse_args()
    if args.worker:
        validate_source(SOURCE)
        worker(read(CONTROL/'run'/f'plan-{args.job_id}.json')); return
    p = plan(args.job_id)
    if args.execute: execute(p)
    else:
        native_check(p)
        write_json(CONTROL/'launch-plan.json',p)
        print(json.dumps({k:p[k] for k in ('resources','primary_tasks','primary_attempts','judge_max_usd','status')},indent=2))


if __name__ == '__main__':
    main()
