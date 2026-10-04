#!/usr/bin/env python3
"""Prepare matched WebVoyager evaluations; execute only in approved allocations."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import time
from types import SimpleNamespace

import run_arm_stealth90_o4 as shared
import run_arm_stealth90_o4_more as parent

base = shared.base
CONTROL = base.RUNTIME / 'arm-turn-bonus-preparation/webvoyager90-stealth-20260929'
SOURCE = CONTROL / 'source-v2-chat-prompts'
METHODS = ('baseline', 'additive', 'gate-b')
RESOURCES = dict(jobs=3, gpus_per_job=1, gpu_type='H200', hours_per_job=12,
                 cpus_per_job=8, memory_gib_per_job=240, max_concurrent_jobs=3,
                 total_gpu_hours=36, browsers_per_job=3)
PROTOCOL = 'Actor-only iteration90; Browser Use stealth; GPT-4o/WebVoyager; T0.6 p0.95 k20; 4096 tokens; 30 turns; released FARA595'
DATA_SHA256 = 'ac9253317b76884abbd8d89e9b02570f3f4fbf18e428c335bc8761d778c6fe68'
JUDGE_SHA256 = '433b946f79f5df2b73fffc3068ac4b8453dcb835210d299cce9b2adcea7368dd'


def tasks():
    return [json.loads(line) for line in (SOURCE/'webvoyager595.jsonl').read_text().splitlines()]


def prepare_source():
    shared.validate_protocol_source(parent.SOURCE)
    if shared.sha(CONTROL/'upstream_webvoyager_fara.jsonl') != DATA_SHA256:
        raise ValueError('Released task snapshot changed')
    if shared.sha(CONTROL/'upstream_reward_webvoyager.py') != JUDGE_SHA256:
        raise ValueError('Released WebVoyager judge changed')
    if not SOURCE.exists():
        shutil.copytree(parent.SOURCE, SOURCE, symlinks=True,
            ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.git', '.env*', '.browser_use_sessions'))
        shutil.copy2(CONTROL/'upstream_reward_webvoyager.py', SOURCE/'openwebrl/eval/reward_webvoyager.py')
        shutil.copy2(base.REPO/'scripts/arm_webvoyager_generator.py', SOURCE/'openwebrl/eval_benchmark.py')
        shutil.copy2(CONTROL/'upstream_webvoyager_fara.jsonl', SOURCE/'webvoyager_released.jsonl')
        rows = [json.loads(line) for line in (SOURCE/'webvoyager_released.jsonl').read_text().splitlines()]
        prepared = []
        for row in rows:
            if row['metadata']['intent'] != row['ques'] or row['metadata']['start_url'] != row['web']:
                raise ValueError('Released task metadata differs from instruction/URL')
            if (SOURCE/'openwebrl/env/tasks'/f'{row["id"]}.json').exists():
                raise ValueError('Per-task file would override pinned benchmark data')
            metadata = dict(row['metadata'], task_id=str(row['id']),
                            _browser_task_file=str(SOURCE/'webvoyager_released.jsonl'))
            prepared.append(dict(prompt=[dict(role='user', content=row['ques'])], metadata=metadata))
        (SOURCE/'webvoyager595.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in prepared))
        (SOURCE/'openwebrl/webvoyager_benchmark.yaml').write_text('''eval:
  defaults:
    n_samples_per_eval_prompt: 1
    temperature: 0.6
    top_p: 0.95
    top_k: 20
    max_response_len: 4096
  datasets:
    - name: webvoyager-benchmark
      path: webvoyager595.jsonl
      input_key: prompt
      metadata_key: metadata
      custom_generate_function_path: openwebrl.eval_benchmark.generate
''')
        manifest = base.read(SOURCE/'reference_manifest.json')
        names = ['openwebrl/eval/reward_webvoyager.py', 'openwebrl/eval_benchmark.py',
                 'webvoyager_released.jsonl', 'webvoyager595.jsonl', 'openwebrl/webvoyager_benchmark.yaml']
        manifest['recipe_files_sha256'].update({n:shared.sha(SOURCE/n) for n in names})
        manifest['webvoyager_benchmark'] = dict(protocol=PROTOCOL, parent=str(parent.SOURCE),
            task_sha256=DATA_SHA256, judge_sha256=JUDGE_SHA256, changed_instruction_count_vs_paper=53,
            dataset_decision='Released OpenWebRL tasks; same595 IDs/URLs as FARA paper snapshot, with53 date-updated instructions.')
        manifest['browser_use_evaluation'].update(judge='gpt-4o/WebVoyager', temperature=.6,
            unchanged='Actor-only Browser Use stealth adapter; benchmark changed to released WebVoyager595')
        shared.write_json(SOURCE/'reference_manifest.json', manifest)
    validate_source()


def validate_source():
    import yaml
    shared.validate_source(SOURCE)
    if (shared.sha(SOURCE/'webvoyager_released.jsonl') != DATA_SHA256 or
            shared.sha(SOURCE/'openwebrl/eval/reward_webvoyager.py') != JUDGE_SHA256 or
            shared.sha(SOURCE/'openwebrl/eval_benchmark.py') != shared.sha(base.REPO/'scripts/arm_webvoyager_generator.py')):
        raise ValueError('Frozen dataset, judge or generation source changed')
    args, sampling = runpy.run_path(str(SOURCE/'openwebrl/eval_benchmark.py'))['configure'](
        SimpleNamespace(judge_api_model='o4-mini', judge_prompt_variant='agenttrek'), dict(temperature=0))
    if (args.judge_api_model, args.judge_prompt_variant, args.max_steps, args.judge_max_attached_imgs) != ('gpt-4o','webvoyager',30,30):
        raise ValueError('Wrong executed WebVoyager judge configuration')
    if sampling != dict(temperature=.6,top_p=.95,top_k=20,max_new_tokens=4096,repetition_penalty=1.0):
        raise ValueError('Wrong executed sampling')
    config = yaml.safe_load((SOURCE/'openwebrl/webvoyager_benchmark.yaml').read_text())['eval']
    if config['datasets'] != [dict(name='webvoyager-benchmark',path='webvoyager595.jsonl',
            input_key='prompt',metadata_key='metadata',custom_generate_function_path='openwebrl.eval_benchmark.generate')]:
        raise ValueError('Wrong executed benchmark dataset or generator')
    rows = tasks()
    ids = [r['metadata']['task_id'] for r in rows]
    if len(ids) != 595 or len(set(ids)) != 595:
        raise ValueError('Expected595 unique tasks')
    original = {str(r['id']):r for r in [json.loads(s) for s in (SOURCE/'webvoyager_released.jsonl').read_text().splitlines()]}
    for row in rows:
        meta = row['metadata']; raw = original[meta['task_id']]
        if (row['prompt'] != [dict(role='user', content=raw['ques'])] or meta['intent'] != raw['ques'] or meta['start_url'] != raw['web']
                or meta['_browser_task_file'] != str(SOURCE/'webvoyager_released.jsonl')):
            raise ValueError('Actor/lookup/judge task mismatch')
    return dict(tasks=595, task_sha256=DATA_SHA256, judge_sha256=JUDGE_SHA256,
                sampling=sampling, max_steps=30, judge_max_images=30)


def plan(method, job='PREPARE'):
    if method not in METHODS:
        raise ValueError('Unknown method')
    validate_source()
    checkpoint, report = base.checkpoint(method)
    output = base.RUNTIME/f'evaluations/webvoyager90-gpt4o-t06-{method}-r1-{job}'
    p = base.evaluator.build_plan(SOURCE, checkpoint, output, job, gpus=1, browser_env='browser-use')
    ids = [r['metadata']['task_id'] for r in tasks()]
    p.update(method=method, repeat=1, cohort_label=f'webvoyager-{method}-r1', completed_iteration=90,
        completed_optimizer_updates=report['completed_optimizer_updates'], protocol=PROTOCOL,
        actor_temperature=.6, expected_task_count=595, expected_rollout_task_ids=ids,
        require_task_rollouts=True, parent_training_run=base.METHODS[method]['run_id'],
        wandb_run_id=output.name, browser_concurrency=3, judge_model='gpt-4o',
        judge_prompt_variant='webvoyager', metric_prefix='eval/webvoyager-benchmark', allocation_hours_cap=12)
    p['environment'].update(WANDB_RUN_ID=output.name, WANDB_RESUME='never', JUDGE_MODEL='gpt-4o',
        BROWSER_MAX_STEPS='30', RESPONSE_LEN='4096', BROWSER_CONCURRENCY='3', SGLANG_CONCURRENCY='3',
        BROWSER_TRAIN_CONFIG=str(SOURCE/'openwebrl/browser_training_config.yaml'),
        OPENWEBRL_EXPECTED_SCHEDULER_OFFSET=str(base.METHODS[method]['offset']),
        OPENWEBRL_RECORD_EVAL_EXPECTED_TASK_COUNT='595', PYTHONDONTWRITEBYTECODE='1',
        OPENWEBRL_MULTIMODAL_STORAGE_DIR=str(base.RUNTIME/f'multimodal-scratch/{output.name}'),
        FLASHINFER_WORKSPACE_BASE=str(base.RUNTIME/'flashinfer-webvoyager90'),
        WANDB_CACHE_DIR=str(output/'wandb-cache'), RAY_TMPDIR=f'/tmp/wv-{job}')
    for flag, value in {'--eval-config':str(SOURCE/'openwebrl/webvoyager_benchmark.yaml'),
                        '--wandb-group':'ARM iteration90 matched WebVoyager GPT-4o T0.6'}.items():
        p['command'][p['command'].index(flag)+1] = value
    p['command'].extend(['--seed','1234'])
    base.evaluator.configure_evaluation_tracking(p)
    validate(p)
    return p


def validate(p):
    validate_source()
    if (p['method'] not in METHODS or p['source'] != str(SOURCE) or p['protocol'] != PROTOCOL
            or p['checkpoint'] != str(base.checkpoint(p['method'])[0]) or p['checkpoint_index'] != 89
            or p['optimizer_updates_requested'] != 0 or p['allocation_hours_cap'] != 12
            or p['expected_task_count'] != 595 or p['metric_prefix'] != 'eval/webvoyager-benchmark'
            or p['expected_rollout_task_ids'] != [r['metadata']['task_id'] for r in tasks()]):
        raise ValueError('Wrong source, checkpoint or cohort')
    for k,v in dict(JUDGE_MODEL='gpt-4o',NUM_GPUS='1',TP_SIZE='1',NUM_ROLLOUT='0',
            BROWSER_CONCURRENCY='3',SGLANG_CONCURRENCY='3',WANDB_PROJECT='openwebrl-evals',
            OPENWEBRL_RECORD_EVAL_EXPECTED_TASK_COUNT='595').items():
        if p['environment'].get(k) != v:
            raise ValueError(f'Wrong worker environment: {k}')
    for flag,value in {'--eval-config':str(SOURCE/'openwebrl/webvoyager_benchmark.yaml'),
                       '--wandb-project':'openwebrl-evals','--seed':'1234'}.items():
        if p['command'].count(flag) != 1 or p['command'][p['command'].index(flag)+1] != value:
            raise ValueError('Wrong worker command')


def prepare():
    prepare_source()
    plans = [plan(m) for m in METHODS]
    for p in plans:
        shared.write_json(CONTROL/f'{p["method"]}-preview.json',p)
    request = dict(status='prepared_awaiting_exact_resource_approval', resources=RESOURCES,
        protocol=PROTOCOL, task_attempts=1785, source_validation=validate_source(),
        checkpoints={p['method']:dict(path=p['checkpoint'],adam_updates=p['completed_optimizer_updates']) for p in plans},
        scheduling='Three concurrent jobs,3 browsers each, after all currently approved OM2W repeats; no provider concurrency overlap.',
        estimate='Roughly8-10h each from observed OM2W throughput scaled595/300;12h cap each including startup/retries, release early.',
        reporting='One pass per model:595-task overall, valid-only with denominator, saved task archives/verdicts and matched per-task comparisons.',
        dataset_caveat='Released OpenWebRL dataset changes53 instructions (dates) versus paper-pinned FARA595; all3 models use the same frozen release.',
        api_usage='1785 new Browser Use trajectories and GPT-4o WebVoyager terminal judgements; normal API/browser charges.',
        submission_template='scripts/evaluate_arm_webvoyager90_1gpu.sbatch', prepared_epoch=time.time())
    shared.write_json(CONTROL/'request.json',request)
    return request


def accounting(method,job):
    approval = base.read(CONTROL/'approval.json')
    ids = approval.get('job_ids_by_method',{}).get(method,[])
    if (approval.get('approved') is not True or approval.get('resources') != RESOURCES
            or job not in ids or os.getenv('SLURM_JOB_ID') != job):
        raise ValueError('Exact WebVoyager resource approval and registered attempt required')
    raw = subprocess.check_output(['sacct','-X','-n','-P','-j',','.join(ids),
                                  '--format=JobID,State,ElapsedRaw,ExitCode'],text=True)
    attempts=[]
    for line in raw.splitlines():
        fields=line.split('|')
        if len(fields)>=4 and fields[0] in ids:
            attempts.append(dict(job_id=fields[0],state=fields[1],elapsed_seconds=int(fields[2]),exit_code=fields[3]))
    if {a['job_id'] for a in attempts} != set(ids):
        raise ValueError('Incomplete attempt accounting')
    used=sum(a['elapsed_seconds'] for a in attempts)
    result=dict(approved_seconds=43200,consumed_seconds=used,remaining_seconds=max(0,43200-used),
                attempts=attempts,checked_epoch=time.time())
    shared.write_json(CONTROL/f'{method}-attempts.json',result)
    return result


def audit(p):
    # Reuse the archive/HTTP402/cleanup checks, changing only benchmark identity.
    return shared.audit(p, expected_judge=('gpt-4o','webvoyager'))


def execute(job,method):
    budget=accounting(method,job)
    resources=shared.allocation(subprocess.check_output(['scontrol','show','job',job,'-o'],text=True),
                                job,requested_gpus=1,maximum_hours=12)
    remaining=min(resources['maximum_seconds'],budget['remaining_seconds'])-180
    if remaining<1200:
        raise ValueError('Less than20 minutes remain in original method budget')
    p=plan(method,job)
    root=CONTROL/f'controller-{method}-{job}';root.mkdir(exist_ok=False)
    shared.write_json(root/'provider-balance.json',shared.check_provider_balance())
    shared.write_json(root/'evaluation-plan.json',p)
    state=dict(complete=False,stage='evaluation',iteration=90,job_id=job)
    with (base.SUPERVISOR/'registry-update.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        registry=base.read(base.SUPERVISOR/'registry.json')
        key=f'webvoyager90-{method}'
        item=next((j for j in registry['jobs'] if j['key']==key),None)
        if item is None:
            item=dict(key=key,agent_reviewed_evaluation_roots=[]);registry['jobs'].append(item)
        elif item.get('job_id')!=job:
            item.setdefault('supervised_attempts',[]).append(dict(job_id=item['job_id'],controller_root=item['controller_root']))
        item.update(job_id=job,controller_root=str(root),training_root=p['output'],
            evaluations=[dict(iteration=90,label=p['cohort_label'],root=p['output'],verified=False)],
            approved_total_seconds=43200,budget_receipt=str(CONTROL/'approval.json'),last_budget_check=budget,
            verified_complete=False,requires_user=False,require_agent_completion_review=True,
            agent_completion_reviewed=False,requested_endpoint='595 WebVoyager archives and GPT-4o verdicts')
        shared.previous.atomic(base.SUPERVISOR/'registry.json',registry)
    shared.previous.atomic(root/'status.json',state)
    with (CONTROL/f'{method}-active.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            subprocess.run(['timeout','--signal=INT','--kill-after=120',str(int(remaining)-120),
                'srun',f'--jobid={job}','--nodes=1','--ntasks=1','--cpus-per-task=8',
                '--gres=gpu:h200:1','--exact','--cpu-bind=none',str(base.RUNTIME/'venv/bin/python'),
                '-B',str(Path(__file__).resolve()),'--worker','--manifest',str(root/'evaluation-plan.json')],check=True)
            state.update(complete=True,stage='artifacts_ready',evaluation=audit(p))
        except BaseException as exc:
            state.update(stage='failed',error=type(exc).__name__,detail=str(exc)[:600]);raise
        finally:
            shared.previous.atomic(root/'status.json',state)
            accounting(method,job)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare',action='store_true');parser.add_argument('--execute',action='store_true')
    parser.add_argument('--worker',action='store_true');parser.add_argument('--method',choices=METHODS)
    parser.add_argument('--job-id');parser.add_argument('--manifest',type=Path);args=parser.parse_args()
    if args.prepare: print(json.dumps(prepare(),indent=2))
    elif args.execute: execute(args.job_id,args.method)
    elif args.worker:
        p=base.read(args.manifest);validate(p)
        os.environ['OPENWEBRL_EXPECTED_SCHEDULER_OFFSET']=p['environment']['OPENWEBRL_EXPECTED_SCHEDULER_OFFSET']
        os.environ['OPENWEBRL_RECORD_EVAL_EXPECTED_TASK_COUNT']='595'
        base.evaluator.run(p,base.REPO/'.env')
    else: parser.error('Choose --prepare, --execute or --worker')
