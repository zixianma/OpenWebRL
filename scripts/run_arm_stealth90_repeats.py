#!/usr/bin/env python3
"""Prepare nine matched stealth evaluations; execute only in an approved job."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import evaluate_baseline_checkpoint as evaluator
from resume_baseline import allocation, validate_source, write_json
from resume_arm_mixed_to60 import audit_evaluation

REPO, RUNTIME = evaluator.REPO, evaluator.RUNTIME
CONTROL = RUNTIME / 'arm-turn-bonus-preparation/stealth90-repeats-20260928'
SOURCE = RUNTIME / 'reference-arm-stealth90-repeats-20260928'
PARENT = RUNTIME / 'reference-arm-eval80-additive-20260920-v2'
BROWSER_SOURCE = RUNTIME / 'reference-browseruse-eight-decimal-v2-20260911'
METHODS = {
    'baseline': dict(root='runs/openwebrl-4b-reference-294421-20260913T211532',
                     checkpoint='iter_0000089', report='checkpoint_after90_audit.json',
                     offset=1, run_id='qcq7i4ug'),
    'additive': dict(root='evaluations/arm-failure-additive-311962',
                     checkpoint='runtime/iter_0000089', report='iterations/0089/checkpoint-validation.json',
                     offset=0, run_id='arm-failure-additive-295786'),
    'gate-b': dict(root='evaluations/arm-failure-additive-335729-b100-iter90',
                   checkpoint='runtime/iter_0000089', report='iterations/0089/checkpoint-validation.json',
                   offset=0, run_id='arm-gate-b-309053'),
}
ORDERS = [('baseline', 'additive', 'gate-b'),
          ('additive', 'gate-b', 'baseline'),
          ('gate-b', 'baseline', 'additive')]
RESOURCES = dict(gpus=2, gpu_type='H200', hours=24, gpu_hours=48, cpus=16, memory_gib=480)
SUPERVISOR = RUNTIME/'arm-turn-bonus-preparation/mixed-reweight-20260927/supervisor'


def read(path):
    return json.loads(Path(path).read_text())


def checkpoint(method):
    spec = METHODS[method]
    root = RUNTIME / spec['root']
    cp = root / spec['checkpoint']
    report = read(root / spec['report'])
    if (report['iteration'] != 89 or Path(report['checkpoint']).resolve() != cp.resolve()
            or report['scheduler_minus_optimizer_updates'] != spec['offset']):
        raise ValueError('Checkpoint identity or scheduler offset differs')
    for name, size in report['shard_files'].items():
        if (cp / name).stat().st_size != size:
            raise ValueError('Checkpoint shard changed after validation')
    for path in [cp / 'common.pt', cp / '.metadata', Path(report['dataset_cursor'])]:
        if not path.is_file() or not path.stat().st_size:
            raise ValueError('Checkpoint component or task cursor missing')
    return cp, report


def prepare_source():
    validate_source(PARENT)
    validate_source(BROWSER_SOURCE)
    if SOURCE.exists():
        validate_source(SOURCE)
        manifest = read(SOURCE / 'reference_manifest.json')
        if manifest.get('stealth90_repeats', {}).get('parent') != str(PARENT):
            raise ValueError('Existing source has different provenance')
        if not manifest.get('browser_use_evaluation'):
            manifest['browser_use_evaluation'] = manifest['stealth90_repeats']
            write_json(SOURCE/'reference_manifest.json', manifest)
        return
    shutil.copytree(PARENT, SOURCE, symlinks=True,
        ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.git', '.env*', '.browser_use_sessions'))
    changed = []
    patches = {
        'scripts/run_h200_browser.sh': ('export SLIME_BROWSER_ENV_MODE=local_process',
            'export SLIME_BROWSER_ENV_MODE="${SLIME_BROWSER_ENV_MODE:-local_process}"'),
        'openwebrl/browser_training_config.yaml': ('browser_rollout_concurrency: 32', 'browser_rollout_concurrency: 8'),
        'openwebrl/env/config.yaml': ('  timeout: 10 ', '  timeout: 12 '),
    }
    for name, (old, new) in patches.items():
        p = SOURCE / name
        text = p.read_text()
        if text.count(old) != 1:
            raise ValueError(f'Unexpected source layout: {name}')
        p.write_text(text.replace(old, new, 1)); changed.append(name)
    adapter = 'openwebrl/env/browser_use_env.py'
    shutil.copyfile(BROWSER_SOURCE / adapter, SOURCE / adapter); changed.append(adapter)
    shutil.copytree(BROWSER_SOURCE / 'browser_use_sdk', SOURCE / 'browser_use_sdk',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    changed.extend(str(p.relative_to(SOURCE)) for p in (SOURCE / 'browser_use_sdk').rglob('*.py'))
    manifest = read(SOURCE / 'reference_manifest.json')
    manifest['recipe_files_sha256'].update({n: hashlib.sha256((SOURCE/n).read_bytes()).hexdigest() for n in changed})
    manifest['stealth90_repeats'] = dict(parent=str(PARENT), browser_source=str(BROWSER_SOURCE),
        changed_files=changed, task_concurrency=8, session_timeout_minutes=12,
        backend='Browser Use stealth', requested_proxy_country_code=None,
        unchanged='Same actor-only GPT-4.1/action_history/T0 full300 monitor, prompts and task saver')
    manifest['browser_use_evaluation'] = manifest['stealth90_repeats']
    write_json(SOURCE / 'reference_manifest.json', manifest)
    validate_source(SOURCE)


def build_plan(method, repeat, job='PREPARE', *, gpus=2, source=SOURCE):
    if method not in METHODS or repeat not in (1, 2, 3):
        raise ValueError('Expected one of the three methods and repeat1–3')
    cp, report = checkpoint(method)
    label = f'{method}-r{repeat}'
    output = RUNTIME / f'evaluations/stealth90-{label}-{job}'
    p = evaluator.build_plan(source, cp, output, job, browser_env='browser-use', gpus=gpus)
    run_id = f'stealth90-{label}-{job}'
    tasks = [json.loads(line) for line in (source/'online_mind2web_monitor.jsonl').read_text().splitlines()]
    ids = [str(t['metadata']['task_id']) for t in tasks]
    if len(ids) != 300 or len(set(ids)) != 300:
        raise ValueError('Full300 cohort required')
    p.update(wandb_run_id=run_id, parent_training_run=METHODS[method]['run_id'],
        method=method, repeat=repeat, cohort_label=label, completed_iteration=90,
        expected_task_count=300, require_task_rollouts=True, expected_rollout_task_ids=ids,
        completed_optimizer_updates=report['completed_optimizer_updates'],
        protocol='Actor only; Browser Use stealth; GPT-4.1/action_history; T0; 4096 tokens; 30 turns; full300')
    p['command'][p['command'].index('--wandb-group')+1] = 'ARM iteration90 stealth repeated comparison'
    p['command'][p['command'].index('--eval-config')+1] = str(source/'openwebrl/online_mind2web_monitor.yaml')
    p['environment'].update(WANDB_RUN_ID=run_id, WANDB_RESUME='never',
        OPENWEBRL_EXPECTED_SCHEDULER_OFFSET=str(METHODS[method]['offset']),
        OPENWEBRL_RECORD_EVAL_EXPECTED_TASK_COUNT='300', PYTHONDONTWRITEBYTECODE='1',
        BROWSER_TRAIN_CONFIG=str(source/'openwebrl/browser_training_config.yaml'),
        FLASHINFER_WORKSPACE_BASE=str(RUNTIME/'flashinfer-stealth90'),
        OPENWEBRL_MULTIMODAL_STORAGE_DIR=str(RUNTIME/f'multimodal-scratch/stealth90-{label}-{job}'))
    return evaluator.configure_evaluation_tracking(p)


def prepare():
    CONTROL.mkdir(parents=True, exist_ok=True)
    prepare_source()
    plans = [build_plan(method, repeat) for repeat, order in enumerate(ORDERS, 1) for method in order]
    for p in plans:
        write_json(CONTROL / (p['cohort_label']+'-preview.json'), p)
    request = dict(status='prepared_awaiting_exact_resource_approval', resources=RESOURCES,
        full300_evaluations=9, task_attempts=2700, order=ORDERS,
        protocol=plans[0]['protocol'], source=str(SOURCE),
        api_usage='2700 task attempts with fresh Browser Use sessions and GPT-4.1 terminal judging; service fees additional',
        expected_wall_hours=[18,21], estimate_basis='Historical eight-browser full300 evaluations took about1.9–2.1h each',
        checkpoint_provenance=[dict(method=m, checkpoint=str(checkpoint(m)[0]),
            adam_updates=checkpoint(m)[1]['completed_optimizer_updates'], scheduler_offset=METHODS[m]['offset']) for m in METHODS],
        reporting='Each repeat plus mean/sample SD across3; paired per-task comparisons; no pass@3 or three-training-seed claim',
        existing_training_budgets='No diversion from mixed-pair or Gate B continuation approvals',
        prepared_epoch=time.time(), submission_template='scripts/evaluate_arm_stealth90_repeats_2gpu.sbatch')
    write_json(CONTROL/'request.json', request)
    return request


def audit(plan):
    result = audit_evaluation(plan)
    receipts = Path(plan['output'])/'browser_sessions'
    leftovers = [p.name for p in receipts.iterdir() if p.is_file()] if receipts.exists() else []
    if leftovers:
        raise ValueError('Unconfirmed owned browser-session shutdown; inspect receipts before continuing')
    result.update(method=plan['method'], repeat=plan['repeat'], cohort_label=plan['cohort_label'])
    write_json(Path(plan['output'])/'audit.json', result)
    return result


def register(job, root, state, current=None):
    """Expose this evaluation queue to the existing active repair mechanism."""
    evaluations = [dict(iteration=90, label=a['cohort_label'], root=a['root'], verified=True)
                   for a in state['evaluations']]
    if current is not None:
        evaluations.append(dict(iteration=90, label=current['cohort_label'],
                                root=current['output'], verified=False))
    with (SUPERVISOR/'registry-update.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        registry = read(SUPERVISOR/'registry.json')
        item = next((j for j in registry['jobs'] if j['key']=='stealth90-repeats'), None)
        if item is None:
            item=dict(key='stealth90-repeats', agent_reviewed_evaluation_roots=[])
            registry['jobs'].append(item)
        item.update(job_id=job, controller_root=str(root), training_root=str(root),
            evaluations=evaluations, requested_endpoint='nine full300 evaluation cohorts',
            approved_total_seconds=86400, budget_receipt=str(CONTROL/'approval.json'),
            purpose='Iteration90 baseline/additive/Gate B, three fresh stealth repeats each',
            verified_complete=False, requires_user=False, scheduled_continuations=[])
        write_json(SUPERVISOR/'registry.json',registry)


def execute(job):
    approval = read(CONTROL/'approval.json')
    if (not approval.get('approved') or approval.get('resources') != RESOURCES
            or os.environ.get('SLURM_JOB_ID') != job or job not in approval.get('job_ids', [])):
        raise ValueError('Exact approved resources and owning allocation required')
    resources = allocation(subprocess.check_output(['scontrol','show','job',job,'-o'],text=True),
                           job,requested_gpus=2,maximum_hours=24)
    raw = subprocess.check_output(['sacct','-X','-n','-P','-j',','.join(approval['job_ids']),
                                   '--format=JobID,ElapsedRaw'],text=True)
    used = sum(int(fields[1]) for line in raw.splitlines()
               if len(fields := line.split('|')) >= 2 and fields[1].isdigit()
               and fields[0] in approval['job_ids'])
    resources['maximum_seconds'] = min(resources['maximum_seconds'],86400-used-180)
    root = CONTROL/f'controller-{job}';root.mkdir(exist_ok=True)
    lock = (CONTROL/'execution.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state = dict(job_id=job, complete=False, evaluations=[], remaining_seconds=resources['maximum_seconds'])
    started = time.time()
    try:
        for repeat, order in enumerate(ORDERS, 1):
            for method in order:
                output = RUNTIME/f'evaluations/stealth90-{method}-r{repeat}-{job}'
                if output.exists():
                    p = read(output/'evaluation_manifest.json')
                    if (p.get('method'), p.get('repeat'), p.get('checkpoint')) != (method, repeat, str(checkpoint(method)[0])):
                        raise ValueError('Saved evaluation has different lineage or repeat identity')
                else:
                    p = build_plan(method, repeat, job)
                planfile = root/f'{p["cohort_label"]}-plan.json';write_json(planfile,p)
                state.update(stage='evaluation', cohort=p['cohort_label'], output=p['output'])
                state['iteration'] = 90
                write_json(root/'status.json',state)
                register(job,root,state,p)
                output = Path(p['output'])
                if output.exists():
                    result = audit(p)  # Never blindly overwrite or rerun an incomplete attempt.
                else:
                    remaining = resources['maximum_seconds']-(time.time()-started)
                    if remaining < 3*3600:
                        state.update(stage='budget-stop',remaining_seconds=int(remaining));write_json(root/'status.json',state);return
                    command = ['srun',f'--jobid={job}','--overlap','--nodes=1','--ntasks=1',
                        '--cpus-per-task=16','--gres=gpu:h200:2','--exact','--cpu-bind=none',
                        str(RUNTIME/'venv/bin/python'),'-B',str(Path(__file__).resolve()),
                        '--worker','--manifest',str(planfile)]
                    # Bound even a stalled worker by both the cohort allowance
                    # and the remaining original approval, with cleanup time.
                    command = ['timeout','--signal=INT','--kill-after=120',
                               str(int(min(3*3600,remaining))-120),*command]
                    subprocess.run(command,check=True)
                    result = audit(p)
                state['evaluations'].append(result);write_json(root/'status.json',state)
                register(job,root,state)
        state.update(complete=True,stage='verified_complete');write_json(root/'status.json',state)
    except BaseException as exc:
        state.update(stage='failed',error=type(exc).__name__,detail=str(exc)[:600]);write_json(root/'status.json',state)
        raise


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--job-id')
    parser.add_argument('--manifest',type=Path)
    args=parser.parse_args()
    if args.prepare: print(json.dumps(prepare(),indent=2))
    elif args.worker:
        p=read(args.manifest)
        os.environ['OPENWEBRL_EXPECTED_SCHEDULER_OFFSET']=p['environment']['OPENWEBRL_EXPECTED_SCHEDULER_OFFSET']
        os.environ['OPENWEBRL_RECORD_EVAL_EXPECTED_TASK_COUNT']='300'
        evaluator.run(p,REPO/'.env')
    elif args.execute: execute(args.job_id)
    else: parser.error('Choose --prepare, --execute or --worker')
