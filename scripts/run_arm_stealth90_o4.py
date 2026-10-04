#!/usr/bin/env python3
"""Prepare three matched o4-mini/T0.6 cohorts; execute only with exact approval."""
import argparse
import fcntl
import hashlib
import json
import os
import pickletools
from pathlib import Path
import runpy
import shutil
import subprocess
import time
from types import SimpleNamespace
import zipfile

import run_arm_stealth90_repeats as base
import run_arm_stealth90_parallel as previous
from resume_baseline import allocation, validate_source, write_json

CONTROL = base.RUNTIME / 'arm-turn-bonus-preparation/stealth90-o4-t06-20260929'
SOURCE = base.RUNTIME / 'reference-arm-stealth90-o4-t06-20260929'
JUDGE_SOURCE = base.RUNTIME / 'reference-paper-om2w-20260912'
METHODS = ('baseline', 'additive', 'gate-b')
RESOURCES = dict(jobs=3, gpus_per_job=1, gpu_type='H200', hours_per_job=7,
                 cpus_per_job=8, memory_gib_per_job=240, max_concurrent_jobs=3,
                 total_gpu_hours=21, browsers_per_job=3)
PROTOCOL = 'Actor-only iteration90; Browser Use stealth; o4-mini/AgentTrek; T0.6 p0.95 k20; 4096 tokens; 30 turns; full300'
JUDGE_FILES = ('openwebrl/eval/reward_online_mind2web.py', 'openwebrl/eval/_shared.py')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare_source():
    validate_source(previous.SOURCE)
    validate_source(JUDGE_SOURCE)
    if not SOURCE.exists():
        shutil.copytree(previous.SOURCE, SOURCE, symlinks=True,
            ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.git', '.env*', '.browser_use_sessions'))
        for name in JUDGE_FILES:
            shutil.copy2(JUDGE_SOURCE/name, SOURCE/name)
        shutil.copy2(base.REPO/'scripts/arm_stealth_o4_generator.py', SOURCE/'openwebrl/eval_benchmark.py')
        shutil.copy2(JUDGE_SOURCE/'openwebrl/online_mind2web_benchmark.yaml', SOURCE/'openwebrl/online_mind2web_benchmark.yaml')
        path = SOURCE/'openwebrl/browser_training_config.yaml'
        text = path.read_text()
        if text.count('browser_rollout_concurrency: 4') != 1:
            raise ValueError('Unexpected browser concurrency source')
        path.write_text(text.replace('browser_rollout_concurrency: 4', 'browser_rollout_concurrency: 3'))
        manifest = base.read(SOURCE/'reference_manifest.json')
        changed = [*JUDGE_FILES, 'openwebrl/eval_benchmark.py',
                   'openwebrl/online_mind2web_benchmark.yaml', 'openwebrl/browser_training_config.yaml']
        manifest['recipe_files_sha256'].update({n: sha(SOURCE/n) for n in changed})
        manifest['paper_om2w_benchmark'] = dict(base.read(JUDGE_SOURCE/'reference_manifest.json')['paper_om2w_benchmark'],
            browser_concurrency=3, parent_source=str(JUDGE_SOURCE),
            changed_files_sha256={n: sha(SOURCE/n) for n in changed},
            partial_results='All300 task-addressable archives and judge verdicts, including aborts')
        manifest['browser_use_evaluation'].update(task_concurrency=3,
            unchanged='Actor-only Browser Use stealth; same300 tasks and browser adapter',
            judge='o4-mini/AgentTrek', temperature=0.6)
        manifest['single_gpu_evaluation'].update(browser_concurrency=3)
        manifest['stealth_o4_matched'] = dict(parent=str(previous.SOURCE), judge_source=str(JUDGE_SOURCE),
            protocol=PROTOCOL, changed_files=changed)
        write_json(SOURCE/'reference_manifest.json', manifest)
    validate_protocol_source()


def validate_protocol_source(source=SOURCE):
    import yaml
    validate_source(source)
    manifest = base.read(source/'reference_manifest.json')
    if manifest.get('stealth_o4_matched', {}).get('protocol') != PROTOCOL:
        raise ValueError('Wrong frozen stealth protocol')
    for name in JUDGE_FILES:
        if sha(source/name) != sha(JUDGE_SOURCE/name):
            raise ValueError('Historical AgentTrek judge implementation changed')
    module = runpy.run_path(str(source/'openwebrl/eval_benchmark.py'))
    configured, sampling = module['configure'](SimpleNamespace(judge_api_model='wrong',
        judge_prompt_variant='wrong'), dict(temperature=0, top_p=1, top_k=1))
    expected = dict(temperature=0.6, top_p=0.95, top_k=20, max_new_tokens=4096, repetition_penalty=1.0)
    if sampling != expected or configured.judge_api_model != 'o4-mini' or configured.judge_prompt_variant != 'agenttrek':
        raise ValueError('Executed sampling or judge differs from declared protocol')
    cfg = yaml.safe_load((source/'openwebrl/online_mind2web_benchmark.yaml').read_text())['eval']
    for k, v in dict(temperature=.6, top_p=.95, top_k=20, max_response_len=4096, n_samples_per_eval_prompt=1).items():
        if cfg['defaults'].get(k) != v:
            raise ValueError('Evaluation YAML changed')
    if cfg['datasets'][0]['custom_generate_function_path'] != 'openwebrl.eval_benchmark.generate':
        raise ValueError('Wrong generation entry point')
    return dict(sampling=sampling, judge='o4-mini', judge_prompt='agenttrek', max_steps=configured.max_steps,
                source_sha256=sha(source/'reference_manifest.json'))


def prepare_recovery_source(method):
    evidence = base.read(CONTROL/f'{method}-provider-recovery.json')
    target = CONTROL/f'recovery-source-{method}'
    tasks = [json.loads(line) for line in (SOURCE/'online_mind2web_monitor.jsonl').read_text().splitlines()]
    wanted = set(evidence['recovery_task_ids'])
    subset = [task for task in tasks if str(task['metadata']['task_id']) in wanted]
    if not wanted or len(subset) != len(wanted) or evidence['other_invalid'] + evidence['valid'] + len(wanted) != 300:
        raise ValueError('Recovery must contain exactly the independently identified credit-blocked tasks')
    if not target.exists():
        shutil.copytree(SOURCE, target, symlinks=True,
            ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.git', '.env*', '.browser_use_sessions'))
        dataset = target/'online_mind2web_monitor.jsonl'
        if dataset.is_symlink():
            raise ValueError('Refuse to modify shared dataset')
        dataset.write_text(''.join(json.dumps(t)+'\n' for t in subset))
        shutil.copy2(base.REPO/'scripts/arm_stealth_o4_generator.py', target/'openwebrl/eval_benchmark.py')
        manifest = base.read(target/'reference_manifest.json')
        for n in ['online_mind2web_monitor.jsonl', 'openwebrl/eval_benchmark.py']:
            manifest['recipe_files_sha256'][n] = sha(target/n)
        manifest['provider_recovery'] = dict(method=method, original_root=evidence['root'],
            expected_task_ids=[str(t['metadata']['task_id']) for t in subset],
            selection='Only HTTP402 credit-blocked tasks; retain all original valid outcomes and ordinary invalids',
            evidence_sha256=sha(CONTROL/f'{method}-provider-recovery.json'), parent_source=str(SOURCE))
        write_json(target/'reference_manifest.json', manifest)
    validate_protocol_source(target)
    provenance = base.read(target/'reference_manifest.json')['provider_recovery']
    if provenance['evidence_sha256'] != sha(CONTROL/f'{method}-provider-recovery.json'):
        raise ValueError('Recovery evidence changed')
    return target


def plan(method, job='PREPARE', recovery=False):
    if method not in METHODS:
        raise ValueError('Expected baseline, additive or gate-b')
    source = prepare_recovery_source(method) if recovery else SOURCE
    validate_protocol_source(source)
    checkpoint, report = base.checkpoint(method)
    label = f'o4-t06-{method}-r1' + ('-recovery' if recovery else '')
    output = base.RUNTIME/f'evaluations/stealth90-{label}-{job}'
    p = base.evaluator.build_plan(source, checkpoint, output, job, gpus=1,
                                 browser_env='browser-use', protocol='benchmark')
    ids = [str(json.loads(line)['metadata']['task_id']) for line in
           (source/'online_mind2web_monitor.jsonl').read_text().splitlines()]
    if not recovery and (len(ids) != 300 or len(set(ids)) != 300):
        raise ValueError('Expected exactly300 unique tasks')
    run_id = output.name
    p.update(method=method, repeat=1, cohort_label=label, completed_iteration=90,
        completed_optimizer_updates=report['completed_optimizer_updates'], protocol=PROTOCOL,
        actor_temperature=.6, expected_task_count=len(ids), expected_rollout_task_ids=ids,
        require_task_rollouts=True, parent_training_run=base.METHODS[method]['run_id'],
        wandb_run_id=run_id, browser_concurrency=3, judge_model='o4-mini', judge_prompt_variant='agenttrek')
    p['environment'].update(WANDB_RUN_ID=run_id, WANDB_RESUME='never', JUDGE_MODEL='o4-mini',
        BROWSER_CONCURRENCY='3', SGLANG_CONCURRENCY='3',
        BROWSER_TRAIN_CONFIG=str(source/'openwebrl/browser_training_config.yaml'),
        OPENWEBRL_EXPECTED_SCHEDULER_OFFSET=str(base.METHODS[method]['offset']),
        OPENWEBRL_RECORD_EVAL_EXPECTED_TASK_COUNT=str(len(ids)), PYTHONDONTWRITEBYTECODE='1',
        OPENWEBRL_MULTIMODAL_STORAGE_DIR=str(base.RUNTIME/f'multimodal-scratch/{run_id}'),
        FLASHINFER_WORKSPACE_BASE=str(base.RUNTIME/'flashinfer-stealth90'),
        WANDB_CACHE_DIR=str(output/'wandb-cache'), RAY_TMPDIR=f'/tmp/so4-{job}')
    p['command'][p['command'].index('--wandb-group')+1] = 'ARM iteration90 matched stealth o4-mini T0.6'
    if recovery:
        p['provider_recovery'] = base.read(source/'reference_manifest.json')['provider_recovery']
    p = base.evaluator.configure_evaluation_tracking(p)
    validate_plan(p)
    return p


def validate_plan(p):
    validate_protocol_source(Path(p['source']))
    env, cmd = p['environment'], p['command']
    expected_source = CONTROL/f'recovery-source-{p["method"]}' if p.get('provider_recovery') else SOURCE
    if (p['source'] != str(expected_source) or p['actor_temperature'] != .6 or
            p['protocol'] != PROTOCOL or p['optimizer_updates_requested'] != 0 or
            p['completed_iteration'] != 90 or p['checkpoint_index'] != 89):
        raise ValueError('Wrong source, checkpoint or declared protocol')
    for k, v in dict(JUDGE_MODEL='o4-mini', NUM_GPUS='1', TP_SIZE='1', NUM_ROLLOUT='0',
                     BROWSER_CONCURRENCY='3', SGLANG_CONCURRENCY='3', WANDB_PROJECT='openwebrl-evals').items():
        if env.get(k) != v:
            raise ValueError(f'Wrong worker environment: {k}')
    # Ray appends /ray/session_<timestamp>_<pid>/sockets/plasma_store.
    # Reserve room for a ten-digit PID before any allocation is submitted.
    socket_path = str(Path(env['RAY_TMPDIR'])/'ray/session_2026-09-29_09-45-06_123456_1234567890/sockets/plasma_store')
    if len(socket_path.encode()) > 107:
        raise ValueError('Ray temporary root exceeds the AF_UNIX socket path limit')
    if p.get('provider_recovery'):
        provenance = base.read(expected_source/'reference_manifest.json')['provider_recovery']
        if p['provider_recovery'] != provenance or p['expected_rollout_task_ids'] != provenance['expected_task_ids']:
            raise ValueError('Recovery differs from audited outage task IDs')
    if (cmd[cmd.index('--eval-config')+1] != str(expected_source/'openwebrl/online_mind2web_benchmark.yaml')
            or cmd[cmd.index('--wandb-project')+1] != 'openwebrl-evals'):
        raise ValueError('Wrong worker command')


def prepare():
    CONTROL.mkdir(parents=True, exist_ok=True)
    prepare_source()
    plans = [plan(m) for m in METHODS]
    for p in plans:
        write_json(CONTROL/f'{p["method"]}-preview.json', p)
    assert all(p['expected_rollout_task_ids'] == plans[0]['expected_rollout_task_ids'] for p in plans)
    request = dict(status='prepared_awaiting_exact_resource_approval', resources=RESOURCES,
        protocol=PROTOCOL, full300_evaluations=3, task_attempts=900,
        estimate='About5–6h per cohort plus queue time;7h cap each including startup and recovery.',
        scheduling='Three concurrent single-GPU jobs with3 browser sessions each,9 total; record actual overlap/dates.',
        source=str(SOURCE), api_usage='900 fresh Browser Use task attempts and o4-mini terminal judgements, with normal service charges',
        previous_evaluations='Preserved separately; no reuse of old verdicts or trajectories for new comparisons',
        reporting='Overall and valid-only, fixed100 and full300; paired per-task tests; primary three pairwise tests with Holm correction',
        checkpoint_provenance=[dict(method=p['method'],checkpoint=p['checkpoint'],adam_updates=p['completed_optimizer_updates']) for p in plans],
        source_validation=validate_protocol_source(), prepared_epoch=time.time(),
        submission_template='scripts/evaluate_arm_stealth90_o4_1gpu.sbatch')
    write_json(CONTROL/'request.json', request)
    return request


def audit(p, expected_judge=('o4-mini', 'agenttrek')):
    root = Path(p['output']);status = base.read(root/'status.json')
    if (root/'provider-blocked.json').exists():
        raise ValueError('Browser Use credit exhaustion: cohort requires recovery')
    if not status.get('complete') or status.get('returncode') != 0:
        raise ValueError('Worker did not complete')
    records = [base.read(f) for f in (root/'rollouts').glob('*.json')]
    ids = [r['task_id'] for r in records]
    count = p['expected_task_count']
    if len(ids) != count or len(set(ids)) != count or set(ids) != set(p['expected_rollout_task_ids']):
        raise ValueError('Task cohort incomplete or different')
    for r in records:
        if (r['judge_model'], r['judge_prompt_variant']) != expected_judge:
            raise ValueError('Wrong saved judge identity')
        archive = (root/'rollouts'/r['rollout_file']).resolve()
        if not archive.is_relative_to((root/'rollouts').resolve()):
            raise ValueError('Archive outside rollout directory')
        with zipfile.ZipFile(archive) as z:
            if not z.namelist():
                raise ValueError('Empty archive')
            # Historical workers swallowed HTTP402 into an ABORTED turn. Inspect
            # metadata without loading tensors or executing pickle payloads.
            from arm_stealth_o4_generator import credits_exhausted
            for name in z.namelist():
                if name.endswith('/data.pkl') and any(
                        isinstance(value, str) and value.startswith('generation_error:')
                        and credits_exhausted(value)
                        for _, value, _ in pickletools.genops(z.read(name))):
                    raise ValueError('Browser Use credit exhaustion: cohort requires recovery')
    sessions = root/'browser_sessions'
    if sessions.exists() and any(f.is_file() for f in sessions.iterdir()):
        raise ValueError('Browser cleanup still pending')
    successes = sum(r['metrics']['successes'] for r in records)
    valid = sum(r['metrics']['valid_trajectories'] for r in records)
    result = dict(root=str(root), iteration=90, tasks=count, successes=successes, valid=valid,
        invalid=count-valid, overall=successes/count, valid_only=successes/valid if valid else None,
        saved_rollouts=count, saved_task_records=count, checkpoint=p['checkpoint'], protocol=p.get('protocol', PROTOCOL))
    write_json(root/'audit.json', result)
    return result


def register(job, method, root, p):
    with (base.SUPERVISOR/'registry-update.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        registry = base.read(base.SUPERVISOR/'registry.json')
        key = f'stealth90-o4-t06-{method}-r1'
        item = next((j for j in registry['jobs'] if j['key'] == key), None)
        if item is None:
            item = dict(key=key, agent_reviewed_evaluation_roots=[]);registry['jobs'].append(item)
        elif item['job_id'] != job:
            item.setdefault('supervised_attempts', []).append(dict(job_id=item['job_id'],controller_root=item['controller_root']))
        item.update(job_id=job, controller_root=str(root), training_root=str(root),
            evaluations=[dict(iteration=90,label=p['cohort_label'],root=p['output'],verified=False)],
            approved_total_seconds=25200, budget_receipt=str(CONTROL/'approval.json'),
            verified_complete=False, requires_user=False, require_agent_completion_review=True,
            agent_completion_reviewed=False, requested_endpoint='Fresh full300 o4-mini/T0.6 archives and verdicts')
        previous.atomic(base.SUPERVISOR/'registry.json', registry)


def check_provider_balance():
    from dotenv import load_dotenv
    load_dotenv(base.REPO/'.env')
    import sys
    sys.path.insert(0, str(base.RUNTIME/'browser-use-sdk-3.11.3'))
    from browser_use_sdk import BrowserUse
    account = BrowserUse(api_key=os.environ['BROWSER_USE_API_KEY']).billing.account()
    balance = float(account.total_credits_balance_usd)
    if balance < .01:
        raise ValueError('Browser Use credits below the minimum session balance; requires funded account')
    return dict(balance_usd=balance, checked_epoch=time.time())


def merge_recovery(p):
    old = Path(p['provider_recovery']['original_root'])
    original = {r['task_id']:r for f in (old/'rollouts').glob('*.json') for r in [base.read(f)]}
    retry = {r['task_id']:r for f in (Path(p['output'])/'rollouts').glob('*.json') for r in [base.read(f)]}
    wanted = set(p['provider_recovery']['expected_task_ids'])
    if set(retry) != wanted or not wanted <= set(original) or len(original) != 300:
        raise ValueError('Recovery merge identities differ')
    if any(original[k]['metrics']['valid_trajectories'] for k in wanted):
        raise ValueError('Refuse to replace a valid original outcome')
    merged = dict(original, **retry)
    successes = sum(r['metrics']['successes'] for r in merged.values())
    valid = sum(r['metrics']['valid_trajectories'] for r in merged.values())
    result = dict(tasks=300, successes=successes, valid=valid, invalid=300-valid,
        overall=successes/300, valid_only=successes/valid if valid else None,
        original_root=str(old), recovery_root=p['output'], retry_count=len(wanted),
        retained_original_records=300-len(wanted), protocol=PROTOCOL,
        task_sources={k:p['output'] if k in wanted else str(old) for k in merged},
        requires_independent_agent_review=True)
    write_json(Path(p['output'])/'merged-audit.json', result)
    return result


def execute(job, method, recovery=False):
    approval = base.read(CONTROL/'approval.json')
    ids = approval.get('job_ids_by_method', {}).get(method, [])
    if (approval.get('approved') is not True or approval.get('resources') != RESOURCES
            or job not in ids or os.getenv('SLURM_JOB_ID') != job):
        raise ValueError('Exact resource approval and registered attempt required')
    resources = allocation(subprocess.check_output(['scontrol','show','job',job,'-o'],text=True),
                           job,requested_gpus=1,maximum_hours=7)
    raw = subprocess.check_output(['sacct','-X','-n','-P','-j',','.join(ids),
                                  '--format=JobID,ElapsedRaw'],text=True)
    used = sum(int(f[1]) for line in raw.splitlines() if len(f:=line.split('|'))>=2 and f[0] in ids and f[1].isdigit())
    remaining = min(resources['maximum_seconds'],25200-used)-180
    if remaining < 1200:
        raise ValueError('Less than20 minutes remain in original cohort budget')
    root = CONTROL/f'controller-{method}-{job}';root.mkdir(exist_ok=False)
    write_json(root/'provider-balance.json', check_provider_balance())
    p = plan(method,job,recovery=recovery);write_json(root/'evaluation-plan.json',p)
    state = dict(complete=False,stage='evaluation',iteration=90,job_id=job,consumed_seconds_at_start=used)
    previous.atomic(root/'status.json',state);register(job,method,root,p)
    with (CONTROL/f'{method}-active.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            command = ['timeout','--signal=INT','--kill-after=120',str(int(remaining)-120),
                'srun',f'--jobid={job}','--nodes=1','--ntasks=1','--cpus-per-task=8',
                '--gres=gpu:h200:1','--exact','--cpu-bind=none',str(base.RUNTIME/'venv/bin/python'),
                '-B',str(Path(__file__).resolve()),'--worker','--manifest',str(root/'evaluation-plan.json')]
            subprocess.run(command,check=True)
            state.update(complete=True,stage='artifacts_ready',evaluation=audit(p))
            if recovery:
                state['merged_evaluation'] = merge_recovery(p)
        except BaseException as exc:
            state.update(stage='failed',error=type(exc).__name__,detail=str(exc)[:600]);raise
        finally:
            previous.atomic(root/'status.json',state)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare',action='store_true');parser.add_argument('--execute',action='store_true')
    parser.add_argument('--worker',action='store_true');parser.add_argument('--method',choices=METHODS)
    parser.add_argument('--job-id');parser.add_argument('--manifest',type=Path)
    parser.add_argument('--recovery',action='store_true')
    args=parser.parse_args()
    if args.prepare:print(json.dumps(prepare(),indent=2))
    elif args.worker:
        p=base.read(args.manifest);validate_plan(p)
        os.environ['OPENWEBRL_EXPECTED_SCHEDULER_OFFSET']=p['environment']['OPENWEBRL_EXPECTED_SCHEDULER_OFFSET']
        os.environ['OPENWEBRL_RECORD_EVAL_EXPECTED_TASK_COUNT']=str(p['expected_task_count'])
        base.evaluator.run(p,base.REPO/'.env')
    elif args.execute:execute(args.job_id,args.method,args.recovery)
    else:parser.error('Choose --prepare, --execute or --worker')
