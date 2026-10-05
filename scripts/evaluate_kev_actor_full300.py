#!/usr/bin/env python3
"""Prepare/submit/own the direct Kev27B full300 evaluation; exact approval required."""
from __future__ import annotations
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from urllib.parse import urlsplit

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from openwebrl import jev_eval as io, kev_eval as kev
import evaluate_jev_ultrafast as cohort
import evaluate_kev_pair as pair

RUNTIME = pair.RUNTIME
ROOT = RUNTIME / 'evaluations/kev27b-actor-full300-20261004'
POINTER = RUNTIME / 'current_kev27b_actor_full300.json'
SOURCE = RUNTIME / 'reference-kev27b-actor-full300-20261004-v1'
ENV_FILE = Path('/gpfs/projects/krishna/zixianma/OpenWebRL/.env')
RESOURCES = dict(gpus=1, gpu_type='H200', cpus=8, memory_gib=120, total_seconds=14400)
LIMITS = dict(browser_sessions=330, concurrent_browsers=2, browser_expiry_minutes=12,
    browser_minutes=3960, local_kev_requests=59410, text_http_attempts=10000,
    text_completion_tokens=1024, judge_http_attempts=1320, judge_completion_tokens=4096)
FILES = ('openwebrl/__init__.py', 'openwebrl/jev_eval.py', 'openwebrl/kev_eval.py',
    'openwebrl/jev_harness.py', 'openwebrl/selection_budget.py', 'openwebrl/decision_selection.py',
    'openwebrl/arm_inference.py', 'openwebrl/eval/reward_online_mind2web.py',
    'scripts/evaluate_jev_ultrafast.py', 'scripts/evaluate_kev_pair.py',
    'scripts/evaluate_kev_actor_full300.py', 'scripts/track_kev_actor_full300.py',
    'scripts/supervise_kev_actor_full300.py')
TERMINAL = {'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL', 'PREEMPTED', 'BOOT_FAIL'}


def read(path):
    return json.loads(Path(path).read_text())


def check_models():
    audit = read(pair.PREPARATION / 'verified-models.json')['27b']
    if not audit['verified'] or audit['spec'] != kev.model_spec('27b'):
        raise ValueError('Kev27B identity differs from verified weights')
    for filename, receipt in audit['files'].items():
        stat = Path(filename).stat()
        if (stat.st_size, stat.st_mtime_ns) != (receipt['size'], receipt['mtime_ns']):
            raise ValueError('Model file changed after independent hash verification')
    return io.digest(audit)


def prepare(root=ROOT):
    root.mkdir(parents=True, exist_ok=True)
    dataset = REPO / 'openwebrl/data/eval/online-mind2web.jsonl'
    tasks = io.load_tasks(dataset)
    if len(tasks) != 300:
        raise ValueError('Expected all300 unique benchmark tasks')
    dependencies = read(pair.PREPARATION / 'environment.json')
    versions = subprocess.check_output([str(RUNTIME/'kev-eval-venv/bin/python'), '-c',
        'import importlib.metadata as m,json,sys; print(json.dumps({p:m.version(p) for p in sys.argv[1:]}))',
        *dependencies['packages']], text=True)
    if json.loads(versions) != dependencies['packages']:
        raise ValueError('Kev serving environment changed')
    args = argparse.Namespace(browser='browser-use', text_provider='openai',
        upstream=RUNTIME/'jev-ultrafast-upstream', harness_revision='upstream-v1',
        decision_provider='kev', kev_variant='27b', kev_endpoint='http://127.0.0.1:18763/v1/systemone')
    cfg = cohort.configuration(args)
    budget = dict(root=str(root/'budget'), limits={k:LIMITS[k] for k in
        ('browser_sessions', 'local_kev_requests', 'text_http_attempts', 'judge_http_attempts')})
    cfg.update(budget=budget)
    io.load_credentials(ENV_FILE, cfg)  # Presence check only, no requests.
    # Freeze a minimal runtime so ongoing experiments and subsequent edits stay isolated.
    for name in FILES:
        source, target = REPO/name, SOURCE/name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.read_bytes() != source.read_bytes():
            raise ValueError('Frozen source changed; preserve it and create a new revision')
        if not target.exists(): shutil.copy2(source, target)
    hashes = {name:kev.file_hash(SOURCE/name) for name in FILES}
    cfg['code_sha256'].update(hashes)
    cp = dict(config=cfg, tasks=tasks, task_file_sha256=kev.file_hash(dataset),
              workers=2, wall_budget_seconds=RESOURCES['total_seconds'])
    warmup = read(RUNTIME/'evaluations/kev-pair-om2w-pilot-20261004/warmup-requests.json')
    plan = dict(experiment='kev27b_actor', status='prepared_awaiting_exact_resource_approval',
        resources=RESOURCES, limits=LIMITS, source=str(SOURCE), source_sha256=hashes,
        tasks=tasks, task_file_sha256=cp['task_file_sha256'], cohort_plan_sha256=io.digest(cp),
        kev=kev.model_spec('27b'), model_audit_sha256=check_models(), upstream=pair.source_identity(),
        dependencies=dependencies, protocol=cfg, budget=budget, warmup_sha256=io.digest(warmup),
        wandb_project='openwebrl-evals', wandb_id='kev27b-actor-full300-20261004',
        pilot=dict(successes=3, tasks=10, mean_total_seconds=25.786215553293005,
            linear_estimate_seconds_for_300_at_concurrency2=3867.932332993951),
        comparison='Fresh all300, including pilot10; same direct pilot policy, not SFT candidate selection.',
        recovery='Charge all scheduler elapsed and every reservation. Never reroll valid failures. '
                 'Preserve invalid/interrupted attempts for diagnosis before a separate retry; no transferred pilot budget.',
        completion_requires='All300 independently audited with rollouts, final screenshots and verdicts '
                            'or explicit invalid diagnoses; closed browsers, scheduler accounting, W&B, docs/private review.')
    for path, value in ((root/'actor/plan.json', cp),
                        (root/'warmup-requests.json', warmup)):
        if path.exists() and read(path) != value: raise ValueError('Prepared manifest changed: '+str(path))
        io.write_json(path, value)
    batch=f'''#!/usr/bin/env bash
# Prepared only; 4 hours TOTAL including all failed/replacement allocations.
#SBATCH --job-name=kev27-actor-300
#SBATCH --account=zixianma
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --signal=B:TERM@180
#SBATCH --output={RUNTIME}/logs/kev27-actor-300-%j.out
set -euo pipefail
cd {SOURCE}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export WANDB_PROJECT=openwebrl-evals
export CPATH={RUNTIME}/src/python-headers/Include:{RUNTIME}/src/python-headers
export PLAYWRIGHT_BROWSERS_PATH={RUNTIME}/browsers
exec {RUNTIME}/jev-eval-venv/bin/python scripts/evaluate_kev_actor_full300.py --root {root} --execute
'''
    (root/'launch.sbatch').write_text(batch)
    plan['launch_sha256']=kev.file_hash(root/'launch.sbatch')
    if (root/'plan.json').exists() and read(root/'plan.json')!=plan:
        raise ValueError('Prepared plan changed; preserve it before revising')
    io.write_json(root/'plan.json',plan)
    if not (root/'approval.json').exists():
        io.write_json(root/'approval.json', dict(approved=False, resources=RESOURCES, limits=LIMITS,
            plan_sha256=kev.file_hash(root/'plan.json'), attempts=[], authorization='Exact new allocation/API approval pending'))
    io.write_json(root/'preflight.json',dict(prepared_unix=time.time(),tasks=300,
        unique_task_ids=300, frozen_files=len(hashes), checkpoint_verified=True,
        source_verified=True, dependencies_verified=True, credentials_present=True,
        submitted=False, paid_calls=0, plan_sha256=kev.file_hash(root/'plan.json')))
    return plan


def verify(root):
    plan=read(root/'plan.json')
    if kev.file_hash(root/'launch.sbatch')!=plan['launch_sha256']:
        raise ValueError('Prepared resource request changed')
    for name, sha in plan['source_sha256'].items():
        if kev.file_hash(Path(plan['source'])/name)!=sha: raise ValueError('Frozen source changed: '+name)
    if io.digest(read(root/'actor/plan.json'))!=plan['cohort_plan_sha256']:
        raise ValueError('Cohort plan changed')
    if io.digest(read(root/'warmup-requests.json'))!=plan['warmup_sha256']:
        raise ValueError('Warmup requests changed')
    if plan['model_audit_sha256']!=check_models() or plan['upstream']!=pair.source_identity():
        raise ValueError('Kev source/model changed')
    versions=subprocess.check_output([str(RUNTIME/'kev-eval-venv/bin/python'),'-c',
        'import importlib.metadata as m,json,sys; print(json.dumps({p:m.version(p) for p in sys.argv[1:]}))',
        *plan['dependencies']['packages']],text=True)
    if json.loads(versions)!=plan['dependencies']['packages']:
        raise ValueError('Kev dependencies changed after preparation')
    return plan


def approval(root, plan):
    value=read(root/'approval.json')
    if (not value['approved'] or value['resources']!=plan['resources'] or
            value['limits']!=plan['limits'] or value['plan_sha256']!=kev.file_hash(root/'plan.json')):
        raise ValueError('Exact resource/API approval required before submission or execution')
    return value


def scheduler(job):
    if not str(job).isdecimal():raise ValueError('Invalid scheduler job ID')
    text=subprocess.check_output(['sacct','-X','-n','-P','-j',str(job),
        '--format=JobIDRaw,State,ElapsedRaw'],text=True,timeout=30)
    row=next((x.split('|') for x in text.splitlines() if x.split('|')[0]==str(job)),None)
    if row is None:raise ValueError('Missing scheduler record')
    return dict(job_id=str(job),state=row[1].split()[0],scheduler_seconds=int(row[2]))


def previous_usage(value, current=None):
    rows=[]
    for attempt in value['attempts']:
        if attempt['job_id']==current:continue
        row=scheduler(attempt['job_id'])
        if row['state'] not in TERMINAL:raise ValueError('Previous attempt still active; concurrent writers forbidden')
        rows.append(row)
    return sum(x['scheduler_seconds'] for x in rows),rows


def submit(root):
    plan=verify(root);value=approval(root,plan)
    used,rows=previous_usage(value)
    minutes=(plan['resources']['total_seconds']-used)//60
    if minutes<20:raise ValueError('Insufficient remaining approved scheduler budget')
    job=subprocess.check_output(['sbatch','--hold','--parsable',f'--time={minutes}',str(root/'launch.sbatch')],text=True).strip().split(';')[0]
    if not job.isdecimal():raise ValueError('Invalid sbatch receipt')
    value['attempts'].append(dict(job_id=job,time_limit_seconds=minutes*60,submitted_unix=time.time()))
    io.write_json(root/'approval.json',value)
    io.write_json(root/'scheduler-ledger.json',dict(attempts=rows,charged_seconds=used,total_seconds=plan['resources']['total_seconds']))
    io.write_json(POINTER,dict(job_id=job,status='submitted',run_directory=str(root),verified_complete=False,
        consumed_scheduler_seconds=used,remaining_approved_seconds=plan['resources']['total_seconds']-used,
        continuation_active=False,supervision_required=True,updated_unix=time.time()))
    subprocess.run(['scontrol','release',job],check=True)
    return read(POINTER)


def pending_tasks(root, cp):
    pending=[]
    diagnoses=None
    for task in cp['tasks']:
        directory=root/'actor/tasks'/io.digest(task['task_id'])
        if (directory/'result.json').exists():
            result=read(directory/'result.json')
            if result.get('task_id')!=task['task_id']:
                raise ValueError('Invalid result requires diagnosis and preservation before retry')
            if result.get('valid'):
                if result.get('provider_blocked'):
                    raise ValueError('Invalid result requires diagnosis and preservation before retry')
                # A scored negative is completed work, never a retry candidate.
                continue
            # This is acceptance of an explicit owner's diagnosis, not an
            # automatic diagnosis or a substitute for the final evidence audit.
            if diagnoses is None:
                path=root/'diagnosed-invalids.json'
                diagnoses=read(path) if path.exists() else {}
            diagnosis=diagnoses.get(task['task_id']) if isinstance(diagnoses,dict) else None
            required_fields={'task_id','reason','result_sha256','expected_actor_error',
                             'expected_judge_error','evidence_sha256'}
            if (not isinstance(diagnosis,dict) or not required_fields<=set(diagnosis)
                    or diagnosis['task_id']!=task['task_id'] or result.get('valid') is not False
                    or result.get('completed') is not True or 'score' not in result or result['score'] is not None
                    or not isinstance(diagnosis['reason'],str) or not diagnosis['reason'].strip()
                    or diagnosis['result_sha256']!=kev.file_hash(directory/'result.json')
                    or diagnosis['expected_actor_error']!=result.get('actor_error')
                    or diagnosis['expected_judge_error']!=result.get('judge_error')
                    or not (result.get('actor_error') or result.get('judge_error') or result.get('cleanup_errors'))):
                raise ValueError('Invalid result requires exact hash-bound owner diagnosis before it can be skipped')
            evidence=diagnosis['evidence_sha256']
            required={str((directory/name).relative_to(root)) for name in ('result.json','trajectory.json','worker.log')}
            if not isinstance(evidence,dict) or not required<=set(evidence):
                raise ValueError('Invalid diagnosis lacks immutable result/trajectory/log evidence')
            for name,expected in evidence.items():
                if not isinstance(name,str):
                    raise ValueError('Invalid diagnosis evidence path')
                path=(root/name).resolve()
                if (not path.is_relative_to(directory.resolve()) or not path.is_file()
                        or kev.file_hash(path)!=expected):
                    raise ValueError('Invalid diagnosis evidence changed or escapes its task directory')
            marker=directory/'browser-session.json'
            if not marker.exists() or read(marker).get('stopped') is not True:
                raise ValueError('Invalid diagnosis requires a stopped owned browser')
            if read(directory/'task.json')!=task or read(directory/'trajectory.json').get('task')!=task:
                raise ValueError('Invalid diagnosis task/trajectory identity mismatch')
        elif (directory/'task.json').exists():
            raise ValueError('Interrupted attempt requires diagnosis and preservation before retry')
        else:pending.append(task)
    return pending


def run(root):
    import httpx
    plan=verify(root);value=approval(root,plan);job=os.environ.get('SLURM_JOB_ID')
    if not job or job not in {a['job_id'] for a in value['attempts']}:
        raise ValueError('Execution requires this approved scheduler allocation')
    used,rows=previous_usage(value,current=job);current=scheduler(job)
    remaining=plan['resources']['total_seconds']-used-current['scheduler_seconds']
    cp=read(root/'actor/plan.json');cfg=cp['config'];pending=pending_tasks(root,cp)
    cohort.check_worker_source(cfg);io.load_credentials(ENV_FILE,cfg)
    io.write_json(root/'scheduler-ledger.json',dict(attempts=rows+[current],charged_seconds=used+current['scheduler_seconds'],total_seconds=plan['resources']['total_seconds']))
    attempt_root=root/'attempts'/job;attempt_root.mkdir(parents=True,exist_ok=True)
    started=time.monotonic();stop=False;server=tracker=gpu=None;active={};handles=[]
    def signal_stop(*_):
        nonlocal stop
        stop=True
    old={s:signal.signal(s,signal_stop) for s in (signal.SIGTERM,signal.SIGINT)}
    def heartbeat(stage):
        left=remaining-(time.monotonic()-started)
        io.write_json(root/'heartbeat.json',dict(job_id=job,stage=stage,updated_unix=time.time(),
            previous_scheduler_seconds=used,remaining_seconds=max(0,left),active_workers=len(active),pending_tasks=len(pending)))
        if stop or left<=120:raise TimeoutError('Shutdown or approved total scheduler budget reached')
        return left
    def launch(command, log, **kwargs):
        handle=log.open('a');handles.append(handle)
        return subprocess.Popen(command,stdout=handle,stderr=handle,**kwargs)
    try:
        gpu=launch(['nvidia-smi','--query-gpu=timestamp,index,name,utilization.gpu,memory.used,power.draw',
            '--format=csv','-l','15'],attempt_root/'gpu.csv')
        # Tracker uses the existing W&B environment; actor/server dependencies stay isolated.
        tracker=launch([str(RUNTIME/'venv/bin/python'),str(REPO/'scripts/track_kev_actor_full300.py'),
            '--root',str(root),'--env-file',str(ENV_FILE)],attempt_root/'wandb.log',cwd=REPO)
        endpoint=cfg['decision_endpoint'];pair.check_port_available(endpoint)
        server=launch([str(RUNTIME/'kev-eval-venv/bin/python'),'-m','kev.serve','--run',cfg['kev']['server_run'],
            '--host','127.0.0.1','--port',str(urlsplit(endpoint).port)],attempt_root/'server.log',
            cwd=RUNTIME/'kev-upstream',env=pair.server_environment())
        deadline=time.monotonic()+600
        while True:
            heartbeat('loading_kev27b')
            if server.poll() is not None:raise RuntimeError('Kev server exited before startup')
            try:card=kev.check_server(endpoint,cfg['kev']);break
            except httpx.TransportError:
                if time.monotonic()>deadline:raise TimeoutError('Kev startup exceeded600 seconds')
                time.sleep(5)
        io.write_json(attempt_root/'server-identity.json',card)
        for request in read(root/'warmup-requests.json'):
            heartbeat('warmup_kev27b')
            io.reserve_budget(cfg,'local_kev_requests',warmup=True,job_id=job)
            io.append_json(attempt_root/'warmup-attempts.jsonl',dict(request=request,started_unix=time.time()))
            response=httpx.post(endpoint,json=request,timeout=240,trust_env=False)
            io.append_json(attempt_root/'warmup-responses.jsonl',dict(status=response.status_code,response=response.json()))
            response.raise_for_status();kev.validate_answers(request,response.json())
        io.write_json(attempt_root/'server-warm.json',kev.check_server(endpoint,cfg['kev']))
        if tracker.poll() is not None or not (root/'wandb-ready.json').exists() or read(root/'wandb-ready.json')['job_id']!=job:
            raise RuntimeError('Current W&B tracker has not confirmed online readiness')
        drain=False
        while pending or active:
            left=heartbeat('browsers_kev27b')
            if server.poll() is not None:raise RuntimeError('Kev server stopped during collection')
            if tracker.poll() is not None:raise RuntimeError('W&B tracker stopped; inspect preserved log')
            drain=drain or left<1260
            while pending and len(active)<2 and not drain:
                task=pending.pop(0);task_hash=io.digest(task['task_id']);directory=root/'actor/tasks'/task_hash
                directory.mkdir(parents=True,exist_ok=True)
                process=launch([sys.executable,str(REPO/'scripts/evaluate_jev_ultrafast.py'),'--output',str(root/'actor'),
                    '--env-file',str(ENV_FILE),'--worker',task_hash],directory/'worker.log',cwd=REPO,start_new_session=True)
                active[process.pid]=(process,directory,time.monotonic())
            for pid,(process,directory,began) in list(active.items()):
                if time.monotonic()-began>1200:cohort.stop_owned_worker(process)
                if process.poll() is not None:
                    active.pop(pid)
                    cleanup=io.stop_recorded_session(directory)
                    result=read(directory/'result.json') if (directory/'result.json').exists() else {}
                    if process.returncode or cleanup or not result.get('valid') or result.get('provider_blocked'):
                        raise RuntimeError('Task needs diagnosis: '+directory.name)
            summary=io.summarize(cp['tasks'],root/'actor','kev')
            # Independent agent audit, scheduler shutdown and W&B are still required.
            summary.update(verified_complete=False,collection_complete=not pending and not active,
                active_workers=len(active),pending_tasks=len(pending),state='running' if active else 'draining' if drain else 'running')
            io.write_json(root/'actor/summary.json',summary)
            if not active and (not pending or drain):break
            time.sleep(5)
        io.write_json(attempt_root/'server-final.json',kev.check_server(endpoint,cfg['kev']))
        io.write_json(root/'collection-status.json',dict(complete=not pending and not active,verified_complete=False,
            job_id=job,updated_unix=time.time(),reason='Await independent audit' if not pending else 'Compute reserve reached'))
        return 0 if not pending else 2
    except Exception as error:
        io.write_json(attempt_root/'failure.json',dict(error_type=type(error).__name__,error=str(error)[:500],unix=time.time()))
        io.write_json(root/'failure.json',dict(job_id=job,attempt=str(attempt_root),error_type=type(error).__name__,error=str(error)[:500],unix=time.time()))
        raise
    finally:
        try:
            for process,directory,_ in active.values():
                cohort.stop_owned_worker(process);io.stop_recorded_session(directory)
            summary=io.summarize(cp['tasks'],root/'actor','kev')
            summary.update(verified_complete=False,collection_complete=summary['completed']==len(cp['tasks']),active_workers=0,state='awaiting_audit')
            io.write_json(root/'actor/summary.json',summary)
        finally:
            pair.stop_process(server,20);pair.stop_process(gpu,10);pair.stop_process(tracker,90)
            for handle in handles:handle.close()
            for sig,handler in old.items():signal.signal(sig,handler)
            io.write_json(attempt_root/'controller-finished.json',dict(unix=time.time(),controller_seconds=time.monotonic()-started))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT)
    action=p.add_mutually_exclusive_group();action.add_argument('--submit',action='store_true')
    action.add_argument('--execute',action='store_true');action.add_argument('--verify',action='store_true')
    args=p.parse_args()
    if args.execute or args.submit:
        with (args.root/('owner.lock' if args.execute else 'submit.lock')).open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            if args.execute:raise SystemExit(run(args.root))
            print(json.dumps(submit(args.root),indent=2))
    else:
        plan=verify(args.root) if args.verify else prepare(args.root)
        print(json.dumps({k:plan[k] for k in ('experiment','status','resources','limits','pilot')},indent=2))
