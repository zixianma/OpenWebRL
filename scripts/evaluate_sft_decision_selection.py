#!/usr/bin/env python3
"""Prepare/own the four-cohort SFT proposal-selection pilot. Never submits jobs."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from openwebrl.decision_selection import selection_payload, selected_index, write_json
from openwebrl.decision_selection_eval import PROTOCOL
from openwebrl.kev_eval import model_spec, file_hash, check_server
from evaluate_kev_pair import server_environment, source_identity, check_port_available
from prepare_arm_turn_bonus import copy_plain
from resume_baseline import validate_source, source_command, allocation, clean_environment
from runtime_ports import lease_ports

RUNTIME = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime')
CONTROL = RUNTIME / 'evaluations/sft-decision-selection-pilot-20261004'
PARENT = RUNTIME / 'reference-arm-task-rescue-20261002-v1'
SOURCE = RUNTIME / 'reference-sft-decision-selection-20261004-v1'
ACTOR = Path('/gpfs/scrubbed/zixianma/checkpoints/web/OpenWebRL-4B-SFT')
MODES = ('sft', 'jev', 'kev-0.8b', 'kev-27b')
RESOURCES = dict(gpus=2, gpu_type='H200', cpus=16, memory_gib=240, total_seconds=3600)
LIMITS = dict(browser_sessions=40, concurrent_browsers=2, browser_expiry_minutes=12,
    jev_requests=300, local_kev_requests=602, judge_http_attempts=160,
    judge_completion_tokens=4096, actor_proposals=4800, text_helper_calls=0)
CHANGED = ['openwebrl/decision_selection.py', 'openwebrl/decision_selection_eval.py',
    'openwebrl/kev_eval.py', 'openwebrl/arm_inference.py', 'openwebrl/arm_eval.py',
    'openwebrl/run_evaluate.py', 'openwebrl/eval/reward_online_mind2web.py',
    'openwebrl/env/web_env.py', 'openwebrl/env/browser_use_env.py', 'openwebrl/env/browser_runtime.py']


def read(path):
    return json.loads(Path(path).read_text())


def environment():
    from dotenv import dotenv_values
    env = {k:v for k,v in clean_environment().items() if not k.startswith('OPENWEBRL_ARM_')}
    env.setdefault('FLASHINFER_WORKSPACE_BASE', str(RUNTIME))
    for key, value in dotenv_values(REPO / '.env').items():
        if value and key.startswith(('OPENAI_', 'JUDGE_', 'WANDB_')):
            env.setdefault(key, value)
    return env


def stop(child):
    if child is None or child.poll() is not None:
        return
    try:
        os.killpg(child.pid, signal.SIGTERM)
        child.wait(timeout=30)
    except ProcessLookupError:
        pass
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGKILL)
        child.wait()


def warmup_request():
    return selection_payload('kev-latest', 'Open the search page',
        dict(active_tab_url='https://example.com', selection_page=dict(url='https://example.com',
            text='Search', interactive_elements=[])), [],
        [dict(thought=f'Alternative {i}', action='{"name":"wait","arguments":{}}') for i in range(5)])


def prepare():
    import yaml
    validate_source(PARENT); CONTROL.mkdir(parents=True, exist_ok=True)
    if not SOURCE.exists():
        shutil.copytree(PARENT, SOURCE, symlinks=True, copy_function=copy_plain,
            ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.git', '.env*', '.browser_use_sessions'))
        for name in CHANGED:
            copy_plain(REPO / name, SOURCE / name)
        config = yaml.safe_load((SOURCE / 'openwebrl/env/config.yaml').read_text())
        config.update(mode='browser-use', width=1280, height=1000, dpr=1)
        config['browser_use'] = dict(api_key='', timeout=12, proxy_country_code=None, profile_id=None)
        (SOURCE / 'openwebrl/env/config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
        manifest = read(SOURCE / 'reference_manifest.json')
        changed = [*CHANGED, 'openwebrl/env/config.yaml']
        manifest['recipe_files_sha256'].update({n: file_hash(SOURCE / n) for n in changed})
        manifest['decision_selection'] = dict(parent=str(PARENT), changed=changed, protocol=PROTOCOL)
        write_json(SOURCE / 'reference_manifest.json', manifest)
    validate_source(SOURCE)
    if any(file_hash(REPO / n) != file_hash(SOURCE / n) for n in CHANGED):
        raise ValueError('Prepared source differs: archive and prepare a new source revision')
    rows = [json.loads(line) for line in (REPO / 'openwebrl/data/eval/online-mind2web.jsonl').read_text().splitlines()][:10]
    ids = [r['metadata']['task_id'] for r in rows]
    direct = read(RUNTIME / 'evaluations/jev-ultrafast-om2w-pilot-20261004/plan.json')
    if ids != [r['task_id'] for r in direct['tasks']]:
        raise ValueError('Must reuse the direct pilots\' task identities')
    task_text = ''.join(json.dumps(r) + '\n' for r in rows)
    tasks = CONTROL / 'tasks.jsonl'
    if tasks.exists() and tasks.read_text() != task_text:
        raise ValueError('Prepared tasks changed')
    tasks.write_text(task_text)
    weights = {str(p.relative_to(ACTOR)): dict(size=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns)
               for p in ACTOR.glob('*.safetensors')}
    if not weights:
        raise ValueError('Missing original SFT weights')
    model_audit = read(RUNTIME / 'kev-preparation-20261004/verified-models.json')
    for variant in ('0.8b', '27b'):
        if not model_audit[variant]['verified'] or model_audit[variant]['spec'] != model_spec(variant):
            raise ValueError('Kev audit changed')
        for path, record in model_audit[variant]['files'].items():
            stat = Path(path).stat()
            if (stat.st_size, stat.st_mtime_ns) != (record['size'], record['mtime_ns']):
                raise ValueError('Kev weights changed since independent hash audit')
    plan = dict(status='prepared_awaiting_exact_resource_approval', protocol=PROTOCOL,
        modes=MODES, resources=RESOURCES, limits=LIMITS, actor=str(ACTOR), actor_weights=weights,
        actor_config_sha256=file_hash(ACTOR / 'config.json'), source=str(SOURCE),
        source_manifest_sha256=file_hash(SOURCE / 'reference_manifest.json'),
        tasks=str(tasks), task_ids=ids, tasks_sha256=file_hash(tasks),
        kev_source=source_identity(), kev_specs={v: model_spec(v) for v in ('0.8b', '27b')},
        code_sha256={p: file_hash(REPO / p) for p in ('scripts/evaluate_sft_decision_selection.py',
            'scripts/evaluate_sft_decision_selection_2gpu.sbatch')}, wandb_project='openwebrl-evals',
        budget_policy='One hour total including startup/recovery; no automatic extension or extra browser attempts',
        comparison='Ten fresh episodes per condition; same seed schedule, live states diverge after different actions')
    # JSON-normalize tuples before comparing a saved plan.
    plan = json.loads(json.dumps(plan))
    path = CONTROL / 'plan.json'
    if path.exists() and read(path) != plan:
        raise ValueError('Prepared plan changed')
    write_json(path, plan)
    return plan


def run(plan):
    import httpx
    from dotenv import dotenv_values
    job = os.environ.get('SLURM_JOB_ID')
    approval = read(CONTROL / 'approval.json')
    if (not job or not approval.get('approved') or approval['resources'] != RESOURCES or
            approval['limits'] != LIMITS or approval['plan_sha256'] != file_hash(CONTROL / 'plan.json')):
        raise ValueError('Exact new allocation and API/browser approval required')
    resources = allocation(subprocess.check_output(['scontrol', 'show', 'job', job, '-o'], text=True),
                           job, requested_gpus=2, maximum_hours=1)
    if resources['cpus'] != 16 or resources['allocated_memory_gib'] != 240:
        raise ValueError('Allocation differs from the approved CPU/memory request')
    attempts = approval['attempts']
    current = [a for a in attempts if a['job_id'] == job]
    if len(current) != 1:
        raise ValueError('Allocation absent from approval ledger')
    used = 0
    for previous in (a for a in attempts if a['job_id'] != job):
        rows = subprocess.check_output(['sacct', '-X', '-n', '-P', '-j', previous['job_id'],
            '--format=JobIDRaw,State,ElapsedRaw'], text=True).strip().splitlines()
        row = next(r.split('|') for r in rows if r.split('|')[0] == previous['job_id'])
        if row[1].split()[0] not in ('COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT'):
            raise ValueError('A previous attempt is still active')
        used += int(row[2])
    if used + current[0]['time_limit_seconds'] > RESOURCES['total_seconds']:
        raise ValueError('Recovery exceeds the original allocation budget')
    deadline = time.monotonic() + min(resources['maximum_seconds'], current[0]['time_limit_seconds'] - 90)
    env = environment()
    for key, value in dotenv_values(REPO / '.env').items():
        if value and key in ('OPENAI_API_KEY','JUDGE_API_KEY','BROWSER_USE_API_KEY','TYPESAFE_API_KEY','JEV_API_KEY'):
            env[key] = value
    env.update(PYTHONPATH=str(SOURCE), WANDB_PROJECT='openwebrl-evals', OMP_NUM_THREADS='1',
        OPENBLAS_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1',
        CPATH=str(RUNTIME / 'src/python-headers/Include') + ':' + str(RUNTIME / 'src/python-headers'))
    devices = env.get('CUDA_VISIBLE_DEVICES', '').split(',')
    if len(devices) != 2 or not all(devices):
        raise ValueError('Exactly two allocated GPUs required')
    lease, port = lease_ports(job); children = []; handles = []
    actor = None

    def spawn(command, label, child_env, cwd=SOURCE, activate=False):
        log = (CONTROL / (label + '.log')).open('a'); handles.append(log)
        if activate: command = source_command(SOURCE, command)
        process = subprocess.Popen(command, cwd=cwd, env=child_env, stdout=log,
            stderr=subprocess.STDOUT, start_new_session=True)
        children.append(process); return process

    def heartbeat(stage):
        write_json(CONTROL / 'heartbeat.json', dict(job_id=job, stage=stage, updated_unix=time.time(),
            previous_scheduler_seconds=used, remaining_seconds=deadline-time.monotonic()))
        if time.monotonic() >= deadline:
            raise TimeoutError('Approved allocation is ending; preserve partial results')
        if actor is not None and actor.poll() is not None:
            raise RuntimeError('SFT actor server exited')

    def terminated(*unused):
        raise TimeoutError('Allocation termination signal')
    old_signals = {s: signal.signal(s, terminated) for s in (signal.SIGTERM, signal.SIGINT)}
    try:
        actor = spawn([str(RUNTIME / 'venv/bin/python'), '-m', 'sglang.launch_server',
            '--model-path', str(ACTOR), '--host', '127.0.0.1', '--port', str(port), '--dtype', 'bfloat16',
            '--tp', '1', '--mem-fraction-static', '.45', '--context-length', '32768',
            '--max-running-requests', '10', '--chunked-prefill-size', '4096', '--disable-cuda-graph'],
            'actor', dict(env, CUDA_VISIBLE_DEVICES=devices[0]), activate=True)
        loading_deadline = min(deadline, time.monotonic()+600)
        while True:
            heartbeat('actor_loading')
            try:
                response = httpx.get(f'http://127.0.0.1:{port}/get_model_info', timeout=5, trust_env=False)
                response.raise_for_status(); break
            except (httpx.TransportError, httpx.HTTPStatusError):
                if time.monotonic() >= loading_deadline: raise TimeoutError('Actor startup timeout')
                time.sleep(5)
        spawn(['nvidia-smi', '--query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw',
               '--format=csv', '-l', '15'], 'devices', env)
        for mode in MODES:
            output = CONTROL / mode; output.mkdir(exist_ok=True)
            if (output / 'summary.json').exists() and read(output / 'summary.json')['complete']:
                continue
            server = None
            endpoint = f'http://127.0.0.1:{port+1}/v1/systemone'
            if mode.startswith('kev-'):
                spec = model_spec(mode.removeprefix('kev-'))
                check_port_available(endpoint)
                kev_env = server_environment()
                kev_env.update(CUDA_VISIBLE_DEVICES=devices[1], CPATH=env['CPATH'])
                server = spawn([str(RUNTIME / 'kev-eval-venv/bin/python'), '-m', 'kev.serve',
                    '--run', spec['server_run'], '--host', '127.0.0.1', '--port', str(port+1)],
                    mode+'-server', kev_env, RUNTIME / 'kev-upstream')
                loading_deadline = min(deadline, time.monotonic()+600)
                while True:
                    heartbeat(mode+'_loading')
                    if server.poll() is not None: raise RuntimeError('Kev server exited')
                    try:
                        card = check_server(endpoint, spec); break
                    except httpx.TransportError:
                        if time.monotonic() >= loading_deadline: raise TimeoutError('Kev startup timeout')
                        time.sleep(5)
                write_json(output / 'server-identity.json', card)
                warmup = output / 'warmup-request.json'
                if not warmup.exists():
                    request = warmup_request(); write_json(warmup, request)
                    response = httpx.post(endpoint, json=request, timeout=180, trust_env=False)
                    response.raise_for_status(); selected_index(request, response.json())
                    write_json(output / 'warmup-response.json', response.json())
                elif not (output / 'warmup-response.json').exists():
                    raise ValueError('Interrupted warmup needs diagnosis; do not reset request budget')
            cfg = dict(protocol=PROTOCOL, output=str(output), tasks=plan['tasks'], task_ids=plan['task_ids'],
                actor=str(ACTOR), actor_port=port, mode=mode, endpoint=endpoint)
            write_json(output / 'config.json', cfg)
            worker_env = dict(env, CUDA_VISIBLE_DEVICES=devices[0],
                OPENWEBRL_BROWSER_USE_SESSION_DIR=str(output / 'browser_sessions'),
                OPENWEBRL_MULTIMODAL_STORAGE_DIR=str(output / 'multimodal'))
            worker = spawn([str(RUNTIME / 'venv/bin/python'), '-m', 'openwebrl.decision_selection_eval',
                '--config', str(output / 'config.json')], mode+'-worker', worker_env, activate=True)
            while worker.poll() is None:
                heartbeat(mode)
                if server is not None and server.poll() is not None: raise RuntimeError('Kev serving failed')
                time.sleep(5)
            if worker.returncode: raise RuntimeError('Cohort failed; inspect preserved task artifacts')
            stop(server)
        write_json(CONTROL / 'summary.json', dict(complete=True,
            modes={m: read(CONTROL / m / 'summary.json') for m in MODES}))
    finally:
        for s, handler in old_signals.items(): signal.signal(s, handler)
        for child in reversed(children): stop(child)
        for handle in handles: handle.close()
        lease.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    plan = prepare()
    if args.execute:
        with (CONTROL / 'owner.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            run(plan)
    else:
        print(json.dumps({k: plan[k] for k in ('status','resources','limits','modes','protocol')}, indent=2))
