#!/usr/bin/env python3
"""Prepare and own an explicitly approved Qwen/Luna pilot; never submits jobs."""
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
from openwebrl.luna_qwen_metrics import PRICING, digest, schedule, write_json
from openwebrl.luna_qwen_eval import PROTOCOL
from prepare_arm_turn_bonus import copy_plain
from resume_baseline import validate_source, source_command, allocation
from run_arm_quality_audit import environment, stop
from runtime_ports import lease_ports
from browser_prompt_assets import freeze_browser_prompt_assets, validate_browser_prompt_assets
from openwebrl.browser_actor_protocol import validate_browser_harness_protocol

RUNTIME = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime')
CONTROL = RUNTIME / 'luna-qwen-inference-20261004'
SOURCE = RUNTIME / 'reference-luna-qwen-inference-20261004-v1'
PARENT = RUNTIME / 'reference-arm-controlled-inference-20261004-v1'
ACTOR = Path('/gpfs/scrubbed/zixianma/checkpoints/web/Qwen3-VL-4B-Thinking')
RESOURCES = dict(gpus=4, gpu_type='H200', cpus=32, memory_gib=480, total_seconds=5400)
LIMITS = dict(luna_usd=15., luna_calls=902, judge_usd=5., judge_calls=160,
              tasks=10, episodes=40, browser_concurrency=4)
FILES = ['openwebrl/luna_qwen_metrics.py', 'openwebrl/luna_qwen_policy.py',
         'openwebrl/responses_actor_history.py', 'openwebrl/browser_actor_protocol.py',
         'openwebrl/luna_qwen_eval.py', 'openwebrl/arm_inference_flops.py', 'scripts/report_luna_qwen_inference.py',
         'scripts/browser_prompt_assets.py']
PLOT_DEPS = CONTROL / 'plot-deps'


def terminate(signum, frame):
    raise SystemExit(128 + signum)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare():
    import yaml
    validate_source(PARENT)
    ready = read(CONTROL / 'model-ready.json')
    if ready['revision'] != PROTOCOL['actor_revision'] or not ready['weights_sha256_verified']:
        raise ValueError('Qwen model has not passed its independent hash audit')
    if not read(CONTROL / 'luna-access.json')['read_only_model_lookup']:
        raise ValueError('Luna is absent from the account model listing')
    if not SOURCE.exists():
        shutil.copytree(PARENT, SOURCE, symlinks=True, copy_function=copy_plain,
            ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.git', '.env*', '.browser_use_sessions'))
    if (CONTROL / 'approval.json').exists():
        raise ValueError('Do not rewrite a plan after approval; preserve its source/lineage')
    if not PLOT_DEPS.exists():
        shutil.copytree('/tmp/arm-critic-plot-20261004', PLOT_DEPS, copy_function=copy_plain,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for name in FILES:
        copy_plain(REPO / name, SOURCE / name)
    name = 'openwebrl/generate_browser.py'
    text = (PARENT / name).read_text()
    marker = '                selected_output, arm_metadata = await selector(\n'
    if text.count(marker) != 1 or text.count('await env.reset()') != 2 or text.count('await _controlled_timed_step(args, env, actions)') != 2:
        raise ValueError('Preserved generator changed')
    text = text.replace(marker, '                selector.set_context(mm_messages_filtered, tools_info, tokenizer, len(input_tokens))\n' + marker)
    text = text.replace('await env.reset()', 'await _luna_measured_reset(args, env)')
    text = text.replace('await _controlled_timed_step(args, env, actions)', 'await _luna_measured_step(args, env, actions)')
    text += '''

async def _luna_measured_reset(args, env):
    result = await env.reset()
    args.luna_episode_observation['screenshot'] = result[0].get('screenshot')
    return result

async def _luna_measured_step(args, env, actions):
    import time
    started = time.monotonic()
    observation = args.luna_episode_observation
    observation['browser_steps'] += 1
    try:
        result = await env.step(actions)
        if result[0].get('screenshot'):
            observation['screenshot'] = result[0]['screenshot']
        return result
    finally:
        observation['browser_step_seconds'] += time.monotonic() - started
'''
    (SOURCE / name).write_text(text)
    env_file = 'openwebrl/env/config.yaml'
    env_config = yaml.safe_load((PARENT / env_file).read_text())
    env_config.update(mode='local_process', width=1280, height=720, dpr=1)
    (SOURCE / env_file).write_text(yaml.safe_dump(env_config, sort_keys=False))
    manifest = read(SOURCE / 'reference_manifest.json')
    manifest['recipe_files_sha256'].update({f: sha(SOURCE / f) for f in [*FILES, name, env_file]})
    prompt_hashes = freeze_browser_prompt_assets(SOURCE, REPO)
    validate_browser_harness_protocol(PROTOCOL, SOURCE)
    manifest['recipe_files_sha256'].update(prompt_hashes)
    manifest['luna_qwen_inference'] = dict(protocol=PROTOCOL, parent=str(PARENT), changes=[*FILES, name, env_file])
    manifest['luna_qwen_inference']['required_browser_prompt_sha256'] = prompt_hashes
    manifest['luna_qwen_inference']['api_actor_history'] = 'native_responses_v1'
    write_json(SOURCE / 'reference_manifest.json', manifest)
    validate_source(SOURCE)
    validate_browser_prompt_assets(SOURCE)
    task_file = REPO / 'openwebrl/data/eval/online-mind2web.jsonl'
    rows = [json.loads(line) for line in task_file.read_text().splitlines()]
    task_id = lambda row: row.get('task_id', row['metadata']['task_id'])
    full_schedule = schedule([task_id(r) for r in rows])
    assert len(full_schedule) == 300
    pilot = full_schedule[:10]
    ids = {row['task_id'] for row in pilot}
    private_tasks = CONTROL / 'pilot-tasks.jsonl'
    private_tasks.write_text(''.join(json.dumps(r) + '\n' for r in rows if task_id(r) in ids))
    write_json(CONTROL / 'full300-schedule.json', full_schedule)
    model_files = {p.name: dict(size=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns) for p in ACTOR.glob('*.safetensors')}
    plan = dict(status='prepared_not_submitted', protocol=PROTOCOL, resources=RESOURCES, limits=LIMITS,
        schedule=pilot, task_file=str(private_tasks), task_file_sha256=sha(private_tasks),
        actor_path=str(ACTOR), actor_files=model_files, actor_revision=ready['revision'],
        actor_config_sha256=sha(ACTOR / 'config.json'), pricing=PRICING,
        pricing_sources=['https://developers.openai.com/api/docs/models/gpt-6-luna',
                        'openwebrl/docs/RL_RUNTIME.md#h200-testing--approved-continuation-287371-2026-09-10-1957-pdt'],
        selector_prompt_source=str(RUNTIME / 'arm-reproduction/source'),
        source=str(SOURCE), source_manifest_sha256=sha(SOURCE / 'reference_manifest.json'),
        controller_sha256=sha(__file__), batch_sha256=sha(REPO / 'scripts/run_luna_qwen_inference_4gpu.sbatch'),
        final_target=dict(tasks=300, episodes=1200, allocation_not_requested=True),
        supervision='Active agent must own queue/startup/progress/recovery/completion; no unattended repair service is claimed',
        cost_definition='Exclusive one-episode local GPU reservation plus billed-usage API estimate; browser CPU not separately priced; terminal judge and research overhead separate',
        metrics=['cost', 'latency', 'input_output_tokens', 'browser_steps', 'local_compute'],
        plots=['success_vs_cost', 'success_vs_latency', 'success_vs_input_and_output_tokens'])
    write_json(CONTROL / 'plan.json', plan)
    env = environment()
    env.update(PYTHONPATH=str(SOURCE) + ':' + str(PLOT_DEPS), PYTHONPYCACHEPREFIX='/tmp/luna-qwen-pycache', OPENBLAS_NUM_THREADS='1',
               MPLCONFIGDIR='/tmp/luna-qwen-mpl', OMP_NUM_THREADS='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', CUDA_VISIBLE_DEVICES='')
    command = [str(RUNTIME / 'venv/bin/python'), '-c',
        'import matplotlib; from transformers import AutoProcessor; from openwebrl.luna_qwen_eval import PROTOCOL; '
        'from openwebrl import run_evaluate; p=AutoProcessor.from_pretrained(' + repr(str(ACTOR)) + ',local_files_only=True); '
        'assert p.chat_template; print("CPU imports, plot dependencies and Qwen processor passed")']
    subprocess.run(command, cwd=SOURCE, env=env, check=True)
    write_json(CONTROL / 'readiness.json', dict(cpu_import_passed=True, gpu_api_browser_preflight_pending=True,
        plan_sha256=digest(plan), source_manifest_sha256=plan['source_manifest_sha256']))
    return plan


def check_approval(approval, plan, job, used):
    if (not approval.get('approved') or approval['resources'] != RESOURCES or approval['limits'] != LIMITS
            or approval['plan_sha256'] != digest(plan)):
        raise ValueError('Exact allocation and API budget approval required')
    current = [a for a in approval['attempts'] if a['job_id'] == job]
    if len(current) != 1 or used + current[0]['time_limit_seconds'] > RESOURCES['total_seconds']:
        raise ValueError('Retry would exceed the original 90-minute total')
    return current[0]


def worker_command(job, config):
    # Override SLURM_GPUS inherited from the four-GPU batch allocation as well
    # as the per-node GRES. Otherwise Slurm can serialize nominally TP1 steps.
    return ['srun', '--jobid=' + job, '--nodes=1', '--ntasks=1', '--cpus-per-task=8',
            '--gpus=h200:1', '--gres=gpu:h200:1', '--mem=120G', '--exact', '--cpu-bind=cores',
            sys.executable, str(Path(__file__).resolve()), '--worker', str(config)]


def worker(config):
    if os.getenv('SLURM_JOB_ID') != config['job_id'] or f'/job_{config["job_id"]}/' not in Path('/proc/self/cgroup').read_text():
        raise ValueError('Worker must run inside its approved allocation')
    signal.signal(signal.SIGTERM, terminate)
    env = environment()
    env.update(PYTHONPATH=str(SOURCE), WANDB_PROJECT='openwebrl-evals', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    root = CONTROL / 'workers' / config['job_id'] / str(config['worker_id'])
    root.mkdir(parents=True, exist_ok=True)
    devices = env.get('CUDA_VISIBLE_DEVICES', '').split(',')
    if len(devices) != 1 or not devices[0]:
        raise ValueError('Expected one allocated H200')
    # NVML enumerates only this step's cgroup-visible GPU, with index zero.
    # SLURM_STEP_GPUS retains the host index and cannot index that namespace.
    gpu_uuid = subprocess.check_output(['nvidia-smi',
        '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip()
    if not gpu_uuid.startswith('GPU-') or '\n' in gpu_uuid:
        raise ValueError('Expected exactly one identifiable physical GPU')
    write_json(root / 'device-assignment.json', dict(worker_id=config['worker_id'],
        cuda_visible_devices=devices, slurm_step_gpus=env.get('SLURM_STEP_GPUS'), gpu_uuid=gpu_uuid))
    lease, port = lease_ports(config['job_id'])
    children, handles = [], []
    def spawn(command, label, activate=False):
        handle = (root / (label + '.log')).open('a')
        handles.append(handle)
        if activate:
            command = source_command(SOURCE, command)
        process = subprocess.Popen(command, cwd=SOURCE, env=env, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
        children.append(process)
        return process
    try:
        actor = spawn([str(RUNTIME / 'venv/bin/python'), '-m', 'sglang.launch_server', '--model-path', str(ACTOR),
            '--host', '127.0.0.1', '--port', str(port), '--dtype', 'bfloat16', '--tp', '1', '--mem-fraction-static', '.60',
            '--context-length', '32768', '--max-running-requests', '10', '--chunked-prefill-size', '4096', '--disable-cuda-graph'],
            'actor', True)
        import urllib.request
        deadline = time.monotonic() + 900
        while True:
            if actor.poll() is not None:
                raise RuntimeError('Actor startup failed')
            try:
                with urllib.request.urlopen(f'http://127.0.0.1:{port}/get_model_info', timeout=5) as response:
                    json.load(response)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise TimeoutError('Actor startup exceeded 15 minutes')
                time.sleep(3)
        spawn(['nvidia-smi', '-i', gpu_uuid, '--query-gpu=timestamp,index,uuid,utilization.gpu,memory.used,power.draw',
               '--format=csv', '-l', '5'], 'devices')
        cfg = dict(output=str(CONTROL / 'pilot'), plan=str(CONTROL / 'plan.json'),
                   worker_id=config['worker_id'], actor_port=port, browser_port=port + 10)
        write_json(root / 'worker-config.json', cfg)
        collection = spawn([str(RUNTIME / 'venv/bin/python'), '-m', 'openwebrl.luna_qwen_eval',
                            '--config', str(root / 'worker-config.json')], 'collection', True)
        while collection.poll() is None:
            if actor.poll() is not None:
                raise RuntimeError('Actor exited during collection')
            time.sleep(5)
        if collection.returncode:
            raise RuntimeError('Collection failed; retain every artifact')
    finally:
        for process in reversed(children):
            stop(process)
        for handle in handles:
            handle.close()
        lease.close()


def execute(job):
    signal.signal(signal.SIGTERM, terminate)
    plan = read(CONTROL / 'plan.json')
    approval = read(CONTROL / 'approval.json')
    used = 0
    for attempt in approval['attempts']:
        if attempt['job_id'] == job:
            continue
        rows = subprocess.check_output(['sacct', '-X', '-n', '-P', '-j', attempt['job_id'], '--format=JobIDRaw,State,ElapsedRaw'], text=True).splitlines()
        row = next(r.split('|') for r in rows if r.split('|')[0] == attempt['job_id'])
        if row[1].split()[0] not in ('COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT'):
            raise ValueError('Previous attempt remains active')
        used += int(row[2])
    current = check_approval(approval, plan, job, used)
    res = allocation(subprocess.check_output(['scontrol', 'show', 'job', job, '-o'], text=True), job,
                     requested_gpus=4, maximum_hours=current['time_limit_seconds']/3600)
    if res['cpus'] != 32 or res['allocated_memory_gib'] != 480:
        raise ValueError('Wrong CPU/RAM allocation')
    validate_source(SOURCE)
    readiness = read(CONTROL / 'readiness.json')
    validate_browser_prompt_assets(SOURCE)
    if (readiness['plan_sha256'] != digest(plan) or plan['controller_sha256'] != sha(__file__)
            or plan['source_manifest_sha256'] != sha(SOURCE / 'reference_manifest.json')
            or plan['actor_config_sha256'] != sha(ACTOR / 'config.json')):
        raise ValueError('Prepared code or plan changed')
    for name, expected in plan['actor_files'].items():
        stat = (ACTOR / name).stat()
        if (stat.st_size, stat.st_mtime_ns) != (expected['size'], expected['mtime_ns']):
            raise ValueError('Actor changed after independent hash verification')
    if plan.get('execution_strategy') == 'parallel_sft_priority':
        sft_source = Path(read(CONTROL / 'sft-extension/plan.json')['source'])
        env = environment()
        env.update(PYTHONPATH=str(sft_source) + ':' + str(PLOT_DEPS),
                   MPLCONFIGDIR='/tmp/luna-qwen-mpl', OPENBLAS_NUM_THREADS='1')
        subprocess.run([sys.executable, str(sft_source / 'scripts/run_luna_qwen_parallel.py'),
                        '--execute', job], cwd=sft_source, env=env, check=True)
        return
    with (CONTROL / 'owner.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        children = []
        deadline = time.monotonic() + res['maximum_seconds'] - 120
        try:
            for worker_id in range(4):
                cfg = dict(job_id=job, worker_id=worker_id)
                cfg_path = CONTROL / f'worker-{job}-{worker_id}.json'
                write_json(cfg_path, cfg)
                children.append(subprocess.Popen(worker_command(job, cfg_path), start_new_session=True))
            while any(child.poll() is None for child in children):
                if any(child.poll() not in (None, 0) for child in children):
                    raise RuntimeError('A worker failed; stop siblings and preserve partial tasks')
                assignments = [CONTROL / 'workers' / job / str(w) / 'device-assignment.json' for w in range(4)]
                if all(p.exists() for p in assignments) and len({read(p)['gpu_uuid'] for p in assignments}) != 4:
                    raise ValueError('Workers do not own four distinct physical GPUs')
                if time.monotonic() >= deadline:
                    raise TimeoutError('Original allocation budget ending')
                write_json(CONTROL / 'current.json', dict(job_id=job, stage='running', updated_unix=time.time(),
                    worker_states=[child.poll() for child in children], remaining_seconds=deadline-time.monotonic()))
                time.sleep(20)
            if any(child.returncode for child in children):
                raise RuntimeError('Worker failed')
            if not all(read(CONTROL / 'pilot/workers' / str(w) / 'status.json')['complete'] for w in range(4)):
                raise ValueError('Missing worker completion sentinel')
            report_env = environment()
            report_env.update(PYTHONPATH=str(SOURCE) + ':' + str(PLOT_DEPS), MPLCONFIGDIR='/tmp/luna-qwen-mpl',
                              OPENBLAS_NUM_THREADS='1')
            subprocess.run([str(RUNTIME / 'venv/bin/python'), str(SOURCE / 'scripts/report_luna_qwen_inference.py'),
                '--root', str(CONTROL), '--output', str(CONTROL / 'report')], cwd=SOURCE, env=report_env, check=True)
            write_json(CONTROL / 'pilot/status.json', dict(complete=True, episodes=40, tasks=10, workers=4))
            write_json(CONTROL / 'current.json', dict(job_id=job, stage='completed', complete=True, updated_unix=time.time()))
        except BaseException as exc:
            write_json(CONTROL / 'current.json', dict(job_id=job, stage='failed_or_interrupted', complete=False,
                error_type=type(exc).__name__, updated_unix=time.time()))
            raise
        finally:
            for child in children:
                stop(child)
            result = subprocess.run(['sacct', '-X', '-n', '-P', '-j', job,
                '--format=JobIDRaw,State,ElapsedRaw,AllocTRES'], text=True, capture_output=True)
            write_json(CONTROL / 'accounting' / (job + '.json'), dict(job_id=job,
                final_scheduler_audit_required=True, captured_unix=time.time(), sacct=result.stdout))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--execute', metavar='JOB_ID')
    parser.add_argument('--worker', type=Path)
    args = parser.parse_args()
    if args.prepare:
        p = prepare()
        print(json.dumps({k:p[k] for k in ('status', 'resources', 'limits', 'metrics', 'plots')}, indent=2))
    elif args.worker:
        worker(read(args.worker))
    elif args.execute:
        execute(args.execute)
    else:
        parser.error('Choose --prepare, --execute or --worker')
