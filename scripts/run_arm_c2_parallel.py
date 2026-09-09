#!/usr/bin/env python3
"""Collect C2 with paired actor/teacher replicas in an existing multi-GPU allocation."""
import argparse
from datetime import datetime, timezone
import fcntl
import getpass
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

from scripts.run_arm_c2 import REPO, choose_teacher, freeze_execution_config, training_command
from scripts.run_arm_retry import atomic_json, digest, get_health, process_identity, stop_owned
from openwebrl.arm_c2 import collection_endpoints


def command(argv):
    return subprocess.check_output(argv, text=True, timeout=20).strip()


def validate_resources(queue, require_free=False):
    job = str(queue['allocation'])
    if os.getenv('SLURM_JOB_ID') != job or f'/job_{job}/' not in Path('/proc/self/cgroup').read_text():
        raise ValueError('Parallel collector must run in its authorized allocation')
    info = command(['scontrol', 'show', 'job', job, '-o'])
    if f'UserId={getpass.getuser()}(' not in info or 'JobState=RUNNING' not in info:
        raise ValueError('Authorized allocation is not running under this user')
    expected = [r['gpu_uuid'] for r in collection_endpoints(queue)]
    devices = command(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader']).splitlines()
    if len(set(expected)) != len(expected) or devices != expected:
        raise ValueError('Assigned GPUs do not match the replica inventory')
    if require_free:
        for gpu in devices:
            if command(['nvidia-smi', '--id='+gpu, '--query-compute-apps=pid', '--format=csv,noheader']):
                raise ValueError(f'Assigned GPU {gpu} is occupied; leave its workload untouched')
    return devices


def verify_sources(queue):
    for name, expected in queue['source_sha256'].items():
        if digest(REPO / name) != expected:
            raise ValueError(f'C2 source changed after preflight: {name}')
    if digest(queue['task_file']) != queue['task_file_sha256']:
        raise ValueError('C2 task inventory changed')
    training_command(queue)


def outcome_hashes(source):
    return {str(p.relative_to(source)): digest(p) for mode in ('baseline', 'scalar', 'selection')
            for p in (source / mode / 'results').glob('*.json')}


def execute(queue_path):
    queue_path = Path(queue_path)
    queue = json.loads(queue_path.read_text())
    if queue.get('authorized') is not True or queue.get('ready') is not True:
        raise ValueError('Require an authorized, checked queue')
    output, source = Path(queue['output']).resolve(), Path(queue['source_full']).resolve()
    if output == source or source.parent in output.parents or output in source.parents:
        raise ValueError('C2 artifacts must remain separate from the evaluation')
    lock = (output / 'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    replicas = collection_endpoints(queue)
    deadline = datetime.fromisoformat(queue['stop_utc']).timestamp()
    stopped = False
    def interrupt(signum, frame):
        nonlocal stopped
        stopped = True
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    def record(phase, **extra):
        value = dict(phase=phase, updated_utc=datetime.now(timezone.utc).isoformat(),
                     allocation=queue['allocation'], output=str(output), **extra)
        atomic_json(output / 'status.json', value)
        print(phase, json.dumps(extra), flush=True)
    validate_resources(queue, require_free=True)
    verify_sources(queue)
    if not (output / 'synthetic-training-smoke/smoke-passed.json').exists():
        raise ValueError('The validated C2 training smoke is required before continuation')
    if time.time() >= deadline - 600:
        record('paused_before_collection', reason='Insufficient startup time'); return
    teacher, manifest, decision = choose_teacher(source)
    atomic_json(output / 'teacher-decision.json', dict(teacher=teacher, manifest=manifest, **decision))
    queue.update(slurm_step=os.environ['SLURM_STEP_ID'], teacher=teacher, teacher_manifest=manifest)
    freeze_execution_config(output, queue)
    atomic_json(queue_path, queue)
    atomic_json(output / 'queue-manifest.json', queue)
    original = outcome_hashes(source)
    old_inventory = output / 'original-eval-sha256.json'
    if old_inventory.exists() and json.loads(old_inventory.read_text()) != original:
        raise ValueError('Original evaluation outcomes changed before resume')
    preserved = {p.name: digest(p) for p in (output / 'results').glob('*.json')}
    session = output / 'execution-sessions' / queue['allocation']
    session.mkdir(exist_ok=True)
    atomic_json(session / 'before-resume-outcomes.json', preserved)
    services, streams = [], []
    child = None
    def spawn(argv, log_name, gpu=None):
        env = dict(os.environ)
        if gpu is not None: env['CUDA_VISIBLE_DEVICES'] = gpu
        stream = (output / log_name).open('a'); streams.append(stream)
        process = subprocess.Popen(argv, cwd=REPO, env=env, stdout=stream,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        return process
    def check_services():
        for process in services:
            if process.poll() is not None:
                raise RuntimeError(f'Inference service {process.pid} exited {process.returncode}')
    def stop_collector():
        nonlocal child
        if child is not None and child.poll() is None:
            # Allow cancellation-safe artifact writes and the final dataset audit.
            os.killpg(child.pid, signal.SIGTERM)
            try: child.wait(timeout=120)
            except subprocess.TimeoutExpired: stop_owned(child)
    try:
        record('starting_replicas', teacher=teacher, replicas=replicas, parallel=queue['parallel'])
        for replica in replicas:
            for key in ('actor_port', 'selector_port'):
                with socket.socket() as probe: probe.bind(('127.0.0.1', replica[key]))
        for i, replica in enumerate(replicas):
            gpu = replica['gpu_uuid']
            services.append(spawn([queue['actor_python'], '-u', '-m', 'sglang.launch_server',
                '--model-path', manifest['actor'], '--host', '127.0.0.1', '--port', str(replica['actor_port']),
                '--dtype', 'bfloat16', '--tp', '1', '--mem-fraction-static', '0.4', '--context-length', '32768',
                '--max-running-requests', '48', '--chunked-prefill-size', '4096', '--cuda-graph-max-bs', '48'],
                f'actor-replica-{i}.log', gpu))
            services.append(spawn([sys.executable, '-u', str(REPO / 'scripts/serve_arm.py'), '--mode', teacher,
                '--model', manifest['selector_health']['model'], '--base', manifest['actor'],
                '--port', str(replica['selector_port'])], f'teacher-replica-{i}.log', gpu))
        startup_deadline = min(deadline - 180, time.time() + 600)
        ready = set()
        while len(ready) < len(replicas):
            check_services()
            if stopped or time.time() >= startup_deadline:
                raise RuntimeError('Replica startup interrupted or timed out')
            for i, replica in enumerate(replicas):
                if i in ready: continue
                try:
                    get_health(f"http://127.0.0.1:{replica['actor_port']}/health_generate")
                    model = get_health(f"http://127.0.0.1:{replica['actor_port']}/get_model_info")
                    health = get_health(f"http://127.0.0.1:{replica['selector_port']}/health")
                except Exception: continue
                if os.path.realpath(model['model_path']) != os.path.realpath(manifest['actor']) or health != manifest['selector_health']:
                    raise ValueError('Replica differs from pinned actor/teacher')
                ready.add(i)
            time.sleep(1)
        atomic_json(session / 'service-identities.json', [process_identity(p.pid) for p in services])
        record('collecting', teacher=teacher, tasks=queue['task_count'], replicas=replicas, parallel=queue['parallel'])
        child = spawn([sys.executable, '-u', '-m', 'openwebrl.arm_c2', 'collect',
                       '--config', str(output / 'frozen-config.json')], 'collection.log')
        atomic_json(session / 'collector-identity.json', process_identity(child.pid))
        while child.poll() is None and not stopped and time.time() < deadline:
            check_services()
            time.sleep(5)
        if child.poll() is None: stop_collector()
        if child.returncode:
            raise RuntimeError(f'C2 collection exited {child.returncode}; see collection.log')
        for name, expected in preserved.items():
            if digest(output / 'results' / name) != expected:
                raise ValueError('A completed C2 outcome changed during collection')
        if outcome_hashes(source) != original:
            raise ValueError('Original evaluation outcomes changed during collection')
        progress = json.loads((output / 'collection-summary.json').read_text())
        complete = progress['attempted'] == queue['task_count']
        record('collection_complete' if complete else 'collection_paused', teacher=teacher, summary=progress)
        if not complete or stopped: return
        if time.time() >= deadline - 900:
            record('dataset_complete_waiting_for_training_compute', teacher=teacher); return
        verify_sources(queue)
        for process in reversed(services): stop_owned(process)
        for _ in range(30):
            try:
                validate_resources(queue, require_free=True)
                break
            except ValueError as exc:
                if 'occupied' not in str(exc): raise
                time.sleep(1)
        else: raise ValueError('Inference GPUs did not become free; leave remaining workloads untouched')
        train_argv, train_cwd = training_command(queue)
        argv = train_argv + ['--config', str(output / 'frozen-config.json')]
        latest = output / 'student/latest-checkpoint.json'
        if latest.exists(): argv += ['--resume', json.loads(latest.read_text())['path']]
        training_log = (output / 'training.log').open('a'); streams.append(training_log)
        train_env = dict(os.environ, CUDA_VISIBLE_DEVICES=queue['gpu_uuid'])
        record('training_student', teacher=teacher, synthetic_data_only=False, gpu_uuid=queue['gpu_uuid'])
        child = subprocess.Popen(argv, cwd=train_cwd, env=train_env, stdout=training_log,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        while child.poll() is None and not stopped and time.time() < deadline: time.sleep(5)
        if child.poll() is None: stop_collector()
        if child.returncode: raise RuntimeError('C2 trainer failed; see training.log')
        record('complete' if (output / 'student/complete.json').exists() else 'training_paused', teacher=teacher)
    except BaseException as exc:
        record('failed', error_type=type(exc).__name__, reason=str(exc)); raise
    finally:
        stop_collector()
        for process in reversed(services): stop_owned(process)
        for stream in streams: stream.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue', required=True)
    execute(parser.parse_args().queue)


if __name__ == '__main__': main()
