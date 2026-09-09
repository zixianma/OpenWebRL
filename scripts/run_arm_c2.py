#!/usr/bin/env python3
"""Continue a completed ARM evaluation into authorized, resumable C2 collection."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from scripts.run_arm_retry import atomic_json, digest, get_health, load_rows, stop_owned, validate_allocation, process_identity

REPO = Path(__file__).resolve().parents[1]


def choose_teacher(source):
    manifests, rows = {}, {}
    for mode in ('baseline', 'scalar', 'selection'):
        directory = source / mode
        manifests[mode] = json.loads((directory / 'manifest.json').read_text())
        summary = json.loads((directory / 'summary.json').read_text())
        rows[mode] = load_rows(directory)
        ids = manifests[mode]['task_ids']
        if len(ids) != 300 or len(set(ids)) != 300 or set(rows[mode]) != set(ids) or summary['attempted'] != 300:
            raise ValueError(f'{mode} evaluation has not completed all 300 tasks')
    for mode in ('scalar', 'selection'):
        for field in ('actor', 'task_file_sha256', 'task_ids', 'seed', 'sampling', 'max_steps', 'judge'):
            if manifests[mode][field] != manifests['baseline'][field]:
                raise ValueError(f'Unmatched source field: {field}')
    wins = {mode: sum(bool(r.get('valid')) and r.get('reward') == 1 for r in records.values())
            for mode, records in rows.items()}
    common = [t for t in rows['scalar'] if rows['scalar'][t].get('valid') and rows['selection'][t].get('valid')]
    paired_difference = sum(int(rows['selection'][t]['reward'] == 1) - int(rows['scalar'][t]['reward'] == 1) for t in common)
    teacher = 'selection' if wins['selection'] > wins['scalar'] and paired_difference > 0 else 'scalar'
    return teacher, manifests[teacher], dict(successes=wins, scalar_selection_common_valid=len(common),
        selection_minus_scalar_successes_on_common_valid=paired_difference,
        rule='Selection if more successes on all 300 and a positive difference on common-valid tasks; otherwise Scalar. This is teacher selection, not a significance claim.')


def execute(queue_path):
    queue_path = Path(queue_path)
    q = json.loads(queue_path.read_text())
    if q.get('authorized') is not True:
        raise ValueError('C2 continuation is not authorized')
    source, output = Path(q['source_full']).resolve(), Path(q['output']).resolve()
    if output == source or source.parent in output.parents or output in source.parents:
        raise ValueError('C2 artifacts must be separate from original evaluation')
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / 'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    deadline = datetime.fromisoformat(q['stop_utc']).timestamp()
    def record(phase, **extra):
        atomic_json(output / 'status.json', dict(phase=phase, updated_utc=datetime.now(timezone.utc).isoformat(),
                    allocation=q['allocation'], output=str(output), **extra))
        print(phase, json.dumps(extra), flush=True)
    teacher, manifest, decision = choose_teacher(source)
    atomic_json(output / 'teacher-decision.json', dict(teacher=teacher, manifest=manifest, **decision))
    validate_allocation(q)
    record('waiting_for_preflight', teacher=teacher)
    while True:
        current = json.loads(queue_path.read_text())
        if current.get('authorized') is not True:
            record('held'); return
        if time.time() >= deadline - 120:
            record('paused_before_collection', reason='Existing allocation budget exhausted during preparation'); return
        if current.get('ready') is True:
            q = current
            break
        time.sleep(5)
    for name, sha in q['source_sha256'].items():
        if digest(REPO / name) != sha:
            raise ValueError(f'C2 source changed after preflight: {name}')
    if digest(q['task_file']) != q['task_file_sha256']:
        raise ValueError('C2 task manifest changed')
    validate_allocation(q)
    atomic_json(output / 'frozen-config.json', dict(q, teacher=teacher, teacher_manifest=manifest))
    before = {str(p.relative_to(source)): digest(p) for mode in ('baseline', 'scalar', 'selection')
              for p in (source / mode / 'results').glob('*.json')}
    atomic_json(output / 'original-eval-sha256.json', before)
    server = child = None
    server_log = collection_log = None
    try:
        smoke_passed = output / 'synthetic-training-smoke/smoke-passed.json'
        if not smoke_passed.exists():
            record('training_engineering_smoke', teacher=teacher, synthetic_data_only=True)
            with (output / 'training-smoke.log').open('a') as smoke_log:
                child = subprocess.Popen([q['training_python'], str(REPO / 'scripts/train_arm_c2.py'),
                        '--config', str(output / 'frozen-config.json'), '--smoke'], cwd=REPO,
                        stdout=smoke_log, stderr=subprocess.STDOUT, start_new_session=True)
                smoke_deadline = min(time.time() + 600, deadline - 120)
                while child.poll() is None and time.time() < smoke_deadline:
                    time.sleep(2)
                if child.poll() is None:
                    stop_owned(child)
                    raise TimeoutError('Synthetic training smoke exceeded its budget')
                if child.returncode or not smoke_passed.exists():
                    raise RuntimeError('Synthetic training smoke failed; see training-smoke.log')
                child = None
            validate_allocation(q)
        server_log = (output / 'teacher-server.log').open('a')
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='0')
        server = subprocess.Popen([sys.executable, str(REPO / 'scripts/serve_arm.py'), '--mode', teacher,
                '--model', manifest['selector_health']['model'], '--base', manifest['actor']], cwd=REPO,
                env=env, stdout=server_log, stderr=subprocess.STDOUT, start_new_session=True)
        startup = min(time.time() + 600, deadline)
        while True:
            if server.poll() is not None:
                raise RuntimeError('C2 teacher server failed to start')
            try:
                health = get_health('http://127.0.0.1:19101/health')
            except Exception:
                if time.time() > startup: raise TimeoutError('C2 teacher startup timeout')
                time.sleep(1); continue
            if health != manifest['selector_health']:
                raise ValueError('C2 teacher differs from validated evaluation teacher')
            break
        record('collecting', teacher=teacher, tasks=q['task_count'], actor_reused=True)
        collection_log = (output / 'collection.log').open('a')
        argv = [sys.executable, '-m', 'openwebrl.arm_c2', 'collect', '--config', str(output / 'frozen-config.json')]
        child = subprocess.Popen(argv, cwd=REPO, env=env, stdout=collection_log, stderr=subprocess.STDOUT, start_new_session=True)
        while child.poll() is None:
            if time.time() >= deadline:
                stop_owned(child)
                record('paused_at_allocation_deadline', teacher=teacher, reason='Resume incomplete collection within a subsequently assigned allocation')
                break
            time.sleep(5)
        else:
            if child.returncode:
                raise RuntimeError(f'C2 collection exited {child.returncode}; see collection.log')
            progress = json.loads((output / 'collection-summary.json').read_text())
            record('collection_complete' if progress['attempted'] == q['task_count'] else 'collection_paused', teacher=teacher, summary=progress)
            if progress['attempted'] == q['task_count'] and time.time() < deadline - 900:
                # The full student starts only after the entire fixed task pool
                # is processed. Free only the verified evaluation services.
                stop_owned(server); server = None
                validate_allocation(q)
                actor_parents = []
                for identity in q['resident_actor_processes']:
                    pid = identity['pid']
                    if process_identity(pid) != identity:
                        raise ValueError('Actor identity changed before training transition')
                    if b'sglang.launch_server' in (Path('/proc') / str(pid) / 'cmdline').read_bytes():
                        actor_parents.append(pid)
                if len(actor_parents) != 1:
                    raise ValueError('Cannot uniquely identify the owned actor parent')
                os.kill(actor_parents[0], signal.SIGTERM)
                for _ in range(30):
                    gpu_pids = subprocess.check_output(['nvidia-smi', '--id=' + q['gpu_uuid'],
                               '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
                    if not gpu_pids: break
                    time.sleep(1)
                if gpu_pids:
                    raise ValueError('Evaluation GPU did not become free; leave remaining processes untouched')
                record('training_student', teacher=teacher, synthetic_data_only=False)
                with (output / 'training.log').open('a') as training_log:
                    argv = [q['training_python'], str(REPO / 'scripts/train_arm_c2.py'),
                            '--config', str(output / 'frozen-config.json')]
                    latest = output / 'student/latest-checkpoint.json'
                    if latest.exists():
                        argv += ['--resume', json.loads(latest.read_text())['path']]
                    child = subprocess.Popen(argv, cwd=REPO, stdout=training_log,
                                             stderr=subprocess.STDOUT, start_new_session=True)
                    while child.poll() is None and time.time() < deadline:
                        time.sleep(5)
                    if child.poll() is None:
                        stop_owned(child)
                        record('training_paused_at_deadline', teacher=teacher)
                    elif child.returncode:
                        raise RuntimeError('C2 trainer failed; see training.log')
                    else:
                        record('complete' if (output / 'student/complete.json').exists() else 'training_paused', teacher=teacher)
            elif progress['attempted'] == q['task_count']:
                record('dataset_complete_waiting_for_training_compute', teacher=teacher)
        after = {str(p.relative_to(source)): digest(p) for mode in ('baseline', 'scalar', 'selection')
                 for p in (source / mode / 'results').glob('*.json')}
        if before != after:
            raise ValueError('Original evaluation results changed during C2 collection')
    except BaseException as exc:
        record('failed', error_type=type(exc).__name__, reason=str(exc)); raise
    finally:
        stop_owned(child); stop_owned(server)
        if server_log: server_log.close()
        if collection_log: collection_log.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue', required=True)
    args = parser.parse_args()
    def interrupted(signum, frame):
        raise InterruptedError(f'Received signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    execute(args.queue)


if __name__ == '__main__':
    main()
