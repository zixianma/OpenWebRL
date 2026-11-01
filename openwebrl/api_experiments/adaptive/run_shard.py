"""Own one CPU-only Slurm allocation: browser preflight, worker slots, teardown."""
from bootstrap import ROOT, CONFIG
import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path

from adaptive_worker import read, shard_root, validate_sources
from openwebrl.luna_qwen_metrics import write_json


def env(root):
    e = os.environ.copy()
    e.update(PYTHONPATH=os.pathsep.join([str(ROOT), CONFIG['cuawright_source'], CONFIG['frozen_support_source']]),
             PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='',
             WANDB_PROJECT='openwebrl-evals', TMPDIR='/tmp', DO_NOT_TRACK='1',
             PATH=str(Path(CONFIG['python']).parent) + os.pathsep + e.get('PATH', '/usr/bin:/bin'))
    for key in ['XDG_CACHE_HOME', 'XDG_CONFIG_HOME', 'XDG_DATA_HOME', 'XDG_STATE_HOME', 'WANDB_DIR', 'WANDB_CONFIG_DIR',
                'WANDB_CACHE_DIR', 'WANDB_DATA_DIR', 'HF_HOME', 'TORCH_HOME', 'MPLCONFIGDIR',
                'BUN_RUNTIME_TRANSPILER_CACHE_PATH', 'BUN_INSTALL_CACHE_DIR', 'PIP_CACHE_DIR']:
        path = root / 'runtime-cache' / key.lower()
        path.mkdir(parents=True, exist_ok=True)
        e[key] = str(path)
    return e


def allocation(job_id):
    raw = subprocess.check_output(['scontrol', 'show', 'job', '-o', str(job_id)], text=True)
    fields = dict(re.findall(r'(\w+)=(\S+)', raw))
    tres = dict(x.split('=', 1) for x in fields['AllocTRES'].split(','))
    assert fields['JobState'] == 'RUNNING' and fields['Account'] == 'vlm', 'Unexpected allocation'
    assert not any(k.startswith('gres/gpu') and float(v) for k, v in tres.items()), 'CPU-only allocation required'
    limit = fields['TimeLimit']
    days, clock = limit.split('-') if '-' in limit else ('0', limit)
    return int(days) * 86400 + sum(int(v) * 60 ** i for i, v in enumerate(reversed(clock.split(':'))))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--shard', required=True)
    p.add_argument('--slots', type=int, default=2)
    a = p.parse_args()
    validate_sources()
    plan = read(ROOT / 'campaign-plan.json')
    assert a.shard in plan['shards'], 'Unknown shard'
    job = os.environ['SLURM_JOB_ID']
    seconds = allocation(job)
    started = time.monotonic()
    attempts = shard_root(a.shard) / 'attempts'
    attempts.mkdir(parents=True, exist_ok=True)
    root = attempts / f'attempt-{len(list(attempts.iterdir())):03d}-job{job}'
    root.mkdir()
    os.environ.update(env(root))
    from worker import preflight
    from environments import stop, record_owned, cleanup_registered_groups
    from openwebrl.luna_qwen_policy import Budget
    caps = plan['budget']
    children = []
    try:
        write_json(root / 'current.json', {'epoch': time.time(), 'stage': 'mechanical_browser_preflight', 'job_id': job})
        browser = Budget(shard_root(a.shard) / 'budget/browser', caps['browser_starts_per_shard'], caps['browser_starts_per_shard'])
        preflight(root, browser)
        for slot in range(a.slots):
            log = (root / f'worker{slot}.log').open('ab')
            children.append(subprocess.Popen([CONFIG['python'], '-B', str(ROOT / 'adaptive_worker.py'), '--shard', a.shard,
                                              '--slot', str(slot), '--slots', str(a.slots), '--attempt-dir', str(root)],
                                             env=env(root), stdout=log, stderr=log, start_new_session=True))
            record_owned(children[-1], root, f'worker{slot}')
        deadline = started + seconds - 300
        while any(c.poll() is None for c in children):
            if time.monotonic() > deadline:
                raise TimeoutError('Allocation time nearly exhausted; stopping workers for a later attempt')
            outcomes = len(list((root / 'episodes').glob('*/result.json')))
            write_json(root / 'heartbeat.json', {'epoch': time.time(), 'stage': 'running', 'outcomes': outcomes, 'job_id': job,
                                                 'workers_alive': sum(c.poll() is None for c in children),
                                                 'worker_returncodes': [c.poll() for c in children]})
            time.sleep(15)
        codes = [c.returncode for c in children]
        write_json(root / 'current.json', {'epoch': time.time(), 'stage': 'finished' if not any(codes) else 'worker_failed',
                                           'worker_returncodes': codes, 'job_id': job,
                                           'outcomes': len(list((root / 'episodes').glob('*/result.json')))})
        if any(codes):
            raise RuntimeError('A worker failed; diagnose before another attempt')
    except BaseException as exc:
        write_json(root / 'failure.json', {'epoch': time.time(), 'error_type': type(exc).__name__, 'error': str(exc)})
        raise
    finally:
        rows = []
        for child in children:
            try:
                rows.append({'pid': child.pid, 'cleanup': stop(child, grace_seconds=90), 'returncode': child.returncode})
            except BaseException as exc:
                rows.append({'pid': child.pid, 'error_type': type(exc).__name__})
        write_json(root / 'owned-worker-teardown.json', {'epoch': time.time(), 'workers': rows})
        cleanup_registered_groups(root)


if __name__ == '__main__':
    main()
