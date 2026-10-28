#!/usr/bin/env python3
"""Prepare or enter one explicitly approved, actively supervised CPU debug allocation.

No website image or entrypoint runs automatically. The interactive allocation is
for original Reddit image import, service diagnosis and reset validation. This
initial-attempt launcher rejects a second launch; recovery must reconcile actual
scheduler consumption and use only the remaining approved seconds.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import tempfile
import time

ROOT = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-webarena-20261011')
SCRIPT = Path(__file__).resolve()
REQUEST = {
    'schema_version': 1,
    'scope': 'Original WebArena Reddit import, startup diagnosis and reset validation; no benchmark collection',
    'account': 'zixianma', 'partition': 'gpu-h200', 'qos': 'normal',
    'nodes': 1, 'cpus': 4, 'memory_gib': 16, 'gpus': 0,
    'total_allocation_seconds_including_retries': 7200,
    'api_budget_usd': 0, 'actor_judge_calls': 0,
    'original_reddit_archive_bytes': 53435097088,
    'original_reddit_archive_download_cap_bytes': 54000000000,
    'node_local_free_space_required_gib': 300,
    'persistent_artifacts_planning_ceiling_gib': 100,
    'older_om2w_budget_seconds_added': 0,
    'supervision': 'An active agent owns the interactive salloc/srun session, every worker and all stage handoffs.',
    'endpoint': 'Saved import/config hashes, website login/startup/reset evidence or diagnosed blocker within the cap.',
    'limitations': ['CPU-only scheduler acceptance and node-local space are not yet verified on a compute node.',
                   'Original service entrypoint must be inspected before execution; minimal fakeroot is not website validation.',
                   'This request does not cover the other four sites, full evaluation, GPU inference or paid API calls.'],
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def safe_path(path):
    path = Path(path).resolve()
    allowed = (Path('/gpfs/scrubbed/zixianma/openwebrl-runtime'), Path('/tmp'))
    if not any(path == root or root in path.parents for root in allowed):
        raise ValueError('Use an approved runtime or /tmp directory')
    return path


def write_new(path, value):
    if path.is_symlink():
        raise ValueError('Refusing artifact symlink')
    with path.open('x') as handle:
        json.dump(value, handle, indent=2)
        handle.write('\n')


def prepare(root):
    root.mkdir(parents=True, exist_ok=True)
    for filename, payload in {
        'resource-request.json': REQUEST,
        'authorization.json': {'status': 'pending', 'request_sha256': digest(REQUEST),
                               'exact_user_approval': None},
    }.items():
        path = root / filename
        if path.exists() or path.is_symlink():
            if path.is_symlink() or json.loads(path.read_text()) != payload:
                raise ValueError(f'Preserve differing existing artifact: {path.name}')
        else:
            write_new(path, payload)
    print(json.dumps({'status': 'prepared_not_submitted', 'request_sha256': digest(REQUEST),
                      'launch_after_exact_approval': ['python3', '-B', str(SCRIPT), '--launch', '--root', str(root)]}))


def check_authorization(root):
    for name in ('resource-request.json', 'authorization.json'):
        if (root / name).is_symlink():
            raise ValueError('Refusing authorization symlink')
    if json.loads((root / 'resource-request.json').read_text()) != REQUEST:
        raise ValueError('Resource request changed')
    auth = json.loads((root / 'authorization.json').read_text())
    if (auth.get('status') != 'approved' or auth.get('request_sha256') != digest(REQUEST)
            or not isinstance(auth.get('exact_user_approval'), str)
            or not auth['exact_user_approval'].strip()):
        raise ValueError('Exact resource approval is pending; no allocation requested')


def node_shell(root):
    check_authorization(root)
    if not os.environ.get('SLURM_JOB_ID') or not os.environ.get('SLURM_STEP_ID'):
        raise ValueError('Node shell requires the owned srun allocation')
    if int(os.environ.get('SLURM_CPUS_PER_TASK', '0')) != REQUEST['cpus']:
        raise ValueError('Unexpected CPU allocation')
    if int(os.environ.get('SLURM_MEM_PER_NODE', '0')) != REQUEST['memory_gib'] * 1024:
        raise ValueError('Unexpected memory allocation')
    if any(os.environ.get(key, '').strip() not in ('', '0')
           for key in ('SLURM_GPUS', 'SLURM_GPUS_ON_NODE', 'SLURM_GPUS_PER_NODE')):
        raise ValueError('Unexpected GPU allocation')
    if any(os.environ.get(key, '').strip() for key in ('SLURM_JOB_GPUS', 'SLURM_STEP_GPUS')):
        raise ValueError('Unexpected assigned GPU devices')
    if shutil.disk_usage('/tmp').free < REQUEST['node_local_free_space_required_gib'] * 2**30:
        raise ValueError('Insufficient free node-local scratch; stop before downloading')
    scratch = Path(tempfile.mkdtemp(prefix='webprm-wa-', dir='/tmp'))
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(('APPTAINER_', 'SINGULARITY_'))}
    for key, suffix in {
        'TMPDIR': 'tmp', 'APPTAINER_TMPDIR': 'apptainer-tmp',
        'APPTAINER_CONFIGDIR': 'apptainer-config', 'APPTAINER_CACHEDIR': 'apptainer-cache',
        'XDG_CONFIG_HOME': 'xdg-config', 'XDG_DATA_HOME': 'xdg-data',
        'XDG_CACHE_HOME': 'xdg-cache', 'XDG_RUNTIME_DIR': 'xdg-runtime',
        'PIP_CACHE_DIR': 'pip-cache',
    }.items():
        path = scratch / suffix
        path.mkdir(mode=0o700)
        env[key] = str(path)
    env.update(PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1',
               HISTFILE='/dev/null', HISTSIZE='0', HISTFILESIZE='0', BASH_ENV='/dev/null', ENV='/dev/null',
               APPTAINER_NO_HOME='true', APPTAINER_CONTAINALL='true',
               APPTAINER_NO_MOUNT='bind-paths,home,cwd,hostfs,tmp',
               WANDB_PROJECT='openwebrl-evals', WEBPRM_SITE_SCRATCH=str(scratch))
    receipt = {'status': 'active_agent_debug_shell_ready', 'job_id': env['SLURM_JOB_ID'],
               'hostname': socket.gethostname(), 'scratch': str(scratch),
               'started_unix': time.time(), 'request_sha256': digest(REQUEST),
               'automatic_home_and_gpfs_container_binds_disabled': True}
    write_new(root / f'node-ready-{env["SLURM_JOB_ID"]}.json', receipt)
    print(json.dumps(receipt), flush=True)
    os.chdir(root)
    os.execvpe('/bin/bash', ['/bin/bash', '--noprofile', '--norc'], env)


def launch(root):
    check_authorization(root)
    if (root / 'initial-launch.json').exists():
        raise ValueError('Initial launch already recorded; reconcile consumed allocation time before recovery')
    argv = ['salloc', '--account=zixianma', '--partition=gpu-h200', '--qos=normal',
            '--nodes=1', '--ntasks=1', '--cpus-per-task=4', '--mem=16G', '--time=02:00:00',
            '--job-name=wa-reddit-debug', '--chdir=' + str(root),
            'srun', '--ntasks=1', '--cpus-per-task=4', '--pty',
            'python3', '-B', str(SCRIPT), '--node-shell', '--root', str(root)]
    # Prevent inherited submission options from adding a GPU or changing caps.
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(('SALLOC_', 'SBATCH_', 'SRUN_', 'SLURM_'))}
    env.update(HISTFILE='/dev/null', PYTHONDONTWRITEBYTECODE='1',
               BASH_ENV='/dev/null', ENV='/dev/null')
    write_new(root / 'initial-launch.json', {'status': 'submission_started_not_allocation_verified',
              'started_unix': time.time(), 'argv': argv, 'request_sha256': digest(REQUEST),
              'script_sha256': hashlib.sha256(SCRIPT.read_bytes()).hexdigest()})
    os.chdir(root)
    os.execvpe('salloc', argv, env)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT / 'local-debug')
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare', action='store_true')
    mode.add_argument('--launch', action='store_true')
    mode.add_argument('--node-shell', action='store_true')
    args = parser.parse_args()
    root = safe_path(args.root)
    (prepare if args.prepare else node_shell if args.node_shell else launch)(root)


if __name__ == '__main__':
    main()
