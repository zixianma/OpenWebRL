#!/usr/bin/env python3
"""Resume the preserved reference baseline inside an explicitly authorized job.

Defaults to read-only preflight. --launch uses an EXISTING Slurm allocation;
this program cannot submit, extend, or cancel allocations. Requires the project's
own installed runtime and the preserved reference source recorded in the pointer.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
import zipfile

REPO = Path(__file__).resolve().parents[1]
RUNTIME = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime')
DEFAULT_STATE = RUNTIME / 'current_baseline.json'


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def capture(argv):
    return subprocess.check_output(argv, text=True, stderr=subprocess.PIPE).strip()


def allocation(info, job_id, now=None, uid=None, requested_gpus=2):
    """Validate ownership, capacity, state and the existing paid time boundary."""
    if requested_gpus not in (2, 4):
        raise ValueError('Supported profiles use 2 or 4 H200 GPUs.')
    fields = dict(re.findall(r'(\w+)=(\S+)', info))
    uid = os.getuid() if uid is None else uid
    now = time.time() if now is None else now
    if fields.get('JobId') != job_id or fields.get('JobState') != 'RUNNING':
        raise ValueError('The specified allocation is not RUNNING.')
    if not fields.get('UserId', '').endswith(f'({uid})'):
        raise ValueError('The allocation belongs to another user.')
    if fields.get('NumNodes') != '1':
        raise ValueError('This tested workflow requires a single-node allocation.')
    tres = dict(part.split('=', 1) for part in fields['AllocTRES'].split(','))
    cpus, gpus = int(fields['NumCPUs']), int(tres.get('gres/gpu', '0'))
    memory = re.fullmatch(r'([\d.]+)([KMGT])', tres.get('mem', ''))
    gib = float(memory[1]) * {'K': 2**-20, 'M': 2**-10, 'G': 1, 'T': 1024}[memory[2]] if memory else 0
    if cpus < 4 * requested_gpus or gpus < requested_gpus or gib < 120 * requested_gpus:
        raise ValueError(f'This profile needs {requested_gpus} GPUs, {4 * requested_gpus} CPUs and {120 * requested_gpus} GiB in the existing allocation.')
    if int(tres.get('gres/gpu:h200', '0')) < requested_gpus:
        raise ValueError(f'This profile requires {requested_gpus} allocated H200 GPUs.')
    end = datetime.strptime(fields['EndTime'], '%Y-%m-%dT%H:%M:%S').timestamp()
    seconds = min(8 * 3600, int(end - now) - 180)
    if seconds < 600:
        raise ValueError('Less than 10 minutes remain after the three-minute shutdown margin.')
    return {'job_id': job_id, 'host': fields['NodeList'], 'cpus': min(cpus, 16),
            'gpus': requested_gpus, 'tensor_parallel_size': requested_gpus, 'allocated_memory_gib': gib, 'end_time': fields['EndTime'],
            'maximum_seconds': seconds, 'shutdown_margin_seconds': 180}


def active_steps(output, *, allow_batch=False):
    ignored = {'interactive', 'extern'} | ({'batch'} if allow_batch else set())
    return [line.strip() for line in output.splitlines()
            if line.strip() and line.split('|', 1)[0].rsplit('.', 1)[-1] not in ignored]


def lineage(state):
    """Follow only this run's recorded ancestry, never unrelated checkpoint dirs."""
    roots, current = [], Path(state['run_directory']).resolve()
    while current not in roots:
        roots.append(current)
        manifest = current / 'launch_manifest.json'
        parent = read_json(manifest).get('resume_from') if manifest.exists() else None
        if not parent:
            break
        current = Path(parent).resolve()
    last = Path(state['last_valid_checkpoint']).resolve().parent
    if last not in roots:
        raise ValueError('Last verified checkpoint is outside the recorded run ancestry.')
    return roots


def checkpoint_root(roots):
    candidates = []
    for root in roots:
        marker = root / 'latest_checkpointed_iteration.txt'
        if marker.exists():
            number = int(marker.read_text().strip())
            if number < 0:
                raise ValueError('Invalid checkpoint iteration.')
            candidates.append((number, root))
    if not candidates:
        raise ValueError('No completed checkpoint marker in this run lineage.')
    # Preserve newest continuation when two directories contain the same iteration.
    number, root = max(candidates, key=lambda item: item[0])
    return root, number


def validate_source(source):
    manifest = read_json(source / 'reference_manifest.json')
    hashes = manifest['recipe_files_sha256']
    if not hashes:
        raise ValueError('Reference source has no recipe hash manifest.')
    for relative, expected in hashes.items():
        path = (source / relative).resolve()
        if not path.is_relative_to(source.resolve()):
            raise ValueError('Recipe file escapes the preserved source.')
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Reference recipe changed: {relative}')
    launcher = source / 'scripts/run_small_baseline.py'
    text = launcher.read_text()
    for required in ['OPENWEBRL_MULTIMODAL_STORAGE_DIR', '--save-debug-rollout-data',
                     '--skip-eval-before-train', "'shutdown_margin_seconds'", "'durable_optimizer_updates_at_start'"]:
        if required not in text:
            raise ValueError(f'Preserved launcher lacks required resume support: {required}')
    collector = (source / 'slime/rollout/sglang_rollout.py').read_text()
    transport = (source / 'slime/utils/rollout_transport.py').read_text()
    if 'await asyncio.to_thread(file_back_completed_group, group)' not in collector or 'malloc_trim' not in transport:
        raise ValueError('Preserved source lacks collection-time image mapping and allocator trim required for 240 GiB.')
    return hashlib.sha256(launcher.read_bytes()).hexdigest()


def replay_batch(state, roots, root, iteration):
    next_id = iteration + 1
    pending = state.get('pending_replay_batch')
    paths = ([Path(pending)] if pending and Path(pending).stem == str(next_id) else [])
    paths += [p / 'rollout_recovery' / f'{next_id}.pt' for p in roots]
    for batch in dict.fromkeys(paths):
        if not batch.exists():
            continue
        if batch.resolve().parent.parent not in roots:
            raise ValueError('Replay batch is outside this run lineage.')
        # Central-directory validation detects interrupted torch.save without reading 60+ GB.
        with zipfile.ZipFile(batch) as archive:
            if not any(name.endswith('/data.pkl') for name in archive.namelist()):
                raise ValueError('Recovery file is not a complete torch.save archive.')
        provenance = batch.with_suffix('.provenance.json')
        if provenance.exists():
            data = read_json(provenance)
            if (data['rollout_id_zero_based'] != next_id
                    or data['preceding_checkpoint_iteration'] != iteration
                    or Path(data['preceding_checkpoint_root']).resolve() != root
                    or data['batch_bytes'] != batch.stat().st_size):
                raise ValueError('Recovery provenance disagrees with the checkpoint or saved batch.')
            consumed = data['submitted_groups']
        else:
            progress = batch.parent.parent / 'progress.log'
            pattern = rf'\[GenerateProgress\] rollout={next_id + 1}/\d+ event=done groups=48/48 completed_groups=(\d+) pending_groups=(\d+)'
            matches = re.findall(pattern, progress.read_text())
            if len(matches) != 1:
                raise ValueError('Cannot determine replay cursor advance from a unique completed collection; inspect it manually.')
            consumed = sum(map(int, matches[0]))
        if not isinstance(consumed, int) or consumed < 48:
            raise ValueError('Invalid submitted-group count for replay.')
        return {'batch': str(batch.resolve()), 'rollout_id': next_id, 'consumed_groups': consumed}
    return None


def source_command(source, argv):
    # shlex.join quotes literal paths; no shell interpolation of credentials.
    return ['bash', '-c', f'set -euo pipefail\nsource {shlex.quote(str(source / "scripts/h200_env.sh"))}\nexec {shlex.join(argv)}']


def step_command(job, cpus, gpus=2):
    return ['srun', f'--jobid={job}', '--overlap', '--nodes=1', '--ntasks=1',
            f'--cpus-per-task={cpus}', f'--gres=gpu:{gpus}', '--exact']


def clean_environment():
    env = os.environ.copy()
    for key in list(env):
        if key.startswith('OPENWEBRL_REPLAY_') or key in {
            'OPENWEBRL_VERIFY_RESUME_ONLY', 'OPENWEBRL_STOP_AFTER_SAVED_ROLLOUT',
            'WANDB_RUN_ID', 'DRY_RUN', 'SLIME_LOAD_CHECKPOINT', 'SLIME_CKPT_STEP',
            'OVERRIDE_OPT_PARAM_SCHEDULER'}:
            env.pop(key, None)
    return env


def prepare(args):
    state = read_json(args.state)
    gpus = getattr(args, 'gpus', 2)
    verification = getattr(args, 'verify_resume_only', False)
    job = allocation(capture(['scontrol', 'show', 'job', args.job_id, '-o']), args.job_id, requested_gpus=gpus)
    batch_driver = (os.environ.get('OPENWEBRL_BATCH_DRIVER_JOB') == args.job_id
                    and os.environ.get('SLURM_JOB_ID') == args.job_id)
    busy = active_steps(capture(['squeue', '--steps', f'--jobs={args.job_id}', '--noheader', '--format=%i|%j']), allow_batch=batch_driver)
    other_live_steps = []
    previous_job = str(state.get('allocation', ''))
    if previous_job and previous_job != args.job_id and not (Path(state['run_directory']) / 'exit_status.json').exists():
        own_steps = capture(['squeue', '--steps', '--user', str(os.getuid()), '--noheader', '--format=%i|%j'])
        other_live_steps = active_steps('\n'.join(line for line in own_steps.splitlines() if line.startswith(previous_job + '.')))
    source = Path(args.source or state['source_directory']).resolve()
    source_hash = validate_source(source)
    topology_capable = 'OPENWEBRL_RESUME_TOPOLOGY_V1' in (source / 'scripts/run_small_baseline.py').read_text()
    if gpus != 2 and not topology_capable:
        raise ValueError('Prepare a topology-capable preserved source with scripts/prepare_resume_topology.py first.')
    roots = lineage(state)
    root, iteration = checkpoint_root(roots)
    offset = state.get('scheduler_offset_updates', 0)
    inspection = source_command(source, ['python', str(REPO / 'scripts/inspect_training_checkpoint.py'),
                               str(root), '--expected-scheduler-offset-updates', str(offset)])
    report = json.loads(capture(inspection))
    if report['iteration'] != iteration:
        raise ValueError('Checkpoint marker changed during preflight; retry after training stops.')
    replay = None if verification else replay_batch(state, roots, root, iteration)
    run_id = state.get('wandb_run_id') or state['wandb_url'].rstrip('/').rsplit('/', 1)[-1]
    if args.wandb_run_id and args.wandb_run_id != run_id:
        raise ValueError('W&B ID differs from the recorded lineage; use a separate state file for another run.')
    if not re.fullmatch(r'[A-Za-z0-9_-]+', run_id):
        raise ValueError('Invalid recorded W&B run ID.')
    extra = ['--gpus', str(gpus)] if topology_capable else []
    if verification:
        extra.append('--verify-resume-only')
    command = step_command(args.job_id, job['cpus'], gpus) + source_command(source, [
        'python', str(source / 'scripts/run_small_baseline.py'), '--profile', 'reference',
        '--resume-from', str(root), '--wandb-run-id', run_id, '--env-file', str(args.env_file.resolve()), *extra])
    return {'allocation': job, 'active_steps': busy, 'other_live_steps': other_live_steps, 'source': str(source),
            'launcher_sha256': source_hash, 'resume_from': str(root),
            'checkpoint_report': report, 'replay': replay, 'wandb_run_id': run_id,
            'wandb_url': state['wandb_url'], 'command': command, 'verify_resume_only': verification}, state


def verification_receipt(job_id, gpus=4):
    return RUNTIME / 'logs' / f'resume-{job_id}-{gpus}gpu-verification.json'


def verification_identity(plan):
    return {key: plan[key] for key in ('source', 'launcher_sha256', 'resume_from')} | {
        'checkpoint': plan['checkpoint_report']['checkpoint'],
        'optimizer_updates': plan['checkpoint_report']['completed_optimizer_updates'],
        'gpus': plan['allocation'].get('gpus', 2),
        'job_id': plan['allocation']['job_id'],
    }


def validate_verification_receipt(plan, job_id):
    path = verification_receipt(job_id)
    if not path.is_file() or read_json(path).get('identity') != verification_identity(plan):
        raise ValueError('Four-GPU continuation requires a successful --verify-resume-only --launch for this checkpoint, source and allocation first.')


def record_verification(plan, job_id, run):
    report = read_json(run / 'resume_verification.json')
    expected = plan['checkpoint_report']['iteration']
    if (report.get('full_model_and_optimizer_load') != 'passed'
            or report.get('loaded_iteration') != expected
            or report.get('next_rollout_id') != expected + 1
            or report.get('gpus') != plan['allocation']['gpus']
            or report.get('source_checkpoint_root') != plan['resume_from']
            or report.get('optimizer_updates_executed') != 0
            or report.get('browser_collections_executed') != 0):
        raise ValueError('GPU verification report does not match the requested checkpoint/topology.')
    receipt = verification_receipt(job_id, plan['allocation']['gpus'])
    write_json(receipt, {'identity': verification_identity(plan), 'verification_run': str(run), 'report': report})
    print(f'GPU restore verified; receipt: {receipt}', flush=True)


def launch(plan, state, args):
    if plan.get('other_live_steps') and not plan.get('verify_resume_only', False):
        raise ValueError('The recorded training run is still active in another allocation; stop its trainer at a checkpoint boundary before continuing the same lineage.')
    if plan['active_steps']:
        raise ValueError('Allocation already has active steps; refusing to launch a duplicate trainer: '
                         + ', '.join(plan['active_steps']))
    env = clean_environment()
    if plan['replay']:
        replay = plan['replay']
        env.update(OPENWEBRL_REPLAY_FIRST_BATCH=replay['batch'],
                   OPENWEBRL_REPLAY_ROLLOUT_ID=str(replay['rollout_id']),
                   OPENWEBRL_REPLAY_CONSUMED_GROUPS=str(replay['consumed_groups']))
    verification = plan.get('verify_resume_only', False)
    if not verification and plan['allocation'].get('gpus', 2) == 4:
        validate_verification_receipt(plan, args.job_id)
    stamp = time.strftime('%Y%m%dT%H%M%S', time.gmtime())
    prefix = RUNTIME / 'logs' / f'resume-{args.job_id}-{stamp}'
    prefix.parent.mkdir(parents=True, exist_ok=True)
    write_json(prefix.with_suffix('.json'), plan)
    profile = 'resume-check' if verification else 'reference'
    pattern = f'openwebrl-4b-{profile}-{args.job_id}-*'
    before = set((RUNTIME / 'runs').glob(pattern))
    # Foreground supervisor: survives terminal disconnect when invoked with nohup.
    with prefix.with_suffix('.log').open('x') as output:
        proc = subprocess.Popen(plan['command'], env=env, stdout=output, stderr=subprocess.STDOUT)
        monitor = None
        monitor_log = None
        run = None
        print(f'Launcher log: {prefix.with_suffix(".log")}', flush=True)
        while proc.poll() is None:
            if run is None:
                found = set((RUNTIME / 'runs').glob(pattern)) - before
                ready = [p for p in found if (p / 'launch_manifest.json').exists()]
                if len(ready) == 1:
                    run = ready[0]
                    manifest = read_json(run / 'launch_manifest.json')
                    if manifest['resume_from'] != plan['resume_from']:
                        raise RuntimeError('Unexpected concurrent launcher detected; inspect allocation manually.')
                    updated = dict(state)
                    updated.update(run_directory=str(run), allocation=args.job_id, host=plan['allocation']['host'],
                                   source_directory=plan['source'], wandb_run_id=plan['wandb_run_id'],
                                   progress=str(run / 'progress.log'), health=str(run / 'health.jsonl'),
                                   launcher_log=str(prefix.with_suffix('.log')), allocation_status='RUNNING',
                                   last_valid_checkpoint=plan['checkpoint_report']['checkpoint'],
                                   durable_optimizer_updates=plan['checkpoint_report']['completed_optimizer_updates'],
                                   status_note='Resume launched; inspect training.log for actual GPU restore and checkpoint progress.')
                    report_path = run / 'resume_checkpoint_validation.json'
                    write_json(report_path, plan['checkpoint_report'])
                    updated['last_valid_checkpoint_report'] = str(report_path)
                    verified = state.get('full_resume_verified_checkpoint') == plan['checkpoint_report']['checkpoint']
                    updated['full_resume_verified'] = bool(verified and state.get('full_resume_verified'))
                    updated['last_valid_checkpoint_full_tensor_reload_verified'] = updated['full_resume_verified']
                    updated['pending_replay_batch'] = plan['replay']['batch'] if plan['replay'] else None
                    updated['pending_replay_provenance'] = None
                    if plan['replay']:
                        provenance = Path(plan['replay']['batch']).with_suffix('.provenance.json')
                        if provenance.exists():
                            updated['pending_replay_provenance'] = str(provenance)
                    if not verification:
                        write_json(args.state, updated)
                    write_json(run / 'resume_plan.json', plan)
                    monitor_log = prefix.with_suffix('.health.log').open('x')
                    command = step_command(args.job_id, 1, plan['allocation'].get('gpus', 2)) + source_command(Path(plan['source']), [
                        'python', str(REPO / 'scripts/monitor_baseline.py'), str(run),
                        '--seconds', str(plan['allocation']['maximum_seconds'] + 120)])
                    monitor = subprocess.Popen(command, env=env, stdout=monitor_log, stderr=subprocess.STDOUT)
                    tracking = 'Offline checkpoint verification' if verification else f'W&B: {plan["wandb_url"]}'
                    print(f'Run directory: {run}\n{tracking}', flush=True)
            if run and (run / 'progress.log').exists():
                lines = (run / 'progress.log').read_text(errors='replace').splitlines()
                if lines:
                    print(lines[-1], flush=True)
            time.sleep(30)
        write_json(prefix.with_suffix('.exit.json'), {'launcher_exit_code': proc.returncode, 'run_directory': str(run) if run else None})
        if verification and proc.returncode == 0:
            if run is None:
                raise ValueError('Verification exited without a recorded run directory.')
            record_verification(plan, args.job_id, run)
            if monitor is not None:
                monitor.wait(timeout=45)
        if monitor_log:
            monitor_log.close()
        # The read-only recorder exits on exit_status or its own allocation-bounded timer.
        print(f'Launcher exited {proc.returncode}; inspect logs and validate the latest checkpoint before reporting progress.', flush=True)
        return proc.returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job-id', required=True, help='Existing allocation explicitly authorized by the user')
    parser.add_argument('--state', type=Path, default=DEFAULT_STATE)
    parser.add_argument('--source', type=Path, help='Preserved, tested reference source; defaults to persistent pointer')
    parser.add_argument('--wandb-run-id', help='Optional assertion against the recorded W&B ID')
    parser.add_argument('--env-file', type=Path, default=REPO / '.env')
    parser.add_argument('--gpus', type=int, choices=[2, 4], default=2, help='Use TP2 or TP4 in the existing allocation')
    parser.add_argument('--verify-resume-only', action='store_true', help='Restore on GPUs without training, browsers, online W&B, or changing the run pointer')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--launch', action='store_true')
    mode.add_argument('--dry-run', action='store_true', help='Read-only preflight (default)')
    args = parser.parse_args()
    if not re.fullmatch(r'\d+', args.job_id):
        parser.error('--job-id must be a numeric existing allocation ID')
    try:
        if not args.launch:
            plan, _ = prepare(args)
            print(json.dumps(plan, indent=2))
            return 0
        # Guard simultaneous invocations before checking active Slurm steps.
        lock = RUNTIME / f'resume-{args.job_id}.lock'
        with lock.open('a') as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            plan, state = prepare(args)
            print(json.dumps(plan, indent=2), flush=True)
            return launch(plan, state, args)
    except (ValueError, OSError, KeyError, subprocess.CalledProcessError, zipfile.BadZipFile) as exc:
        # Do not dump subprocess output, environments, or credential-bearing configs.
        print(f'Resume stopped: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
