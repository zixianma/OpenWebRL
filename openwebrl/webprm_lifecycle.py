"""External episode watchdogs; preserve outcomes and terminate owned browsers."""
from __future__ import annotations

import ctypes
import errno
import json
import math
import multiprocessing
from multiprocessing import resource_tracker
import os
from pathlib import Path
import signal
import time
import uuid

from .jev_eval import append_json, write_json


EPISODE_WATCHDOG = dict(browser_phase_timeout_seconds=180, cleanup_timeout_seconds=20,
                        termination_grace_seconds=2, poll_seconds=.25)


def mark_episode_phase(config, phase, **details):
    """A phase starts once per operation, never as a liveness-only heartbeat."""
    if '_episode_phase_path' not in config:
        return
    record = dict(phase=phase, started_at=time.time(), pid=os.getpid(),
                  run_token=config['_episode_run_token'],
                  identity_sha256=config['identity_sha256'], **details)
    path = Path(config['_episode_phase_path'])
    write_json(path, record)
    append_json(path.with_name('lifecycle.jsonl'), record)


def save_outcome_and_close(result, folder, env, config):
    """Persist the verdict/error before cleanup, which the parent can interrupt."""
    result.update(finished_at=time.time(), cleanup_status='pending')
    write_json(Path(folder) / 'result.json', result)
    cleanup_started_at = time.time()
    mark_episode_phase(config, 'cleanup', cleanup_started_at=cleanup_started_at)
    try:
        if env is not None:
            env.close()
        result['cleanup_status'] = 'completed'
    except Exception as error:
        result.update(cleanup_status='error', browser_close_error=type(error).__name__)
    result['cleanup_finished_at'] = time.time()
    write_json(Path(folder) / 'result.json', result)
    mark_episode_phase(config, 'completed', cleanup_started_at=cleanup_started_at)


def _process(pid):
    stat = Path(f'/proc/{pid}/stat').read_text()
    fields = stat[stat.rfind(')') + 2:].split()
    return dict(state=fields[0], ppid=int(fields[1]), pgrp=int(fields[2]),
                session=int(fields[3]), start_ticks=int(fields[19]))


def _processes():
    # procfs can transiently reject enumeration while children are exiting.
    # Never convert a failed scan into an empty, apparently clean process tree.
    for attempt in range(3):
        try:
            found = {}
            for entry in Path('/proc').iterdir():
                if not entry.name.isdigit():
                    continue
                try:
                    found[int(entry.name)] = _process(int(entry.name))
                except (FileNotFoundError, ProcessLookupError):
                    continue
            return found
        except OSError as error:
            if error.errno not in (errno.ENOMEM, errno.EINTR) or attempt == 2:
                raise
            time.sleep(.02)


class OwnedEpisodeTree:
    """Track descendants, including browser helpers that create new sessions.

    The supervisor is a Linux subreaper for this episode. Its existing children
    are excluded; newly adopted descendants remain attributable after a browser
    double-forks. PID start times prevent signaling a reused PID.
    """
    def __init__(self, child, baseline):
        self.child, self.baseline = child, baseline
        self.parent = os.getpid()
        self.owned = {}
        self.seen = set()

    def refresh(self):
        current = _processes()
        selected = ({self.child.pid} if self.child.pid not in self.owned or
                    current.get(self.child.pid, {}).get('start_ticks') == self.owned[self.child.pid] else set())
        selected.update(pid for pid, value in current.items()
                        if value['ppid'] == self.parent and pid not in self.baseline)
        selected.update(pid for pid, value in current.items()
                        if value['session'] == self.child.pid)
        selected.update(pid for pid, start in self.owned.items()
                        if current.get(pid, {}).get('start_ticks') == start)
        while True:
            descendants = {pid for pid, value in current.items() if value['ppid'] in selected}
            if descendants <= selected:
                break
            selected.update(descendants)
        for pid in selected:
            if pid in current and pid not in self.baseline and pid != self.parent:
                if pid in self.owned and self.owned[pid] != current[pid]['start_ticks']:
                    continue
                self.owned[pid] = current[pid]['start_ticks']
                self.seen.add(pid)
        return {pid: current[pid] for pid, start in self.owned.items()
                if current.get(pid, {}).get('start_ticks') == start}

    def signal(self, sig):
        current = self.refresh()
        # The child creates this group itself; never signal the supervisor's
        # inherited group if startup failed before setsid().
        if any(row['pgrp'] == self.child.pid for row in current.values()):
            try:
                os.killpg(self.child.pid, sig)
            except ProcessLookupError:
                pass
        for pid, row in current.items():
            try:
                if _process(pid)['start_ticks'] == row['start_ticks']:
                    os.kill(pid, sig)
            except (FileNotFoundError, ProcessLookupError):
                pass

    def reap(self):
        self.child.join(timeout=0)
        for pid in self.refresh():
            if pid == self.child.pid:
                continue
            try:
                os.waitpid(pid, os.WNOHANG)
            except ChildProcessError:
                pass

    def stop(self, grace, *, reap_deadline=None):
        if reap_deadline is not None and (not math.isfinite(reap_deadline) or reap_deadline < 0):
            raise ValueError('Finite monotonic owned-reap deadline required')
        try:
            return self._stop_tracked(grace, reap_deadline=reap_deadline)
        except OSError as error:
            # A persistent procfs failure must not bypass termination of the
            # child and already identified helpers. Verification stays failed,
            # so the caller persists evidence and stops the whole collection.
            problems = [f'{type(error).__name__}: {error}']
            for sig in (signal.SIGTERM, signal.SIGKILL):
                if self.child.is_alive():
                    try:
                        if os.getpgid(self.child.pid) == self.child.pid:
                            os.killpg(self.child.pid, sig)
                        os.kill(self.child.pid, sig)
                    except ProcessLookupError:
                        pass
                for pid, start in self.owned.items():
                    try:
                        if _process(pid)['start_ticks'] == start:
                            os.kill(pid, sig)
                    except (FileNotFoundError, ProcessLookupError):
                        pass
                    except OSError as failure:
                        problems.append(f'PID {pid}: {failure}')
                self.child.join(timeout=max(1., grace))
            for pid in self.owned:
                if pid != self.child.pid:
                    try:
                        os.waitpid(pid, os.WNOHANG)
                    except ChildProcessError:
                        pass
            return dict(owned_pids=sorted(self.seen), child_exitcode=self.child.exitcode,
                remaining_owned_processes={'verification_failed': problems})

    def _stop_tracked(self, grace, *, reap_deadline=None):
        self.signal(signal.SIGTERM)
        until = time.monotonic() + grace
        while self.refresh() and time.monotonic() < until:
            self.reap()
            time.sleep(min(.05, grace))
        self.signal(signal.SIGKILL)
        until = time.monotonic() + max(1., grace)
        while self.refresh() and time.monotonic() < until:
            self.reap()
            time.sleep(.02)
        self.reap()
        remaining = self.refresh()
        initial_remaining = remaining
        # SIGKILL is pending while a process is in uninterruptible I/O. Keep
        # ownership and reap within the caller's fixed shutdown deadline; no
        # new episode, provider call, signal grace or deadline refresh occurs.
        while remaining and reap_deadline is not None and time.monotonic() < reap_deadline:
            self.reap()
            time.sleep(min(.05, max(0., reap_deadline - time.monotonic())))
            remaining = self.refresh()
        receipt = dict(owned_pids=sorted(self.seen), remaining_owned_processes=remaining,
                       child_exitcode=self.child.exitcode)
        if initial_remaining and reap_deadline is not None:
            receipt.update(initial_stop_remaining_processes=initial_remaining,
                           bounded_reap_deadline=reap_deadline,
                           recovered_after_initial_stop=not remaining)
        return receipt


def _subreaper(enabled=None):
    libc = ctypes.CDLL(None, use_errno=True)
    previous = ctypes.c_int()
    if libc.prctl(37, ctypes.byref(previous), 0, 0, 0):  # PR_GET_CHILD_SUBREAPER
        raise OSError(ctypes.get_errno(), 'Cannot query child subreaper')
    if enabled is not None and libc.prctl(36, int(enabled), 0, 0, 0):
        raise OSError(ctypes.get_errno(), 'Cannot configure child subreaper')
    return previous.value


def _child_entry(target, task, arm, config):
    os.setsid()
    mark_episode_phase(config, 'startup')
    target(task, arm, config)


def run_supervised_episode(task, arm, config, target):
    """Run one cell once; no provider calls, retries or budget edits in parent."""
    folder = Path(config['output']) / task['task_id'] / arm
    path = folder / 'result.json'
    identity = config['identity_sha256']
    # A parent-generated failure still belongs to the assigned actor/selector.
    # Keep these immutable identities when the child cannot write its result.
    fixed_identity = {key: config[key] for key in ('actor_model', 'selector') if key in config}
    if path.exists():
        previous = json.loads(path.read_text())
        if previous.get('identity_sha256') != identity:
            raise ValueError('Episode resume identity changed')
        return previous
    settings = config['episode_watchdog']
    if set(settings) != set(EPISODE_WATCHDOG) or any(
            not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0
            for value in settings.values()):
        raise ValueError('Explicit positive episode watchdog settings required')
    if not math.isfinite(config['deadline']):
        raise ValueError('Finite allocation deadline required')
    folder.mkdir(parents=True, exist_ok=True)
    started = time.time()
    if started >= config['deadline']:
        result = dict(task_id=task['task_id'], arm=arm, success=0, valid=False, **fixed_identity,
            attempted=False, identity_sha256=identity, started_at=started,
            finished_at=started, turns=0, error_type='TimeoutError',
            error='Pilot deadline reached before episode process startup',
            watchdog=dict(reason='allocation_deadline_before_start', settings=settings,
                          owned_pids=[], remaining_owned_processes={}, child_exitcode=None))
        write_json(folder / 'watchdog.json', result['watchdog'])
        write_json(path, result)
        return result
    token = uuid.uuid4().hex
    child_config = dict(config, _episode_run_token=token,
                        _episode_phase_path=str(folder / 'phase.json'))
    phase = dict(phase='startup', started_at=started, run_token=token,
                 identity_sha256=identity)
    child = None
    tree = None
    reason = None
    receipt = dict(owned_pids=[], remaining_owned_processes={}, child_exitcode=None)
    supervisor_error = None
    previous_subreaper = _subreaper(True)
    stop_signal = None
    def request_stop(signum, frame):
        nonlocal stop_signal
        stop_signal = signum
    previous_handlers = {sig: signal.signal(sig, request_stop)
                         for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        # Establish the multiprocessing service before identifying preexisting
        # children, so it is never mistaken for an adopted browser helper.
        resource_tracker.ensure_running()
        baseline = {pid for pid, row in _processes().items() if row['ppid'] == os.getpid()}
        child = multiprocessing.get_context('spawn').Process(
            target=_child_entry, args=(target, task, arm, child_config))
        child.start()
        tree = OwnedEpisodeTree(child, baseline)
        while True:
            tree.refresh()
            phase_path = Path(child_config['_episode_phase_path'])
            if phase_path.exists():
                observed = json.loads(phase_path.read_text())
                if observed.get('run_token') == token:
                    phase = observed
            if not child.is_alive():
                break
            now = time.time()
            if stop_signal is not None:
                reason = 'supervisor_termination'
                break
            if now >= config['deadline']:
                reason = 'allocation_deadline'
                break
            cleanup = phase['phase'] in {'cleanup', 'completed'}
            limit = settings['cleanup_timeout_seconds'] if cleanup else settings['browser_phase_timeout_seconds']
            phase_started = phase.get('cleanup_started_at', phase['started_at']) if cleanup else phase['started_at']
            if now - phase_started >= limit:
                reason = 'cleanup_timeout' if cleanup else 'phase_timeout'
                break
            time.sleep(min(settings['poll_seconds'], max(.001, config['deadline'] - now),
                           max(.001, limit - (now - phase_started))))
    except Exception as error:
        supervisor_error = f'{type(error).__name__}: {error}'
        reason = 'supervisor_error'
    finally:
        try:
            if tree is not None:
                # Reaping gets at most the existing cleanup allowance, bounded by
                # the allocation deadline. The phase watchdog and signal grace
                # remain unchanged; an unreaped tree still stops collection.
                reap_seconds = min(settings['cleanup_timeout_seconds'],
                                   max(0., config['deadline'] - time.time()))
                receipt = tree.stop(settings['termination_grace_seconds'],
                                    reap_deadline=time.monotonic() + reap_seconds)
        finally:
            _subreaper(previous_subreaper)
            for sig, handler in previous_handlers.items():
                signal.signal(sig, handler)
    if path.exists():
        result = json.loads(path.read_text())
        if result.get('identity_sha256') != identity:
            raise ValueError('Child result identity changed')
    else:
        rollout_path = folder / 'rollout.json'
        turns = len(json.loads(rollout_path.read_text())) if rollout_path.exists() else 0
        result = dict(task_id=task['task_id'], arm=arm, success=0, valid=False, **fixed_identity,
                      attempted=child is not None and child.pid is not None,
                      identity_sha256=identity, started_at=started,
                      finished_at=time.time(), turns=turns,
                      error_type=('EpisodeSupervisorError' if supervisor_error else
                                  'EpisodeWatchdogTimeout' if reason else 'EpisodeWorkerExited'),
                      error=(f"{reason} in {phase['phase']}" if reason else
                             f"Episode child exited without result (exit {receipt['child_exitcode']})"))
    if result.get('cleanup_status') == 'pending':
        result.update(cleanup_status='interrupted', browser_close_error=reason or 'worker_exit')
    result['watchdog'] = dict(reason=reason, last_phase=phase, settings=settings,
                             supervisor_signal=stop_signal, supervisor_error=supervisor_error, **receipt)
    write_json(folder / 'watchdog.json', result['watchdog'])
    write_json(path, result)
    if receipt.get('remaining_owned_processes'):
        raise RuntimeError('Episode watchdog could not reap all owned processes; collection must stop')
    if supervisor_error:
        raise RuntimeError('Episode watchdog supervision failed; collection must stop: ' + supervisor_error)
    return result
