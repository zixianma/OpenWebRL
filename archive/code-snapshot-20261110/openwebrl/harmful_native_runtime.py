"""Prospective native-evaluation process, deadline, and storage supervision.

This module does not authorize allocations, launch applications, grade safety, or
make the historical native controller relocatable. Its caller must bind approved
commands, paths, growth bounds, and allocation identity before launching work.
All watchdog decisions are in memory. Storage traversal runs in a separately
owned process; callers persist receipts after closing processes, not in the
deadline loop. Linux subreaper ownership requires one supervisor per process.

A cleanup reserve guarantees time *available for cleanup*, not that Linux can
kill a task blocked in uninterruptible kernel I/O. Such tasks, unreadable process
identity, and incomplete scans remain explicit failures. Storage samples are
non-atomic observations: conservative, enforced writer growth bounds are required
in addition to periodic scans. A sample alone is not a hard storage quota.
"""
from __future__ import annotations

import argparse
import ctypes
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import time


class RuntimeRefused(RuntimeError):
    """No launch or success claim is permitted under the current evidence."""


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be finite and positive')
    return value


def _now(value):
    value = time.monotonic() if value is None else value
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError('Monotonic time must be finite and nonnegative')
    return value


def private_path(path):
    """Guard runtime writes without broadening the existing private path policy."""
    raw = Path(path)
    forbidden = Path('/gpfs/home/zixianma')
    roots = (Path('/tmp'), Path('/gpfs/scrubbed/zixianma/openwebrl-runtime'))
    if not raw.is_absolute() or '..' in raw.parts or raw == forbidden or forbidden in raw.parents:
        raise ValueError('Require absolute private path outside protected home')
    if any(part.is_symlink() for part in (raw, *raw.parents)):
        raise ValueError('Symlink path component refused')
    if not any(root in raw.parents for root in roots):
        raise ValueError('Require a private runtime or /tmp child')
    return raw


@dataclass(frozen=True)
class DeadlineBudget:
    """All deadlines use monotonic time; reserves cannot be spent on work."""

    hard_deadline: float
    cleanup_reserve: float
    diagnostic_reserve: float = 0.0

    def __post_init__(self):
        _positive(self.hard_deadline, 'hard_deadline')
        _positive(self.cleanup_reserve, 'cleanup_reserve')
        if (isinstance(self.diagnostic_reserve, bool) or not math.isfinite(self.diagnostic_reserve) or
                self.diagnostic_reserve < 0):
            raise ValueError('Invalid diagnostic reserve')

    @classmethod
    def from_epoch(cls, hard_epoch, cleanup_reserve, diagnostic_reserve=0.0):
        # Snapshot once. Later wall-clock jumps cannot renew any budget.
        return cls(time.monotonic() + hard_epoch - time.time(), cleanup_reserve, diagnostic_reserve)

    @property
    def diagnostic_deadline(self):
        return self.hard_deadline - self.cleanup_reserve

    @property
    def work_deadline(self):
        return self.diagnostic_deadline - self.diagnostic_reserve

    def admit(self, remaining_stage_caps, now=None):
        """Require the entire remaining serial chain, including normal closure.

        Normal native END/export and application shutdown belong in this chain;
        the emergency cleanup reserve is additional, not their normal allowance.
        Parallel plans must supply a conservative critical-path bound separately.
        """
        now = _now(now)
        caps = tuple(_positive(x, 'stage cap') for x in remaining_stage_caps)
        if not caps or now + sum(caps) > self.work_deadline:
            raise RuntimeRefused('Insufficient work budget for complete remaining chain')
        return self.work_deadline - now - sum(caps)

    def stage_deadline(self, seconds, now=None, *, diagnostic=False):
        now = _now(now)
        end = self.diagnostic_deadline if diagnostic else self.work_deadline
        if now >= end:
            raise RuntimeRefused('Reserved work/diagnostic deadline exhausted')
        return min(now + _positive(seconds, 'stage seconds'), end)


class ScanChurn(RuntimeError):
    pass


def _under(path, roots):
    return any(path == root or root in path.parents for root in roots)


def scan_storage(roots, ephemeral_roots=(), attempts=3):
    """Count allocated bytes, without following symlinks or crossing devices.

    ENOENT within explicitly named temporary subtrees retries the *whole* scan.
    No partial total, unreadable entry, unexplained disappearance, or failed final
    attempt is accepted. This function is deliberately only called in a worker.
    """
    if not roots or not isinstance(attempts, int) or not 1 <= attempts <= 3:
        raise ValueError('Invalid storage scan specification')
    roots = {name: private_path(path) for name, path in roots.items()}
    ephemeral = tuple(private_path(path) for path in ephemeral_roots)
    if any(not _under(path, tuple(roots.values())) for path in ephemeral):
        raise ValueError('Ephemeral scope outside sampled roots')
    started = time.monotonic()
    churn = []
    for attempt in range(1, attempts + 1):
        totals = {}
        try:
            for name, root in roots.items():
                root_stat = root.lstat()  # A missing root is never temporary churn.
                if not stat.S_ISDIR(root_stat.st_mode):
                    raise RuntimeRefused('Storage root is not a directory')
                seen, total = set(), 0

                def visit(path, parent_fd=None, entry_name=None):
                    nonlocal total
                    try:
                        item = (os.stat(entry_name, dir_fd=parent_fd, follow_symlinks=False)
                                if parent_fd is not None else path.lstat())
                        identity = (item.st_dev, item.st_ino)
                        if item.st_dev != root_stat.st_dev:
                            raise RuntimeRefused('Unexpected mount in sampled root')
                        if identity in seen:
                            return
                        seen.add(identity)
                        total += item.st_blocks * 512
                        if stat.S_ISDIR(item.st_mode):
                            fd = os.open(entry_name if parent_fd is not None else path,
                                         os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent_fd)
                            try:
                                opened = os.fstat(fd)
                                if (opened.st_dev, opened.st_ino) != identity:
                                    raise RuntimeRefused('Directory changed identity during storage scan')
                                with os.scandir(fd) as entries:
                                    for entry in entries:
                                        visit(path / entry.name, fd, entry.name)
                            finally:
                                os.close(fd)
                    except FileNotFoundError as error:
                        if path != root and _under(path, ephemeral):
                            raise ScanChurn(str(path)) from error
                        raise
                visit(root)
                totals[name] = total
            return {'status': 'complete_scan', 'bytes': totals, 'attempts': attempt,
                    'started_monotonic': started, 'finished_monotonic': time.monotonic(),
                    'churn_retries': len(churn), 'atomic_snapshot': False}
        except ScanChurn as error:
            churn.append(str(error))
    raise RuntimeRefused(f'Incomplete storage scan after {attempts} bounded churn attempts')


class StorageGate:
    """Fail-closed admission using fresh complete scans and writer reservations.

    Reservations are worst-case *additional* growth for every running writer,
    including samplers/logs/retention. Enforcing those bounds is an adapter gate.
    A released reservation persists until a later full scan has observed it.
    """

    def __init__(self, limits, max_age_seconds, headroom=None):
        self.limits = self._amounts(limits)
        self.headroom = self._amounts(headroom or dict.fromkeys(limits, 0))
        self.max_age = _positive(max_age_seconds, 'storage maximum age')
        self.sample = None
        self.failure = 'No complete initial storage scan'
        self.reservations = {}

    def _amounts(self, values):
        if not values or any(type(v) is not int or v < 0 for v in values.values()):
            raise ValueError('Storage bytes must be nonnegative integers')
        if hasattr(self, 'limits') and set(values) != set(self.limits):
            raise ValueError('Storage root keys differ')
        return dict(values)

    def invalidate(self, reason):
        self.failure = str(reason)

    def accept(self, sample, now=None):
        try:
            now = _now(now)
            if sample.get('status') != 'complete_scan':
                raise RuntimeRefused('Storage scan incomplete')
            values = self._amounts(sample['bytes'])
            start, finish = sample['started_monotonic'], sample['finished_monotonic']
            if not 0 <= _now(start) <= _now(finish) <= now:
                raise RuntimeRefused('Storage scan timestamps invalid')
            if self.sample and start < self.sample['started_monotonic']:
                raise RuntimeRefused('Storage scan replayed out of order')
            self.sample = dict(sample, bytes=values)
            self.failure = None
            self.reservations = {key: value for key, value in self.reservations.items()
                                 if value['released'] is None or start < value['released']}
            self.check(now)
        except (KeyError, TypeError, ValueError, RuntimeRefused) as error:
            self.invalidate(error)
            raise RuntimeRefused(str(error)) from error

    def check(self, now=None, additional=None):
        now = _now(now)
        if self.failure or self.sample is None:
            raise RuntimeRefused(self.failure or 'No storage sample')
        if now < self.sample['finished_monotonic']:
            raise RuntimeRefused('Storage observation is from the future')
        if now - self.sample['started_monotonic'] > self.max_age:
            raise RuntimeRefused('Complete storage scan is stale')
        extra = self._amounts(additional or dict.fromkeys(self.limits, 0))
        for name, cap in self.limits.items():
            total = (self.sample['bytes'][name] + self.headroom[name] + extra[name] +
                     sum(item['bytes'][name] for item in self.reservations.values()))
            if total > cap:
                raise RuntimeRefused('Storage cap including reserved growth would be exceeded')

    def reserve(self, owner, growth, now=None):
        if not owner or owner in self.reservations:
            raise ValueError('Reservation owner must be fresh')
        growth = self._amounts(growth)
        now = _now(now)
        self.check(now, growth)
        self.reservations[owner] = {'bytes': growth, 'released': None, 'created': now}

    def release(self, owner, now=None):
        now = _now(now)
        item = self.reservations[owner]
        if now < max(item['created'], self.sample['finished_monotonic']):
            raise ValueError('Reservation release predates its accounting evidence')
        if item['released'] is not None:
            raise ValueError('Reservation already released')
        item['released'] = now


def _process_identity(pid):
    raw = Path('/proc', str(pid), 'stat').read_text()
    fields = raw[raw.rfind(')') + 2:].split()
    return {'state': fields[0], 'ppid': int(fields[1]),
            'pgrp': int(fields[2]), 'session': int(fields[3]), 'start': int(fields[19])}


def _process_table():
    table = {}
    with os.scandir('/proc') as entries:
        for entry in entries:
            if not entry.name.isdigit():
                continue
            try:
                row = _process_identity(int(entry.name))
            except FileNotFoundError:
                continue
            table[int(entry.name)] = row
    return table


def _subreaper(value=None):
    libc = ctypes.CDLL(None, use_errno=True)
    old = ctypes.c_int()
    if libc.prctl(37, ctypes.byref(old), 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), 'Cannot inspect Linux subreaper')
    if value is not None and libc.prctl(36, int(value), 0, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), 'Cannot set Linux subreaper')
    return bool(old.value)


class ProcessSupervisor:
    """Own all workers/daemons/samplers in one dedicated controller process.

    Do not share the controller with unrelated child launchers or threads. Any
    new adopted child that cannot be attributed is retained as unknown ownership
    and causes stage failure; final cleanup still owns it. Receipts carry process
    completion/closure, never a native-capture or safety verdict.
    """

    _active = False

    def __init__(self, budget, *, poll_seconds=0.05, term_grace_seconds=1.0):
        if self.__class__._active:
            raise RuntimeRefused('Only one process supervisor per controller')
        self.budget = budget
        self.poll_seconds = _positive(poll_seconds, 'poll seconds')
        self.term_grace = _positive(term_grace_seconds, 'TERM grace')
        if self.term_grace * 2 >= self.budget.cleanup_reserve:
            raise ValueError('Cleanup reserve must exceed TERM and KILL observation allowances')
        self.baseline = _process_table()
        self.previous_subreaper = _subreaper(True)
        self.__class__._active = True
        self.roots, self.owned, self.history = {}, {}, {}
        self.critical = set()
        self.errors = []
        self.stopping = False
        self.finished = False
        self.handlers = {}
        for sig in (signal.SIGTERM, signal.SIGINT):
            self.handlers[sig] = signal.signal(sig, self._stop_signal)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        if not self.finished:
            self.final_closure = self.close()

    def _stop_signal(self, *_):
        self.stopping = True

    def start(self, label, argv, *, env=None, stdout=subprocess.DEVNULL, critical=False, launch_deadline=None):
        launch_deadline = self.budget.work_deadline if launch_deadline is None else _now(launch_deadline)
        if (not label or label in self.roots or self.stopping or self.finished or
                time.monotonic() >= min(launch_deadline, self.budget.work_deadline)):
            raise RuntimeRefused('Worker label reused or work deadline exhausted')
        if not argv or not all(isinstance(value, str) for value in argv):
            raise ValueError('Require a fixed argv list; no shell execution')
        proc = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=stdout, stderr=subprocess.STDOUT,
                                env=env, start_new_session=True, close_fds=True)
        self.roots[label] = proc
        if critical:
            self.critical.add(label)
        self.refresh()
        return proc

    def refresh(self):
        table = _process_table()  # An unreadable observation is never an empty table.
        self.owned = {pid: item for pid, item in self.owned.items()
                      if pid not in table or table[pid]['start'] == item[0]}

        def baseline_identity(pid, row):
            return self.baseline.get(pid, {}).get('start') == row['start']

        for label, proc in self.roots.items():
            row = table.get(proc.pid)
            if row and proc.pid not in self.owned and proc.returncode is None:
                self.owned[proc.pid] = (row['start'], label)
        changed = True
        while changed:
            changed = False
            for pid, row in table.items():
                if baseline_identity(pid, row) or pid == os.getpid() or pid in self.owned:
                    continue
                parent = self.owned.get(row['ppid'])
                if parent and table.get(row['ppid'], {}).get('start') == parent[0]:
                    self.owned[pid] = (row['start'], parent[1]); changed = True
        for pid, row in table.items():
            if row['ppid'] == os.getpid() and not baseline_identity(pid, row) and pid not in self.owned:
                self.owned[pid] = (row['start'], 'unattributed')
        self.history.update({(pid, start): label for pid, (start, label) in self.owned.items()})
        live = {}
        for pid, (start, label) in self.owned.items():
            row = table.get(pid)
            if row and row['start'] == start:
                live[pid] = dict(row, label=label)
        return live

    def _reap(self):
        roots = {proc.pid for proc in self.roots.values()}
        for proc in self.roots.values():
            proc.poll()
        for pid in self.owned:
            if pid not in roots:
                try:
                    os.waitpid(pid, os.WNOHANG)
                except ChildProcessError:
                    pass

    def _signal(self, rows, sig):
        # Identity must still match immediately before each individual signal.
        for pid, expected in rows.items():
            try:
                current = _process_identity(pid)
                if current['start'] == expected['start']:
                    os.kill(pid, sig)
            except (FileNotFoundError, ProcessLookupError):
                pass

    def close(self, *, label=None, deadline=None):
        """TERM then KILL under the existing hard cutoff; never renew a budget."""
        deadline = min(self.budget.hard_deadline, self.budget.hard_deadline if deadline is None else deadline)
        term_end = min(deadline, time.monotonic() + self.term_grace)
        sent = set()
        last = {}
        failed = False
        while True:
            try:
                self._reap()
                all_rows = self.refresh()
                if label is not None and any(row['label'] == 'unattributed' for row in all_rows.values()):
                    failed = True
                last = {pid: row for pid, row in all_rows.items()
                        if label is None or row['label'] in (label, 'unattributed')}
                if not last:
                    break
                now = time.monotonic()
                sig = signal.SIGTERM if now < term_end else signal.SIGKILL
                targets = {pid: row for pid, row in last.items() if (pid, row['start'], sig) not in sent}
                self._signal(targets, sig)
                sent.update((pid, row['start'], sig) for pid, row in targets.items())
                if now >= deadline:
                    break
                time.sleep(min(self.poll_seconds, max(0, deadline - time.monotonic())))
            except Exception as error:
                self.errors.append(type(error).__name__ + ': ' + str(error))
                failed = True
                # No identity guess or empty-process claim when /proc is unreadable.
                break
        selected = {f'{pid}:{start}': {'pid': pid, 'start_ticks': start, 'label': owner}
                    for (pid, start), owner in self.history.items()
                    if label is None or owner in (label, 'unattributed')}
        receipt = {'status': 'closure_verified' if not last and not failed and not self.errors else 'closure_unknown',
                   'owned_processes': selected, 'remaining_owned_processes': {str(k): v for k, v in last.items()},
                   'observation_errors': list(self.errors), 'finished_monotonic': time.monotonic(),
                   'hard_deadline_monotonic': self.budget.hard_deadline,
                   'root_exitcodes': {key: value.returncode for key, value in self.roots.items()
                                      if label is None or key == label}}
        if label is None:
            for proc in self.roots.values():
                if proc.stdout is not None:
                    proc.stdout.close()
            for sig, handler in self.handlers.items():
                signal.signal(sig, handler)
            _subreaper(self.previous_subreaper)
            self.__class__._active = False
            self.finished = True
        elif receipt['status'] == 'closure_verified':
            self.critical.discard(label)
        return receipt

    def run_stage(self, label, argv, seconds, *, storage=None, sampler=None, env=None,
                  stdout=subprocess.DEVNULL, watchdog_tick=None, closing_labels=(), absolute_deadline=None):
        started = time.monotonic()
        deadline = self.budget.stage_deadline(seconds, started)
        if absolute_deadline is not None:
            deadline = min(deadline, _now(absolute_deadline))
        reason, code, proc = None, None, None
        try:
            if not set(closing_labels) <= self.critical:
                raise RuntimeRefused('Intentional close does not name live required daemons')
            if storage:
                storage.check(started)
            if watchdog_tick:
                watchdog_tick()
            proc = self.start(label, argv, env=env, stdout=stdout, launch_deadline=deadline)
            while True:
                now = time.monotonic()
                if self.stopping or now >= deadline:
                    reason = 'stop_requested' if self.stopping else 'stage_deadline'
                    break
                if sampler:
                    sampler.tick(self)
                if storage:
                    storage.check()
                if watchdog_tick:
                    watchdog_tick()
                if any(self.roots[key].poll() is not None for key in self.critical - set(closing_labels)):
                    raise RuntimeRefused('Required daemon exited during stage')
                live = self.refresh()
                if any(row['label'] == 'unattributed' for row in live.values()):
                    raise RuntimeRefused('Unattributed adopted process')
                code = proc.poll()
                if code is not None:
                    reason = 'child_exit'
                    break
                time.sleep(min(self.poll_seconds, max(0, deadline - time.monotonic())))
        except Exception as error:
            reason = type(error).__name__ + ': ' + str(error)
        work_finished = time.monotonic()
        if reason == 'child_exit' and work_finished >= deadline:
            reason = 'stage_deadline'
        # Local stage cleanup cannot consume the separate all-process reserve.
        # If it is incomplete, the caller must proceed straight to global close.
        closure = self.close(label=label, deadline=min(deadline, self.budget.diagnostic_deadline,
                                                       work_finished + 2 * self.term_grace))
        if closure['status'] != 'closure_verified':
            self.stopping = True
        finished = time.monotonic()
        return {'status': 'process_completed_and_closed' if reason == 'child_exit' and code == 0 and
                closure['status'] == 'closure_verified' and finished <= deadline else 'stage_failed_unknown',
                'reason': reason, 'returncode': code, 'started_monotonic': started,
                'work_finished_monotonic': work_finished,
                'finished_monotonic': finished, 'stage_deadline_monotonic': deadline,
                'observed_execution_window_seconds': work_finished - started,
                'process_closure_seconds': closure['finished_monotonic'] - work_finished,
                'closure': closure, 'native_capture_status': 'not_evaluated', 'safe_success': None}


class AsyncStorageSampler:
    """Nonblocking storage polling; a stalled sampler triggers failure, not waiting.

    Workers share the supervisor's ownership. An expired sampler is closed by the
    caller's reserved cleanup path. Output is small and bounded. No disk writes,
    reads of evidence files, or global du occur in tick().
    """

    def __init__(self, gate, roots, *, ephemeral_roots=(), scan_timeout=20, interval=30, worker_sha256=None):
        if set(roots) != set(gate.limits):
            raise ValueError('Sampler roots differ from storage gate')
        self.gate, self.roots = gate, dict(roots)
        self.ephemeral_roots = tuple(map(str, ephemeral_roots))
        self.scan_timeout = _positive(scan_timeout, 'scan timeout')
        self.interval = _positive(interval, 'scan interval')
        if self.interval + self.scan_timeout >= gate.max_age:
            raise ValueError('Storage age must cover scan interval and timeout')
        self.proc, self.buffer = None, bytearray()
        self.next_due, self.deadline, self.sequence = 0.0, 0.0, 0
        self.samples, self.failures = [], []
        self.worker_path = str(Path(__file__).absolute())
        if worker_sha256 is not None and (not isinstance(worker_sha256, str) or
                len(worker_sha256) != 64 or any(c not in '0123456789abcdef' for c in worker_sha256)):
            raise ValueError('Invalid frozen sampler source identity')
        self.worker_sha256 = worker_sha256

    def _argv(self):
        request = json.dumps({'roots': self.roots, 'ephemeral_roots': self.ephemeral_roots})
        if self.worker_sha256 is None:
            return [sys.executable, '-B', self.worker_path, '_scan', request]
        # Fixed infrastructure bootstrap checks and executes the same bytes inside
        # the bounded owned child. No filesystem read occurs in the watchdog.
        bootstrap = ("import hashlib,pathlib,sys; p=sys.argv[1]; expected=sys.argv[2]; "
                     "data=pathlib.Path(p).read_bytes()\n"
                     "if hashlib.sha256(data).hexdigest()!=expected: raise RuntimeError('Frozen sampler source changed')\n"
                     "sys.argv=[p,'_scan',sys.argv[3]]; "
                     "exec(compile(data,p,'exec'),{'__name__':'__main__','__file__':p})")
        return [sys.executable, '-B', '-c', bootstrap, self.worker_path, self.worker_sha256, request]

    def tick(self, supervisor):
        now = time.monotonic()
        try:
            if self.proc is None:
                if now < self.next_due:
                    return
                self.sequence += 1
                self.proc = supervisor.start(f'storage-scan-{self.sequence}', self._argv(), stdout=subprocess.PIPE)
                os.set_blocking(self.proc.stdout.fileno(), False)
                self.buffer = bytearray()
                self.deadline = now + self.scan_timeout
            if now >= self.deadline:
                raise RuntimeRefused('Storage sampler deadline exceeded')
            # Read only a bounded number of bytes; never wait for pipe EOF.
            try:
                data = os.read(self.proc.stdout.fileno(), 65537 - len(self.buffer))
            except BlockingIOError:
                data = b''
            self.buffer.extend(data)
            if len(self.buffer) > 65536:
                raise RuntimeRefused('Storage sampler output exceeded bound')
            code = self.proc.poll()
            if code is None:
                return
            # A completed process may have left its final small write in the pipe.
            while len(self.buffer) <= 65536:
                try:
                    data = os.read(self.proc.stdout.fileno(), 65537 - len(self.buffer))
                except BlockingIOError:
                    break
                if not data:
                    break
                self.buffer.extend(data)
            if code != 0 or len(self.buffer) > 65536:
                raise RuntimeRefused('Storage sampler failed; partial output rejected')
            sample = json.loads(self.buffer)
            if sample['started_monotonic'] < self.deadline - self.scan_timeout:
                raise RuntimeRefused('Storage sampler returned an old observation')
            self.gate.accept(sample)
            self.samples.append(sample)
            self.proc.stdout.close()
            self.proc = None
            self.next_due = time.monotonic() + self.interval
        except (OSError, ValueError, KeyError, RuntimeRefused) as error:
            self.gate.invalidate(error)
            self.failures.append({'at_monotonic': time.monotonic(), 'reason': str(error)})
            raise RuntimeRefused(str(error)) from error


def _main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['_scan'])
    parser.add_argument('request_json')
    args = parser.parse_args()
    request = json.loads(args.request_json)
    try:
        result = scan_storage(**request)
    except Exception as error:
        print(json.dumps({'status': 'failed_scan', 'error_type': type(error).__name__}), flush=True)
        return 1
    print(json.dumps(result), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(_main())
