"""Durable JSON and namespace mutual exclusion for shared experiment state.

A holder that crashes leaves the lock in place. There is no lease expiry or
automatic stale-lock takeover; a timeout stops admission for diagnosis.
"""
from contextlib import contextmanager
from pathlib import Path
import json
import os
import threading
import time
import uuid

_locks = {}
_guard = threading.Lock()


def writable(path):
    path = Path(path).resolve()
    if path.is_relative_to(Path('/gpfs/home/zixianma')):
        raise ValueError('Home is read-only')
    return path


@contextmanager
def namespace_lock(root, *, timeout=30):
    root = writable(root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / 'financial.namespace-lock'
    with _guard:
        lock = _locks.setdefault(str(path), threading.RLock())
    deadline = time.monotonic() + timeout
    if not lock.acquire(timeout=max(0, timeout)):
        raise TimeoutError('Shared state thread lock timed out')
    acquired = False
    try:
        while True:
            try:
                path.mkdir()
                acquired = True
                break
            except FileExistsError:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError('Shared namespace lock timed out; no stale takeover')
                time.sleep(min(.01, remaining))
        if time.monotonic() > deadline:
            raise TimeoutError('Shared namespace acquisition exceeded its deadline')
        yield
    finally:
        try:
            if acquired:
                path.rmdir()
        finally:
            lock.release()


def atomic_write(path, value):
    path = Path(path).absolute()
    writable(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + str(os.getpid()) + '.' + uuid.uuid4().hex + '.tmp')
    with temporary.open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, default=str)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
