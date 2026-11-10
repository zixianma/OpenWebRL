"""Owned, isolated local-browser processes for the action-credit pilot.

The parent never receives trajectory tensors over a pipe. Each child persists
its exact episode and verdict; the parent loads the committed private pointer.
Failure leaves the existing attempt for reconciliation, never implicit retry.
No scheduler allocation or API approval is supplied by this module.
"""
from __future__ import annotations

import argparse
import asyncio
from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import signal
import stat
import sys
import time
from types import SimpleNamespace
import uuid

from .action_credit import CollectionContract, digest
from .action_credit_collect import (EpisodeRequest, FrozenActor,
    physical_verifier_sha256, prefix_sha256)
from .action_credit_episode import load_pending_episode
from .action_credit_state import CampaignManifest, CampaignStore, _safe_path


_CHILD_REGISTRY = 'OPENWEBRL_ACTION_CREDIT_CHILD_REGISTRY'


def _process_identity(pid):
    """Linux birth identity, not just a reusable PID."""
    try:
        text = Path(f'/proc/{pid}/stat').read_text()
    except FileNotFoundError:
        return None
    fields = text[text.rfind(')') + 2:].split()
    return dict(pid=int(pid), state=fields[0], ppid=int(fields[1]),
                pgid=int(fields[2]), session=int(fields[3]), start_ticks=int(fields[19]))


def _group_members(pgid):
    result = []
    for path in Path('/proc').iterdir():
        if path.name.isdigit():
            try:
                if path.stat().st_uid != os.getuid():
                    continue
            except FileNotFoundError:
                continue
            item = _process_identity(int(path.name))
            if item and item['pgid'] == pgid and item['session'] == pgid and item['state'] != 'Z':
                result.append(item)
    return result


def _descendants(pid):
    """Exact current ancestry, including children that created new sessions."""
    processes = {}
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        try:
            if path.stat().st_uid != os.getuid():
                continue
        except FileNotFoundError:
            continue
        item = _process_identity(int(path.name))
        if item is not None:
            processes[item['pid']] = item
    owned = {pid}
    while True:
        added = {p for p, item in processes.items() if item['ppid'] in owned} - owned
        if not added:
            break
        owned.update(added)
    return [processes[p] for p in owned - {pid} if processes[p]['state'] != 'Z']


def _signal_identity(identity, sig):
    """A pidfd pins the process while its recorded birth identity is checked."""
    try:
        fd = os.pidfd_open(identity['pid'])
    except ProcessLookupError:
        return
    try:
        current = _process_identity(identity['pid'])
        if current is None:
            return
        if current['start_ticks'] != identity['start_ticks']:
            raise RuntimeError('Owned descendant PID was reused; refusing to signal')
        try:
            signal.pidfd_send_signal(fd, sig)
        except ProcessLookupError:
            pass
    finally:
        os.close(fd)


def _live_children(pid):
    # The guardian is single-threaded. This kernel child list also catches a
    # child just reparented while the /proc ancestry snapshot was being read.
    path = Path(f'/proc/{pid}/task/{pid}/children')
    return [item for child in path.read_text().split()
            if (item := _process_identity(int(child))) and item['state'] != 'Z']


def _registry_owner(registry):
    path = _safe_path(registry) / 'owner.json'
    info = path.stat()
    if info.st_uid != os.getuid() or info.st_mode & 0o077 or info.st_nlink != 1:
        raise ValueError('Detached process owner must be private and unaliased')
    return json.loads(path.read_text())


@contextmanager
def detached_process_scope(registry):
    """Wrap detached native launches only inside this isolated worker process.

    The guardian registers before launching the browser. Cancellation while
    asyncio creates the process therefore cannot lose ownership of a browser.
    The normal local transport still owns health checks and graceful shutdown.
    """
    original = asyncio.create_subprocess_exec

    async def create(*argv, **kwargs):
        if kwargs.get('start_new_session') is True:
            owner_path = _safe_path(registry) / 'owner.json'
            for _ in range(100):
                if owner_path.exists():
                    break
                await asyncio.sleep(.01)
            owner = _registry_owner(registry)
            identity = _process_identity(os.getpid())
            if (owner['worker_pid'] != os.getpid()
                    or owner['worker_start_ticks'] != identity['start_ticks']):
                raise ValueError('Detached launcher differs from its exact worker owner')
            argv = (sys.executable, '-B', '-m', 'openwebrl.action_credit_process',
                    '--detached-proxy', str(registry), owner['nonce'], '--', *argv)
        return await original(*argv, **kwargs)

    asyncio.create_subprocess_exec = create
    try:
        yield
    finally:
        asyncio.create_subprocess_exec = original


def _detached_proxy(registry, nonce, argv):
    """An asyncio transport can SIGKILL this proxy without killing its guardian.

    The native client still owns the returned proxy process and its group. A
    proxy death triggers the guardian's parent-death cleanup, including when
    asyncio destroys a subprocess transport during worker interpreter exit.
    """
    import ctypes
    import subprocess
    owner = _registry_owner(registry)
    if nonce != owner['nonce']:
        raise ValueError('Detached proxy nonce mismatch')
    stopping = False

    def stop(*_):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    if ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGTERM, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), 'PR_SET_PDEATHSIG failed')
    parent = _process_identity(os.getppid())
    if (not parent or parent['pid'] != owner['worker_pid']
            or parent['start_ticks'] != owner['worker_start_ticks'] or stopping):
        return 125
    identity = _process_identity(os.getpid())
    _write(_safe_path(registry) / f"proxy-{identity['pid']}-{identity['start_ticks']}.json",
           dict(schema_version=1, nonce=nonce, worker_pid=owner['worker_pid'],
                worker_start_ticks=owner['worker_start_ticks'], proxy=identity))
    # A proxy registered after the parent's census cannot spawn anything: its
    # owner has already exited, so this second check fails. Registered live
    # proxies are drained before the parent's final guardian census.
    if stopping or os.getppid() != owner['worker_pid']:
        return 125
    process = subprocess.Popen([sys.executable, '-B', '-m', 'openwebrl.action_credit_process',
        '--detached-guardian', str(registry), nonce, str(identity['pid']),
        str(identity['start_ticks']), '--', *argv], start_new_session=True)
    notified = False
    while process.poll() is None:
        if stopping and not notified:
            try:
                process.send_signal(signal.SIGTERM)
            except ProcessLookupError:
                pass
            notified = True
        time.sleep(.05)
    return process.returncode


def _detached_guardian(registry, nonce, launcher_pid, launcher_start_ticks, argv):
    """Keep an authenticated group leader alive until its children are gone."""
    import ctypes
    import subprocess
    registry = _safe_path(registry)
    owner = _registry_owner(registry)
    identity = _process_identity(os.getpid())
    if nonce != owner['nonce'] or identity['pgid'] != os.getpid() or identity['session'] != os.getpid():
        raise ValueError('Detached guardian requires its own bound session')
    stopping = False

    def stop(*_):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    # Playwright launches Chromium in a new session. Adopting orphaned
    # descendants keeps its process identity owned when the server/Node exits;
    # a process-group-only kill cannot establish browser cleanup.
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
        raise OSError(ctypes.get_errno(), 'PR_SET_CHILD_SUBREAPER failed')
    probe = os.pidfd_open(os.getpid())
    os.close(probe)
    if not hasattr(signal, 'pidfd_send_signal'):
        raise RuntimeError('Exact descendant cleanup requires Linux pidfd signaling')
    # Install parent-death notification before checking parent identity. If the
    # owner died earlier, the check aborts before a browser can be spawned.
    if libc.prctl(1, signal.SIGTERM, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), 'PR_SET_PDEATHSIG failed')
    parent = _process_identity(os.getppid())
    if (not parent or parent['pid'] != launcher_pid
            or parent['start_ticks'] != launcher_start_ticks or stopping):
        return 125
    _write(registry / f"child-{identity['pid']}-{identity['start_ticks']}.json",
           dict(schema_version=1, nonce=nonce, worker_pid=owner['worker_pid'],
                worker_start_ticks=owner['worker_start_ticks'], guardian=identity,
                launcher=parent))
    observed = {}

    def closed():
        _write(registry / f"closed-{identity['pid']}-{identity['start_ticks']}.json",
               dict(schema_version=1, nonce=nonce, guardian=identity,
                    owned_descendants_closed=True, observed_descendants=list(observed.values())))

    if stopping or os.getppid() != launcher_pid:
        closed()
        return 125
    try:
        process = subprocess.Popen(argv)  # Initial server inherits this group.
    except BaseException:
        if not _descendants(os.getpid()):
            closed()
        raise
    deadline = None
    signalled = set()
    while True:
        code = process.poll()
        if (stopping or code is not None) and deadline is None:
            deadline = time.monotonic() + 5
        if deadline is not None:
            members = _descendants(os.getpid())
            for item in members:
                key = (item['pid'], item['start_ticks'])
                observed[key] = item
                if key not in signalled or time.monotonic() >= deadline:
                    _signal_identity(item, signal.SIGKILL if time.monotonic() >= deadline else signal.SIGTERM)
                    signalled.add(key)
            if not members:
                if _live_children(os.getpid()):
                    time.sleep(.1)
                    continue
                process.wait()
                # Reap adopted zombies before leaving the guardian. Zombies
                # retain no browser socket or executable workload.
                try:
                    while os.waitpid(-1, os.WNOHANG)[0]:
                        pass
                except ChildProcessError:
                    pass
                closed()
                return code if code is not None and code >= 0 else 125
        time.sleep(.1)


async def _cleanup_detached(registry, owner):
    """Signal only registered live group leaders with an exact birth identity."""
    if _registry_owner(registry) != owner:
        raise ValueError('Detached registry owner changed')
    proxies = []
    for path in sorted(Path(registry).glob('proxy-*.json')):
        info = path.stat()
        if info.st_uid != os.getuid() or info.st_mode & 0o077 or info.st_nlink != 1:
            raise ValueError('Detached proxy record must be private and unaliased')
        record = json.loads(path.read_text())
        proxy = record['proxy']
        if (record['nonce'] != owner['nonce'] or record['worker_pid'] != owner['worker_pid']
                or record['worker_start_ticks'] != owner['worker_start_ticks']
                or proxy['pid'] != proxy['pgid'] or proxy['pid'] != proxy['session']
                or path.name != f"proxy-{proxy['pid']}-{proxy['start_ticks']}.json"):
            raise ValueError('Detached proxy has a different owner or session')
        proxies.append(record)
        current = _process_identity(proxy['pid'])
        if current and current['state'] != 'Z':
            if any(current[k] != proxy[k] for k in ('start_ticks', 'pgid', 'session')):
                raise ValueError('Detached proxy PID was reused; refusing to signal')
            try:
                os.kill(proxy['pid'], signal.SIGTERM)
            except ProcessLookupError:
                pass
    proxy_deadline = time.monotonic() + 8
    while True:
        pending = []
        for record in proxies:
            proxy = record['proxy']
            current = _process_identity(proxy['pid'])
            if current and current['state'] != 'Z':
                if any(current[k] != proxy[k] for k in ('start_ticks', 'pgid', 'session')):
                    raise ValueError('Detached proxy identity changed while draining')
                pending.append(proxy)
        if not pending:
            break
        if time.monotonic() >= proxy_deadline:
            for proxy in pending:
                try:
                    os.kill(proxy['pid'], signal.SIGKILL)
                except ProcessLookupError:
                    pass
            await asyncio.sleep(.1)
            if any((p := _process_identity(x['pid'])) and p['state'] != 'Z' for x in pending):
                raise RuntimeError('Detached proxy survived forced cleanup')
            break
        await asyncio.sleep(.1)
    records = []
    for path in sorted(Path(registry).glob('child-*.json')):
        info = path.stat()
        if info.st_uid != os.getuid() or info.st_mode & 0o077 or info.st_nlink != 1:
            raise ValueError('Detached guardian record must be private and unaliased')
        record = json.loads(path.read_text())
        guardian = record['guardian']
        if (record['nonce'] != owner['nonce'] or record['worker_pid'] != owner['worker_pid']
                or record['worker_start_ticks'] != owner['worker_start_ticks']
                or guardian['pid'] != guardian['pgid'] or guardian['pid'] != guardian['session']
                or path.name != f"child-{guardian['pid']}-{guardian['start_ticks']}.json"):
            raise ValueError('Detached guardian has a different owner or session')
        records.append(record)
        current = _process_identity(guardian['pid'])
        if current and current['state'] != 'Z':
            if any(current[k] != guardian[k] for k in ('start_ticks', 'pgid', 'session')):
                raise ValueError('Detached guardian PID was reused; refusing to signal')
            try:
                os.killpg(guardian['pid'], signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + 8
    while True:
        alive = []
        for record in records:
            guardian = record['guardian']
            members = _group_members(guardian['pid'])
            if not members:
                continue
            current = _process_identity(guardian['pid'])
            if not current or any(current[k] != guardian[k] for k in ('start_ticks', 'pgid', 'session')):
                # The guardian may finish after the first member census. It
                # exits only once that census is empty; verify afresh before
                # treating a concurrently reaped leader as lost ownership.
                if not _group_members(guardian['pid']):
                    continue
                raise RuntimeError('Detached group lost its registered leader; manual reconciliation required')
            alive.append(guardian)
        if not alive:
            for record in records:
                guardian = record['guardian']
                path = Path(registry)/f"closed-{guardian['pid']}-{guardian['start_ticks']}.json"
                if not path.is_file():
                    raise RuntimeError('Detached guardian has no descendant cleanup acknowledgement')
                ack = json.loads(path.read_text())
                if (ack.get('nonce') != owner['nonce'] or ack.get('guardian') != guardian
                        or ack.get('owned_descendants_closed') is not True):
                    raise ValueError('Detached descendant acknowledgement differs from registered owner')
                for item in ack['observed_descendants']:
                    current = _process_identity(item['pid'])
                    if current and current['start_ticks'] == item['start_ticks'] and current['state'] != 'Z':
                        raise RuntimeError('Owned detached descendant survived its cleanup acknowledgement')
                record['cleanup_acknowledgement'] = dict(path=str(path), sha256=_hash(path))
            return dict(proxies=proxies, guardians=records)
        if time.monotonic() >= deadline:
            for guardian in alive:
                os.killpg(guardian['pid'], signal.SIGKILL)
            # Keep the receipt fail closed until the kernel reports no live
            # members. Zombies carry no browser, socket, or execution state.
            await asyncio.sleep(.1)
            if any(_group_members(x['pid']) for x in alive):
                raise RuntimeError('Detached browser processes survived forced cleanup')
            raise RuntimeError('Guardian required forced owner kill; descendant reconciliation is mandatory')
        await asyncio.sleep(.1)


def _hash(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def _private_file(reference, root):
    path, root = _safe_path(reference['path']), _safe_path(root)
    if root not in path.parents:
        raise ValueError('Worker artifacts must remain inside their exact campaign')
    info = path.stat()
    if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
            or info.st_uid != os.getuid() or info.st_mode & 0o077):
        raise ValueError('Worker artifacts require a private unaliased owned file')
    if _hash(path) != reference['sha256']:
        raise ValueError('Worker artifact SHA256 mismatch')
    return path


def _write(path, value, *, tensor=False):
    path = _safe_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'wb') as stream:
        if tensor:
            import torch
            torch.save(value, stream)
        else:
            stream.write((json.dumps(value, sort_keys=True, allow_nan=False) + '\n').encode())
        stream.flush()
        os.fsync(stream.fileno())
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return dict(path=str(path), sha256=_hash(path))


def _no_credentials(value):
    """Runtime configuration can reference authority artifacts, never secrets."""
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = str(key).lower()
            if any(s in normalized for s in ('api_key', 'access_token', 'password', 'secret_key')):
                if item not in (None, ''):
                    raise ValueError('Credentials must remain in the worker environment')
            _no_credentials(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            _no_credentials(item)


async def owned_process(argv, *, cwd, env, log_path, deadline_seconds,
                        detached_registry=None, attempt_id=None):
    """Await a child and its cancellation; stdout is a bounded-memory log file."""
    if deadline_seconds <= 0:
        raise ValueError('A positive worker deadline is required')
    path = _safe_path(log_path)
    owner = None
    cleanup_reference = None
    if detached_registry is not None:
        registry = _safe_path(detached_registry)
        registry.mkdir(mode=0o700, parents=False, exist_ok=False)
        env = dict(env, **{_CHILD_REGISTRY: str(registry)})
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'wb') as log:
        spawn = asyncio.create_task(asyncio.create_subprocess_exec(*argv, cwd=str(cwd), env=env,
            stdout=log, stderr=asyncio.subprocess.STDOUT, start_new_session=True))
        cancelled_start = False
        try:
            process = await asyncio.shield(spawn)
        except asyncio.CancelledError:
            process = await asyncio.shield(spawn)
            cancelled_start = True
        waiter = asyncio.create_task(process.wait())
        try:
            if detached_registry is not None:
                identity = _process_identity(process.pid)
                if identity is None:
                    raise RuntimeError('Worker disappeared before ownership registration')
                owner = dict(schema_version=1, nonce=uuid.uuid4().hex,
                             worker_pid=process.pid, worker_start_ticks=identity['start_ticks'],
                             attempt_id=attempt_id)
                _write(registry/'owner.json', owner)
            if cancelled_start:
                raise asyncio.CancelledError()
            code = await asyncio.wait_for(asyncio.shield(waiter), deadline_seconds)
        except BaseException:
            # Give the worker's owned browser cleanup a chance to finish, then
            # reap its group. Detached browser groups are handled separately
            # using their registered guardians, including startup failures.
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(asyncio.shield(waiter), 30)
            except asyncio.TimeoutError:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                await asyncio.shield(waiter)
            raise
        finally:
            if owner is not None:
                cleanup = asyncio.create_task(_cleanup_detached(registry, owner))
                try:
                    records = await asyncio.shield(cleanup)
                except asyncio.CancelledError:
                    records = await asyncio.shield(cleanup)
                    raise
                cleanup_reference = _write(registry/'cleanup.json',
                    dict(schema_version=1, attempt_id=attempt_id, owner=owner,
                         owned_processes_closed=True, detached_guardians=records['guardians'],
                         detached_proxies=records['proxies'],
                         worker_returncode=process.returncode))
    if code != 0:
        raise RuntimeError(f'Isolated episode exited {code}; reconcile retained ledger and {path}')
    return cleanup_reference


class IsolatedWorkerPool:
    """A bounded subprocess owner; factories resolve only pinned native inputs.

    ``sample_factory(request)`` returns JSON ``prompt, metadata, index``.
    ``native_args_factory(actor)`` returns explicit JSON native runtime args
    pointing to that actor's direct engine. The parent driver holds weights
    frozen throughout collection and awaits every concurrent operation.
    """
    def __init__(self, *, store, contract, template, sample_factory,
                 native_args_factory, concurrency, deadline_seconds,
                 python=sys.executable, source_root=None):
        if type(concurrency) is not int or not 1 <= concurrency <= 64:
            raise ValueError('Explicit worker concurrency must be within 1..64')
        if deadline_seconds <= 0:
            raise ValueError('Explicit worker deadline required')
        from .action_credit_worker import verify_worker_environment
        verify_worker_environment()
        self.store, self.contract = store, contract
        self.template = json.loads(json.dumps(template, allow_nan=False))
        self.sample_factory, self.native_args_factory = sample_factory, native_args_factory
        self.semaphore = asyncio.Semaphore(concurrency)
        self.concurrency = concurrency
        self.deadline_seconds, self.python = deadline_seconds, str(python)
        self.source_root = Path(source_root or Path(__file__).resolve().parents[1])
        self.active = set()

    async def _run(self, request, prefix, actor, operation):
        actor.validate(self.contract)
        if operation == 'collect' and prefix_sha256(prefix) != request.prefix_sha256:
            raise ValueError('Dispatch prefix differs from the planned request')
        async with self.semaphore:
            from .action_credit_tensor_refs import tensor_inventory
            prefix_owners = tensor_inventory(self.store, (turn.tensors for turn in prefix))
            root = _safe_path(self.store.root/'worker_dispatch'/digest(request.attempt_id)/uuid.uuid4().hex)
            root.mkdir(parents=True, exist_ok=False)
            packet = dict(self.template, schema_version=1, operation=operation,
                store_root=str(self.store.root), manifest_sha256=self.store.manifest.sha256,
                contract=asdict(self.contract), actor=asdict(actor), request=asdict(request),
                verifier_sha256=physical_verifier_sha256(),
                native_args=self.native_args_factory(actor), sample=self.sample_factory(request),
                prefix_ref=_write(root/'prefix.pt', dict(prefix=tuple(prefix), tensor_owners=prefix_owners),
                                  tensor=True) if prefix else None)
            _no_credentials(packet)
            reference = _write(root/'packet.json', packet)
            env = dict(os.environ)
            # Inherited credentials remain in process memory, never in packet.
            env['PYTHONDONTWRITEBYTECODE'] = '1'
            process = asyncio.create_task(owned_process(
                [self.python, '-B', '-m', 'openwebrl.action_credit_process',
                 '--packet', reference['path'], '--sha256', reference['sha256']],
                cwd=self.source_root, env=env, log_path=root/'worker.log',
                deadline_seconds=self.deadline_seconds,
                detached_registry=root/'detached-processes', attempt_id=request.attempt_id))
            self.active.add(process)
            try:
                cleanup_reference = await process
            finally:
                self.active.discard(process)
            failure = _safe_path(self.store.root/'online_artifacts'/digest(request.attempt_id)/'clean-preactor-failure.json')
            if failure.exists():
                from .action_credit_worker import settle_clean_pre_actor_failure
                settle_clean_pre_actor_failure(self.store, request,
                    process_cleanup_reference=cleanup_reference)
                return None
            return load_pending_episode(self.store, self.contract, actor, request.attempt_id,
                                        verifier_sha256=physical_verifier_sha256())

    async def collect(self, request, prefix, actor):
        return await self._run(request, prefix, actor, 'collect')

    async def resume_pending(self, episode, actor):
        return await self._run(episode.request, (), actor, 'judge_existing')

    async def close(self):
        """An owning driver must await this even when a sibling stage fails."""
        results = await asyncio.gather(*tuple(self.active), return_exceptions=True)
        for result in results:
            if isinstance(result, BaseException):
                raise result


async def run_packet(reference):
    # Inspect JSON only before opening the exact campaign; private packet/hash
    # checks precede any trusted tensor load or external side effect.
    raw_path = _safe_path(reference['path'])
    if _hash(raw_path) != reference['sha256']:
        raise ValueError('Packet SHA256 mismatch')
    packet = json.loads(raw_path.read_text())
    root = _safe_path(packet['store_root'])
    _private_file(reference, root)
    if packet.get('schema_version') != 1 or packet.get('operation') not in ('collect', 'judge_existing'):
        raise ValueError('Unsupported worker packet')
    state = json.loads((root/'state.json').read_text())
    manifest = CampaignManifest.from_dict(state['manifest'])
    if manifest.sha256 != packet['manifest_sha256']:
        raise ValueError('Worker campaign identity mismatch')
    store = CampaignStore(root, manifest)
    value = dict(packet['contract'])
    for key in ('train_task_ids', 'heldout_task_ids'):
        value[key] = tuple(value[key])
    contract, request, actor = CollectionContract(**value), EpisodeRequest(**packet['request']), FrozenActor(**packet['actor'])
    if packet['verifier_sha256'] != physical_verifier_sha256():
        raise ValueError('Physical collector changed after worker dispatch')
    from .action_credit_worker import (run_collect_and_judge, judge_existing_pending_episode,
                                      verify_worker_environment)
    from .action_credit_judge import JudgeHTTPProtocol
    verify_worker_environment()
    args = SimpleNamespace(**packet['native_args'])
    common = dict(store=store, contract=contract, actor=actor, args=args,
        approval_reference=packet['approval_reference'],
        judge_protocol=JudgeHTTPProtocol(**packet['judge_protocol']),
        canonical_reward_source_sha256=packet['canonical_reward_source_sha256'],
        tree_terminal=manifest.to_dict()['arm_id'] in ('tree_outcome_sampling', 'tree_grpo'))
    if packet['operation'] == 'judge_existing':
        if packet['prefix_ref'] is not None:
            raise ValueError('Judge recovery cannot recollect or replay a prefix')
        return await judge_existing_pending_episode(**common, attempt_id=request.attempt_id)
    prefix = ()
    if packet['prefix_ref'] is not None:
        import torch
        path = _private_file(packet['prefix_ref'], root)
        payload = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
        prefix = payload['prefix']
        from .action_credit_tensor_refs import bind_tensor_inventory
        bind_tensor_inventory(store, (turn.tensors for turn in prefix), payload['tensor_owners'])
    if type(prefix) is not tuple or prefix_sha256(prefix) != request.prefix_sha256:
        raise ValueError('Worker prefix differs from dispatch request')
    sample_data = packet['sample']
    if set(sample_data) != {'prompt', 'metadata', 'index'} or sample_data['metadata'].get('task_id') != request.task_id:
        raise ValueError('Worker sample must match the fixed task')
    from slime.utils.types import Sample
    return await run_collect_and_judge(**common, request=request, sample=Sample(**sample_data),
        sampling=packet['sampling'], browser_config=packet['browser_config'],
        browser_manifest=packet['browser_manifest'], prefix=prefix,
        reserved_cost=dict(gpu_seconds=0, api_requests=0, api_microdollars=0, browser_dispatches=1))


def main():
    if len(sys.argv) > 1 and sys.argv[1] == '--detached-proxy':
        if len(sys.argv) < 6 or sys.argv[4] != '--':
            raise ValueError('Malformed detached proxy command')
        raise SystemExit(_detached_proxy(sys.argv[2], sys.argv[3], sys.argv[5:]))
    if len(sys.argv) > 1 and sys.argv[1] == '--detached-guardian':
        if len(sys.argv) < 8 or sys.argv[6] != '--':
            raise ValueError('Malformed detached guardian command')
        raise SystemExit(_detached_guardian(sys.argv[2], sys.argv[3],
            int(sys.argv[4]), int(sys.argv[5]), sys.argv[7:]))
    parser = argparse.ArgumentParser()
    parser.add_argument('--packet', required=True)
    parser.add_argument('--sha256', required=True)
    args = parser.parse_args()
    async def run():
        registry = os.environ.get(_CHILD_REGISTRY)
        if not registry:
            raise ValueError('Isolated episode requires a parent-owned detached-process registry')
        with detached_process_scope(registry):
            task = asyncio.create_task(run_packet(dict(path=args.packet, sha256=args.sha256)))
            loop = asyncio.get_running_loop()
            for sig in (signal.SIGTERM, signal.SIGINT):
                loop.add_signal_handler(sig, task.cancel)
            await task
    asyncio.run(run())


if __name__ == '__main__':
    main()
