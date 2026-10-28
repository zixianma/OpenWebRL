"""Fixed-cell concurrent admission and durable attempt ownership.

The host supplies domain readiness, resource admission, and a bounded worker
coroutine. Existing webprm_broker / SharedBudget remain the only authorities for
per-call limits and money; this queue neither allocates nor reserves model calls.
Every worker coroutine must own and close its service/browser process tree before
returning, including cancellation. The controller awaits every admitted worker.
"""
from __future__ import annotations

import asyncio
from copy import deepcopy
import fcntl
import hashlib
import inspect
import json
import os
from pathlib import Path
import secrets
import time

ROOTS=(Path('/tmp'),Path('/gpfs/scrubbed/zixianma/openwebrl-runtime'),
       Path('/gpfs/projects/krishna/zixianma/OpenWebRL'))


def encoded(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def identity(value):return hashlib.sha256(encoded(value)).hexdigest()


def checked(value):
    path=Path(value)
    if not path.is_absolute() or path.resolve()!=path or not any(r in path.parents for r in ROOTS) or any(p.is_symlink() for p in (path,*path.parents)):
        raise ValueError('Queue path must be canonical, nonsymlink and below an approved root')
    return path


def write_once(path,value):
    path=checked(path);raw=encoded(value)+b'\n'
    if len(raw)>2**20:raise ValueError('Queue receipt exceeds one MiB')
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'wb') as out:out.write(raw);out.flush();os.fsync(out.fileno())
    fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)


def fixed_cells(cells):
    if not isinstance(cells,list) or not cells:raise ValueError('A nonempty fixed cohort is required')
    copied=deepcopy(cells);ids=set()
    for cell in copied:
        if (not isinstance(cell,dict) or not isinstance(cell.get('cell_id'),str) or not cell['cell_id'] or
            not isinstance(cell.get('domain'),str) or not cell['domain'] or cell['cell_id'] in ids):
            raise ValueError('Every fixed cell needs a unique ID and domain')
        ids.add(cell['cell_id']);encoded(cell)
    return copied


class ClaimStore:
    """One owning controller; a durable claim is never a fresh primary again."""
    def __init__(self,root,cells):
        self.root=checked(root);self.cells=fixed_cells(cells)
        self.by_id={c['cell_id']:c for c in self.cells}
        self.token=secrets.token_hex(16);self.fd=None

    def __enter__(self):
        self.root.mkdir(mode=0o700,parents=True,exist_ok=True)
        self.fd=os.open(checked(self.root/'owner.lock'),os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW,0o600)
        try:
            if os.fstat(self.fd).st_nlink!=1:raise ValueError('Owner lock has an unsafe link count')
            fcntl.flock(self.fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
            manifest=dict(cells_sha256=identity(self.cells),cell_count=len(self.cells))
            path=self.root/'manifest.json'
            if path.exists():
                if json.loads(checked(path).read_text())!=manifest:raise ValueError('Queue cohort/order changed across controllers')
            else:write_once(path,manifest)
            for name in ('claims','outcomes'):checked(self.root/name).mkdir(mode=0o700,exist_ok=True)
            return self
        except BaseException:
            os.close(self.fd);self.fd=None;raise

    def __exit__(self,*args):
        if self.fd is not None:os.close(self.fd);self.fd=None

    def _path(self,group,cell):
        if self.fd is None:raise RuntimeError('ClaimStore must hold its exclusive owner lock')
        if self.by_id.get(cell['cell_id'])!=cell:raise ValueError('Cell does not match the fixed manifest')
        return checked(self.root/group/(identity(cell['cell_id'])+'.json'))

    def claim(self,cell):
        path=self._path('claims',cell)
        if path.exists():
            old=json.loads(path.read_text())
            if old.get('cell_sha256')!=identity(cell) or old.get('cell_id')!=cell['cell_id']:
                raise ValueError('Existing claim identity mismatch')
            return False
        write_once(path,dict(cell_id=cell['cell_id'],cell_sha256=identity(cell),owner=self.token,epoch=time.time()))
        return True

    def finish(self,cell,result):
        claim=json.loads(self._path('claims',cell).read_text())
        if claim.get('owner')!=self.token or claim.get('cell_sha256')!=identity(cell):
            raise ValueError('Cannot finish an attempt owned by another controller')
        write_once(self._path('outcomes',cell),dict(cell_id=cell['cell_id'],cell_sha256=identity(cell),
            owner=self.token,result=result,epoch=time.time()))

    def state(self,cell):
        claim=self._path('claims',cell);outcome=self._path('outcomes',cell)
        claim_value=None
        if claim.exists():
            claim_value=json.loads(claim.read_text())
            if claim_value.get('cell_sha256')!=identity(cell) or claim_value.get('cell_id')!=cell['cell_id']:
                raise ValueError('Claim identity mismatch')
        if outcome.exists():
            if not claim.exists():raise ValueError('Outcome without durable attempt claim')
            value=json.loads(outcome.read_text())
            if (value.get('cell_sha256')!=identity(cell) or value.get('cell_id')!=cell['cell_id'] or
                value.get('owner')!=claim_value.get('owner')):
                raise ValueError('Outcome identity mismatch')
            return 'recorded'
        if claim.exists():
            return 'attempted_without_outcome'
        return 'pending'


async def call(function,*args):
    value=function(*args)
    return await value if inspect.isawaitable(value) else value


async def gate(function,*args):
    value=await call(function,*args)
    if type(value) is not bool:raise ValueError('Admission gates must return an explicit boolean')
    return value


async def run_queue(cells,store,*,domain_ready,can_start,run_one,release=None,
                    max_workers=8,stop_admission=lambda:False,poll_seconds=1.0):
    """Drain admitted work, preserve failed primaries, and leave blocked cells pending.

    domain_ready(domain) must verify a pinned service/auth/reset readiness record.
    can_start(cell, active_count) returns a bool and must conservatively check
    shared broker allowance, node RAM/disk, and shutdown time. It may acquire an
    episode resource lease; release(cell) releases that lease after worker closure.
    Admission checks do not replace the broker's atomic per-dispatch reservations.
    A false can_start with active work waits for an owned completion before
    checking all admission gates again. A false with no active work closes
    admission. A false domain_ready blocks only that domain. No active/pending-ready work means exit:
    a GPU owner must not wait idle for independently running CPU setup jobs.

    run_one exceptions become recorded invalid attempts. Host admission/readiness
    exceptions stop admission, await every existing worker, then propagate. The
    caller retains active ownership during this drain. Cancellation requests
    worker cleanup and awaits it before returning or releasing any resource lease.
    """
    cells=fixed_cells(cells)
    if cells!=store.cells:raise ValueError('Queue/store manifests differ')
    if type(max_workers) is not int or not 1<=max_workers<=8:raise ValueError('One to eight workers required')
    if not isinstance(poll_seconds,(float,int)) or isinstance(poll_seconds,bool) or not 0<poll_seconds<=60:
        raise ValueError('Bounded positive poll interval required')
    pending=[c for c in cells if store.state(c)=='pending']
    active={};admitted=[];closed=[];admission_closed=False;capacity_wait=False;reason='completed'

    async def owned(cell):
        try:
            try:
                result=await call(run_one,deepcopy(cell))
                if not isinstance(result,dict):raise ValueError('Worker must return a structured outcome')
            except asyncio.CancelledError:
                # The bounded worker coroutine must close its descendants in
                # its own finally before cancellation reaches this wrapper.
                result=dict(status='invalid',reason='controller_cancelled')
                store.finish(cell,result)
                raise
            except Exception as error:
                result=dict(status='invalid',reason='worker_exception',error_type=type(error).__name__)
            store.finish(cell,result)
            return cell['cell_id']
        finally:
            if release is not None:await call(release,deepcopy(cell))

    async def reap(done):
        for task in done:
            cell=active.pop(task)
            await task  # Receipt/release failures stop host admission, not silently ignored.
            closed.append(cell['cell_id'])

    try:
        while pending or active:
            if await gate(stop_admission):admission_closed=True;reason='stop_admission'
            while pending and len(active)<max_workers and not admission_closed and not capacity_wait:
                if await gate(stop_admission):
                    admission_closed=True;reason='stop_admission';break
                selected=None
                for index,cell in enumerate(pending):
                    if await gate(domain_ready,cell['domain']):selected=index;break
                if selected is None:break
                cell=pending[selected]
                if not await gate(can_start,deepcopy(cell),len(active)):
                    if active:
                        # Live states and resource leases can disappear only
                        # after their owning workers have closed. Do not turn
                        # transient pressure into a terminal campaign result,
                        # or repeatedly sample admission while they are live.
                        capacity_wait=True
                    else:
                        admission_closed=True;reason='resource_admission_closed'
                    break
                try:claimed=store.claim(cell)
                except BaseException:
                    if release is not None:await call(release,deepcopy(cell))
                    raise
                pending.pop(selected)
                if not claimed:
                    if release is not None:await call(release,deepcopy(cell))
                    continue
                admitted.append(cell['cell_id'])
                task=asyncio.create_task(owned(cell),name='webprm-cell-'+identity(cell['cell_id'])[:16])
                active[task]=cell
            if not active:
                if pending and not admission_closed:reason='domains_not_ready'
                break
            done,_=await asyncio.wait(active,timeout=poll_seconds,return_when=asyncio.FIRST_COMPLETED)
            await reap(done)
            if done:capacity_wait=False
    except asyncio.CancelledError:
        for task in active:task.cancel()
        await asyncio.gather(*active,return_exceptions=True)
        raise
    except BaseException:
        # A host-side gate error must not leave sibling workers orphaned.
        await asyncio.gather(*active,return_exceptions=True)
        raise
    return dict(reason=reason,admitted=admitted,closed=closed,
        pending=[c['cell_id'] for c in cells if store.state(c)=='pending'],
        attempted_without_outcome=[c['cell_id'] for c in cells if store.state(c)=='attempted_without_outcome'],
        recorded=[c['cell_id'] for c in cells if store.state(c)=='recorded'],remaining_owned_coroutines=0,
        process_closure_verified_by_queue=False)
