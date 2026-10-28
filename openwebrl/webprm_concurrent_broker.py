"""Prospective bounded broker; the legacy pilot server remains unchanged.

One process owns all provider handlers, the dispatch log and financial journal.
No implicit migration, automatic retry, credential forwarding or client-selected
budget settings. Concurrency here is a capacity bound, not measured throughput.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from copy import deepcopy
import fcntl
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import socket
import threading
import time
import uuid

from .webprm_broker import (
    BrokerError, BrokerProcess, MAX_FRAME_BYTES, MAX_RESPONSE_BYTES, PAYLOAD_KEYS, PROTOCOL,
    _append, _deadline, _path, _peer, _receive, _send, _timeout,
    _validate_cells, _validate_prices, _validate_routes, payload_sha256,
)
from .webprm_budget_journal import JournalBudget, VERSION as BUDGET_VERSION

VERSION = 'bounded-single-owner-broker-v2'

REJECTION_LOG_BYTES=1024*1024
REJECTION_REASONS={
    'Unexpected broker request fields/protocol':'protocol',
    'Cell identity mismatch':'identity',
    'Request extends host deadline':'deadline',
    'Invalid request ID':'request_id',
    'Unsupported payload':'payload_fields',
    'Unregistered model/purpose':'route',
    'Route mismatch':'route_mode',
    'Output token/response count cap rejected':'output_limit',
    'Messages required':'messages',
    'Input token cap rejected':'input_limit',
    'Broker admission closed':'admission_closed',
    'Overlapping requests for one cell rejected':'cell_overlap',
    'Paid dispatch pacing deadline reached':'pacing_deadline',
}


class PaidDispatchPacer:
    """Space paid admission starts without holding the global admission lock.

    A slot remains locked through the durable admission record, then releases
    before inference so responses may overlap. Local calls never take this lock.
    Waiting consumes neither a dispatch nor a financial reservation. No request
    is retried. The monotonic clock prevents wall-clock adjustments from making
    a burst, while each wait also observes the absolute request deadline.
    """
    def __init__(self, interval):
        if (type(interval) not in (int, float) or not math.isfinite(interval)
                or not 0 <= interval <= 60):
            raise ValueError('Paid dispatch interval must be finite and within 0–60 seconds')
        self.interval=float(interval)
        self.lock=threading.Lock()
        self.last_started=None

    @contextmanager
    def slot(self, *, deadline, closed):
        acquired=False
        try:
            while True:
                if closed():raise BrokerError('Broker admission closed')
                remaining=deadline-time.time()
                if remaining<=0:raise BrokerError('Paid dispatch pacing deadline reached')
                if self.lock.acquire(timeout=min(.05,remaining)):
                    acquired=True
                    break
            while True:
                if closed():raise BrokerError('Broker admission closed')
                remaining=deadline-time.time()
                delay=0 if self.last_started is None else max(0,self.last_started+self.interval-time.monotonic())
                if remaining<=0 or delay>=remaining:
                    raise BrokerError('Paid dispatch pacing deadline reached')
                if delay<=0:break
                time.sleep(min(.05,delay,remaining))
            yield
            # Called only after successful durable admission. Rejected calls do
            # not reserve a future slot or increase the cumulative dispatch cap.
            self.last_started=time.monotonic()
        finally:
            if acquired:self.lock.release()


def rejection_metadata(error,request,stage,cells,routes):
    """No payload, provider text, credentials or arbitrary client strings."""
    request=request if isinstance(request,dict) else {}
    cell_id=request.get('cell_id');cell_id=cell_id if isinstance(cell_id,str) and cell_id in cells else None
    payload=request.get('payload');payload=payload if isinstance(payload,dict) else {}
    model,purpose=payload.get('model'),request.get('purpose')
    known=isinstance(model,str) and isinstance(purpose,str) and (model,purpose) in routes
    ident=request.get('request_id')
    if not isinstance(ident,str) or len(ident)!=32 or any(c not in '0123456789abcdef' for c in ident):ident=None
    number=getattr(error,'errno',None)
    return dict(schema_version=1,event='request_rejected_before_dispatch',at_epoch=time.time(),
        stage=stage,error_type=type(error).__name__[:64],
        reason_code=REJECTION_REASONS.get(str(error),'unclassified') if isinstance(error,BrokerError) else 'unclassified',
        errno=number if type(number) is int else None,cell_id=cell_id,request_id=ident,
        model=model if known else None,purpose=purpose if known else None,broker_dispatch_recorded=False)




def append_rejection(path,row,*,limit=REJECTION_LOG_BYTES):
    """Caller holds admission lock; reject a write before exceeding its cap."""
    size=len(json.dumps(row,sort_keys=True,separators=(',',':'),allow_nan=False).encode())+1
    previous=path.stat().st_size if path.exists() else 0
    if previous+size>limit:raise BrokerError('Rejection diagnostic cap exhausted')
    _append(path,row)


def _receipt(cell,payload,purpose,offset,local,budget,meta):
    journal=Path(cell['journal'])
    with journal.open('rb') as stream:
        stream.seek(offset)
        rows=[json.loads(line) for line in stream if line.strip()]
    requests=[r for r in rows if r.get('event')=='request']
    responses=[r for r in rows if r.get('event')=='response']
    if len(requests)!=1 or len(responses)!=1:
        raise BrokerError('Expected exactly one durable request/response pair')
    request,response=requests[0],responses[0]
    if (request.get('payload')!=payload or request.get('purpose')!=purpose or request.get('local')!=local or
            response.get('reservation')!=request.get('reservation')):
        raise BrokerError('Cell journal identity mismatch')
    result=dict(reservation_id=request.get('reservation'),journal_sha256=hashlib.sha256(journal.read_bytes()).hexdigest())
    if local:
        if result['reservation_id'] is not None:raise BrokerError('Local call reserved paid budget')
        return dict(result,reservation_state='local',usage=response['response'].get('usage'))
    proof=budget.receipt(result['reservation_id'])
    if (proof.pop('model')!=payload['model'] or proof['usage']!=response['response'].get('usage') or
            proof.pop('request_id')!=meta['request_id'] or proof.pop('cell_id')!=meta['cell_id']):
        raise BrokerError('Financial receipt and exact cell request disagree')
    return dict(result,**{k:v for k,v in proof.items() if k!='reservation_id'})


def serve_concurrent_broker(socket_path, *, cells, routes, deadline, max_dispatches, audit_path,
        max_concurrency, financial_contract, epoch_id, previous_closed_epoch_sha256,
        expected_uid=None, completion_fn=None, ready=None, stop_event=None,
        max_paid_dispatches=None, max_local_dispatches=None,
        paid_dispatch_interval_seconds=0.0):
    """Blocking owned service: close admission, await handlers, then export budget.

    The outer ConcurrentBrokerProcess enforces a hard deadline/termination bound
    if a handler does not return. In that case durable reservations remain charged
    and replay, rather than a missing final snapshot, is the financial authority.
    """
    socket_path,audit_path=_path(socket_path),_path(audit_path)
    paid_pacer=PaidDispatchPacer(paid_dispatch_interval_seconds)
    _deadline(deadline);_validate_cells(cells,deadline);_validate_routes(routes);_validate_prices(cells,routes)
    if type(max_concurrency) is not int or not 1<=max_concurrency<=8:
        raise ValueError('Explicit broker concurrency must be within 1–8')
    if type(max_dispatches) is not int or max_dispatches<0:raise ValueError('Fixed nonnegative dispatch cap required')
    max_paid_dispatches=max_dispatches if max_paid_dispatches is None else max_paid_dispatches
    max_local_dispatches=max_dispatches if max_local_dispatches is None else max_local_dispatches
    if any(type(n) is not int or not 0<=n<=max_dispatches for n in (max_paid_dispatches,max_local_dispatches)):
        raise ValueError('Paid/local dispatch caps must be within total cap')
    financial={(str(_path(c['config']['budget_path'])),c['config']['api_usd_cap']) for c in cells.values()}
    if len(financial)!=1:raise ValueError('Exactly one shared financial authority required')
    budget_path,cap=next(iter(financial))
    authority_paths=[Path(budget_path),_path(budget_path+'.events.jsonl'),_path(Path(budget_path).with_suffix('.lock')),
                     audit_path,_path(str(audit_path)+'.lock'),socket_path]
    if len(set(authority_paths))!=len(authority_paths) or any(Path(c['journal']) in authority_paths for c in cells.values()):
        raise ValueError('Cell journals, financial authority, audit and socket paths must be distinct')
    expected_caps=dict(api_usd_cap=cap,max_dispatches=max_dispatches,max_paid_dispatches=max_paid_dispatches,
                       max_local_dispatches=max_local_dispatches,budget_backend=BUDGET_VERSION)
    if not isinstance(financial_contract,dict) or any(financial_contract.get(k)!=v for k,v in expected_caps.items()):
        raise ValueError('Broker must use the exact cumulative financial contract caps')
    for cell in cells.values():
        for model,purpose in cell['allowed_routes']:
            if not routes[model,purpose]['local'] and cell['config']['prices_usd_per_token'][model]!=financial_contract['prices_usd_per_token'].get(model):
                raise ValueError('Registered paid route price differs from frozen financial contract')
    if audit_path.exists() or audit_path.is_symlink():raise ValueError('Each broker epoch requires a fresh audit path')
    rejection_path=_path(str(audit_path)+'.rejections.jsonl')
    if rejection_path in authority_paths or any(Path(c['journal'])==rejection_path for c in cells.values()):
        raise ValueError('Rejection diagnostics must be separate from authority and cell journals')
    if rejection_path.exists() or rejection_path.is_symlink():raise ValueError('Each broker epoch requires fresh rejection diagnostics')
    if (len(os.fsencode(socket_path))>=108 or socket_path.exists() or socket_path.is_symlink() or
            not socket_path.parent.is_dir() or socket_path.parent.stat().st_uid!=os.getuid() or
            socket_path.parent.stat().st_mode & 0o077):
        raise ValueError('A fresh socket in an owned private directory is required')
    _timeout(deadline)
    expected_uid=os.getuid() if expected_uid is None else expected_uid
    cells,routes=deepcopy(cells),deepcopy(routes)
    source_files=[Path(__file__),Path(__file__).with_name('webprm_budget_journal.py'),Path(__file__).with_name('webprm_broker.py'),
                  Path(__file__).with_name('webprm_worker.py')]
    sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
    policy_hash=payload_sha256(dict(version=VERSION,budget_version=BUDGET_VERSION,implementation_sha256=sources,
        cells=cells,routes=[list(k)+[v] for k,v in sorted(routes.items())],deadline=deadline,
        max_dispatches=max_dispatches,max_paid_dispatches=max_paid_dispatches,max_local_dispatches=max_local_dispatches,
        expected_uid=expected_uid,max_concurrency=max_concurrency,epoch_id=epoch_id,
        paid_dispatch_interval_seconds=paid_pacer.interval,
        financial_contract_sha256=payload_sha256(financial_contract)))
    lock_path=_path(str(audit_path)+'.lock')
    fd=os.open(lock_path,os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'a+') as audit_lock:
        fcntl.flock(audit_lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        if completion_fn is None:
            from .webprm_worker import completion as completion_fn
        admission=threading.Lock();active_cells=set();slots=threading.BoundedSemaphore(max_concurrency)
        audit_failed=threading.Event()
        def record(event):
            try:_append(audit_path,event)
            except BaseException:
                audit_failed.set()
                raise
        with JournalBudget(budget_path,cap,policy_sha256=payload_sha256(financial_contract),financial_contract=financial_contract) as budget:
            budget.start_epoch(epoch_id=epoch_id,broker_policy_sha256=policy_hash,deadline=deadline,
                implementation_sha256=sources,registered_cells_sha256=payload_sha256(cells),
                previous_closed_epoch_sha256=previous_closed_epoch_sha256)
            record(dict(event='epoch_started',policy_sha256=policy_hash,epoch_id=epoch_id,
                        paid_dispatch_interval_seconds=paid_pacer.interval))
            def handle(conn):
                meta=None;claimed_cell=None;request=None;stage='peer'
                reply_deadline=min(deadline,time.time()+100)
                with conn:
                    try:
                        _peer(conn,expected_uid)
                        stage='frame'
                        request=_receive(conn,MAX_FRAME_BYTES,min(reply_deadline,time.time()+10))
                        if request==dict(protocol=PROTOCOL,operation='status'):
                            with admission:
                                summary=budget.summary()
                                status=dict(policy_sha256=policy_hash,budget=summary,dispatches=summary['dispatches'],
                                    paid_dispatches=summary['paid_dispatches'],local_dispatches=summary['local_dispatches'],
                                    active_cells=len(active_cells),active_cell_ids=sorted(active_cells),max_concurrency=max_concurrency,
                                    paid_dispatch_interval_seconds=paid_pacer.interval,
                                    admission_open=not audit_failed.is_set() and not (stop_event is not None and stop_event.is_set()))
                            _send(conn,dict(ok=True,status=status),MAX_RESPONSE_BYTES,reply_deadline)
                            return
                        stage='identity'
                        if set(request)!={'protocol','cell_id','identity_sha256','request_id','purpose','local','payload','expires_at'} or request['protocol']!=PROTOCOL:
                            raise BrokerError('Unexpected broker request fields/protocol')
                        cell=cells[request['cell_id']]
                        if request['identity_sha256']!=cell['config']['identity_sha256']:raise BrokerError('Cell identity mismatch')
                        expiry=_deadline(request['expires_at'])
                        if expiry>cell['config']['deadline'] or expiry>deadline:raise BrokerError('Request extends host deadline')
                        _timeout(expiry);reply_deadline=min(reply_deadline,expiry)
                        ident=request['request_id']
                        if not isinstance(ident,str) or len(ident)!=32 or uuid.UUID(hex=ident).hex!=ident:
                            raise BrokerError('Invalid request ID')
                        stage='payload'
                        payload,purpose=request['payload'],request['purpose']
                        if not isinstance(payload,dict) or set(payload)-PAYLOAD_KEYS:raise BrokerError('Unsupported payload')
                        stage='route'
                        route_key=(payload['model'],purpose)
                        if route_key not in map(tuple,cell['allowed_routes']):raise BrokerError('Unregistered model/purpose')
                        route=routes[route_key]
                        if type(request['local']) is not bool or request['local']!=route['local']:raise BrokerError('Route mismatch')
                        stage='output_limit'
                        tokens=payload.get('max_completion_tokens',payload.get('max_tokens'))
                        if (type(tokens) is not int or not 0<tokens<=route['max_output_tokens'] or payload.get('n',1)!=1 or
                                'max_tokens' in payload and 'max_completion_tokens' in payload):
                            raise BrokerError('Output token/response count cap rejected')
                        if not isinstance(payload.get('messages'),list) or not payload['messages']:raise BrokerError('Messages required')
                        stage='input_limit'
                        from .webprm_worker import input_token_upper_bound
                        input_bound=input_token_upper_bound(payload['messages'])
                        if input_bound>route['max_input_tokens']:raise BrokerError('Input token cap rejected')
                        stage='cell_configuration'
                        _validate_cells({request['cell_id']:cell},deadline)
                        stage='admission'
                        with admission:
                            _timeout(expiry)
                            if audit_failed.is_set() or stop_event is not None and stop_event.is_set():raise BrokerError('Broker admission closed')
                            if request['cell_id'] in active_cells:raise BrokerError('Overlapping requests for one cell rejected')
                            active_cells.add(request['cell_id']);claimed_cell=request['cell_id']
                        stage='paid_pacing' if not route['local'] else 'admission'
                        slot=(paid_pacer.slot(deadline=reply_deadline,
                            closed=lambda:audit_failed.is_set() or stop_event is not None and stop_event.is_set())
                            if not route['local'] else nullcontext())
                        with slot, admission:
                            _timeout(expiry)
                            if audit_failed.is_set() or stop_event is not None and stop_event.is_set():raise BrokerError('Broker admission closed')
                            candidate_meta=dict(schema_version=1,policy_sha256=policy_hash,request_id=ident,cell_id=request['cell_id'],
                                identity_sha256=request['identity_sha256'],model=payload['model'],purpose=purpose,local=route['local'],
                                payload_sha256=payload_sha256(payload),broker_version=VERSION)
                            budget.admit_dispatch(candidate_meta)
                            meta=candidate_meta
                            record(dict(meta,event='dispatch_started',at_epoch=time.time()))
                        journal=Path(cell['journal']);offset=journal.stat().st_size if journal.exists() else 0
                        stage='completion'
                        result=completion_fn(dict(cell['config'],deadline=expiry),payload,journal,local=route['local'],
                            endpoint=route['endpoint'],purpose=purpose,budget=budget.for_request(ident,claimed_cell))
                        usage=result.get('usage')
                        if not route['local'] and (not isinstance(usage,dict) or any(type(usage.get(k)) is not int or usage[k]<0
                                for k in ('prompt_tokens','completion_tokens')) or usage['prompt_tokens']>input_bound or usage['completion_tokens']>tokens):
                            budget.halt('broker_observed_token_cap_overrun_or_invalid_usage')
                            raise BrokerError('Observed token cap overrun; shared paid budget halted')
                        stage='receipt'
                        receipt=dict(meta,budget_owner='webprm_worker.completion',
                            **_receipt(cell,payload,purpose,offset,route['local'],budget,meta))
                        result={k:result[k] for k in ('model','choices','usage') if k in result}
                        # Release the cell before replying: a worker's next request can arrive
                        # milliseconds after this reply (359620/361041 Shepherd cells were rejected
                        # as overlapping when release happened only after _send).
                        with admission:
                            record(dict(receipt,event='dispatch_completed',at_epoch=time.time()))
                            active_cells.discard(claimed_cell);claimed_cell=None
                        stage='reply'
                        _send(conn,dict(ok=True,result=result,receipt=receipt),MAX_RESPONSE_BYTES,reply_deadline)
                    except Exception as error:
                        with admission:
                            if meta is None:
                                # Keep rejection diagnostics separate from the admitted
                                # dispatch/financial journals. Never retry the request.
                                try:
                                    diagnostic=rejection_metadata(error,request,stage,cells,routes)
                                    append_rejection(rejection_path,diagnostic)
                                except Exception:
                                    audit_failed.set()  # drain owned handlers; no silent diagnostic loss
                            if meta is not None:
                                record(dict(meta,event='dispatch_failed',at_epoch=time.time(),error_type=type(error).__name__))
                            if claimed_cell is not None:active_cells.discard(claimed_cell);claimed_cell=None
                        try:_send(conn,dict(ok=False,error=type(error).__name__),MAX_RESPONSE_BYTES,reply_deadline)
                        except (OSError,TimeoutError,BrokerError):pass
                    finally:
                        with admission:
                            if claimed_cell is not None:active_cells.discard(claimed_cell)
                        slots.release()
            server=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)
            try:
                server.bind(str(socket_path));os.chmod(socket_path,0o600)
                # Connection bursts can precede the first accept(). Keep a bounded
                # waiting queue separate from the semaphore-limited active handlers.
                # A full AF_UNIX backlog otherwise makes connect() fail with EAGAIN.
                server.listen(max(8,2*max_concurrency))
                with ThreadPoolExecutor(max_workers=max_concurrency,thread_name_prefix='webprm-provider') as pool:
                    if ready is not None:
                        ready.send(dict(status='ready',policy_sha256=policy_hash,pid=os.getpid(),max_concurrency=max_concurrency,
                                        paid_dispatch_interval_seconds=paid_pacer.interval,
                                        budget_backend=BUDGET_VERSION,implementation_sha256=sources));ready.close()
                    while time.time()<deadline and not audit_failed.is_set() and not (stop_event is not None and stop_event.is_set()):
                        if not slots.acquire(timeout=.1):continue
                        try:
                            server.settimeout(min(.1,_timeout(deadline)))
                            conn,_=server.accept()
                        except (socket.timeout,TimeoutError):
                            slots.release();continue
                        try:pool.submit(handle,conn)
                        except BaseException:
                            conn.close();slots.release();raise
            finally:server.close()
            if not audit_failed.is_set():
                record(dict(event='epoch_closed',policy_sha256=policy_hash,epoch_id=epoch_id))
                budget.close_epoch(audit_path)


def _child(kwargs,ready,stop):
    os.setsid()
    try:serve_concurrent_broker(**kwargs,ready=ready,stop_event=stop)
    except BaseException as error:
        try:ready.send(dict(status='failed',error=type(error).__name__))
        except (OSError,ValueError):pass
        raise


class ConcurrentBrokerProcess(BrokerProcess):
    """Opt-in process owner; graceful drain then bounded inherited group reap."""
    def start(self):
        if self.process is not None:raise BrokerError('BrokerProcess is single-use')
        _timeout(_deadline(self.kwargs['deadline']))
        context=multiprocessing.get_context('spawn')
        parent,child=context.Pipe(duplex=False)
        self._admission_stop=context.Event()
        self.process=context.Process(target=_child,args=(self.kwargs,child,self._admission_stop))
        try:
            self.process.start();child.close()
            if not parent.poll(min(self.startup_seconds,_timeout(self.kwargs['deadline']))):raise BrokerError('Broker startup timed out')
            self.ready=parent.recv()
            if self.ready.get('status')!='ready':raise BrokerError('Broker startup failed')
            self._watchdog=threading.Thread(target=self._watch_deadline,daemon=True);self._watchdog.start()
            return self
        except BaseException:
            self.close();raise
        finally:parent.close();child.close()

    def _close(self):
        if getattr(self,'_admission_stop',None) is not None:self._admission_stop.set()
        if self.process is not None and self.process.pid is not None and self.process.is_alive():
            self.process.join(timeout=min(self.shutdown_seconds,max(0,self.kwargs['deadline']-time.time())))
        return super()._close()

    def status(self):
        return read_broker_status(self.kwargs['socket_path'],deadline=self.kwargs['deadline'])


def read_broker_status(socket_path, *, deadline):
    """Same-UID read-only hint; queued handlers may delay this bounded request."""
    until=min(_deadline(deadline),time.time()+5)
    with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as conn:
        conn.settimeout(_timeout(until,5));conn.connect(str(_path(socket_path)))
        _peer(conn,os.getuid());_send(conn,dict(protocol=PROTOCOL,operation='status'),MAX_FRAME_BYTES,until)
        result=_receive(conn,MAX_RESPONSE_BYTES,until)
    if result.get('ok') is not True or not isinstance(result.get('status'),dict):
        raise BrokerError('Broker status unavailable; no implied admission')
    return result['status']
