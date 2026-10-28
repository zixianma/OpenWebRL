"""Prospective, single-owner API accounting with durable append-only events.

The lifetime filesystem lock excludes competing owners. Reservations are fsynced
before returning to the provider caller; failed/unknown calls keep their full
charge. Startup replays once, and each later operation touches only new bytes.
The JSON export is a derived audit snapshot, never the financial authority.
"""
from __future__ import annotations

from copy import deepcopy
from decimal import Decimal, InvalidOperation, localcontext
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import threading
import time
import uuid

from .webprm_broker import _encoded, _path

VERSION = 'single-owner-journal-v2'
ZERO_SHA = '0' * 64


def _sha(value):
    return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)


def _money(value):
    if isinstance(value, bool) or not isinstance(value, (int, float, str, Decimal)):
        raise ValueError('Explicit finite nonnegative financial value required')
    try:
        result = Decimal(str(value))
    except InvalidOperation as error:
        raise ValueError('Invalid financial value') from error
    if not result.is_finite() or result < 0:
        raise ValueError('Finite nonnegative financial value required')
    return result


def _usage(value):
    if not isinstance(value, dict) or any(type(value.get(k)) is not int or value[k] < 0
            for k in ('prompt_tokens', 'completion_tokens')):
        raise ValueError('Exact nonnegative token usage required')
    # Check serializability/finite ancillary fields before appending any event.
    _encoded(value)
    return deepcopy(value)


def _sum(*values):
    # Do not depend on a caller's thread-local Decimal precision. Preserve even
    # a tiny reservation added to a large already charged amount exactly.
    precision=max(1,max(v.adjusted() for v in values)-min(v.as_tuple().exponent for v in values)+3)
    with localcontext() as context:
        context.prec=precision
        return sum(values,Decimal(0))


class JournalBudget:
    """One broker owns this object until every request handler has stopped."""
    def __init__(self, path, cap, *, policy_sha256, financial_contract=None):
        self.path = _path(path)
        self.event_path = _path(str(self.path) + '.events.jsonl')
        self.lock_path = _path(self.path.with_suffix('.lock'))
        if (not isinstance(policy_sha256, str) or len(policy_sha256) != 64 or
                any(x not in '0123456789abcdef' for x in policy_sha256)):
            raise ValueError('Fixed broker policy SHA256 required')
        self.cap = _money(cap)
        self.policy = policy_sha256
        self.contract = deepcopy(financial_contract)
        if self.contract is not None:
            required={'schema_version','budget_backend','cohort_sha256','protocol_sha256','prices_usd_per_token',
                      'api_usd_cap','max_dispatches','max_paid_dispatches','max_local_dispatches'}
            if (set(self.contract)!=required or self.contract['schema_version']!=1 or self.contract['budget_backend']!=VERSION or
                    hashlib.sha256(_encoded(self.contract)).hexdigest()!=policy_sha256 or _money(self.contract['api_usd_cap'])!=self.cap):
                raise ValueError('Exact stable financial contract required')
            for name in ('cohort_sha256','protocol_sha256'):
                value=self.contract[name]
                if not isinstance(value,str) or len(value)!=64 or any(x not in '0123456789abcdef' for x in value):
                    raise ValueError('Frozen cohort/protocol hashes required')
            for name in ('max_dispatches','max_paid_dispatches','max_local_dispatches'):
                if type(self.contract[name]) is not int or not 0<=self.contract[name]<=self.contract['max_dispatches']:
                    raise ValueError('Stable nonnegative cumulative dispatch caps required')
            prices=self.contract['prices_usd_per_token']
            if not isinstance(prices,dict) or not prices or any(not isinstance(m,str) or not m or not isinstance(r,list) or
                    len(r)!=2 or any(_money(v)<=0 for v in r) for m,r in prices.items()):
                raise ValueError('Fixed positive model prices required')
        self._mutex = threading.RLock()
        self._closed = False
        self._poisoned = False
        self._calls = []
        self._reserved_requests = set()
        self._charged = Decimal(0)
        self._reserved_count = 0
        self._reserved_amount = Decimal(0)
        self._halt_reason = None
        # A positive residual does not imply another episode can fund its next
        # request. Keep this owner-local admission latch separate from charges
        # and hard halts: already admitted work may still settle or fit.
        self._reservation_rejection = None
        self._sequence = 0
        self._previous = ZERO_SHA
        self._prefix_hash = hashlib.sha256()
        self._prefix_bytes = 0
        self._header = False
        self._dispatches = {}
        self._dispatch_counts = {False:0,True:0}
        self._epoch_ids = set()
        self._active_epoch = None
        self._owned_epoch = None
        self._last_closed_epoch = None
        self._stream = self._file_lock = None
        try:
            fd = os.open(self.lock_path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
            self._file_lock = os.fdopen(fd, 'a+')
            fcntl.flock(self._file_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if self.path.exists() and not self.event_path.exists():
                raise ValueError('Legacy budget cannot be silently migrated')
            fd = os.open(self.event_path, os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW, 0o600)
            self._stream = os.fdopen(fd, 'a+b')
            self._stream.seek(0)
            for raw in self._stream:
                if not raw.endswith(b'\n'):
                    raise ValueError('Incomplete budget tail: preserve and review; no automatic repair')
                event = json.loads(raw)
                self._verify(event)
                self._apply(event)
                self._advance(event, raw)
            if not self._header:
                self._append(dict(event='header', version=VERSION, cap=str(self.cap), policy_sha256=self.policy,
                                  financial_contract=self.contract))
        except BaseException:
            if self._stream is not None: self._stream.close()
            if self._file_lock is not None: self._file_lock.close()
            self._closed = True
            raise

    def _verify(self, event):
        core = {k:v for k,v in event.items() if k != 'event_sha256'}
        if (event.get('sequence') != self._sequence or event.get('previous_sha256') != self._previous or
                event.get('event_sha256') != hashlib.sha256(_encoded(core)).hexdigest()):
            raise ValueError('Budget event sequence/hash mismatch')

    def _apply(self, event):
        kind = event.get('event')
        if not self._header:
            if (kind != 'header' or event.get('version') != VERSION or _money(event.get('cap')) != self.cap or
                    event.get('policy_sha256') != self.policy or event.get('financial_contract')!=self.contract):
                raise ValueError('Budget header version/cap/policy mismatch')
            self._header = True
            return
        if kind == 'reserve':
            amount = _money(event['amount'])
            if (self._halt_reason or type(event['id']) is not int or event['id'] != len(self._calls) or
                    not isinstance(event['model'], str) or not event['model'] or _sum(self._charged,amount) > self.cap):
                raise ValueError('Invalid durable reservation')
            if self.contract is not None:
                dispatch=self._dispatches.get(event.get('request_id'))
                if (self._active_epoch is None or dispatch is None or dispatch['local'] or
                        dispatch['epoch_id']!=self._active_epoch['epoch_id'] or
                        dispatch['cell_id']!=event.get('cell_id') or dispatch['model']!=event['model'] or
                        event['request_id'] in self._reserved_requests):
                    raise ValueError('Reservation lacks one exact admitted paid request')
                self._reserved_requests.add(event['request_id'])
            self._charged = _sum(self._charged,amount)
            self._reserved_count += 1
            self._reserved_amount = _sum(self._reserved_amount,amount)
            self._calls.append(dict(id=event['id'],model=event['model'],reserved=amount,state='reserved',
                request_id=event.get('request_id'),cell_id=event.get('cell_id')))
        elif kind == 'settle':
            ident = event['id']
            if type(ident) is not int or not 0 <= ident < len(self._calls):
                raise ValueError('Unknown reservation')
            row = self._calls[ident]
            actual, usage = _money(event['actual']), _usage(event['usage'])
            if row['state'] != 'reserved':raise ValueError('Reservation already settled')
            self._charged = _sum(self._charged,actual,row['reserved'].copy_negate())
            self._reserved_count -= 1
            self._reserved_amount = _sum(self._reserved_amount,row['reserved'].copy_negate())
            overrun = actual > row['reserved']
            row.update(state='overrun' if overrun else 'settled',charged=actual,usage=usage,
                       settlement_event_sha256=event['event_sha256'])
            if overrun:self._halt_reason = 'observed_usage_exceeded_reservation'
        elif kind == 'halt':
            if not isinstance(event.get('reason'), str) or not event['reason']:
                raise ValueError('Explicit halt reason required')
            self._halt_reason = event['reason']
        elif kind == 'epoch_started':
            self._validate_epoch_start(event)
            self._active_epoch=deepcopy(event)
            self._epoch_ids.add(event['epoch_id'])
        elif kind == 'dispatch':
            meta=event['dispatch']
            local=self._validate_dispatch(meta)
            if event['epoch_id']!=self._active_epoch['epoch_id']:raise ValueError('Wrong dispatch epoch')
            self._dispatches[meta['request_id']]=dict(deepcopy(meta),epoch_id=event['epoch_id'])
            self._dispatch_counts[local]+=1
        elif kind == 'epoch_closed':
            if self._active_epoch is None or event['epoch_id']!=self._active_epoch['epoch_id']:
                raise ValueError('No matching open epoch')
            self._last_closed_epoch=deepcopy(event)
            self._active_epoch=None
        else:
            raise ValueError('Unknown or repeated budget header/event')

    def _advance(self, event, raw):
        self._sequence += 1
        self._previous = event['event_sha256']
        self._prefix_hash.update(raw)
        self._prefix_bytes += len(raw)

    def _check(self):
        if self._closed or self._poisoned:
            raise RuntimeError('Budget owner closed or durability uncertain; no dispatch')

    def _append(self, fields):
        self._check()
        event = dict(fields,sequence=self._sequence,previous_sha256=self._previous,at_epoch=time.time())
        event['event_sha256'] = hashlib.sha256(_encoded(event)).hexdigest()
        raw = _encoded(event) + b'\n'
        try:
            self._stream.write(raw)
            self._stream.flush()
            os.fsync(self._stream.fileno())
            if self._sequence == 0:
                directory = os.open(self.path.parent,os.O_RDONLY | os.O_DIRECTORY)
                try:os.fsync(directory)
                finally:os.close(directory)
            self._apply(event)
            self._advance(event,raw)
        except BaseException:
            self._poisoned = True
            raise

    def reserve(self, amount, model, *, request_id=None, cell_id=None):
        amount = _money(amount)
        if not isinstance(model,str) or not model:raise ValueError('Model required')
        with self._mutex:
            self._check()
            if not self._halt_reason and _sum(self._charged,amount) > self.cap:
                self._reservation_rejection = dict(amount=str(amount),model=model,
                    remaining=str(max(Decimal(0),_sum(self.cap,self._charged.copy_negate()))))
            if self._halt_reason or _sum(self._charged,amount) > self.cap:
                raise RuntimeError('Shared API budget exhausted or halted')
            if self.contract is not None:
                dispatch=self._dispatches.get(request_id)
                if (self._active_epoch is None or self._owned_epoch!=self._active_epoch['epoch_id'] or
                        dispatch is None or dispatch['local'] or dispatch['cell_id']!=cell_id or
                        dispatch['epoch_id']!=self._owned_epoch or
                        dispatch['model']!=model or request_id in self._reserved_requests):
                    raise ValueError('Reservation must belong to exactly one admitted paid request')
            ident = len(self._calls)
            self._append(dict(event='reserve',id=ident,amount=str(amount),model=model,
                              request_id=request_id,cell_id=cell_id))
            return ident

    def settle(self, ident, actual, usage, *, request_id=None):
        actual, usage = _money(actual), _usage(usage)
        with self._mutex:
            self._check()
            if (type(ident) is not int or not 0 <= ident < len(self._calls) or
                    self._calls[ident]['state'] != 'reserved'):
                raise ValueError('Unknown or already settled reservation')
            if self.contract is not None:
                row=self._calls[ident]
                if (self._owned_epoch is None or row['request_id']!=request_id or
                        self._dispatches[request_id]['epoch_id']!=self._owned_epoch):
                    raise ValueError('Only the exact request owner may settle its current-epoch reservation')
            self._append(dict(event='settle',id=ident,actual=str(actual),usage=usage))
            if self._calls[ident]['state'] == 'overrun':
                raise RuntimeError('Observed usage exceeded reservation; paid calls halted')

    def halt(self, reason):
        if not isinstance(reason,str) or not reason:raise ValueError('Halt reason required')
        with self._mutex:self._append(dict(event='halt',reason=reason))

    def receipt(self, ident):
        with self._mutex:
            self._check()
            if type(ident) is not int or not 0 <= ident < len(self._calls):raise ValueError('Unknown reservation')
            row = self._calls[ident]
            if row['state'] != 'settled':raise ValueError('A settled reservation is required')
            return dict(reservation_id=ident,reservation_state='settled',model=row['model'],usage=deepcopy(row['usage']),
                request_id=row['request_id'],cell_id=row['cell_id'],
                settlement_event_sha256=row['settlement_event_sha256'],budget_ledger_sha256=self._prefix_hash.hexdigest(),
                budget_prefix_bytes=self._prefix_bytes,budget_backend=VERSION)

    def for_request(self, request_id, cell_id):
        owner = self
        class Adapter:
            def reserve(self,amount,model):return owner.reserve(amount,model,request_id=request_id,cell_id=cell_id)
            def settle(self,ident,actual,usage):return owner.settle(ident,actual,usage,request_id=request_id)
        return Adapter()

    def snapshot(self):
        """Explicit O(n) audit export; never used on the per-request path."""
        with self._mutex:
            self._check()
            calls = deepcopy(self._calls)
            for row in calls:
                for field in ('reserved','charged'):
                    if field in row:row[field]=float(row[field])
            result = dict(cap=float(self.cap),charged=float(self._charged),calls=calls,backend=VERSION,
                policy_sha256=self.policy,event_journal=str(self.event_path),
                budget_prefix_bytes=self._prefix_bytes,budget_ledger_sha256=self._prefix_hash.hexdigest(),
                cap_exact=str(self.cap),charged_exact=str(self._charged),financial_contract=deepcopy(self.contract),
                dispatches=len(self._dispatches),paid_dispatches=self._dispatch_counts[False],local_dispatches=self._dispatch_counts[True],
                active_epoch_id=self._active_epoch['epoch_id'] if self._active_epoch else None,
                last_closed_epoch_sha256=self._last_closed_epoch['event_sha256'] if self._last_closed_epoch else None)
            if self._halt_reason:result['halt_reason']=self._halt_reason
            return result

    def summary(self):
        """O(1), read-only admission hint; reserve remains the sole authority."""
        with self._mutex:
            self._check()
            return dict(cap=str(self.cap),charged=str(self._charged),remaining=str(max(Decimal(0),_sum(self.cap,self._charged.copy_negate()))),
                reservations=len(self._calls),unsettled_reservations=self._reserved_count,
                unsettled_reserved=str(self._reserved_amount),halt_reason=self._halt_reason,
                reservation_rejection=deepcopy(self._reservation_rejection),
                budget_prefix_bytes=self._prefix_bytes,budget_ledger_sha256=self._prefix_hash.hexdigest(),
                budget_backend=VERSION,dispatches=len(self._dispatches),paid_dispatches=self._dispatch_counts[False],
                local_dispatches=self._dispatch_counts[True],active_epoch_id=self._active_epoch['epoch_id'] if self._active_epoch else None,
                last_closed_epoch_sha256=self._last_closed_epoch['event_sha256'] if self._last_closed_epoch else None)

    def start_epoch(self, *, epoch_id, broker_policy_sha256, deadline, implementation_sha256,
                    registered_cells_sha256, previous_closed_epoch_sha256):
        with self._mutex:
            self._check()
            fields=dict(event='epoch_started',epoch_id=epoch_id,broker_policy_sha256=broker_policy_sha256,
                deadline=deadline,implementation_sha256=deepcopy(implementation_sha256),registered_cells_sha256=registered_cells_sha256,
                previous_closed_epoch_sha256=previous_closed_epoch_sha256)
            self._validate_epoch_start(fields)
            if self._last_closed_epoch:
                prior=self._last_closed_epoch
                path=_path(prior['audit_path'])
                raw=path.read_bytes()
                if len(raw)!=prior['audit_prefix_bytes'] or hashlib.sha256(raw).hexdigest()!=prior['audit_sha256']:
                    raise ValueError('Previous closed audit changed; no successor epoch')
            self._append(fields)
            self._owned_epoch=epoch_id

    def _validate_epoch_start(self, fields):
        expected=self._last_closed_epoch['event_sha256'] if self._last_closed_epoch else None
        ident=fields.get('epoch_id');sources=fields.get('implementation_sha256');deadline=fields.get('deadline')
        if (self.contract is None or self._active_epoch is not None or not isinstance(ident,str) or not 1<=len(ident)<=128 or
                ident in self._epoch_ids or fields.get('previous_closed_epoch_sha256')!=expected or
                not _sha(fields.get('broker_policy_sha256')) or not _sha(fields.get('registered_cells_sha256')) or
                type(deadline) not in (int,float) or not math.isfinite(deadline) or deadline<=0 or
                not isinstance(sources,dict) or not sources or any(not isinstance(k,str) or not k or not _sha(v) for k,v in sources.items())):
            raise ValueError('Unique epoch, source/policy hashes, deadline and exact closed predecessor required')

    def _validate_dispatch(self, meta):
        if not isinstance(meta,dict):raise ValueError('Explicit dispatch identity required')
        ident=meta.get('request_id');local=meta.get('local')
        if (self._active_epoch is None or not isinstance(ident,str) or len(ident)!=32 or
                any(c not in '0123456789abcdef' for c in ident) or type(local) is not bool or
                not isinstance(meta.get('cell_id'),str) or not meta['cell_id'] or
                not isinstance(meta.get('model'),str) or not meta['model'] or
                meta.get('policy_sha256')!=self._active_epoch['broker_policy_sha256'] or
                not local and meta['model'] not in self.contract['prices_usd_per_token']):
            raise ValueError('Exact current epoch/cell/model/request identity required')
        if (ident in self._dispatches or len(self._dispatches)>=self.contract['max_dispatches'] or
                self._dispatch_counts[local]>=self.contract['max_local_dispatches' if local else 'max_paid_dispatches']):
            raise RuntimeError('Cumulative dispatch cap exhausted or request duplicated')
        return local

    def admit_dispatch(self, meta):
        with self._mutex:
            self._check()
            self._validate_dispatch(meta)
            if self._owned_epoch!=self._active_epoch['epoch_id']:raise ValueError('Replayed epoch needs recovery proof')
            self._append(dict(event='dispatch',epoch_id=self._active_epoch['epoch_id'],dispatch=deepcopy(meta)))

    def close_epoch(self, audit_path):
        with self._mutex:
            self._check()
            if self._active_epoch is None or self._owned_epoch!=self._active_epoch['epoch_id']:
                raise ValueError('Only this owning process can close an epoch without recovery proof')
            path=_path(audit_path)
            raw=path.read_bytes()
            self._append(dict(event='epoch_closed',epoch_id=self._active_epoch['epoch_id'],audit_path=str(path),
                audit_prefix_bytes=len(raw),audit_sha256=hashlib.sha256(raw).hexdigest()))
            self._owned_epoch=None
            return self._last_closed_epoch['event_sha256']

    def recover_epoch(self, *, audit_path, closure_receipt_path, audit_receipt_path):
        """Explicit offline recovery after independent closure and receipt audits.

        This verifies supplied audit artifacts and their exact prefix bindings;
        it does not contact Slurm or establish process death itself. The trusted
        controller/auditor must verify scheduler terminal state and reap all
        owned workers before producing these receipts. No costs are refunded,
        IDs/caps reset, tail bytes repaired, or requests retried by this method.
        """
        with self._mutex:
            self._check()
            if self._active_epoch is None or self._owned_epoch is not None:
                raise ValueError('Only a replayed, unowned open epoch needs recovery')
            epoch=self._active_epoch
            proof_paths=[_path(closure_receipt_path),_path(audit_receipt_path)]
            raws=[p.read_bytes() for p in proof_paths]
            closure,audit=[json.loads(raw) for raw in raws]
            for proof in (closure,audit):
                if (proof.get('schema_version')!=1 or proof.get('verified') is not True or
                        proof.get('epoch_id')!=epoch['epoch_id'] or proof.get('broker_policy_sha256')!=epoch['broker_policy_sha256'] or
                        proof.get('errors')!=[]):
                    raise ValueError('Verified matching closure and receipt audits required')
            if (closure.get('scheduler_terminal') is not True or closure.get('all_owned_processes_reaped') is not True or
                    closure.get('scheduler_state') not in {'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED'} or
                    not isinstance(closure.get('job_id'),str) or not closure['job_id']):
                raise ValueError('Terminal scheduler and owned-process closure proof required')
            for name in ('scheduler_evidence','process_closure_evidence'):
                evidence=closure.get(name)
                if not isinstance(evidence,dict) or set(evidence)!={'path','sha256'}:
                    raise ValueError('Closure evidence paths and hashes required')
                if hashlib.sha256(_path(evidence['path']).read_bytes()).hexdigest()!=evidence['sha256']:
                    raise ValueError('Underlying closure evidence changed')
            path=_path(audit_path);raw=path.read_bytes()
            summary=self.summary()
            expected=dict(audit_path=str(path),audit_prefix_bytes=len(raw),audit_sha256=hashlib.sha256(raw).hexdigest(),
                budget_prefix_bytes=self._prefix_bytes,budget_ledger_sha256=self._prefix_hash.hexdigest(),
                dispatches=summary['dispatches'],paid_dispatches=summary['paid_dispatches'],local_dispatches=summary['local_dispatches'],
                unknown_reservations=summary['unsettled_reservations'],unknown_reservations_retained=True)
            if (any(audit.get(k)!=v for k,v in expected.items()) or raw and not raw.endswith(b'\n')):
                raise ValueError('Final receipt audit does not bind exact durable counts/prefixes')
            for line in raw.splitlines():json.loads(line)
            self._append(dict(event='epoch_closed',epoch_id=epoch['epoch_id'],audit_path=str(path),
                audit_prefix_bytes=len(raw),audit_sha256=expected['audit_sha256'],closure='audited_recovery',
                recovery_evidence=[dict(path=str(p),sha256=hashlib.sha256(raw).hexdigest()) for p,raw in zip(proof_paths,raws)]))
            return self._last_closed_epoch['event_sha256']

    def close(self):
        with self._mutex:
            if self._closed:return
            try:
                # A tainted append must not create a deceptively complete snapshot.
                if not self._poisoned:
                    temporary = _path(str(self.path)+'.export-'+uuid.uuid4().hex)
                    fd=os.open(temporary,os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,0o600)
                    with os.fdopen(fd,'wb') as stream:
                        stream.write(_encoded(self.snapshot())+b'\n');stream.flush();os.fsync(stream.fileno())
                    _path(self.path)
                    os.replace(temporary,self.path)
                    directory=os.open(self.path.parent,os.O_RDONLY | os.O_DIRECTORY)
                    try:os.fsync(directory)
                    finally:os.close(directory)
            finally:
                self._closed=True
                self._stream.close();self._file_lock.close()

    def __enter__(self):return self
    def __exit__(self,*_):self.close()
