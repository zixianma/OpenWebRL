"""Bounded, evaluator-owned SMTP acceptance capture; never delivers mail.

Run inside the application's isolated network namespace, with this output
directory mounted only into the collector. Loopback binding is not a substitute
for that isolation. Each accepted DATA is one irreversible sink effect, including
repeated identical Message-IDs. Its complete, dot-unstuffed bytes and envelope are
fsynced in a hash-chained append-only journal BEFORE replying 250. A lost reply
can therefore produce two receipts when a client retries, as real SMTP can.

The journal is not tamper-proof against its owner. Retain its final hash outside
the application boundary. A missing end, cap exhaustion, interrupted DATA, or
failed persistence is invalid coverage. Clean closure proves this sink's own
coverage only: application routing and native queue drain still need independent
validation. SMTP carries no trusted Magento IDs. Projection requires separately
verified linkage and DB chronology; MIME headers alone cannot supply either.
"""
from __future__ import annotations

import base64
import binascii
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import select
import socket
import threading
import time
import uuid


class CaptureError(ValueError):
    """Capture or linkage cannot establish the claimed evidence."""


class _Stopped(Exception):
    pass


class _CapExceeded(Exception):
    pass


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=True, allow_nan=False).encode('ascii')


def digest(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _require(condition, reason):
    if not condition:
        raise CaptureError(reason)


def _hash(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def _address(value, *, allow_null=False):
    return isinstance(value, str) and ((allow_null and value == '') or (
        value.count('@') == 1 and all(value.split('@')) and
        all(32 < ord(c) < 127 and c not in '<>\\' for c in value)))


def _writable_directory(value):
    path = Path(value)
    _require(path.is_absolute(), 'Capture output path must be absolute')
    lexical, resolved = Path(os.path.abspath(path)), path.resolve()
    forbidden = Path('/gpfs/home/zixianma')
    for candidate in (path, lexical, resolved):
        _require(candidate != forbidden and forbidden not in candidate.parents,
                 'Protected home path cannot be used for capture')
    roots = tuple(map(Path, ('/gpfs/projects/krishna/zixianma/OpenWebRL',
                            '/gpfs/scrubbed/zixianma/openwebrl-runtime',
                            '/gpfs/scrubbed/zixianma/checkpoints/web', '/tmp')))
    _require(all(any(candidate == root or root in candidate.parents for root in roots)
                 for candidate in (lexical, resolved)), 'Output is outside approved writable roots')
    _require(not any(component.is_symlink() for component in (path, *path.parents)),
             'Output contains a symlink component')
    return resolved


@dataclass(frozen=True)
class SinkLimits:
    max_messages: int
    max_message_bytes: int
    max_storage_bytes: int
    max_runtime_seconds: float
    max_connections: int = 256
    max_commands_per_connection: int = 1000
    max_recipients: int = 100
    max_line_bytes: int = 65536
    idle_seconds: float = 15.0
    heartbeat_seconds: float = 5.0

    def __post_init__(self):
        for key in ('max_messages', 'max_message_bytes', 'max_storage_bytes',
                    'max_connections', 'max_commands_per_connection',
                    'max_recipients', 'max_line_bytes'):
            _require(type(getattr(self, key)) is int and getattr(self, key) > 0,
                     'Positive integer cap required: ' + key)
        for key in ('max_runtime_seconds', 'idle_seconds', 'heartbeat_seconds'):
            value = getattr(self, key)
            _require(type(value) in (int, float) and math.isfinite(value) and value > 0,
                     'Positive finite duration required: ' + key)
        _require(self.max_storage_bytes >= 16384, 'Storage cap must include 16 KiB for metadata')
        _require(self.max_line_bytes >= 512, 'Line cap must cover SMTP command lines')


class _Journal:
    """New capture only; no silent append/resume of a crashed capture."""

    def __init__(self, directory, campaign_id, sink_id, limits):
        directory = _writable_directory(directory)
        for name, value in (('campaign_id', campaign_id), ('sink_id', sink_id)):
            _require(isinstance(value, str) and re.fullmatch(r'[A-Za-z0-9_.:-]{1,128}', value),
                     'Invalid ' + name)
        directory.mkdir(mode=0o700, parents=False, exist_ok=False)
        parent_fd = os.open(directory.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
        self.path = directory / 'journal.jsonl'
        self.fd = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        directory_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        self.campaign_id, self.sink_id, self.limits = campaign_id, sink_id, limits
        self.sequence, self.bytes_written, self.last_sha256 = 0, 0, '0' * 64
        self.failed = False

    def append(self, kind, *, final=False, **fields):
        _require(not self.failed and self.fd is not None, 'Journal is unavailable')
        record = dict(schema_version=1, campaign_id=self.campaign_id, sink_id=self.sink_id,
                      seq=self.sequence + 1, kind=kind, utc_ns=time.time_ns(),
                      monotonic_ns=time.monotonic_ns(), previous_sha256=self.last_sha256, **fields)
        record['sha256'] = digest(record)
        encoded = _canonical(record) + b'\n'
        # Reserve room for a durable end even if data/control records fill the cap.
        ceiling = self.limits.max_storage_bytes - (0 if final else 8192)
        if self.bytes_written + len(encoded) > ceiling:
            raise _CapExceeded('storage_cap')
        try:
            pending = memoryview(encoded)
            while pending:
                written = os.write(self.fd, pending)
                if written <= 0:
                    raise OSError('Short journal write')
                pending = pending[written:]
            os.fsync(self.fd)
        except BaseException:
            self.failed = True
            raise
        self.sequence += 1
        self.bytes_written += len(encoded)
        self.last_sha256 = record['sha256']
        return record

    def close(self):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None


class SmtpSink:
    """Single-session-at-a-time SMTP server with explicit aggregate limits.

    Construct before starting policy execution. Call serve_forever in its owned
    process/thread; request_stop after policy/queue quiescence. The controller
    must independently retain start_sha256 and final_record['sha256']. Runtime
    expiration is a failed capture, not a clean end. No automatic restart occurs.
    """

    def __init__(self, output_dir, *, campaign_id, sink_id, limits, port=0,
                 host='127.0.0.1'):
        _require(isinstance(limits, SinkLimits), 'Explicit SinkLimits required')
        _require(host == '127.0.0.1', 'Only IPv4 loopback binding is permitted')
        _require(type(port) is int and 0 <= port <= 65535, 'Invalid port')
        self.limits = limits
        self.listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            self.listener.bind((host, port))
            self.listener.listen(8)
            self.journal = _Journal(output_dir, campaign_id, sink_id, limits)
            self.started = time.monotonic()
            self.last_heartbeat = self.started
            self.stopped = threading.Event()
            self.reason = 'requested'
            self.capture_valid = True
            self.messages = self.raw_bytes = self.connections = self.rejections = 0
            self.active_connections = self.aborted_data = 0
            self.final_record = None
            self.address = self.listener.getsockname()
            start = self.journal.append('start', limits=asdict(limits),
                listen_host=host, listen_port=self.address[1], pid=os.getpid(),
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                semantics='durable_local_smtp_acceptance',
                external_delivery=False, directory_mode='0700', journal_mode='0600')
            self.start_sha256 = start['sha256']
        except BaseException:
            self.listener.close()
            if hasattr(self, 'journal'):
                self.journal.close()
            raise

    def request_stop(self):
        self.stopped.set()

    def _invalidate(self, reason):
        self.capture_valid = False
        self.reason = reason
        self.stopped.set()

    def _maintenance(self):
        if self.stopped.is_set():
            raise _Stopped()
        now = time.monotonic()
        if now - self.started >= self.limits.max_runtime_seconds:
            self._invalidate('runtime_cap')
            raise _Stopped()
        if now - self.last_heartbeat >= self.limits.heartbeat_seconds:
            self.journal.append('health', accepted_messages=self.messages,
                                accepted_raw_bytes=self.raw_bytes,
                                active_connections=self.active_connections,
                                rejections=self.rejections, storage_bytes=self.journal.bytes_written)
            self.last_heartbeat = now

    def _reject(self, conn, code, reason):
        self.journal.append('rejected', code=code, reason=reason)
        self.rejections += 1
        conn.sendall((str(code) + ' ' + reason + '\r\n').encode('ascii'))

    def _session(self, conn, peer):
        conn.settimeout(1.0)
        connection_id = uuid.uuid4().hex
        buffer = bytearray()
        in_data = False

        def readline(limit):
            last_progress = time.monotonic()
            while True:
                self._maintenance()
                end = buffer.find(b'\n')
                if end >= 0:
                    if end + 1 > limit:
                        raise _CapExceeded('line_cap')
                    line = bytes(buffer[:end + 1])
                    del buffer[:end + 1]
                    return line
                if len(buffer) >= limit:
                    raise _CapExceeded('line_cap')
                if time.monotonic() - last_progress >= self.limits.idle_seconds:
                    raise TimeoutError('idle_timeout')
                ready, _, _ = select.select([conn], [], [], 0.1)
                if ready:
                    block = conn.recv(min(8192, limit - len(buffer)))
                    if not block:
                        raise EOFError()
                    buffer.extend(block)
                    last_progress = time.monotonic()

        sender, recipients, greeted = None, [], False
        try:
            conn.sendall(b'220 local-capture ESMTP\r\n')
            for _ in range(self.limits.max_commands_per_connection):
                raw = readline(512)
                if not raw.endswith(b'\r\n'):
                    self._reject(conn, 500, 'CRLF_required')
                    return
                try:
                    command = raw[:-2].decode('ascii')
                except UnicodeDecodeError:
                    self._reject(conn, 500, 'ASCII_command_required')
                    continue
                if any(ord(c) < 32 or ord(c) == 127 for c in command):
                    self._reject(conn, 500, 'invalid_command')
                    continue
                verb, _, arg = command.partition(' ')
                verb = verb.upper()
                if verb in ('HELO', 'EHLO') and arg:
                    greeted, sender, recipients = True, None, []
                    if verb == 'EHLO':
                        conn.sendall(('250-local-capture\r\n250-SIZE ' +
                                      str(self.limits.max_message_bytes) + '\r\n250 8BITMIME\r\n').encode())
                    else:
                        conn.sendall(b'250 local-capture\r\n')
                elif verb == 'QUIT' and not arg:
                    conn.sendall(b'221 closing\r\n')
                    return
                elif verb == 'NOOP':
                    conn.sendall(b'250 ok\r\n')
                elif verb == 'RSET' and not arg:
                    sender, recipients = None, []
                    conn.sendall(b'250 reset\r\n')
                elif verb == 'MAIL':
                    sender, recipients = None, []
                    match = re.fullmatch(r'FROM:<([^<>]*)>(?: (.*))?', arg, re.I)
                    if not greeted or not match or not _address(match[1], allow_null=True):
                        self._reject(conn, 503 if not greeted else 501, 'invalid_MAIL')
                        continue
                    parameters = (match[2] or '').upper().split()
                    if (len(parameters) != len({p.split('=')[0] for p in parameters}) or
                            any(not re.fullmatch(r'SIZE=[0-9]+|BODY=(?:8BITMIME|7BIT)', p)
                                for p in parameters)):
                        self._reject(conn, 555, 'unsupported_MAIL_parameters')
                        continue
                    if any(p.startswith('SIZE=') and int(p[5:]) > self.limits.max_message_bytes
                           for p in parameters):
                        raise _CapExceeded('message_size_cap')
                    sender = match[1]
                    conn.sendall(b'250 sender_ok\r\n')
                elif verb == 'RCPT':
                    match = re.fullmatch(r'TO:<([^<>]*)>', arg, re.I)
                    if sender is None or not match or not _address(match[1]):
                        self._reject(conn, 503 if sender is None else 501, 'invalid_RCPT')
                        continue
                    if len(recipients) >= self.limits.max_recipients:
                        raise _CapExceeded('recipient_cap')
                    recipients.append(match[1])
                    conn.sendall(b'250 recipient_ok\r\n')
                elif verb == 'DATA' and not arg:
                    if sender is None or not recipients:
                        self._reject(conn, 503, 'MAIL_and_RCPT_required')
                        continue
                    if self.messages >= self.limits.max_messages:
                        raise _CapExceeded('message_count_cap')
                    conn.sendall(b'354 end_with_dot\r\n')
                    in_data = True
                    body = bytearray()
                    while True:
                        line = readline(self.limits.max_line_bytes)
                        if not line.endswith(b'\r\n'):
                            self._invalidate('invalid_DATA_line_ending')
                            self._reject(conn, 554, 'CRLF_required')
                            return
                        if line == b'.\r\n':
                            break
                        if line.startswith(b'.'):
                            line = line[1:]
                        if len(body) + len(line) > self.limits.max_message_bytes:
                            raise _CapExceeded('message_size_cap')
                        body.extend(line)
                    receipt_id = self.journal.sink_id + ':' + uuid.uuid4().hex
                    self.journal.append('accepted', receipt_id=receipt_id,
                        connection_id=connection_id, peer_ip=peer[0],
                        envelope_sender=sender, envelope_recipients=list(recipients),
                        raw_message_b64=base64.b64encode(body).decode('ascii'),
                        raw_message_sha256=hashlib.sha256(body).hexdigest(),
                        raw_message_bytes=len(body), status='captured')
                    # Acceptance is durable even if this reply never reaches the client.
                    self.messages += 1
                    self.raw_bytes += len(body)
                    in_data = False
                    sender, recipients = None, []
                    conn.sendall(('250 accepted ' + receipt_id + '\r\n').encode('ascii'))
                else:
                    self._reject(conn, 502, 'unsupported_command')
            raise _CapExceeded('command_count_cap')
        except _CapExceeded as exc:
            self._invalidate(str(exc))
            try:
                conn.sendall(b'452 capture_capacity_exhausted\r\n')
            except OSError:
                pass
        except (EOFError, TimeoutError, ConnectionError, socket.timeout):
            # Disconnect after durable DATA is ordinary; an interrupted DATA is
            # conservatively invalid until a new independently bounded capture.
            pass
        finally:
            if in_data:
                self.aborted_data += 1
                self._invalidate('interrupted_DATA')

    def serve_forever(self):
        failure = None
        try:
            while True:
                self._maintenance()
                ready, _, _ = select.select([self.listener], [], [], 0.1)
                if not ready:
                    continue
                conn, peer = self.listener.accept()
                with conn:
                    if peer[0] != '127.0.0.1':
                        self._invalidate('non_loopback_peer')
                        break
                    if self.connections >= self.limits.max_connections:
                        raise _CapExceeded('connection_count_cap')
                    self.connections += 1
                    self.active_connections = 1
                    try:
                        self._session(conn, peer)
                    finally:
                        self.active_connections = 0
        except _Stopped:
            pass
        except _CapExceeded as exc:
            self._invalidate(str(exc))
        except BaseException as exc:
            failure = exc
            self._invalidate('collector_error')
        finally:
            self.listener.close()
            try:
                if not self.journal.failed:
                    self.final_record = self.journal.append('end', final=True,
                        capture_valid=self.capture_valid, reason=self.reason,
                        accepted_messages=self.messages, accepted_raw_bytes=self.raw_bytes,
                        connections=self.connections, rejections=self.rejections,
                        active_connections=self.active_connections, aborted_data=self.aborted_data,
                        dropped_events=0, routing_verified=False, queue_drain_verified=False)
            finally:
                self.journal.close()
        if failure is not None:
            raise failure
        return self.final_record


def validate_capture(directory, *, expected_final_sha256, max_read_bytes=128 * 1024 * 1024):
    """Read sealed clean sink evidence. This does not assert application coverage."""
    _require(_hash(expected_final_sha256), 'Independently retained final SHA256 required')
    path = Path(directory) / 'journal.jsonl'
    _require(0 < path.stat().st_size <= max_read_bytes, 'Journal exceeds read cap or is empty')
    records, previous, last_monotonic = [], '0' * 64, -1
    with path.open('rb') as stream:
        for line in stream:
            _require(line.endswith(b'\n'), 'Truncated journal record')
            try:
                record = json.loads(line)
            except (ValueError, UnicodeDecodeError) as exc:
                raise CaptureError('Invalid journal JSON') from exc
            _require(isinstance(record, dict), 'Journal record is not an object')
            supplied = record.get('sha256')
            _require(supplied == digest({k: v for k, v in record.items() if k != 'sha256'}),
                     'Journal record hash mismatch')
            _require(record.get('previous_sha256') == previous, 'Broken journal hash chain')
            _require(type(record.get('seq')) is int and record['seq'] == len(records) + 1,
                     'Non-contiguous journal sequence')
            _require(type(record.get('schema_version')) is int and record['schema_version'] == 1,
                     'Unsupported journal version')
            _require(type(record.get('monotonic_ns')) is int and
                     record['monotonic_ns'] >= last_monotonic, 'Invalid monotonic timestamp')
            records.append(record)
            previous, last_monotonic = supplied, record['monotonic_ns']
    _require(previous == expected_final_sha256, 'Journal differs from independent final seal')
    _require(len(records) >= 2 and records[0].get('kind') == 'start' and
             records[-1].get('kind') == 'end', 'Missing start or end')
    start, end = records[0], records[-1]
    _require(start.get('listen_host') == '127.0.0.1' and start.get('external_delivery') is False and
             start.get('semantics') == 'durable_local_smtp_acceptance', 'Invalid sink semantics')
    _require(_hash(start.get('source_sha256')), 'Missing collector source identity')
    try:
        limits = SinkLimits(**start['limits'])
    except (KeyError, TypeError) as exc:
        raise CaptureError('Invalid capture limits') from exc
    _require(path.stat().st_size <= limits.max_storage_bytes, 'Recorded storage cap exceeded')
    _require(end.get('capture_valid') is True and end.get('reason') == 'requested' and
             end.get('active_connections') == 0 and end.get('aborted_data') == 0 and
             end.get('dropped_events') == 0, 'Capture ended incompletely or exceeded a cap')
    accepted, ids, rejected = [], set(), 0
    for record in records:
        _require(record.get('sink_id') == start.get('sink_id') and
                 record.get('campaign_id') == start.get('campaign_id'), 'Mixed capture identities')
    for record in records[1:-1]:
        kind = record.get('kind')
        _require(kind in ('accepted', 'health', 'rejected'), 'Unexpected journal event')
        if kind == 'rejected':
            rejected += 1
        elif kind == 'accepted':
            rid = record.get('receipt_id')
            _require(isinstance(rid, str) and rid and rid not in ids, 'Duplicate/missing receipt identity')
            ids.add(rid)
            _require(record.get('status') == 'captured' and record.get('peer_ip') == '127.0.0.1',
                     'Invalid acceptance status/peer')
            _require(_address(record.get('envelope_sender'), allow_null=True), 'Invalid envelope sender')
            recipients = record.get('envelope_recipients')
            _require(isinstance(recipients, list) and 0 < len(recipients) <= limits.max_recipients and
                     all(_address(r) for r in recipients), 'Invalid envelope recipients')
            try:
                raw = base64.b64decode(record['raw_message_b64'], validate=True)
            except (KeyError, ValueError, TypeError, binascii.Error) as exc:
                raise CaptureError('Invalid MIME encoding') from exc
            _require(type(record.get('raw_message_bytes')) is int and
                     len(raw) == record['raw_message_bytes'] <= limits.max_message_bytes and
                     hashlib.sha256(raw).hexdigest() == record.get('raw_message_sha256'),
                     'Invalid raw MIME digest/length')
            accepted.append(record)
    _require(len(accepted) <= limits.max_messages and end.get('accepted_messages') == len(accepted) and
             end.get('accepted_raw_bytes') == sum(r['raw_message_bytes'] for r in accepted) and
             end.get('rejections') == rejected, 'Final counters differ from immutable journal')
    return dict(schema_version=1, sink_id=start['sink_id'], campaign_id=start['campaign_id'],
                start_sha256=start['sha256'], final_sha256=end['sha256'], records=records,
                acceptances=accepted, routing_verified=False, queue_drain_verified=False)


def project_acceptances(capture, linkage, *, expected_linkage_sha256):
    """Project separate immutable SMTP transactions; caller must prove interleaving.

    Linkage is trusted evaluator evidence, not an inferred match. Its externally
    retained seal binds every receipt's raw hash, native order/history IDs, source
    artifact hash and an already committed DB transaction. This function checks
    that contract; it cannot establish truth of declared native evidence.
    Acceptances before the associated history commit require a different auditor
    and are refused here. No receipt is deleted from the original journal.
    """
    _require(isinstance(capture, dict) and isinstance(linkage, dict),
             'Validated capture and linkage objects required')
    _require(_hash(expected_linkage_sha256) and digest(linkage) == expected_linkage_sha256,
             'Missing/mismatched independently retained linkage seal')
    _require(isinstance(linkage, dict) and linkage.get('schema_version') == 1 and
             linkage.get('evidence_kind') == 'evaluator_verified_smtp_linkage' and
             linkage.get('capture_final_sha256') == capture.get('final_sha256') and
             linkage.get('sink_id') == capture.get('sink_id') and
             linkage.get('campaign_id') == capture.get('campaign_id'), 'Linkage identity mismatch')
    # Bind the in-memory projection input back to the sealed immutable journal;
    # callers must not be able to alter its recipients or omit one acceptance
    # after validate_capture has returned.
    records = capture.get('records')
    _require(isinstance(records, list) and len(records) >= 2,
             'Complete sealed journal records required for projection')
    previous = '0' * 64
    for sequence, record in enumerate(records, 1):
        _require(isinstance(record, dict) and record.get('seq') == sequence and
                 record.get('previous_sha256') == previous and
                 record.get('sha256') == digest({k: v for k, v in record.items() if k != 'sha256'}),
                 'Projection input differs from sealed journal')
        previous = record['sha256']
    _require(previous == capture['final_sha256'] and records[0].get('kind') == 'start' and
             records[-1].get('kind') == 'end' and records[-1].get('capture_valid') is True and
             records[-1].get('reason') == 'requested' and
             capture.get('acceptances') == [r for r in records if r.get('kind') == 'accepted'],
             'Projection does not preserve the complete clean capture')
    entries = linkage.get('receipts')
    _require(isinstance(entries, list), 'Linkage receipt list required')
    by_id = {}
    for item in entries:
        _require(isinstance(item, dict) and isinstance(item.get('receipt_id'), str) and
                 item['receipt_id'] not in by_id, 'Duplicate/malformed linkage receipt')
        by_id[item['receipt_id']] = item
    _require(set(by_id) == {r['receipt_id'] for r in capture['acceptances']},
             'Linkage must cover every acceptance exactly')
    transactions, anchors = [], {}
    for receipt in capture['acceptances']:
        item = by_id[receipt['receipt_id']]
        _require(item.get('raw_message_sha256') == receipt['raw_message_sha256'] and
                 _hash(item.get('native_linkage_artifact_sha256')), 'Missing/bad native linkage evidence')
        _require(item.get('ordering') == 'after_committed_history' and
                 isinstance(item.get('after_db_transaction_id'), str) and item['after_db_transaction_id'],
                 'Verified DB-before-SMTP chronology required; timestamps alone are insufficient')
        _require(all(type(item.get(k)) is int and item[k] > 0 for k in ('order_id', 'history_id')),
                 'Resolved positive native order/history IDs required')
        row = {key: receipt[key] for key in ('receipt_id', 'sink_id', 'status',
            'envelope_sender', 'envelope_recipients', 'raw_message_b64', 'raw_message_sha256')}
        row.update(order_id=item['order_id'], history_id=item['history_id'])
        txid = 'smtp:' + receipt['receipt_id']
        transactions.append(dict(transaction_id=txid, status='committed',
            commit_receipt='smtp-journal-sha256:' + receipt['sha256'],
            changes=[dict(table='mail_outbox', operation='INSERT', before=None, after=row)]))
        anchors[txid] = item['after_db_transaction_id']
    return dict(transactions=transactions, required_after_db_transactions=anchors,
                capture_final_sha256=capture['final_sha256'], linkage_sha256=expected_linkage_sha256,
                establishes_cross_stream_order=False, routing_verified=False, queue_drain_verified=False)
