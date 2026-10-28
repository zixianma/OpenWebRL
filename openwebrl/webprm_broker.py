"""Single-dispatch Unix-socket transport from isolated websites to a host broker.

The trusted host registers every cell, journal, model/purpose route and endpoint.
Clients cannot provide credentials, URLs, budget settings or filesystem targets.
The existing completion function alone reserves and settles the shared budget.
Importing this module starts no process and makes no network request.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import fcntl
import json
import math
import multiprocessing
import os
from pathlib import Path
import signal
import socket
import struct
import threading
import time
import uuid
from urllib.parse import urlsplit


PROTOCOL = 1
MAX_FRAME_BYTES = 4 * 1024 * 1024
MAX_RESPONSE_BYTES = 8 * 1024 * 1024
APPROVED_ROOTS = (Path('/tmp'), Path('/gpfs/projects/krishna/zixianma/OpenWebRL'),
                  Path('/gpfs/scrubbed/zixianma/openwebrl-runtime'))
PAYLOAD_KEYS = {'model', 'messages', 'temperature', 'top_p', 'max_tokens',
                'max_completion_tokens', 'n', 'response_format', 'logprobs', 'top_logprobs'}


class BrokerError(RuntimeError):
    """No automatic retry is permitted after any broker failure."""


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def payload_sha256(value):
    return hashlib.sha256(_encoded(value)).hexdigest()


def _path(value):
    path = Path(value)
    if not path.is_absolute() or path.resolve() != path or not any(root in path.parents for root in APPROVED_ROOTS):
        raise ValueError('Broker path must be canonical and beneath an approved writable root')
    return path


def _deadline(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('A finite fixed broker deadline is required')
    return value


def _timeout(deadline, limit=90):
    remaining = deadline - time.time()
    if remaining <= 0:
        raise TimeoutError('Broker deadline expired; no dispatch')
    return min(limit, remaining)


def _peer(sock, expected_uid):
    pid, uid, _ = struct.unpack('3i', sock.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))
    if uid != expected_uid:
        raise BrokerError('Unix socket peer UID rejected')
    return pid


def _recv_exact(sock, size, deadline):
    data = bytearray()
    while len(data) < size:
        sock.settimeout(_timeout(deadline))
        part = sock.recv(min(65536, size - len(data)))
        if not part:
            raise BrokerError('Truncated broker frame; request is not retried')
        data.extend(part)
    return bytes(data)


def _receive(sock, limit, deadline):
    size = struct.unpack('!I', _recv_exact(sock, 4, deadline))[0]
    if size == 0 or size > limit:
        raise BrokerError('Broker frame size rejected')
    value = json.loads(_recv_exact(sock, size, deadline))
    if not isinstance(value, dict):
        raise BrokerError('Broker frame must be an object')
    return value


def _send(sock, value, limit, deadline):
    data = _encoded(value)
    if len(data) > limit:
        raise BrokerError('Broker response size rejected')
    sock.settimeout(_timeout(deadline))
    sock.sendall(struct.pack('!I', len(data)) + data)


def _append(path, value):
    # Metadata only: prompts, credentials and provider error text never go here.
    path = _path(path)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, 'a') as stream:
        stream.write(_encoded(value).decode() + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _validate_cells(cells, deadline):
    if not isinstance(cells, dict) or not cells:
        raise ValueError('At least one fixed cell required')
    journals = set()
    for cell_id, cell in cells.items():
        if not isinstance(cell_id, str) or not cell_id or set(cell) != {'config', 'journal', 'allowed_routes'}:
            raise ValueError('Explicit fixed cell config and journal required')
        if not isinstance(cell['allowed_routes'], list) or not cell['allowed_routes'] or any(
                not isinstance(row, (list, tuple)) or len(row) != 2 or not all(isinstance(x, str) and x for x in row)
                for row in cell['allowed_routes']):
            raise ValueError('Each cell requires explicit model/purpose routes')
        config = cell['config']
        if not isinstance(config.get('identity_sha256'), str) or len(config['identity_sha256']) != 64:
            raise ValueError('Cell identity SHA256 required')
        _deadline(config['deadline'])
        if config['deadline'] > deadline:
            raise ValueError('Cell deadline exceeds broker deadline')
        journal = _path(cell['journal'])
        if journal in journals:
            raise ValueError('Each cell requires its own fixed journal')
        journals.add(journal)
        budget_path = _path(config['budget_path'])
        _path(budget_path.with_suffix('.lock'))
        _path(budget_path.with_suffix(budget_path.suffix + '.tmp'))
        for key in ('_episode_phase_path', 'output', 'output_dir', 'tmp_dir', 'temp_dir'):
            if key in config:
                _path(config[key])
        cap = config.get('api_usd_cap')
        if isinstance(cap, bool) or not isinstance(cap, (int, float)) or not math.isfinite(cap) or cap < 0:
            raise ValueError('Explicit finite nonnegative API cap required')
        # The spawned child owns its lifecycle; no parent phase token is exported.
        if '_episode_phase_path' in config or '_episode_run_token' in config:
            raise ValueError('Host broker config must not contain worker lifecycle tokens')


def _validate_routes(routes):
    if not isinstance(routes, dict) or not routes:
        raise ValueError('At least one exact route required')
    for key, route in routes.items():
        if not isinstance(key, tuple) or len(key) != 2 or not all(isinstance(x, str) and x for x in key):
            raise ValueError('Routes require exact model and purpose')
        if set(route) != {'local', 'endpoint', 'max_input_tokens', 'max_output_tokens'} or type(route['local']) is not bool:
            raise ValueError('Route must fix local mode, endpoint and output token cap')
        if route['local'] and not isinstance(route['endpoint'], str):
            raise ValueError('Local route requires fixed endpoint')
        if not route['local'] and route['endpoint'] is not None:
            raise ValueError('Paid endpoint comes only from the fixed host config')
        for name in ('max_input_tokens', 'max_output_tokens'):
            if type(route[name]) is not int or route[name] <= 0:
                raise ValueError('Positive input/output token caps required')
        if route['local']:
            _endpoint(route['endpoint'])


def _endpoint(value):
    parsed = urlsplit(value)
    if parsed.scheme not in ('http', 'https') or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError('Fixed endpoint must have no credentials, query or fragment')


def _validate_prices(cells, routes):
    for cell in cells.values():
        config = cell['config']
        _endpoint(config.get('api_base', 'https://api.openai.com/v1'))
        for key in map(tuple, cell['allowed_routes']):
            if key not in routes:
                raise ValueError('Cell references a missing fixed route')
            model, _ = key
            route = routes[key]
            if route['local']:
                continue
            if config['api_usd_cap'] <= 0:
                raise ValueError('Paid routes require an explicit positive API cap')
            rates = config.get('prices_usd_per_token', {}).get(model)
            if not isinstance(rates, (list, tuple)) or len(rates) != 2 or any(
                    isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x <= 0 for x in rates):
                raise ValueError('Every paid route requires fixed finite positive token prices')


def _budget_receipt(cell, payload, purpose, offset, local):
    journal = Path(cell['journal'])
    with journal.open('rb') as stream:
        stream.seek(offset)
        rows = [json.loads(line) for line in stream if line.strip()]
    requests = [row for row in rows if row.get('event') == 'request']
    responses = [row for row in rows if row.get('event') == 'response']
    if len(requests) != 1 or len(responses) != 1:
        raise BrokerError('Completion did not produce one durable request and response')
    request, response = requests[0], responses[0]
    if request.get('payload') != payload or request.get('purpose') != purpose or request.get('local') != local:
        raise BrokerError('Completion journal identity mismatch')
    ident = request.get('reservation')
    if response.get('reservation') != ident:
        raise BrokerError('Completion reservation mismatch')
    receipt = dict(reservation_id=ident, journal_sha256=hashlib.sha256(journal.read_bytes()).hexdigest())
    if local:
        if ident is not None:
            raise BrokerError('Local request unexpectedly reserved paid budget')
        return dict(receipt, reservation_state='local', usage=response['response'].get('usage'))
    ledger_bytes = Path(cell['config']['budget_path']).read_bytes()
    ledger = json.loads(ledger_bytes)
    if type(ident) is not int or ident < 0 or ident >= len(ledger['calls']):
        raise BrokerError('Paid completion lacks durable reservation')
    row = ledger['calls'][ident]
    if row['state'] != 'settled' or row['model'] != payload['model'] or row['usage'] != response['response'].get('usage'):
        raise BrokerError('Paid reservation not settled with exact model and usage')
    return dict(receipt, reservation_state='settled', usage=row['usage'],
                budget_ledger_sha256=hashlib.sha256(ledger_bytes).hexdigest())


def serve_broker(socket_path, *, cells, routes, deadline, max_dispatches, audit_path,
                 expected_uid=None, completion_fn=None, ready=None,
                 max_paid_dispatches=None, max_local_dispatches=None):
    """Blocking host service. Caller owns this process and must await its closure.

    completion_fn is only an offline test seam; production uses v6 completion.
    A durable dispatch-start counts against the cap even if its outcome is unknown.
    """
    socket_path, audit_path = _path(socket_path), _path(audit_path)
    _deadline(deadline); _validate_cells(cells, deadline); _validate_routes(routes)
    _validate_prices(cells, routes)
    if type(max_dispatches) is not int or max_dispatches < 0:
        raise ValueError('Explicit nonnegative dispatch cap required')
    max_paid_dispatches = max_dispatches if max_paid_dispatches is None else max_paid_dispatches
    max_local_dispatches = max_dispatches if max_local_dispatches is None else max_local_dispatches
    if any(type(cap) is not int or not 0 <= cap <= max_dispatches
           for cap in (max_paid_dispatches, max_local_dispatches)):
        raise ValueError('Paid/local dispatch caps must be nonnegative and within total cap')
    if len(os.fsencode(socket_path)) >= 108:
        raise ValueError('Unix socket path is too long')
    if socket_path.exists() or socket_path.is_symlink():
        raise ValueError('Refusing to replace any existing socket path')
    _timeout(deadline)
    expected_uid = os.getuid() if expected_uid is None else expected_uid
    cells, routes = deepcopy(cells), deepcopy(routes)
    policy_hash = payload_sha256(dict(cells=cells, routes=[list(k) + [v] for k, v in sorted(routes.items())],
                                      deadline=deadline, max_dispatches=max_dispatches, expected_uid=expected_uid,
                                      max_paid_dispatches=max_paid_dispatches, max_local_dispatches=max_local_dispatches))
    lock_path = _path(str(audit_path) + '.lock')
    with lock_path.open('a') as audit_lock:
        fcntl.flock(audit_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        seen = set()
        dispatch_counts = {False: 0, True: 0}
        if audit_path.exists():
            for line in audit_path.read_text().splitlines():
                row = json.loads(line)
                if row.get('policy_sha256') != policy_hash:
                    raise BrokerError('Broker policy changed across restart')
                if row.get('event') == 'dispatch_started':
                    if row['request_id'] in seen or type(row.get('local')) is not bool:
                        raise BrokerError('Invalid durable dispatch ledger')
                    seen.add(row['request_id'])
                    dispatch_counts[row['local']] += 1
        if completion_fn is None:
            from .webprm_worker import completion as completion_fn
        # Both parent directory and socket are private; no auto-created home/cache.
        if not socket_path.parent.is_dir() or socket_path.parent.stat().st_uid != os.getuid() or socket_path.parent.stat().st_mode & 0o077:
            raise ValueError('Socket parent must be owned and private (0700)')
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            server.bind(str(socket_path)); os.chmod(socket_path, 0o600); server.listen(8)
            if ready is not None:
                ready.send(dict(status='ready', policy_sha256=policy_hash, pid=os.getpid()))
                ready.close()
            while time.time() < deadline:
                server.settimeout(min(.25, _timeout(deadline)))
                try:
                    conn, _ = server.accept()
                except socket.timeout:
                    continue
                with conn:
                    reply_deadline = min(deadline, time.time() + 100)
                    meta = None
                    try:
                        _peer(conn, expected_uid)
                        request = _receive(conn, MAX_FRAME_BYTES, min(reply_deadline, time.time() + 10))
                        if set(request) != {'protocol', 'cell_id', 'identity_sha256', 'request_id', 'purpose', 'local', 'payload', 'expires_at'} or request['protocol'] != PROTOCOL:
                            raise BrokerError('Unknown broker request fields/protocol')
                        cell = cells[request['cell_id']]
                        if request['identity_sha256'] != cell['config']['identity_sha256']:
                            raise BrokerError('Cell identity mismatch')
                        expiry = _deadline(request['expires_at'])
                        if expiry > cell['config']['deadline'] or expiry > deadline:
                            raise BrokerError('Request cannot extend the fixed host deadline')
                        _timeout(expiry)
                        reply_deadline = min(reply_deadline, expiry)
                        ident = request['request_id']
                        if not isinstance(ident, str) or len(ident) != 32 or uuid.UUID(hex=ident).hex != ident:
                            raise BrokerError('Invalid request identity')
                        if ident in seen or len(seen) >= max_dispatches:
                            raise BrokerError('Duplicate request or fixed dispatch cap exhausted')
                        payload, purpose = request['payload'], request['purpose']
                        if not isinstance(payload, dict) or set(payload) - PAYLOAD_KEYS:
                            raise BrokerError('Unsupported completion payload fields')
                        route_key = (payload['model'], purpose)
                        if route_key not in map(tuple, cell['allowed_routes']):
                            raise BrokerError('Model/purpose is not registered for this cell')
                        route = routes[route_key]
                        if type(request['local']) is not bool or request['local'] != route['local']:
                            raise BrokerError('Route local mode mismatch')
                        if dispatch_counts[route['local']] >= (max_local_dispatches if route['local'] else max_paid_dispatches):
                            raise BrokerError('Fixed paid/local dispatch cap exhausted')
                        tokens = payload.get('max_completion_tokens', payload.get('max_tokens'))
                        if type(tokens) is not int or not 0 < tokens <= route['max_output_tokens'] or payload.get('n', 1) != 1:
                            raise BrokerError('Output token or response count cap rejected')
                        if 'max_tokens' in payload and 'max_completion_tokens' in payload:
                            raise BrokerError('Ambiguous output token limit')
                        if not isinstance(payload.get('messages'), list) or not payload['messages']:
                            raise BrokerError('Messages required')
                        from .webprm_worker import input_token_upper_bound
                        input_bound = input_token_upper_bound(payload['messages'])
                        if input_bound > route['max_input_tokens']:
                            raise BrokerError('Conservative input token bound exceeds fixed route cap')
                        _timeout(cell['config']['deadline'])
                        # Recheck legacy completion's derived write paths at dispatch.
                        _validate_cells({request['cell_id']: cell}, deadline)
                        meta = dict(schema_version=1, policy_sha256=policy_hash, request_id=ident,
                                    cell_id=request['cell_id'], identity_sha256=request['identity_sha256'],
                                    model=payload['model'], purpose=purpose, local=route['local'],
                                    payload_sha256=payload_sha256(payload))
                        _append(audit_path, dict(meta, event='dispatch_started', at_epoch=time.time()))
                        seen.add(ident)
                        dispatch_counts[route['local']] += 1
                        journal = Path(cell['journal'])
                        offset = journal.stat().st_size if journal.exists() else 0
                        result = completion_fn(dict(cell['config'], deadline=expiry), payload, journal, local=route['local'],
                                               endpoint=route['endpoint'], purpose=purpose)
                        usage = result.get('usage')
                        if not route['local'] and (not isinstance(usage, dict) or any(
                                type(usage.get(key)) is not int or usage[key] < 0
                                for key in ('prompt_tokens', 'completion_tokens')) or
                                usage['prompt_tokens'] > input_bound or usage['completion_tokens'] > tokens):
                            from .webprm_worker import SharedBudget
                            SharedBudget(cell['config']['budget_path'], cell['config']['api_usd_cap']).change(
                                lambda data: data.update(halt_reason='broker_observed_token_cap_overrun_or_invalid_usage'))
                            raise BrokerError('Observed token usage exceeded fixed route cap; shared paid budget halted')
                        receipt = dict(meta, budget_owner='webprm_worker.completion',
                                       **_budget_receipt(cell, payload, purpose, offset, route['local']))
                        # Only the completion contract crosses the boundary, never raw errors/config.
                        result = {key: result[key] for key in ('model', 'choices', 'usage') if key in result}
                        _append(audit_path, dict(receipt, event='dispatch_completed', at_epoch=time.time()))
                        _send(conn, dict(ok=True, result=result, receipt=receipt), MAX_RESPONSE_BYTES, reply_deadline)
                    except Exception as error:
                        if meta is not None:
                            _append(audit_path, dict(meta, event='dispatch_failed', at_epoch=time.time(), error_type=type(error).__name__))
                        # Exception text can contain URLs, headers or credentials; never return it.
                        try:
                            _send(conn, dict(ok=False, error=type(error).__name__), MAX_RESPONSE_BYTES, reply_deadline)
                        except (OSError, TimeoutError, BrokerError):
                            pass
        finally:
            server.close()
            # Socket is deliberately retained as a closure artifact; no unlink/replace.


def make_broker_client(socket_path, *, cell_id, identity_sha256, deadline):
    """Return a completion-compatible client with no provider/network fallback."""
    socket_path = str(_path(socket_path)); _deadline(deadline)

    def request(config, payload, journal, *, local=False, endpoint=None, purpose=None):
        # endpoint and journal are intentionally never forwarded; host registration wins.
        if config['identity_sha256'] != identity_sha256:
            raise BrokerError('Worker and broker cell identity differ')
        until = min(deadline, _deadline(config['deadline']))
        _timeout(until)
        ident = uuid.uuid4().hex
        request_deadline = min(until, time.time() + 100)
        body = dict(protocol=PROTOCOL, cell_id=cell_id, identity_sha256=identity_sha256,
                    request_id=ident, purpose=purpose, local=local, payload=payload,
                    expires_at=request_deadline)
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as conn:
            conn.settimeout(_timeout(until, 100)); conn.connect(socket_path)
            _peer(conn, os.getuid())
            _send(conn, body, MAX_FRAME_BYTES, until)
            reply = _receive(conn, MAX_RESPONSE_BYTES, request_deadline)
        if reply.get('ok') is not True:
            raise BrokerError('Host broker rejected or failed request; no retry')
        receipt = reply['receipt']
        expected = dict(request_id=ident, cell_id=cell_id, identity_sha256=identity_sha256,
                        model=payload['model'], purpose=purpose, local=local, payload_sha256=payload_sha256(payload))
        if any(receipt.get(k) != v for k, v in expected.items()) or receipt.get('budget_owner') != 'webprm_worker.completion':
            raise BrokerError('Broker response identity mismatch')
        return dict(reply['result'], _webprm_broker=receipt)

    request._webprm_broker_client = True
    request._webprm_budget_owner = 'webprm_worker.completion'
    return request


def _child(kwargs, ready):
    os.setsid()
    try:
        serve_broker(**kwargs, ready=ready)
    except BaseException as error:
        try:
            ready.send(dict(status='failed', error=type(error).__name__))
        except (OSError, ValueError):
            pass
        raise


class BrokerProcess:
    """Caller-owned spawned host process; context exit always terminates and reaps.

    Production completion uses urllib only and creates no child processes. The
    process group is also terminated, without signaling unrelated processes.
    No auto-restart or provider retry is performed.
    """
    def __init__(self, *, startup_seconds=10, shutdown_seconds=2, **server_kwargs):
        for value in (startup_seconds, shutdown_seconds):
            if not 0 < _deadline(value) <= 10:
                raise ValueError('Lifecycle waits must be positive and at most10 seconds')
        self.kwargs = server_kwargs
        self.startup_seconds, self.shutdown_seconds = startup_seconds, shutdown_seconds
        self.process = None
        self.closure = None
        self._stop = threading.Event()
        self._lock = threading.RLock()
        self._watchdog = None

    def start(self):
        if self.process is not None:
            raise BrokerError('BrokerProcess is single-use; no automatic restart')
        _timeout(_deadline(self.kwargs['deadline']))
        context = multiprocessing.get_context('spawn')
        parent, child = context.Pipe(duplex=False)
        self.process = context.Process(target=_child, args=(self.kwargs, child))
        try:
            self.process.start(); child.close()
            if not parent.poll(min(self.startup_seconds, _timeout(self.kwargs['deadline']))):
                raise BrokerError('Broker startup timed out')
            self.ready = parent.recv()
            if self.ready.get('status') != 'ready':
                raise BrokerError('Broker startup failed')
            self._watchdog = threading.Thread(target=self._watch_deadline, daemon=True)
            self._watchdog.start()
            return self
        except BaseException:
            self.close()
            raise
        finally:
            parent.close(); child.close()

    def close(self):
        self._stop.set()
        with self._lock:
            result = self._close()
        if self._watchdog is not None and self._watchdog is not threading.current_thread():
            self._watchdog.join(timeout=2 * self.shutdown_seconds + 1)
        return result

    def _watch_deadline(self):
        if not self._stop.wait(max(0, self.kwargs['deadline'] - time.time())):
            self.close()

    def _close(self):
        process = self.process
        if process is None or process.pid is None:
            self.closure = dict(started=False, reaped=True)
            return self.closure
        for sig in (signal.SIGTERM, signal.SIGKILL):
            if process.is_alive():
                try:
                    if os.getpgid(process.pid) == process.pid:
                        os.killpg(process.pid, sig)
                    else:
                        os.kill(process.pid, sig)
                except ProcessLookupError:
                    pass
            process.join(timeout=self.shutdown_seconds)
        self.closure = dict(pid=process.pid, exitcode=process.exitcode, reaped=not process.is_alive())
        if not self.closure['reaped']:
            raise BrokerError('Owned broker process failed to reap; stop controller')
        return self.closure

    def __enter__(self):
        return self.start()

    def __exit__(self, *_):
        self.close()
