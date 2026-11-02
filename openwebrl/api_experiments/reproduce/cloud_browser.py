"""Owned standalone Browser Use V4 sessions and private billing evidence.

Creating a session is paid. Construction makes no calls; start() requires an
explicit admission callback and overshoot consent. Watchdogs are best effort:
V4 has no documented hard per-session proxy-byte/price cap. This module neither
funds an account nor treats its API-key soft spending cap as a hard ceiling.
"""
from contextlib import contextmanager
from datetime import datetime
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
import os
from pathlib import Path
import threading
import time
import urllib.error
import urllib.request
import urllib.parse
from uuid import UUID

API = 'https://api.browser-use.com/api/v4'


def websocket_endpoint(endpoint, session_id, directory):
    """Resolve only this owned V4 HTTPS CDP endpoint to its browser socket."""
    parsed = urllib.parse.urlsplit(endpoint)
    expected = session_id+'.cdp.browser-use.com'
    need(parsed.scheme == 'https' and parsed.hostname == expected
         and parsed.port in (None,443) and not parsed.username and not parsed.password
         and parsed.path in ('','/') and not parsed.fragment,
         'Owned HTTPS CDP endpoint required')
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *args, **kwargs):
            raise ValueError('CDP discovery redirects are not accepted')
    discovery = urllib.parse.urlunsplit((parsed.scheme,parsed.netloc,'/json/version',parsed.query,''))
    request = urllib.request.Request(discovery, method='GET')
    try:
        with urllib.request.build_opener(NoRedirect()).open(request,timeout=20) as response:
            raw = response.read().decode('utf-8')
            status = response.status
    except Exception as exc:
        raise RuntimeError('Owned CDP discovery failed: '+type(exc).__name__) from None
    write_private(Path(directory)/'cloud-private/cdp-discovery.json',dict(http_status=status,raw_response=raw),exclusive=True)
    need(status == 200, 'Owned CDP discovery failed')
    result = json.loads(raw)
    ws = result.get('webSocketDebuggerUrl')
    need(type(ws) is str, 'Owned browser websocket missing')
    target = urllib.parse.urlsplit(ws)
    need(target.scheme in ('ws','wss') and target.hostname in (expected,'localhost','127.0.0.1','::1')
         and not target.username and not target.password and not target.fragment
         and target.path.startswith('/'), 'CDP discovery must retain owned TLS endpoint')
    # Chrome behind a TLS reverse proxy can advertise its local WS authority.
    # Retain only its browser path/query; the actual connection always uses the
    # already verified owned HTTPS authority, never loopback or another host.
    query = target.query
    if parsed.query and parsed.query != query:
        query = query+'&'+parsed.query if query else parsed.query
    normalized = urllib.parse.urlunsplit(('wss',parsed.netloc,target.path,query,''))
    write_private(Path(directory)/'cloud-private/cdp-normalization.json',
                  dict(normalized_endpoint=normalized, reason='owned_TLS_origin_with_Chrome_browser_path',
                       advertised_authority_replaced=target.netloc != parsed.netloc or target.scheme != 'wss'),exclusive=True)
    return normalized


def need(condition, message):
    if not condition:
        raise ValueError(message)


def amount(value):
    """Absent/null is unknown, including when a provider schema has defaults."""
    if value is None:
        return None
    need(type(value) in (str, int, float, Decimal), 'Invalid decimal amount type')
    try:
        result = Decimal(str(value))
    except InvalidOperation:
        raise ValueError('Invalid decimal amount') from None
    need(result.is_finite() and result >= 0, 'Amount must be finite and nonnegative')
    return str(result)


def provider_duration(started, finished):
    if started is None or finished is None:
        return None
    need(type(started) is str and type(finished) is str, 'Provider timestamps must be strings')
    start = datetime.fromisoformat(started.replace('Z','+00:00'))
    end = datetime.fromisoformat(finished.replace('Z','+00:00'))
    need(start.tzinfo is not None and end.tzinfo is not None, 'Provider timestamps need timezones')
    duration = (end-start).total_seconds()
    need(math.isfinite(duration) and duration >= 0, 'Invalid provider lifetime')
    return duration


def write_private(path, value, *, exclusive=False):
    path = Path(path)
    need(not path.resolve().is_relative_to(Path('/gpfs/home/zixianma')), 'Home is read-only')
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    flags = os.O_WRONLY | os.O_CREAT | (os.O_EXCL if exclusive else os.O_TRUNC)
    with os.fdopen(os.open(path, flags, 0o600), 'w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def interval_union(intervals):
    """Wall-time union, so concurrent N5 calls are never added as latency."""
    ordered = []
    for start, end in intervals:
        need(type(start) in (int, float) and type(end) in (int, float)
             and math.isfinite(start) and math.isfinite(end) and 0 <= start <= end,
             'Invalid timing interval')
        ordered.append((start, end))
    total = 0.0
    left = right = None
    for start, end in sorted(ordered):
        if left is None:
            left, right = start, end
        elif start <= right:
            right = max(right, end)
        else:
            total += right-left
            left, right = start, end
    return total + (right-left if left is not None else 0)


class Timeline:
    def __init__(self, directory, *, clock=time.monotonic):
        self.path = Path(directory)/'timing-spans.jsonl'
        self.clock = clock
        self.events = []
        self.lock = threading.Lock()
        self.phase = 'setup'

    @contextmanager
    def span(self, kind, **metadata):
        start = self.clock()
        error = None
        try:
            yield
        except BaseException as exc:
            error = type(exc).__name__
            raise
        finally:
            row = dict(kind=kind, start_monotonic=start, end_monotonic=self.clock(),
                       error_type=error, **metadata)
            with self.lock:
                self.events.append(row)
                self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                with os.fdopen(os.open(self.path, os.O_WRONLY|os.O_APPEND|os.O_CREAT, 0o600), 'a') as stream:
                    stream.write(json.dumps(row, allow_nan=False)+'\n')
                    stream.flush()

    def summary(self):
        with self.lock:
            events = list(self.events)
        result = {}
        for kind in sorted({r['kind'] for r in events}):
            rows = [r for r in events if r['kind'] == kind]
            result[kind] = dict(calls=len(rows), summed_seconds=sum(r['end_monotonic']-r['start_monotonic'] for r in rows),
                                union_seconds=interval_union((r['start_monotonic'], r['end_monotonic']) for r in rows))
        result['definition'] = 'Monotonic process wall intervals. Nested categories overlap; do not sum them. Actor HTTP union includes concurrent candidate calls only once.'
        return result


class V4HTTP:
    """One explicit HTTP operation, no SDK/redirect/automatic POST retry."""
    def __init__(self, api_key, timeout=30):
        need(type(api_key) is str and bool(api_key), 'Browser Use API key required')
        need(type(timeout) in (int, float) and 0 < timeout <= 60, 'Bounded HTTP timeout required')
        self.api_key, self.timeout = api_key, timeout
        class NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, *args, **kwargs):
                return None
        self.opener = urllib.request.build_opener(NoRedirect())

    def request(self, method, path, payload=None):
        need(path == '/browsers' or path.startswith('/browsers/'), 'Unexpected cloud endpoint')
        data = json.dumps(payload, separators=(',', ':')).encode() if payload is not None else None
        request = urllib.request.Request(API+path, data=data, method=method,
                    headers={'X-Browser-Use-API-Key': self.api_key, 'Content-Type': 'application/json'})
        try:
            with self.opener.open(request, timeout=self.timeout) as response:
                return response.status, response.read().decode('utf-8')
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read().decode('utf-8', errors='replace')
        except Exception as exc:
            # The exception text may contain a capability-bearing URL.
            raise RuntimeError('Cloud transport failed: '+type(exc).__name__) from None


class CloudBrowser:
    """Owned one-attempt session; only the injected admission grants paid work.

    admission(create_payload) must durably consume the campaign session credit
    before returning {approved: True, proxy_overshoot_authorized: True}. The
    caller owns the shared monetary admission guard and exact approved totals.
    No amount here is represented as a hard traffic cap.
    """
    def __init__(self, directory, http, admission, *, metadata, on_halt,
                 max_proxy_usd, on_usage=None, timeout_minutes=20, watchdog_seconds=2,
                 polls=8, minimum_settlement_seconds=10, sleep=time.sleep, clock=time.monotonic, timeline=None, cdp_resolver=websocket_endpoint):
        self.directory = Path(directory)
        need(not self.directory.resolve().is_relative_to(Path('/gpfs/home/zixianma')), 'Home is read-only')
        need(type(timeout_minutes) is int and 1 <= timeout_minutes <= 20, 'At most20 owned provider minutes')
        need(type(polls) is int and 2 <= polls <= 30, 'Bounded readiness/stop polls required')
        need(type(watchdog_seconds) in (int, float) and 0 < watchdog_seconds <= 30, 'Bounded watchdog interval required')
        need(type(metadata) is dict and 0 < len(metadata) <= 10 and all(type(k) is str and type(v) is str for k,v in metadata.items()), 'String ownership labels required')
        self.http, self.admission, self.metadata, self.on_halt = http, admission, dict(metadata), on_halt
        self.on_usage = on_usage
        self.usage_stop_requested = False
        self.max_proxy_usd = amount(max_proxy_usd)
        need(self.max_proxy_usd is not None and Decimal(self.max_proxy_usd) > 0, 'Positive operational proxy guard required')
        need(type(minimum_settlement_seconds) in (int,float) and math.isfinite(minimum_settlement_seconds) and minimum_settlement_seconds >= 0, 'Finite nonnegative settlement window required')
        self.minimum_settlement_seconds = minimum_settlement_seconds
        self.timeout_minutes, self.watchdog_seconds, self.polls = timeout_minutes, watchdog_seconds, polls
        self.sleep, self.clock = sleep, clock
        self.timeline = timeline or Timeline(directory, clock=clock)
        self.session_id = None
        self.sequence = 0
        self.sequence_lock = threading.Lock()
        self.close_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = None
        self.receipt = None
        self.latest = None
        self.cdp_endpoint = None
        self.cdp_resolver = cdp_resolver
        self.started = None
        self.create_dispatched = False
        self.metadata_mismatch = False

    def _request(self, method, path, payload=None):
        with self.sequence_lock:
            self.sequence += 1
            sequence = self.sequence
        record = dict(method=method, path=path, payload=payload, started_epoch=time.time())
        try:
            with self.timeline.span('cloud_control_http', method=method):
                status, raw = self.http.request(method, path, payload)
            record.update(http_status=status, raw_response=raw)
            write_private(self.directory/'cloud-private'/f'http-{sequence:04d}.json', record, exclusive=True)
            need(200 <= status < 300, 'Cloud HTTP failure; inspect private receipt')
            view = json.loads(raw)
            need(type(view) is dict, 'Cloud response must be an object')
            return view
        except BaseException as exc:
            if 'raw_response' not in record:
                record['error_type'] = type(exc).__name__
                write_private(self.directory/'cloud-private'/f'http-{sequence:04d}.json', record, exclusive=True)
            raise

    def _view(self, view, *, allow_metadata_mismatch=False):
        need(str(UUID(view['id'])) == view['id'], 'Invalid provider session identity')
        need(self.session_id is None or view['id'] == self.session_id, 'Foreign provider session')
        if view.get('metadata') != self.metadata:
            self.metadata_mismatch = True
            need(allow_metadata_mismatch, 'Provider ownership metadata differs')
        self.latest = view
        if self.on_usage is not None:
            summary = dict(id=view['id'], status=view.get('status'), startedAt=view.get('startedAt'), finishedAt=view.get('finishedAt'), **{key:amount(view.get(key)) for key in ('browserCost','proxyCost','proxyUsedMb')})
            if self.on_usage(summary) is False:
                self.usage_stop_requested = True
        return view

    def start(self):
        need(self.receipt is None, 'Closed owner cannot create a browser')
        payload = dict(proxyCountryCode='us', timeout=self.timeout_minutes,
                       browserScreenWidth=1280, browserScreenHeight=1800,
                       allowResizing=True, enableRecording=False, metadata=self.metadata)
        intent = dict(schema='owned-browser-use-v4-create-intent-v1', request=payload,
                      request_sha256=hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(),
                      epoch=time.time(), creation_retries=0, traffic_guard='best_effort_not_hard')
        write_private(self.directory/'cloud-private/create-intent.json', intent, exclusive=True)
        grant = self.admission(payload)
        need(grant.get('approved') is True and grant.get('proxy_overshoot_authorized') is True,
             'Exact paid admission and proxy overshoot approval required')
        write_private(self.directory/'cloud-private/admission.json', grant, exclusive=True)
        self.started = self.clock()
        try:
            self.create_dispatched = True
            with self.timeline.span('cloud_create'):
                view = self._request('POST', '/browsers', payload)
            need(str(UUID(view['id'])) == view['id'], 'Invalid provider session identity')
            self.session_id = view['id']
            write_private(self.directory/'cloud-private/owned-session.json',
                          dict(id=self.session_id, metadata=self.metadata, create_request_sha256=intent['request_sha256']), exclusive=True)
            self._view(view)
            need(not self.usage_stop_requested, 'Shared cloud operational guard stopped admission')
            self._ready(view)
            self.thread = threading.Thread(target=self._watch, name='owned-cloud-watchdog', daemon=False)
            self.thread.start()
            return self.cdp_endpoint
        except BaseException as exc:
            try:
                self.on_halt('cloud_start_failed_'+type(exc).__name__)
            finally:
                if self.session_id is not None:
                    self.close()
                else:
                    write_private(self.directory/'cloud-private/ambiguous-create.json',
                                  dict(error_type=type(exc).__name__, new_create_authorized=False, reconciliation_required=True), exclusive=True)
            raise

    def _ready(self, view):
        with self.timeline.span('cloud_ready_wait'):
            for index in range(self.polls):
                self._view(view)
                need(view.get('status') == 'active', 'Created browser not active')
                if type(view.get('cdpUrl')) is str and view['cdpUrl'].startswith('wss://'):
                    self.cdp_endpoint = view['cdpUrl']
                    return
                if type(view.get('cdpUrl')) is str and view['cdpUrl'].startswith('https://'):
                    with self.timeline.span('cloud_cdp_discovery'):
                        self.cdp_endpoint = self.cdp_resolver(view['cdpUrl'], self.session_id, self.directory)
                    return
                self.sleep(self.watchdog_seconds)
                view = self._request('GET', '/browsers/'+self.session_id)
        raise RuntimeError('Cloud CDP readiness timed out')

    def _watch(self):
        while not self.stop_event.wait(self.watchdog_seconds):
            try:
                view = self._view(self._request('GET', '/browsers/'+self.session_id))
                cost = amount(view.get('proxyCost'))
                need(cost is not None, 'Active proxy usage unknown')
                if self.usage_stop_requested or Decimal(cost) >= Decimal(self.max_proxy_usd):
                    raise RuntimeError('Operational proxy threshold reached; overshoot remains possible')
                need(view['status'] == 'active', 'Provider session stopped during actor')
            except BaseException as exc:
                try:
                    self.on_halt('cloud_watchdog_'+type(exc).__name__)
                finally:
                    try:
                        self.close()
                    except BaseException:
                        self.on_halt('cloud_watchdog_stop_unverified')
                return

    def close(self):
        if self.session_id is None:
            need(not self.create_dispatched, 'Ambiguous create needs identity reconciliation before stop')
            return dict(schema='owned-browser-use-v4-not-created-v1', verified=True, remote_session_created=False, creation_dispatched=False)
        self.stop_event.set()
        with self.close_lock:
            if self.receipt is None:
                need(self.session_id is not None, 'Ambiguous create needs identity reconciliation before stop')
                with self.timeline.span('cloud_stop_and_usage'):
                    view = self._view(self._request('PATCH', '/browsers/'+self.session_id, {'action':'stop'}), allow_metadata_mismatch=True)
                    stable = 0
                    previous = None
                    first_stopped_elapsed = None
                    for index in range(self.polls):
                        view = self._view(self._request('GET', '/browsers/'+self.session_id), allow_metadata_mismatch=True)
                        if view.get('status') == 'stopped' and first_stopped_elapsed is None:
                            first_stopped_elapsed = self.clock()-self.started
                        fees = tuple(amount(view.get(k)) for k in ('browserCost','proxyCost','proxyUsedMb'))
                        current = (view.get('status'),view.get('startedAt'),view.get('finishedAt'),fees)
                        if view.get('status') == 'stopped' and view.get('finishedAt') and all(f is not None for f in fees):
                            stable = stable+1 if current == previous else 1
                            if stable >= 2 and self.clock()-self.started-first_stopped_elapsed >= self.minimum_settlement_seconds:
                                break
                        previous = current
                        if view.get('status') != 'stopped' and index%2 == 1:
                            self._view(self._request('PATCH', '/browsers/'+self.session_id, {'action':'stop'}), allow_metadata_mismatch=True)
                        self.sleep(self.watchdog_seconds)
                    need(view.get('status') == 'stopped', 'Owned cloud stop not verified')
                    fees = {key:amount(view.get(key)) for key in ('browserCost','proxyCost','proxyUsedMb')}
                    self.receipt = dict(schema='owned-browser-use-v4-close-v1', id=self.session_id,
                         remote_stopped=True, verified=True, browser_backend='browser-use-v4', proxy_country='us',
                         owned_browser_seconds=first_stopped_elapsed, provider_started_at=view.get('startedAt'),
                         provider_finished_at=view.get('finishedAt'), provider_usage=fees,
                         provider_browser_seconds=provider_duration(view.get('startedAt'),view.get('finishedAt')),
                         billing_fields_complete=all(v is not None for v in fees.values()),
                         post_stop_usage_observed_stable=stable >= 2,
                         minimum_settlement_window_observed=self.clock()-self.started-first_stopped_elapsed >= self.minimum_settlement_seconds,
                         billing_status='provisional_provider_snapshot_requires_postcampaign_requery',
                         metadata_verified=not self.metadata_mismatch,
                         billing_source='post_stop_GET', traffic_guard='best_effort_not_hard')
                    write_private(self.directory/'cloud-private/first-close.json', self.receipt, exclusive=True)
        if self.thread is not None and threading.current_thread() is not self.thread:
            self.thread.join(timeout=65)
            need(not self.thread.is_alive(), 'Cloud watchdog not awaited')
        # Idempotent: first-close lifetime never grows through later Terra calls.
        return dict(self.receipt)

    def reconcile_usage(self):
        self.require_stopped()
        with self.timeline.span('cloud_postcampaign_usage_query'):
            view = self._view(self._request('GET', '/browsers/'+self.session_id), allow_metadata_mismatch=True)
        need(view.get('status') == 'stopped', 'Previously stopped session is no longer stopped')
        result = dict(id=self.session_id, provider_started_at=view.get('startedAt'), provider_finished_at=view.get('finishedAt'),
                      provider_usage={key:amount(view.get(key)) for key in ('browserCost','proxyCost','proxyUsedMb')},
                      provider_browser_seconds=provider_duration(view.get('startedAt'),view.get('finishedAt')),
                      observed_epoch=time.time(), source='postcampaign_GET', actual_cash_payment_asserted=False)
        write_private(self.directory/'cloud-private'/('usage-reconciliation-'+str(self.sequence)+'.json'), result, exclusive=True)
        return result

    def require_stopped(self):
        need(self.receipt is not None and self.receipt.get('remote_stopped') is True
             and (self.thread is None or not self.thread.is_alive()), 'Cloud stop must verify before Terra')
