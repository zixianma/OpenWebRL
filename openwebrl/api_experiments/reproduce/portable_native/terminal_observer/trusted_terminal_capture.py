"""Dedicated post-actor observer. Broker owns bytes and the private receipt.

The RPC accepts no endpoint, filename, task, or screenshot bytes from its caller.
It is authorized against the existing native-owner PID/start ticks, and closes
selected-command admission before observing. It never changes a browser page.
"""
import asyncio
import hashlib
import json
import os
from pathlib import Path
import socket
import struct
import sys
import threading
import time

VERSION = 'native-trusted-terminal-observer-v2-preclose'
MAX_FILE_BYTES = 64 * 1024 * 1024
CDP_CONNECT_TIMEOUT_MS = 15000
SCREENSHOT_TIMEOUT_MS = 60000
PRECLOSE_WAIT_SECONDS = 85
RPC_TIMEOUT_SECONDS = 100


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def regular_bytes(root, name):
    path = root / name
    if path.is_symlink() or path.resolve().parent != root.resolve():
        raise ValueError('Terminal source is not a direct regular episode file')
    if not path.is_file() or path.stat().st_size > MAX_FILE_BYTES:
        raise ValueError('Terminal source absent or oversized: ' + name)
    return path.read_bytes()


def exclusive_write(path, raw):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'wb') as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())


async def observe_page(endpoint, session, helper, playwright_factory):
    """Use exactly the current active page, without resize/navigation/new tabs."""
    async with playwright_factory() as playwright:
        browser = await playwright.chromium.connect_over_cdp(endpoint, timeout=CDP_CONNECT_TIMEOUT_MS)
        try:
            if not browser.contexts:
                raise ValueError('Owned browser has no existing context')
            context = browser.contexts[0]
            page = await helper._select_page(context, session)
            if page is None or page.is_closed():
                raise ValueError('No existing active page for terminal observer')
            target = await helper._page_target_id(context, page)
            if not target:
                raise ValueError('Terminal page target identity missing')
            started = time.time()
            raw = await page.screenshot(full_page=False, timeout=SCREENSHOT_TIMEOUT_MS)
            finished = time.time()
            return raw, {'active_target_id': target, 'page_url': page.url,
                         'capture_started_epoch': started,
                         'capture_finished_epoch': finished}
        finally:
            # For a CDP attachment Playwright disconnects; physical Chrome is
            # subsequently closed by the unchanged persistent-session teardown.
            await browser.close()


def observe_owned(proxy, session):
    upstream = Path(__file__).resolve().parent / 'upstream/src'
    sys.path.insert(0, str(upstream))
    from cuawright.webwright.tools import browser_session as helper
    from playwright.async_api import async_playwright
    return asyncio.run(observe_page(proxy.target, session, helper, async_playwright))


def remember_session(broker):
    """Broker-owned snapshot after an authenticated native command finishes.

    Default CLI close deletes its manifest. The last command-complete snapshot
    preserves exact session identity until preclose; it is usable only while
    the selected-browser manifest remains byte-identical.
    """
    with broker.lock:
        root = Path(broker.cfg['output']).resolve()
        session_raw = regular_bytes(root, '.browser-session.json')
        session = json.loads(session_raw)
        if session.get('state') != 'active' or type(session.get('session_epoch')) is not int:
            raise ValueError('Only active identified sessions can be remembered')
        matches = [proxy for proxy in broker.capture_proxies
                   if session.get('connect_url') in (proxy.endpoint, proxy.target)]
        if len(matches) != 1:
            raise ValueError('Remembered session is not one owned physical Chrome')
        manifest = root / 'browser-steps.jsonl'
        steps_raw = regular_bytes(root, manifest.name) if manifest.exists() else None
        saved = {'session_raw': session_raw, 'browser_identity': matches[0].identity,
                 'browser_steps_sha256': digest(steps_raw) if steps_raw is not None else None}
        broker.terminal_session_cache = saved
        broker.record({'event': 'terminal_session_remembered', 'session_epoch': session['session_epoch'],
                       'session_sha256': digest(session_raw), 'browser_steps_sha256': saved['browser_steps_sha256'],
                       'browser_identity': matches[0].identity})
        return {'verified': True, 'session_epoch': session['session_epoch'],
                'session_sha256': digest(session_raw)}


def preclose_capture(broker, proxy, *, observer=None):
    event = getattr(proxy, 'terminal_preclose_done', None)
    if event is None:
        event = proxy.terminal_preclose_done = threading.Event()
    try:
        return _preclose_capture(broker, proxy, observer=observer)
    finally:
        # Final native-owner RPC waits on this OUTSIDE broker.lock. A failed
        # capture releases the waiter but supplies no qualified fallback frame.
        event.set()


def _preclose_capture(broker, proxy, *, observer=None):
    """Broker lifecycle hook before physical Chrome teardown; never an RPC.

    It saves only a provisional frame. Final native-owner RPC must still prove
    it belongs to the final session epoch and unchanged last selected action.
    """
    with broker.lock:
        root = Path(broker.cfg['output']).resolve()
        evidence = Path(broker.cfg['broker_evidence']).resolve()
        if evidence == root or root in evidence.parents:
            raise ValueError('Preclose evidence is inside actor-writable episode')
        sources = {'browser-steps.jsonl': regular_bytes(root, 'browser-steps.jsonl')}
        if (root / '.browser-session.json').exists():
            sources['.browser-session.json'] = regular_bytes(root, '.browser-session.json')
        else:
            cached = getattr(broker, 'terminal_session_cache', None)
            if not cached or cached['browser_identity'] != proxy.identity or cached['browser_steps_sha256'] != digest(sources['browser-steps.jsonl']):
                raise ValueError('Deleted preclose session lacks matching trusted command-complete snapshot')
            sources['.browser-session.json'] = cached['session_raw']
        session = json.loads(sources['.browser-session.json'])
        if type(session.get('session_epoch')) is not int or session['session_epoch'] <= 0:
            raise ValueError('Preclose session epoch missing')
        if session.get('connect_url') not in (proxy.endpoint, proxy.target):
            raise ValueError('Preclose browser is not the current session')
        rows = [json.loads(line) for line in sources['browser-steps.jsonl'].split(b'\n') if line.strip()]
        if not rows:
            raise ValueError('Preclose session has no selected browser action')
        raw, details = observer(proxy.target, session) if observer else observe_owned(proxy, session)
        if regular_bytes(root, 'browser-steps.jsonl') != sources['browser-steps.jsonl']:
            raise ValueError('Browser actions changed during preclose observation')
        if raw[:8] != b'\x89PNG\r\n\x1a\n' or raw[12:16] != b'IHDR' or len(raw) < 24:
            raise ValueError('Preclose observer returned no original PNG')
        geometry = struct.unpack('>II', raw[16:24])
        if geometry not in ((1280, 1800), (1280, 1657)):
            raise ValueError('Unqualified preclose PNG geometry')
        prefix = 'preclose-' + str(proxy.identity['pid'])
        image = evidence / (prefix + '.png')
        receipt_path = evidence / (prefix + '.json')
        receipt = {'schema': VERSION, 'kind': 'provisional_preclose_frame',
                   'broker_pid': os.getpid(), 'browser_identity': proxy.identity,
                   'source_sha256': digest(Path(__file__).read_bytes()),
                   'episode': str(root), 'session_epoch': session.get('session_epoch'),
                   'session_id': session.get('session_id'), 'session_snapshot': session,
                   'connect_url_sha256': digest(str(session.get('connect_url')).encode()),
                   'browser_steps_sha256': digest(sources['browser-steps.jsonl']),
                   'last_browser_step': rows[-1]['browser_step'],
                   'capture': details, 'screenshot_path': str(image),
                   'screenshot_sha256': digest(raw), 'screenshot_dimensions': list(geometry)}
        exclusive_write(image, raw)
        exclusive_write(receipt_path, json.dumps(receipt, sort_keys=True, indent=2).encode() + b'\n')
        saved = {'path': str(receipt_path), 'sha256': digest(receipt_path.read_bytes()), 'receipt': receipt}
        if not hasattr(broker, 'terminal_preclose_frames'):
            broker.terminal_preclose_frames = []
        broker.terminal_preclose_frames.append(saved)
        broker.record({'event': 'terminal_preclose_frame', 'receipt_path': str(receipt_path),
                       'receipt_sha256': saved['sha256'], 'browser_identity': proxy.identity})
        return saved


def closed_frame(broker, proxy, session, snapshot, rows):
    if not broker.capture_proxies or broker.capture_proxies[-1] is not proxy:
        raise ValueError('A later browser launch supersedes the preclose endpoint')
    matches = [item for item in getattr(broker, 'terminal_preclose_frames', [])
               if item['receipt']['browser_identity'] == proxy.identity]
    if len(matches) != 1:
        raise ValueError('No unique trusted preclose frame for closed terminal session')
    saved = matches[0]
    receipt = saved['receipt']
    if digest(Path(saved['path']).read_bytes()) != saved['sha256']:
        raise ValueError('Private preclose provenance changed')
    if receipt['session_epoch'] != session.get('session_epoch') or receipt['connect_url_sha256'] != digest(str(session.get('connect_url')).encode()):
        raise ValueError('Preclose session epoch/endpoint differs from final session')
    if receipt['browser_steps_sha256'] != digest(snapshot['browser-steps.jsonl']) or receipt['last_browser_step'] != rows[-1]['browser_step']:
        raise ValueError('Later browser actions invalidate the preclose endpoint')
    image = Path(receipt['screenshot_path'])
    if image.is_symlink() or image.resolve().parent != Path(broker.cfg['broker_evidence']).resolve():
        raise ValueError('Preclose PNG is outside trusted host evidence')
    raw = image.read_bytes()
    if digest(raw) != receipt['screenshot_sha256']:
        raise ValueError('Trusted preclose PNG changed')
    return raw, {**receipt['capture'], 'endpoint_kind': 'trusted_preclose_after_final_selected_browser_action',
                 'preclose_receipt_path': saved['path'], 'preclose_receipt_sha256': saved['sha256'],
                 'session_epoch_verified': True, 'no_later_browser_action_verified': True,
                 'no_later_owned_browser_launch_verified': True}


def deleted_closed_session(broker, snapshot):
    """Recover identity from private preclose, never from an older image."""
    frames = getattr(broker, 'terminal_preclose_frames', [])
    proxies = broker.capture_proxies
    if not frames or not proxies:
        raise ValueError('Deleted final session has no trusted preclose identity')
    candidates = [item for item in frames if item['receipt']['browser_identity'] == proxies[-1].identity]
    if len(candidates) != 1:
        raise ValueError('Deleted final session has ambiguous/latest-browser identity')
    receipt = candidates[0]['receipt']
    if digest(Path(candidates[0]['path']).read_bytes()) != candidates[0]['sha256']:
        raise ValueError('Private deleted-session provenance changed')
    events_raw = regular_bytes(Path(broker.cfg['output']), 'browser-sessions.jsonl')
    events = [json.loads(line) for line in events_raw.split(b'\n') if line.strip()]
    if not events or events[-1].get('event') != 'closed' or events[-1].get('session_epoch') != receipt['session_epoch'] or events[-1].get('session_id') != receipt['session_id']:
        raise ValueError('Deleted session lacks exact final closed lifecycle event')
    snapshot['browser-sessions.jsonl'] = events_raw
    return {**receipt['session_snapshot'], 'state': 'closed'}


def capture(broker, peer_pid, peer_uid, actor_finished_epoch, *, observer=None,
            preclose_wait_seconds=PRECLOSE_WAIT_SECONDS):
    """Authenticate/freeze admission, then wait for teardown WITHOUT its lock."""
    owner = broker.command_ownership
    owner.authorize(peer_pid, peer_uid, 'terminal-observer', 'terminal_capture')
    with broker.lock, owner.lock:
        if getattr(broker, 'terminal_observer_requested', False) or getattr(broker, 'terminal_observer_started', False):
            raise ValueError('Terminal observation already attempted; no automatic retries')
        if broker.stopping or owner.commands:
            raise ValueError('Actor commands must be closed before terminal observation')
        broker.terminal_observer_requested = True
        root = Path(broker.cfg['output']).resolve()
        session_path = root / '.browser-session.json'
        session = json.loads(regular_bytes(root, session_path.name)) if session_path.exists() else None
        matches = [proxy for proxy in broker.capture_proxies
                   if session and session.get('connect_url') in (proxy.endpoint, proxy.target)]
        if session is None and broker.capture_proxies:
            matches = [broker.capture_proxies[-1]]
        event = getattr(matches[0], 'terminal_preclose_done', None) if len(matches) == 1 else None
        wait = (session is None or session.get('state') == 'closed') and event is not None
    if wait and not event.wait(timeout=preclose_wait_seconds):
        raise ValueError('Timed out awaiting trusted preclose evidence; no earlier frame accepted')
    return _capture_locked(broker, peer_pid, peer_uid, actor_finished_epoch, observer=observer)


def _capture_locked(broker, peer_pid, peer_uid, actor_finished_epoch, *, observer=None):
    """Bind stable final source files after a completed preclose or fresh capture."""
    owner = broker.command_ownership
    owner.authorize(peer_pid, peer_uid, 'terminal-observer', 'terminal_capture')
    if not isinstance(actor_finished_epoch, (int, float)) or not 0 < actor_finished_epoch <= time.time():
        raise ValueError('Invalid actor-finished timestamp')
    with broker.lock, owner.lock:
        if getattr(broker, 'terminal_observer_started', False):
            raise ValueError('Terminal observation already attempted; no automatic retries')
        if broker.stopping or owner.commands:
            raise ValueError('Actor commands must be closed before terminal observation')
        broker.terminal_observer_started = True
        broker.record({'event': 'terminal_observer_started', 'actor_finished_epoch': actor_finished_epoch,
                       'source_sha256': digest(Path(__file__).read_bytes())})
        root = Path(broker.cfg['output']).resolve()
        snapshot = {name: regular_bytes(root, name) for name in ('browser-steps.jsonl', 'trajectory.json')}
        session_path = root / '.browser-session.json'
        snapshot[session_path.name] = regular_bytes(root, session_path.name) if session_path.exists() else None
        session = json.loads(snapshot[session_path.name]) if snapshot[session_path.name] is not None else deleted_closed_session(broker, snapshot)
        rows = [json.loads(line) for line in snapshot['browser-steps.jsonl'].split(b'\n') if line.strip()]
        if not rows or any(type(row.get('action')) is not str for row in rows):
            raise ValueError('Missing selected browser actions')
        numbers = [row['browser_step'] for row in rows]
        if numbers != sorted(set(numbers)):
            raise ValueError('Browser action identities unordered or repeated')
        if session.get('state') not in ('active', 'closed'):
            raise ValueError('Persistent browser session is neither active nor cleanly closed')
        matches = [proxy for proxy in broker.capture_proxies
                   if session.get('connect_url') in (proxy.endpoint, proxy.target)]
        if len(matches) != 1:
            raise ValueError('Session endpoint is not one unambiguous broker-owned Chrome')
        proxy = matches[0]
        if session.get('state') == 'closed':
            raw, details = closed_frame(broker, proxy, session, snapshot, rows)
        elif observer is None:
            raw, details = observe_owned(proxy, session)
        else:
            raw, details = observer(proxy.target, session)
        if raw[:8] != b'\x89PNG\r\n\x1a\n' or raw[12:16] != b'IHDR' or len(raw) < 24:
            raise ValueError('Terminal observer returned no original PNG')
        width, height = struct.unpack('>II', raw[16:24])
        if (width, height) not in ((1280, 1800), (1280, 1657)):
            raise ValueError('Unexpected native terminal PNG geometry')
        if any((regular_bytes(root, name) if (root / name).exists() else None) != value for name, value in snapshot.items()):
            raise ValueError('Actor source files changed during terminal observation')
        evidence = Path(broker.cfg['broker_evidence']).resolve()
        if evidence == root or root in evidence.parents:
            raise ValueError('Terminal evidence must be outside actor-writable episode')
        image = evidence / 'terminal.png'
        receipt_path = evidence / 'terminal-capture.json'
        receipt = {'schema': VERSION, 'verified': True, 'observer_role': 'post_actor_terminal_evidence',
                   'episode': str(root), 'episode_id': root.name,
                   'broker_pid': os.getpid(), 'native_owner_identity': owner.owner_identity,
                   'browser_identity': proxy.identity, 'owned_endpoint_sha256': digest(proxy.target.encode()),
                   'source_sha256': digest(Path(__file__).read_bytes()),
                   'actor_finished_epoch': actor_finished_epoch,
                   'capture': details, 'screenshot_path': str(image),
                   'screenshot_sha256': digest(raw), 'screenshot_bytes': len(raw),
                   'screenshot_dimensions': [width, height], 'last_browser_step': numbers[-1],
                   'browser_step_count': len(rows),
                   'source_files_sha256': {name: digest(value) if value is not None else None for name, value in snapshot.items()},
                   'no_new_actor_decision': True, 'no_selected_command_in_flight': True,
                   'new_actor_commands_admission_closed': True,
                   'selected_action_paid_binding_still_requires_canonical_audit': True,
                   'capture_changes_actor_status': False}
        exclusive_write(image, raw)
        exclusive_write(receipt_path, json.dumps(receipt, sort_keys=True, indent=2).encode() + b'\n')
        broker.record({'event': 'terminal_observer_finished', 'verified': True,
                       'receipt_path': str(receipt_path), 'receipt_sha256': digest(receipt_path.read_bytes())})
        return {'verified': True, 'receipt_path': str(receipt_path),
                'receipt_sha256': digest(receipt_path.read_bytes()), 'screenshot_sha256': digest(raw)}


def request_capture(cfg, actor_finished_epoch):
    """Trusted native owner invokes once after run_one and before teardown."""
    return _request(cfg, {'op': 'terminal_capture', 'actor_finished_epoch': actor_finished_epoch})


def request_remember_session(cfg):
    """Diagnostic-only owner snapshot before exercising default CLI close."""
    return _request(cfg, {'op': 'terminal_remember_session'})


def _request(cfg, payload):
    connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    connection.settimeout(RPC_TIMEOUT_SECONDS)
    try:
        connection.connect(cfg['browser_broker_socket'])
        connection.sendall(json.dumps(payload).encode() + b'\n')
        line = connection.makefile('rb').readline(65537)
        if len(line) > 65536:
            raise ValueError('Oversized terminal observer reply')
        result = json.loads(line)
        if result.get('verified') is not True:
            raise ValueError(result.get('error', 'Terminal capture unverified'))
        return result
    finally:
        connection.close()
