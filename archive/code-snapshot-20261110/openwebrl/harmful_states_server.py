"""Loopback browser interface; evaluator controls require a separate token."""
from __future__ import annotations

from contextlib import contextmanager
import hmac
from html import escape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import secrets
import threading
from urllib.parse import parse_qs, urlparse, quote
import uuid


def _state_html(value):
    if isinstance(value, dict):
        return '<dl>'+''.join(f'<dt>{escape(str(k).replace("_", " "))}</dt><dd>{_state_html(v)}</dd>' for k, v in value.items())+'</dl>'
    if isinstance(value, list):
        return '<ol>'+''.join('<li>'+_state_html(v)+'</li>' for v in value)+'</ol>' if value else '<span class="empty">None</span>'
    return escape(str(value)) if value is not None else '<span class="empty">None</span>'


def render_page(observation, episode_id, message=''):
    forms = []
    for action in observation['actions']:
        fields = []
        for key, field in action['parameters'].items():
            name = 'p_'+key
            control_id = action['id']+'-'+key
            if field['type'] == 'select':
                control = f'<select id="{escape(control_id)}" name="{escape(name)}">'+''.join(
                    f'<option value="{escape(str(v))}"'+(' selected' if v == field['default'] else '')+'>'+escape(str(v))+'</option>'
                    for v in field['options'])+'</select>'
            else:
                kind = 'number' if field['type'] == 'integer' else 'text'
                control = f'<input id="{escape(control_id)}" type="{kind}" name="{escape(name)}" value="{escape(str(field["default"]), quote=True)}" required>'
            fields.append(f'<label for="{escape(control_id)}">{escape(field["label"])}</label>{control}')
        forms.append(f'<form method="post" action="/action"><input type="hidden" name="action_id" value="{escape(action["id"])}">'
                     f'<input type="hidden" name="episode_id" value="{escape(episode_id)}">'
                     f'<input type="hidden" name="request_id" value="{uuid.uuid4().hex}">'
                     +''.join(fields)+f'<button type="submit">{escape(action["label"])}</button></form>')
    notice = '<p role="status">'+escape(message)+'</p>' if message else ''
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{escape(observation['title'])}</title><style>
body{{font:16px system-ui,sans-serif;background:#f4f6fa;color:#182335;margin:0}}main{{max-width:1000px;margin:35px auto;padding:0 24px}}
header{{border-bottom:1px solid #ccd5e0;margin-bottom:25px}}h1{{font-size:28px}}.task{{font-size:18px;line-height:1.55}}
.columns{{display:grid;grid-template-columns:1fr 1fr;gap:24px}}section{{background:white;border:1px solid #dae1ea;border-radius:10px;padding:22px}}
form{{padding:16px 0;border-bottom:1px solid #e3e8ee}}label{{display:block;margin-bottom:12px;font-weight:600}}input,select{{display:block;box-sizing:border-box;width:100%;padding:9px;margin-top:5px;font:inherit;border:1px solid #9baabd;border-radius:4px}}
button{{background:#245db0;color:white;border:0;border-radius:5px;padding:10px 16px;font:inherit;cursor:pointer}}dt{{font-weight:600;margin-top:12px;text-transform:capitalize}}dd{{margin:4px 0 12px 12px;overflow-wrap:anywhere}}.empty{{color:#6b7788}}[role=status]{{background:#e9eef5;padding:12px;border-radius:5px}}@media(max-width:650px){{.columns{{grid-template-columns:1fr}}}}
</style></head><body><main><header><h1>{escape(observation['title'])}</h1><p class="task">{escape(observation['instruction'])}</p></header>
{notice}<div class="columns"><section><h2>Current information</h2>{_state_html(observation['state'])}</section><section><h2>Actions</h2>{''.join(forms)}</section></div></main></body></html>'''


class SandboxHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, sandbox, campaign_id, *, control_token=None):
        if address[0] != '127.0.0.1':
            raise ValueError('This fixture server binds only to 127.0.0.1')
        self.sandbox, self.campaign_id = sandbox, campaign_id
        self.control_token = control_token or secrets.token_urlsafe(32)
        self.control_lock = threading.RLock()
        super().__init__(address, Handler)
        try:
            self.episode_id = sandbox.start_episode(campaign_id)
        except BaseException:
            self.server_close()
            raise

    @property
    def url(self):
        return f'http://127.0.0.1:{self.server_port}'


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _respond(self, status, value, *, html=False):
        payload = (value if html else json.dumps(value)).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'text/html; charset=utf-8' if html else 'application/json')
        self.send_header('Content-Length', str(len(payload)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Content-Security-Policy', "default-src 'none'; style-src 'unsafe-inline'; form-action 'self'; frame-ancestors 'none'; base-uri 'none'")
        self.end_headers()
        self.wfile.write(payload)

    def _authorized(self):
        supplied = self.headers.get('Authorization', '')
        return hmac.compare_digest(supplied, 'Bearer '+self.server.control_token)

    def do_GET(self):
        path = urlparse(self.path)
        if path.path == '/health':
            self._respond(200, {'ok': True})
        elif path.path == '/api/audit':
            if not self._authorized():
                self._respond(403, {'error': 'Evaluator authorization required'})
            else:
                self._respond(200, self.server.sandbox.audit(self.server.campaign_id))
        elif path.path in ('/', '/api/observation'):
            with self.server.control_lock:
                observation = self.server.sandbox.observe(self.server.campaign_id)
                episode = self.server.episode_id
            if path.path == '/':
                message = parse_qs(path.query).get('message', [''])[0]
                self._respond(200, render_page(observation, episode, message), html=True)
            else:
                self._respond(200, dict(observation, episode_id=episode))
        else:
            self._respond(404, {'error': 'Unknown route'})

    def do_POST(self):
        path = urlparse(self.path).path
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 <= length <= 262144: raise ValueError('Request body too large')
            origin = self.headers.get('Origin')
            if origin and origin != self.server.url:
                self._respond(403, {'error': 'Cross-origin action rejected'})
                return
            raw = self.rfile.read(length)
            if path == '/control/reset':
                if not self._authorized():
                    self._respond(403, {'error': 'Evaluator authorization required'})
                    return
                with self.server.control_lock:
                    self.server.episode_id = self.server.sandbox.start_episode(self.server.campaign_id)
                    episode = self.server.episode_id
                self._respond(200, {'episode_id': episode})
                return
            if path not in ('/action', '/api/action'):
                self._respond(404, {'error': 'Unknown route'})
                return
            if path == '/action':
                form = {k: v[-1] for k, v in parse_qs(raw.decode(), keep_blank_values=True).items()}
                observation = self.server.sandbox.observe(self.server.campaign_id)
                action = next((a for a in observation['actions'] if a['id'] == form.get('action_id')), {})
                params = {k[2:]: v for k, v in form.items() if k.startswith('p_')}
                for key, spec in action.get('parameters', {}).items():
                    if spec['type'] == 'integer' and key in params:
                        try: params[key] = int(params[key])
                        except ValueError: pass
                payload = dict(episode_id=form.get('episode_id'), action_id=form.get('action_id'),
                               request_id=form.get('request_id'), parameters=params)
            else:
                payload = json.loads(raw)
                if not isinstance(payload, dict): raise ValueError('JSON object required')
                if set(payload)-{'episode_id', 'action_id', 'parameters', 'request_id'}:
                    raise ValueError('Unknown request fields')
            receipt = self.server.sandbox.execute(self.server.campaign_id,
                payload['episode_id'], payload['action_id'], payload.get('parameters'), payload.get('request_id'))
            # Internal effects and evaluator labels never enter a policy response.
            public = {k: receipt[k] for k in ('receipt_id', 'status', 'message')}
            if path == '/action':
                self.send_response(303)
                self.send_header('Location', '/?message='+quote(receipt['message']))
                self.send_header('Content-Length', '0')
                self.end_headers()
            else:
                self._respond(200, public)
        except (KeyError, ValueError, TypeError, UnicodeError) as exc:
            self._respond(400, {'error': str(exc)})


@contextmanager
def running_server(sandbox, campaign_id):
    server = SandboxHTTPServer(('127.0.0.1', 0), sandbox, campaign_id)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
