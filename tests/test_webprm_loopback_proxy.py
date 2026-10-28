"""Real HTTP regressions for inherited proxies on owned local services."""
import json
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch
from urllib.error import HTTPError

from openwebrl import webprm_worker as worker
from scripts import launch_webprm_table5_campaign as owner


class Handler(BaseHTTPRequestHandler):
    seen = []

    def log_message(self, *_args):
        pass

    def do_GET(self):
        self.seen.append(('GET', self.path))
        if self.path == '/redirect':
            self.send_response(302)
            self.send_header('Location', 'http://example.invalid/private')
            self.end_headers()
            return
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'{"data":[{"id":"fixture"}]}')

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        self.seen.append(('POST', self.path, body, self.headers['Authorization']))
        self.send_response(200)
        self.end_headers()
        self.wfile.write(json.dumps(dict(model=body['model'],
            choices=[dict(finish_reason='stop', message=dict(content='fixture'))],
            usage=dict(prompt_tokens=1, completion_tokens=1))).encode())


class LoopbackProxyTests(unittest.TestCase):
    def setUp(self):
        Handler.seen = []
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = 'http://127.0.0.1:%s' % self.server.server_port
        # A closed local port makes accidental proxy routing fail deterministically.
        self.env = patch.dict(os.environ, dict(http_proxy='http://127.0.0.1:1',
            HTTP_PROXY='http://127.0.0.1:1', https_proxy='http://127.0.0.1:1',
            HTTPS_PROXY='http://127.0.0.1:1', no_proxy='', NO_PROXY=''))
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def test_readiness_ignores_inherited_proxy(self):
        with owner.loopback_urlopen(self.url + '/v1/models', timeout=2) as reply:
            self.assertEqual(json.load(reply), {'data': [{'id': 'fixture'}]})

    def test_local_dispatch_is_direct_and_journaled(self):
        with tempfile.TemporaryDirectory(dir='/tmp') as folder:
            root = Path(folder)
            config = dict(deadline=time.time()+10, budget_path=str(root/'budget.json'))
            payload = dict(model='fixture', messages=[], max_tokens=1)
            result = worker.completion(config, payload, root/'api.jsonl',
                local=True, endpoint=self.url+'/v1', purpose='selector')
            self.assertEqual(result['model'], 'fixture')
            self.assertEqual(Handler.seen, [('POST', '/v1/chat/completions', payload, 'Bearer local')])
            rows = [json.loads(line) for line in (root/'api.jsonl').read_text().splitlines()]
            self.assertEqual([row['event'] for row in rows], ['request', 'response'])
            self.assertFalse((root/'budget.json').exists())

    def test_paid_dispatch_retains_proxy_and_charged_reservation(self):
        with tempfile.TemporaryDirectory(dir='/tmp') as folder:
            root = Path(folder)
            config = dict(deadline=time.time()+10, budget_path=str(root/'budget.json'),
                api_base=self.url+'/v1')
            with patch.object(worker, 'load_api_credentials', return_value='fixture-secret'):
                with self.assertRaises(OSError):
                    worker.completion(config, dict(model=worker.ACTOR, messages=[], max_tokens=1),
                        root/'api.jsonl', local=False)
            self.assertEqual(Handler.seen, [])
            budget = json.loads((root/'budget.json').read_text())
            self.assertEqual(budget['calls'][0]['state'], 'reserved')
            self.assertGreater(budget['charged'], 0)

    def test_direct_opener_rejects_external_urls_and_redirects(self):
        for url in ['http://example.invalid', 'http://127.0.0.1.example.invalid',
                    'http://user:password@127.0.0.1', 'file:///tmp/test']:
            with self.assertRaises(ValueError):
                worker.loopback_urlopen(url, timeout=2)
        with self.assertRaisesRegex(ValueError, 'must not redirect'):
            worker.loopback_urlopen(self.url+'/redirect', timeout=2)
        self.assertEqual(Handler.seen, [('GET', '/redirect')])
