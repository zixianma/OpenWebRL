"""Zero-paid CPU tests of the patched v12 launcher (browser-env Python, as the campaign controller).

1. v12_preflight through real owrl_uds_relay.py sockets to a fake actor /generate + /select server (the frozen payload,
   a selection reply in range passes; an out-of-range selection and a reply without logprobs are rejected).
2. the patched server stage uses the crossbench serving recipe (selector sglang .30 + xgrammar, actor .45 / seed 4200 /
   context 32768 / max batch 48, serve_arm_sglang /select), relays, and the v12 worker/actor identity.
"""
import http.server
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest

D = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-webarena-20261011/next-stage-preparation/webarena-subset-review-20261021')
PKG = Path(os.environ['V12_PACKAGE'])
sys.path.insert(0, str(PKG / 'source'))
from scripts import launch_webprm_table5_campaign as L  # noqa: E402


class Fake(http.server.BaseHTTPRequestHandler):
    mode = 'ok'

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        if self.path == '/generate':
            meta = dict(finish_reason={'type': 'stop'}, prompt_tokens=20)
            if Fake.mode != 'nologprobs':
                meta['output_token_logprobs'] = [[0.0, 1, None]]
            reply = dict(text='OK', meta_info=meta)
        else:
            n = len(body['candidates'])
            reply = dict(raw=json.dumps({'selection': n + 1 if Fake.mode == 'range' else 1}), input_tokens=10, seconds=.01)
        data = json.dumps(reply).encode()
        self.send_response(200); self.send_header('Content-Length', str(len(data))); self.end_headers(); self.wfile.write(data)


class Launcher(unittest.TestCase):
    def test_preflight_through_relays(self):
        server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Fake)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        tmp = Path(tempfile.mkdtemp(prefix='v12l-', dir='/tmp')); relays = []
        try:
            for label in ('actor', 'select'):
                relays.append(subprocess.Popen([sys.executable, '-B', str(L.V12_RELAY), '--listen', str(tmp / f'{label}.sock'),
                                                '--port', str(server.server_address[1]), '--ready', str(tmp / f'{label}.ready')]))
            for _ in range(100):
                if all((tmp / f'{x}.ready').exists() for x in ('actor', 'select')):
                    break
                time.sleep(.1)
            Fake.mode = 'ok'
            result = L.v12_preflight(tmp)
            self.assertEqual(result['select']['raw'], '{"selection": 1}')
            for mode in ('range', 'nologprobs'):
                Fake.mode = mode
                with self.assertRaises(ValueError):
                    L.v12_preflight(tmp)
            print('v12 preflight through relays passed; out-of-range selection and missing logprobs rejected')
        finally:
            for p in relays:
                p.terminate(); p.wait(10)
            server.shutdown()

    def test_patched_serving_and_identity(self):
        source = inspect.getsource(L.Runner.switch_critic) if hasattr(L, 'Runner') else None
        text = Path(L.__file__).read_text()
        for needle in ("'--mem-fraction-static','.30'", "'--grammar-backend','xgrammar'", "'--mem-fraction-static','.45'",
                       "'--random-seed','4200'", "'--context-length','32768'", "'--cuda-graph-max-bs','48'", "serve_arm_sglang.py'),'serve'",
                       "argv=prefix+[V12_PYTHON,'-B',str(REPO/'scripts/run_owrl_webarena_cell.py'),", "plan['owrl']=dict(template['owrl']",
                       "size=4,directory=ports_dir"):
            self.assertIn(needle, text)
        self.assertNotIn("run_native_webarena_cell.py'),'--plan'", text)
        self.assertEqual(L.V12_ACTOR, 'openwebrl-sft-4b')
        self.assertTrue(Path(L.V12_PIOTR_SGLANG).is_dir() and Path(L.V12_ACTOR_PATH, 'config.json').is_file())
        self.assertTrue((L.V12_SNAPSHOT / 'scripts/serve_arm_sglang.py').is_file() and (L.V12_SNAPSHOT / 'scripts/h200_env.sh').is_file())
        print('v12 serving recipe, worker argv, cell sockets and identity present')


if __name__ == '__main__':
    unittest.main(verbosity=2)
