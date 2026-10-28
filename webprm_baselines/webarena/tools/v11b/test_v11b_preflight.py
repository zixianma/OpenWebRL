"""v11b regression for v11 jobs 392753-392755: the Piotr startup preflight through the real ConcurrentBrokerProcess and the
real webprm_worker.completion against a fake local model server that, like the served SelectionARM, runs out of tokens
('</think>\\n\\nOK, the user just said', finish_reason 'length') on the 8-token 'Reply OK.' probe and answers the
selector-shaped probe with stop '{"selection": 1}'. Asserts the old v11 probe fails (as in production) and the v11b
launcher's piotr_preflight_payload passes. Usage: V11_PACKAGE=<v11b pkg> python test_v11b_preflight.py"""
import json, os, sys, tempfile, threading, time, unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
PKG = Path(os.environ['V11_PACKAGE']); sys.path.insert(0, str(PKG/'source'))
from openwebrl.webprm_broker import make_broker_client
from openwebrl.webprm_budget_journal import VERSION as BUDGET_VERSION
from openwebrl.webprm_concurrent_broker import ConcurrentBrokerProcess
from scripts import launch_webprm_table5_campaign as L
IDENTITY = 'a'*64
MODEL = json.loads((PKG/'proposal.json').read_text())['models'].get('piotr_selectionarm') or json.loads(
    (PKG.parent.parent/'cms-table5-350-bounded-reap-recovery-v10/proposal.json').read_text())['models']['piotr_selectionarm']


class FakeSelectionARM(BaseHTTPRequestHandler):
    seen = []
    def log_message(self, *a): pass
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        FakeSelectionARM.seen.append(body)
        short = body.get('max_tokens', 0) < 16
        content, finish = ('</think>\n\nOK, the user just said', 'length') if short else ('{\n  "selection": 1\n}', 'stop')
        out = json.dumps(dict(id='x', object='chat.completion', model=body['model'], usage=dict(prompt_tokens=50, completion_tokens=8 if short else 10, total_tokens=60),
                              choices=[dict(index=0, finish_reason=finish, message=dict(role='assistant', content=content))])).encode()
        self.send_response(200); self.send_header('Content-Type', 'application/json'); self.send_header('Content-Length', str(len(out)))
        self.end_headers(); self.wfile.write(out)


class PiotrPreflight(unittest.TestCase):
    def setUp(self):
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), FakeSelectionARM)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        endpoint = f'http://127.0.0.1:{self.server.server_address[1]}/v1'
        routes = {}
        for r in L.native_routes({'piotr_selectionarm': MODEL}):  # exactly as the launcher builds its route table
            routes[(r['model'], r['purpose'])] = dict(local=r['local'], endpoint=endpoint if r['local'] else None,
                max_input_tokens=r['max_input_tokens'], max_output_tokens=r['max_output_tokens'], max_choices=r.get('max_choices', 1))
        self.endpoint = endpoint
        self.root = Path(tempfile.mkdtemp(prefix='v11b-preflight-', dir='/tmp')); self.deadline = time.time()+60
        prices = {'gpt-4o-2024-08-06': [2.5e-06, 1e-05], 'gpt-4-1106-preview': [1e-05, 3e-05]}
        self.config = dict(identity_sha256=IDENTITY, deadline=self.deadline, budget_path=str(self.root/'budget.json'), api_usd_cap=5.,
                           prices_usd_per_token=prices)
        local = [[m, p] for (m, p), v in routes.items() if v['local']]  # launcher: local_preflight allows every local route
        cells = {'local_preflight': dict(config=self.config, journal=str(self.root/'local-preflight-api.jsonl'), allowed_routes=local)}
        contract = dict(schema_version=1, budget_backend=BUDGET_VERSION, cohort_sha256='b'*64, protocol_sha256='c'*64,
                        prices_usd_per_token=prices, api_usd_cap=5., max_dispatches=8, max_paid_dispatches=8, max_local_dispatches=8)
        self.kwargs = dict(socket_path=str(self.root/'s'), deadline=self.deadline, cells=cells, routes=routes, max_dispatches=8,
                           max_concurrency=1, audit_path=str(self.root/'broker.jsonl'), financial_contract=contract,
                           epoch_id='v11b-test', previous_closed_epoch_sha256=None)

    def tearDown(self):
        self.server.shutdown()

    def call(self, payload):
        client = make_broker_client(self.kwargs['socket_path'], cell_id='local_preflight', identity_sha256=IDENTITY, deadline=self.deadline)
        return client(self.config, payload, '/unused', purpose='selector_piotr', local=True, endpoint=self.endpoint)

    def test_old_probe_fails_new_probe_passes(self):
        old = dict(model=MODEL['served_model'], messages=[dict(role='user', content='Reply OK.')], temperature=0., top_p=1., max_tokens=8)
        new = L.piotr_preflight_payload(MODEL)
        self.assertEqual(new['max_tokens'], 128); self.assertEqual(new['model'], MODEL['served_model'])
        self.assertIn('web automation arbiter', new['messages'][0]['content'])
        self.assertTrue(any(p.get('type') == 'image_url' for p in new['messages'][1]['content']))
        with ConcurrentBrokerProcess(**self.kwargs):
            with self.assertRaises(Exception) as caught:  # v11 jobs 392753-392755: dispatch_failed ValueError
                self.call(old)
            print('v11 8-token probe rejected:', type(caught.exception).__name__, str(caught.exception)[:120], flush=True)
            out = self.call(new)
        self.assertEqual(out['choices'][0]['finish_reason'], 'stop')
        self.assertEqual(json.loads(out['choices'][0]['message']['content']), dict(selection=1))
        self.assertEqual([b['max_tokens'] for b in FakeSelectionARM.seen], [8, 128])
        audit = [json.loads(x) for x in (self.root/'broker.jsonl').read_text().splitlines()]
        self.assertTrue(any(e.get('kind') == 'dispatch_failed' or e.get('event') == 'dispatch_failed' for e in audit), [e.get('kind') or e.get('event') for e in audit])
        print('v11b selector-shaped probe passed: stop', out['choices'][0]['message']['content'].replace('\n', ' '), flush=True)

    def test_launcher_uses_v11b_probe(self):
        text = (PKG/'source/scripts/launch_webprm_table5_campaign.py').read_text()
        self.assertIn("if name=='piotr_selectionarm':payload=piotr_preflight_payload(model)", text)
        self.assertIn("purpose='native_reward' if name=='arbiter7b' else 'selector_piotr' if name=='piotr_selectionarm' else 'selector_pointwise'", text)
        with self.assertRaises(ValueError):
            L.piotr_preflight_payload(dict(MODEL, served_model='other/model'))


if __name__ == '__main__':
    unittest.main(verbosity=2)
