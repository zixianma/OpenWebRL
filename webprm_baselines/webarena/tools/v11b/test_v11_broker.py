"""v11 (copy of v10-tools/test_v10_broker.py; v11 routes equal v10): start the real ConcurrentBrokerProcess (v9b route validator + v9e concurrency) with exactly the launcher's v10 routes
and send a Piotr-shaped local request (system text + screenshot image + JSON-schema selection) through the real
worker.completion with scripted HTTP; native_policy now admits only n=1. Usage: V11_PACKAGE=<pkg> python test_v11_broker.py"""
import base64, io, json, os, sys, tempfile, time, unittest
from pathlib import Path
from unittest.mock import patch
PKG = Path(os.environ['V11_PACKAGE']); sys.path.insert(0, str(PKG/'source'))
from openwebrl.webprm_broker import BrokerError, make_broker_client
from openwebrl.webprm_budget_journal import VERSION as BUDGET_VERSION
from openwebrl.webprm_concurrent_broker import ConcurrentBrokerProcess
ACTOR = 'gpt-4o-2024-08-06'; IDENTITY = 'a'*64
D = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-webarena-20261011/next-stage-preparation')
MODELS = {'piotr_selectionarm': json.loads((D/'cms-table5-350-bounded-reap-recovery-v10/proposal.json').read_text())['models']['piotr_selectionarm']}


def scripted_completion(config, payload, journal, **kwargs):
    import openwebrl.webprm_worker as worker
    def respond(request, timeout=None):
        body = json.loads(request.data); n = body.get('n', 1)
        content = json.dumps(dict(selection=2)) if body['model'] != ACTOR else 'ok'
        return io.BytesIO(json.dumps(dict(model=body['model'], usage=dict(prompt_tokens=1000, completion_tokens=5*n),
            choices=[dict(index=i, finish_reason='stop', message=dict(role='assistant', content=content)) for i in range(n)])).encode())
    with patch.object(worker, 'load_api_credentials', return_value='offline-key'), patch.object(worker, 'urlopen', side_effect=respond), \
            patch.object(worker, 'loopback_urlopen', side_effect=respond):
        return worker.completion(config, payload, journal, **kwargs)


class BrokerV11(unittest.TestCase):
    def setUp(self):
        from scripts import launch_webprm_table5_campaign as L
        self.routes = {}
        for r in L.native_routes(MODELS):  # launcher: routes[(model,purpose)] = dict(..., max_choices=route.get('max_choices',1))
            self.routes[(r['model'], r['purpose'])] = dict(local=r['local'], endpoint='http://127.0.0.1:9/v1' if r['local'] else None,
                max_input_tokens=r['max_input_tokens'], max_output_tokens=r['max_output_tokens'], max_choices=r.get('max_choices', 1))
        self.piotr = (MODELS['piotr_selectionarm']['served_model'], 'selector_piotr')
        assert self.routes[self.piotr]['local'] and all(v['max_choices'] == 1 for v in self.routes.values()), self.routes
        self.root = Path(tempfile.mkdtemp(prefix='v11-broker-', dir='/tmp')); self.deadline = time.time()+60
        prices = {ACTOR: [2.5e-06, 1e-05], 'gpt-4-1106-preview': [1e-05, 3e-05]}
        config = dict(identity_sha256=IDENTITY, deadline=self.deadline, budget_path=str(self.root/'budget.json'), api_usd_cap=5.,
                      prices_usd_per_token=prices)
        allowed = [[ACTOR, 'native_policy'], [ACTOR, 'native_auxiliary'], list(self.piotr)]
        self.cells = {'c': dict(config=config, journal=str(self.root/'c.jsonl'), allowed_routes=allowed)}
        contract = dict(schema_version=1, budget_backend=BUDGET_VERSION, cohort_sha256='b'*64, protocol_sha256='c'*64,
                        prices_usd_per_token=prices, api_usd_cap=5., max_dispatches=8, max_paid_dispatches=8, max_local_dispatches=8)
        self.kwargs = dict(socket_path=str(self.root/'s'), deadline=self.deadline, cells=self.cells, routes=self.routes,
                           max_dispatches=8, max_concurrency=1, audit_path=str(self.root/'broker.jsonl'), completion_fn=scripted_completion,
                           financial_contract=contract, epoch_id='v11-test', previous_closed_epoch_sha256=None)

    def client(self):
        return make_broker_client(self.kwargs['socket_path'], cell_id='c', identity_sha256=IDENTITY, deadline=self.deadline)

    def test_piotr_local_route_and_policy_one_choice(self):
        from PIL import Image
        buf = io.BytesIO(); Image.new('RGB', (1280, 720), 'white').save(buf, format='PNG')
        image = 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()
        payload = dict(model=self.piotr[0], temperature=0, max_tokens=128, messages=[
            dict(role='system', content='You are a web automation arbiter.'),
            dict(role='user', content=[dict(type='text', text='candidates 1..3'), dict(type='image_url', image_url=dict(url=image))])],
            response_format={'type': 'json_schema', 'json_schema': {'name': 'vision_selection_nocot', 'strict': True,
                'schema': {'type': 'object', 'properties': {'selection': {'type': 'integer', 'minimum': 1, 'maximum': 3}}, 'required': ['selection']}}})
        with ConcurrentBrokerProcess(**self.kwargs):
            out = self.client()(self.cells['c']['config'], payload, '/unused', purpose='selector_piotr', local=True, endpoint='http://127.0.0.1:9/v1')
            self.assertEqual(json.loads(out['choices'][0]['message']['content']), dict(selection=2))
            policy = dict(model=ACTOR, messages=[dict(role='user', content='state')], max_tokens=256, temperature=1.0, top_p=0.9)
            self.assertEqual(len(self.client()(self.cells['c']['config'], dict(policy), '/unused', purpose='native_policy')['choices']), 1)
            with self.assertRaises(BrokerError):
                self.client()(self.cells['c']['config'], dict(policy, n=20), '/unused', purpose='native_policy')
        ledger = json.loads((self.root/'budget.json').read_text())
        self.assertAlmostEqual(ledger['charged'], 1000*2.5e-06 + 5*1e-05, places=9)  # the local ARM call is free


if __name__ == '__main__':
    unittest.main(verbosity=2)
