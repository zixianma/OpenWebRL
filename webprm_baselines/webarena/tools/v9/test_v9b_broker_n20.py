"""v9b: start the real ConcurrentBrokerProcess with the package proposal's paid routes (built exactly as the launcher
builds them, including max_choices) and send n=20 through the real worker.completion with scripted HTTP.
Regression for jobs 389731-389734 (broker startup rejected max_choices). Usage: V9_PACKAGE=<pkg> python test_v9b_broker_n20.py"""
import io, json, os, sys, tempfile, time, unittest
from pathlib import Path
from unittest.mock import patch
PKG = Path(os.environ['V9_PACKAGE']); sys.path.insert(0, str(PKG/'source'))
from openwebrl.webprm_broker import BrokerError, make_broker_client
from openwebrl.webprm_budget_journal import VERSION as BUDGET_VERSION
from openwebrl.webprm_concurrent_broker import ConcurrentBrokerProcess
ACTOR = 'gpt-4o-2024-08-06'; IDENTITY = 'a'*64; N = 20


def scripted_completion(config, payload, journal, **kwargs):
    import openwebrl.webprm_worker as worker
    def respond(request, timeout):
        body = json.loads(request.data); n = body.get('n', 1)
        return io.BytesIO(json.dumps(dict(model=body['model'], usage=dict(prompt_tokens=1000, completion_tokens=40*n),
            choices=[dict(index=i, finish_reason='stop', message=dict(content=f'sample {i}')) for i in range(n)])).encode())
    with patch.object(worker, 'load_api_credentials', return_value='offline-key'), patch.object(worker, 'urlopen', side_effect=respond):
        return worker.completion(config, payload, journal, **kwargs)


class BrokerN20(unittest.TestCase):
    def setUp(self):
        proposal = json.loads((PKG/'proposal.json').read_text())
        self.routes = {}
        from scripts import launch_webprm_table5_campaign as L  # stage-2 proposals use exactly L.native_routes(models)
        for r in L.native_routes(proposal['models']):   # launcher: routes[(model,purpose)] = dict(..., max_choices=route.get('max_choices',1))
            if r['local'] or r['model'] != ACTOR:
                continue
            self.routes[(r['model'], r['purpose'])] = dict(local=False, endpoint=None, max_input_tokens=r['max_input_tokens'],
                                                           max_output_tokens=r['max_output_tokens'], max_choices=r.get('max_choices', 1))
        self.policy = [k for k, v in self.routes.items() if v['max_choices'] == N]
        self.other = [k for k, v in self.routes.items() if v['max_choices'] == 1]
        assert len(self.policy) == 1 and self.other, self.routes
        self.root = Path(tempfile.mkdtemp(prefix='v9b-n20-', dir='/tmp')); self.deadline = time.time()+60
        prices = {ACTOR: [2.5e-06, 1e-05]}
        config = dict(identity_sha256=IDENTITY, deadline=self.deadline, budget_path=str(self.root/'budget.json'), api_usd_cap=5.,
                      prices_usd_per_token=prices)
        self.cells = {'c': dict(config=config, journal=str(self.root/'c.jsonl'), allowed_routes=[list(k) for k in self.routes])}
        contract = dict(schema_version=1, budget_backend=BUDGET_VERSION, cohort_sha256='b'*64, protocol_sha256='c'*64,
                        prices_usd_per_token=prices, api_usd_cap=5., max_dispatches=8, max_paid_dispatches=8, max_local_dispatches=8)
        self.kwargs = dict(socket_path=str(self.root/'s'), deadline=self.deadline, cells=self.cells, routes=self.routes,
                           max_dispatches=8, max_concurrency=1, audit_path=str(self.root/'broker.jsonl'), completion_fn=scripted_completion,
                           financial_contract=contract, epoch_id='v9b-test', previous_closed_epoch_sha256=None)

    def call(self, purpose, n):
        client = make_broker_client(self.kwargs['socket_path'], cell_id='c', identity_sha256=IDENTITY, deadline=self.deadline)
        payload = dict(model=ACTOR, messages=[dict(role='user', content='state')], max_tokens=256, temperature=1.0, top_p=1.0)
        if n != 1: payload['n'] = n
        return client(self.cells['c']['config'], payload, '/unused', purpose=purpose)

    def test_broker_starts_and_serves_n20_only_on_policy(self):
        with ConcurrentBrokerProcess(**self.kwargs):
            result = self.call(self.policy[0][1], N)
            self.assertEqual(len(result['choices']), N)
            with self.assertRaises(BrokerError):
                self.call(self.other[0][1], N)
            self.assertEqual(len(self.call(self.other[0][1], 1)['choices']), 1)
        ledger = json.loads((self.root/'budget.json').read_text())
        self.assertAlmostEqual(ledger['charged'], (1000*2.5e-06+800*1e-05) + (1000*2.5e-06+40*1e-05), places=9)

    def test_invalid_max_choices_rejected(self):
        self.routes[self.policy[0]]['max_choices'] = 21
        with self.assertRaises(BrokerError):
            ConcurrentBrokerProcess(**self.kwargs).start()


if __name__ == '__main__':
    unittest.main(verbosity=2)
