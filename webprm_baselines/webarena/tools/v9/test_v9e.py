"""v9e regressions (jobs 390607 and 390653). With the launcher's max_concurrency=1:
  - status read answers while two workers' calls are in flight (390607: 5 s status timeout);
  - two concurrent calls from ONE cell (WebArbiter's parallel pairwise tournament) both succeed (390653 under v9d).
V9E_TEST_CONCURRENCY overrides the broker concurrency to show the tests detect the old behaviours.
Usage: V9_PACKAGE=<pkg> python test_v9e.py"""
import os, sys, threading, time, unittest
from pathlib import Path
PKG = Path(os.environ['V9_PACKAGE']); sys.path.insert(0, str(PKG/'source')); sys.path.insert(0, str(Path(__file__).parent))
import test_v9b_broker_n20 as base
from openwebrl.webprm_concurrent_broker import ConcurrentBrokerProcess


def slow_ok(config, payload, journal, *, local=False, endpoint=None, purpose=None, budget=None):
    """Offline completion in the package tests' style: durable reservation, 4 s 'provider' call, settlement."""
    import time as t
    from openwebrl.jev_eval import append_json
    ident = budget.reserve(0.01, payload['model'])
    append_json(journal, dict(event='request', payload=payload, purpose=purpose, local=local, reservation=ident))
    t.sleep(4)
    result = dict(model=payload['model'], choices=[dict(finish_reason='stop', message=dict(content='ok'))],
                  usage=dict(prompt_tokens=12, completion_tokens=2))
    append_json(journal, dict(event='response', reservation=ident, response=result))
    budget.settle(ident, .0001, result['usage'])
    return result


class V9e(base.BrokerN20):
    def setUp(self):
        super().setUp()
        text = (PKG/'source/scripts/launch_webprm_table5_campaign.py').read_text()
        assert 'max_concurrency=1,financial_contract=financial' in text, 'launcher must keep max_concurrency=1'
        assert 'time.time()+180' in (PKG/'source/openwebrl/webprm_concurrent_broker.py').read_text(), 'v9e status bound missing'
        self.kwargs.update(max_concurrency=int(os.environ.get('V9E_TEST_CONCURRENCY', 1)), completion_fn=slow_ok)
        self.deadline = time.time()+120; self.kwargs['deadline'] = self.deadline
        for c in self.cells.values(): c['config']['deadline'] = self.deadline
        for name in ('w1', 'w2'):
            self.cells[name] = dict(self.cells['c'], journal=str(self.root/f'{name}.jsonl'))

    def client_call(self, cell, errors):
        try:
            client = base.make_broker_client(self.kwargs['socket_path'], cell_id=cell, identity_sha256=base.IDENTITY, deadline=self.deadline)
            client(self.cells[cell]['config'], dict(model=base.ACTOR, messages=[dict(role='user', content='x')], max_tokens=8),
                   '/unused', purpose=self.other[0][1])
        except Exception as error:
            errors.append(repr(error))

    def test_status_answers_behind_two_workers(self):
        errors = []
        with ConcurrentBrokerProcess(**self.kwargs) as owner:
            threads = [threading.Thread(target=self.client_call, args=(c, errors)) for c in ('w1', 'w2')]
            [t.start() for t in threads]; time.sleep(1)
            t0 = time.time(); status = owner.status(); waited = time.time()-t0
            [t.join() for t in threads]
        self.assertIn('budget', status); self.assertGreater(waited, 2)   # it really waited behind a call
        self.assertEqual(errors, [])

    def test_same_cell_parallel_calls_both_succeed(self):
        errors = []
        with ConcurrentBrokerProcess(**self.kwargs):
            threads = [threading.Thread(target=self.client_call, args=('w1', errors)) for _ in range(2)]
            [t.start() for t in threads]; [t.join() for t in threads]
        self.assertEqual(errors, [])

    test_broker_starts_and_serves_n20_only_on_policy = None
    test_invalid_max_choices_rejected = None


if __name__ == '__main__':
    unittest.main(verbosity=2)
