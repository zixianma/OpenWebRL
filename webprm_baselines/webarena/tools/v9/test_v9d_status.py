"""v9d: with 2 workers' calls in flight, the launcher's broker status read must still answer (job 390607 regression).
Usage: V9_PACKAGE=<pkg> python test_v9d_status.py"""
import os, re, sys, threading, time, unittest
from pathlib import Path
PKG = Path(os.environ['V9_PACKAGE']); sys.path.insert(0, str(PKG/'source')); sys.path.insert(0, str(Path(__file__).parent))
import test_v9b_broker_n20 as base
from openwebrl.webprm_concurrent_broker import ConcurrentBrokerProcess


def slow_completion(config, payload, journal, **kwargs):
    import time as t; t.sleep(8)
    raise RuntimeError('offline stub: no provider call')


class StatusUnderLoad(base.BrokerN20):
    def concurrency(self):
        text = (PKG/'source/scripts/launch_webprm_table5_campaign.py').read_text()
        assert "max_concurrency=min(8,limits['max_workers']+1)" in text, 'v9d launcher patch missing'
        return int(os.environ.get('V9D_TEST_CONCURRENCY', min(8, 2 + 1)))

    def test_status_answers_with_two_calls_in_flight(self):
        self.kwargs.update(max_concurrency=self.concurrency(), completion_fn=slow_completion)
        cells = {}
        for name in ('w1', 'w2'):
            cells[name] = dict(self.cells['c'], journal=str(self.root/f'{name}.jsonl'))
        self.cells.update(cells); self.kwargs['cells'] = self.cells
        with ConcurrentBrokerProcess(**self.kwargs) as owner:
            def call(cell):
                client = base.make_broker_client(self.kwargs['socket_path'], cell_id=cell, identity_sha256=base.IDENTITY, deadline=self.deadline)
                client(self.cells[cell]['config'], dict(model=base.ACTOR, messages=[dict(role='user', content='x')], max_tokens=8), '/unused',
                       purpose=self.other[0][1])
            threads = [threading.Thread(target=call, args=(c,)) for c in ('w1', 'w2')]
            [t.start() for t in threads]; time.sleep(4.5)   # both handlers busy (8 s each; paid pacing 0 here)
            t0 = time.time(); status = owner.status(); self.assertLess(time.time()-t0, 5)
            self.assertIn('budget', status)
            [t.join() for t in threads]

    test_broker_starts_and_serves_n20_only_on_policy = None
    test_invalid_max_choices_rejected = None


if __name__ == '__main__':
    unittest.main(verbosity=2)
