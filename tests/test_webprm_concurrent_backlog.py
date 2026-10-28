"""A bounded connection burst must queue without increasing inference concurrency."""
import os
import signal
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
import test_webprm_concurrent_broker as helpers
from openwebrl.webprm_concurrent_broker import ConcurrentBrokerProcess


class BacklogTests(unittest.TestCase):
    def test_four_clients_can_connect_before_two_handler_broker_accepts(self):
        fixture=helpers.ConcurrentBrokerTests('test_handler_bound_holds_when_more_clients_are_waiting')
        fixture.setUp();self.addCleanup(fixture.doCleanups)
        fixture.kwargs['max_concurrency']=2
        with ConcurrentBrokerProcess(**fixture.kwargs) as owner:
            os.kill(owner.process.pid,signal.SIGSTOP)
            with ThreadPoolExecutor(max_workers=4) as pool:
                futures=[pool.submit(fixture.call,str(i)) for i in range(4)]
                try:
                    # Paused accept loop makes the connection burst reproducible;
                    # no request can have reached synthetic inference yet.
                    time.sleep(.3)
                    self.assertFalse(any(f.done() for f in futures),
                                     'Client failed before bounded queued connections could be accepted')
                finally:
                    fixture.gate.write_text('release')
                    os.kill(owner.process.pid,signal.SIGCONT)
                for future in futures:future.result(timeout=10)
        active=peak=0
        for event in fixture.events():
            if not event['event'].startswith('dispatch_'):continue
            active+=1 if event['event']=='dispatch_started' else -1
            peak=max(peak,active)
        self.assertLessEqual(peak,2)
        self.assertEqual(active,0)
