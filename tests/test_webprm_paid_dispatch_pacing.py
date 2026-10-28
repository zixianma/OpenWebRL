"""Offline proof that pacing limits paid bursts without changing inference data."""
from concurrent.futures import ThreadPoolExecutor
import json
import threading
import time
import unittest

import test_webprm_concurrent_broker as helpers
from openwebrl.webprm_broker import BrokerError, make_broker_client
from openwebrl.webprm_concurrent_broker import ConcurrentBrokerProcess, PaidDispatchPacer


class PaidDispatchPacingTests(unittest.TestCase):
    def fixture(self):
        fixture=helpers.ConcurrentBrokerTests('test_four_provider_handlers_overlap_and_keep_cell_receipts_separate')
        fixture.setUp();self.addCleanup(fixture.doCleanups)
        return fixture

    def test_paid_starts_are_spaced_and_provider_handlers_still_overlap(self):
        f=self.fixture();f.kwargs['paid_dispatch_interval_seconds']=.15
        with ConcurrentBrokerProcess(**f.kwargs) as owner:
            self.assertEqual(owner.ready['paid_dispatch_interval_seconds'],.15)
            with ThreadPoolExecutor(max_workers=4) as pool:
                pending=[pool.submit(f.call,str(i)) for i in range(4)]
                try:
                    f.wait_markers(4)
                    started=[e for e in f.events() if e['event']=='dispatch_started']
                    self.assertEqual(len(started),4)
                    self.assertTrue(all(b['at_epoch']-a['at_epoch']>=.149 for a,b in zip(started,started[1:])))
                    self.assertFalse(any(p.done() for p in pending),'Pacing must not serialize full response lifetimes')
                finally:f.gate.write_text('ready')
                results=[p.result() for p in pending]
            self.assertEqual(owner.status()['paid_dispatch_interval_seconds'],.15)
            self.assertEqual(owner.status()['dispatches'],4)
        for i,result in enumerate(results):
            self.assertEqual(result['choices'][0]['message']['content'],str(i))
            request=json.loads((f.root/f'cell-{i}.jsonl').read_text().splitlines()[0])
            self.assertEqual(request['payload'],f.payload)
            self.assertEqual(request['purpose'],'policy')

    def test_local_critic_bypasses_waiting_paid_slot(self):
        f=self.fixture();f.kwargs['paid_dispatch_interval_seconds']=2
        f.kwargs['routes'][(helpers.MODEL,'critic')]=dict(local=True,endpoint='http://127.0.0.1:1234',
            max_input_tokens=10000,max_output_tokens=20)
        f.cells['2']['allowed_routes']=[[helpers.MODEL,'critic']]
        f.cells['2']['config'].pop('gate')
        local=make_broker_client(f.kwargs['socket_path'],cell_id='2',identity_sha256=helpers.IDENTITY,deadline=f.deadline)
        with ConcurrentBrokerProcess(**f.kwargs) as owner:
            with ThreadPoolExecutor(max_workers=2) as pool:
                first=pool.submit(f.call,'0');f.wait_markers(1)
                second=pool.submit(f.call,'1')
                try:
                    started=time.monotonic()
                    result=local(f.cells['2']['config'],f.payload,'/unused',purpose='critic',local=True)
                    self.assertLess(time.monotonic()-started,1)
                    self.assertEqual(result['_webprm_broker']['reservation_state'],'local')
                    self.assertFalse((f.root/'marker-1').exists())
                finally:f.gate.write_text('ready')
                first.result();second.result()
            self.assertEqual(owner.status()['paid_dispatches'],2)
            self.assertEqual(owner.status()['local_dispatches'],1)

    def test_request_without_time_for_paid_slot_rejects_without_spending(self):
        f=self.fixture();f.kwargs['paid_dispatch_interval_seconds']=2
        with ConcurrentBrokerProcess(**f.kwargs) as owner:
            with ThreadPoolExecutor(max_workers=1) as pool:
                first=pool.submit(f.call,'0');f.wait_markers(1)
                short=make_broker_client(f.kwargs['socket_path'],cell_id='1',identity_sha256=helpers.IDENTITY,
                    deadline=time.time()+.5)
                try:
                    with self.assertRaises(BrokerError):short(f.cells['1']['config'],f.payload,'/unused',purpose='policy')
                    self.assertFalse((f.root/'marker-1').exists())
                    status=owner.status()
                    self.assertEqual(status['paid_dispatches'],1)
                    self.assertEqual(status['budget']['unsettled_reservations'],1)
                    self.assertNotIn('1',status['active_cell_ids'])
                finally:f.gate.write_text('ready')
                first.result()
        rejected=[json.loads(s) for s in (f.root/'broker.jsonl.rejections.jsonl').read_text().splitlines()]
        self.assertEqual(len(rejected),1)
        self.assertEqual(rejected[0]['reason_code'],'pacing_deadline')
        self.assertFalse(rejected[0]['broker_dispatch_recorded'])
        self.assertEqual(len(json.loads((f.root/'budget.json').read_text())['calls']),1)

    def test_failed_paid_requests_are_not_retried_and_keep_reservations(self):
        f=self.fixture();f.kwargs['paid_dispatch_interval_seconds']=.15
        for cell in f.cells.values():cell['config']['behavior']='error'
        with ConcurrentBrokerProcess(**f.kwargs) as owner:
            with ThreadPoolExecutor(max_workers=2) as pool:
                pending=[pool.submit(f.call,str(i)) for i in range(2)]
                for p in pending:
                    with self.assertRaises(BrokerError):p.result()
            self.assertEqual(owner.status()['paid_dispatches'],2)
            self.assertEqual(owner.status()['budget']['unsettled_reservations'],2)
        started=[e for e in f.events() if e['event']=='dispatch_started']
        self.assertEqual(len(started),2)
        self.assertGreaterEqual(started[1]['at_epoch']-started[0]['at_epoch'],.149)
        ledger=json.loads((f.root/'budget.json').read_text())
        self.assertEqual(len(ledger['calls']),2)
        self.assertEqual(ledger['charged'],.2)

    def test_pacing_change_between_epochs_keeps_cumulative_financial_contract(self):
        f=self.fixture();f.gate.write_text('ready')
        with ConcurrentBrokerProcess(**f.kwargs):f.call('0')
        first=json.loads((f.root/'budget.json').read_text())
        f.next_epoch(first['last_closed_epoch_sha256'])
        f.kwargs['paid_dispatch_interval_seconds']=3.0
        with ConcurrentBrokerProcess(**f.kwargs) as owner:
            response=f.call('1')
            self.assertEqual(response['_webprm_broker']['reservation_id'],1)
            self.assertEqual(owner.status()['paid_dispatches'],2)
        final=json.loads((f.root/'budget.json').read_text())
        self.assertEqual(final['charged'],.02)
        self.assertEqual(final['financial_contract'],first['financial_contract'])
        self.assertEqual(final['dispatches'],2)
        events=[json.loads(s) for s in (f.root/'budget.json.events.jsonl').read_text().splitlines()]
        epochs=[e for e in events if e['event']=='epoch_started']
        self.assertEqual(len(epochs),2)
        self.assertEqual(epochs[1]['previous_closed_epoch_sha256'],first['last_closed_epoch_sha256'])

    def test_deadline_also_bounds_wait_for_another_pacing_lock_holder(self):
        pacer=PaidDispatchPacer(3);pacer.lock.acquire()
        began=time.monotonic()
        try:
            with self.assertRaisesRegex(BrokerError,'pacing deadline'):
                with pacer.slot(deadline=time.time()+.1,closed=lambda:False):
                    self.fail('Expired lock waiter must not dispatch')
        finally:pacer.lock.release()
        self.assertLess(time.monotonic()-began,.75)

    def test_stop_interrupts_a_pacing_wait_without_yielding(self):
        pacer=PaidDispatchPacer(3);stop=threading.Event()
        with pacer.slot(deadline=time.time()+5,closed=stop.is_set):pass
        timer=threading.Timer(.1,stop.set);timer.start()
        began=time.monotonic()
        try:
            with self.assertRaisesRegex(BrokerError,'admission closed'):
                with pacer.slot(deadline=time.time()+5,closed=stop.is_set):
                    self.fail('Stopped admission must not dispatch')
        finally:timer.join()
        self.assertLess(time.monotonic()-began,.75)

    def test_rejected_admission_does_not_reserve_a_future_pacing_slot(self):
        pacer=PaidDispatchPacer(3)
        with self.assertRaises(ValueError):
            with pacer.slot(deadline=time.time()+5,closed=lambda:False):raise ValueError('scripted cap rejection')
        self.assertIsNone(pacer.last_started)
        with pacer.slot(deadline=time.time()+.5,closed=lambda:False):pass

    def test_invalid_pacing_configuration_is_rejected(self):
        for value in (-1,True,None,'3',float('nan'),float('inf'),61):
            with self.subTest(value=value),self.assertRaises(ValueError):PaidDispatchPacer(value)


if __name__=='__main__':unittest.main()
