"""Actual process/socket overlap, with provider completion scripted offline."""
from concurrent.futures import ThreadPoolExecutor
import io
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest

from openwebrl.webprm_broker import BrokerError,make_broker_client,payload_sha256
from openwebrl.webprm_budget_journal import JournalBudget,VERSION as BUDGET_VERSION
from openwebrl.webprm_concurrent_broker import ConcurrentBrokerProcess

MODEL='scripted-fixed-model'
IDENTITY='a'*64


def completion(config,payload,journal,*,local=False,endpoint=None,purpose=None,budget=None):
    from openwebrl.jev_eval import append_json
    assert budget is not None
    if config.get('behavior')=='before_reserve':raise RuntimeError('Scripted preparation failure')
    ident=None if local else budget.reserve(config.get('reserve',.1),payload['model'])
    append_json(journal,dict(event='request',payload=payload,purpose=purpose,local=local,reservation=ident))
    marker=Path(config['marker']);marker.write_text('reserved before scripted dispatch')
    if config.get('behavior')=='error':raise RuntimeError('SECRET must never be returned')
    if config.get('behavior')=='hang':time.sleep(30)
    if config.get('gate'):
        until=time.monotonic()+8
        while not Path(config['gate']).exists() and time.monotonic()<until:time.sleep(.01)
        if not Path(config['gate']).exists():raise RuntimeError('Offline concurrency gate timed out')
    result=dict(model=payload['model'],choices=[dict(finish_reason='stop',message=dict(content=config['name']))],
                usage=dict(prompt_tokens=12,completion_tokens=2))
    append_json(journal,dict(event='response',reservation=ident,response=result))
    if not local:budget.settle(ident,.01,result['usage'])
    return result


def actual_completion_with_scripted_http(config,payload,journal,**kwargs):
    from unittest.mock import patch
    import openwebrl.webprm_worker as worker
    def respond(request,timeout):
        assert request.full_url=='https://api.openai.com/v1/chat/completions'
        assert json.loads(request.data)==payload
        # The durable journal contains a reservation before the transport seam.
        events=Path(config['budget_path']+'.events.jsonl').read_text().splitlines()
        assert json.loads(events[-1])['event']=='reserve'
        return io.BytesIO(json.dumps(dict(model=payload['model'],choices=[dict(finish_reason='stop',message=dict(content='native completion'))],
            usage=dict(prompt_tokens=12,completion_tokens=2))).encode())
    with patch.object(worker,'load_api_credentials',return_value='offline-key'),patch.object(worker,'urlopen',side_effect=respond):
        return worker.completion(config,payload,journal,**kwargs)


def completion_with_slow_reply(config,payload,journal,**kwargs):
    """Runs inside the spawned broker: delay the handler after each successful reply."""
    import openwebrl.webprm_concurrent_broker as module
    if not getattr(module._send,'slow_after_reply',False):
        original=module._send
        def slow(conn,value,*args,**kw):
            original(conn,value,*args,**kw)
            if value.get('ok') is True and 'receipt' in value:time.sleep(.3)
        slow.slow_after_reply=True;module._send=slow
    return completion(config,payload,journal,**kwargs)


class ConcurrentBrokerTests(unittest.TestCase):
    def setUp(self):
        self.root=Path(tempfile.mkdtemp(prefix='wcb-',dir='/tmp'))
        self.deadline=time.time()+20
        self.gate=self.root/'release'
        self.payload=dict(model=MODEL,messages=[dict(role='user',content='scripted request')],max_tokens=20,temperature=0)
        self.cells={}
        for i in range(4):
            config=dict(identity_sha256=IDENTITY,deadline=self.deadline,budget_path=str(self.root/'budget.json'),
                api_usd_cap=1.,prices_usd_per_token={MODEL:[1e-6,1e-6]},name=str(i),marker=str(self.root/f'marker-{i}'),gate=str(self.gate))
            self.cells[str(i)]=dict(config=config,journal=str(self.root/f'cell-{i}.jsonl'),allowed_routes=[[MODEL,'policy']])
        self.kwargs=dict(socket_path=str(self.root/'s'),deadline=self.deadline,cells=self.cells,
            routes={(MODEL,'policy'):dict(local=False,endpoint=None,max_input_tokens=10000,max_output_tokens=20)},
            max_dispatches=8,max_concurrency=4,audit_path=str(self.root/'broker.jsonl'),completion_fn=completion)
        self.contract=dict(schema_version=1,budget_backend=BUDGET_VERSION,cohort_sha256='b'*64,protocol_sha256='c'*64,
            prices_usd_per_token={MODEL:[1e-6,1e-6]},api_usd_cap=1.,max_dispatches=8,max_paid_dispatches=8,max_local_dispatches=8)
        self.kwargs.update(financial_contract=self.contract,epoch_id='allocation-1',previous_closed_epoch_sha256=None)

    def call(self,cell):
        client=make_broker_client(self.kwargs['socket_path'],cell_id=cell,identity_sha256=IDENTITY,deadline=self.deadline)
        return client(self.cells[cell]['config'],self.payload,'/unused-client-path',purpose='policy')

    def wait_markers(self,n):
        until=time.monotonic()+5
        while len(list(self.root.glob('marker-*')))<n and time.monotonic()<until:time.sleep(.01)
        self.assertEqual(len(list(self.root.glob('marker-*'))),n)

    def events(self):return [json.loads(x) for x in (self.root/'broker.jsonl').read_text().splitlines()]

    def test_four_provider_handlers_overlap_and_keep_cell_receipts_separate(self):
        with ConcurrentBrokerProcess(**self.kwargs) as owner:
            with ThreadPoolExecutor(max_workers=4) as pool:
                futures=[pool.submit(self.call,str(i)) for i in range(4)]
                try:
                    self.wait_markers(4)
                    self.assertEqual([x['event'] for x in self.events() if x['event'].startswith('dispatch_')],['dispatch_started']*4)
                    self.assertFalse((self.root/'budget.json').exists())
                finally:self.gate.write_text('release')
                results=[f.result() for f in futures]
            for i,result in enumerate(results):
                self.assertEqual(result['choices'][0]['message']['content'],str(i))
                self.assertEqual(result['_webprm_broker']['cell_id'],str(i))
                self.assertEqual(len(result['_webprm_broker']['settlement_event_sha256']),64)
            status=owner.status()
            self.assertEqual(status['dispatches'],4)
            self.assertEqual(status['budget']['unsettled_reservations'],0)
            self.assertEqual(status['budget']['charged'],'0.04')
        self.assertTrue(owner.closure['reaped'])
        self.assertEqual(owner.closure['exitcode'],0)
        self.assertEqual(json.loads((self.root/'budget.json').read_text())['charged'],.04)

    def test_atomic_dispatch_cap_with_concurrent_callers(self):
        self.kwargs['max_dispatches']=2
        self.contract.update(max_dispatches=2,max_paid_dispatches=2,max_local_dispatches=2)
        with ConcurrentBrokerProcess(**self.kwargs):
            with ThreadPoolExecutor(max_workers=4) as pool:
                futures=[pool.submit(self.call,str(i)) for i in range(4)]
                try:self.wait_markers(2)
                finally:self.gate.write_text('release')
                success=0
                for f in futures:
                    try:f.result();success+=1
                    except BrokerError:pass
        self.assertEqual(success,2)
        self.assertEqual(sum(x['event']=='dispatch_started' for x in self.events()),2)
        self.assertEqual(len(json.loads((self.root/'budget.json').read_text())['calls']),2)

    def test_handler_bound_holds_when_more_clients_are_waiting(self):
        self.kwargs['max_concurrency']=2
        with ConcurrentBrokerProcess(**self.kwargs):
            with ThreadPoolExecutor(max_workers=4) as pool:
                futures=[pool.submit(self.call,str(i)) for i in range(4)]
                try:
                    self.wait_markers(2)
                    self.assertEqual(sum(x['event']=='dispatch_started' for x in self.events()),2)
                finally:self.gate.write_text('release')
                for f in futures:f.result()
        active=peak=0
        for event in self.events():
            if not event['event'].startswith('dispatch_'):continue
            active += 1 if event['event']=='dispatch_started' else -1
            peak=max(peak,active)
        self.assertEqual(peak,2)
        self.assertEqual(active,0)

    def test_atomic_financial_cap_unknown_failures_keep_full_reservation(self):
        for cell in self.cells.values():cell['config'].update(reserve=.6,behavior='error')
        with ConcurrentBrokerProcess(**self.kwargs) as owner:
            with ThreadPoolExecutor(max_workers=4) as pool:
                futures=[pool.submit(self.call,str(i)) for i in range(4)]
                for future in futures:
                    with self.assertRaises(BrokerError):future.result()
            status=owner.status()
            self.assertEqual(status['budget']['charged'],'0.6')
            self.assertEqual(status['budget']['unsettled_reservations'],1)
        with JournalBudget(self.root/'budget.json',1,policy_sha256=payload_sha256(self.contract),financial_contract=self.contract) as budget:
            self.assertEqual(budget.summary()['charged'],'0.6')
            with self.assertRaises(RuntimeError):budget.reserve(.6,MODEL)
        self.assertNotIn('SECRET',(self.root/'broker.jsonl').read_text())

    def test_same_cell_overlap_rejected_before_second_reservation(self):
        with ConcurrentBrokerProcess(**self.kwargs) as owner:
            with ThreadPoolExecutor(max_workers=2) as pool:
                first=pool.submit(self.call,'0')
                try:
                    self.wait_markers(1)
                    self.assertEqual(owner.status()['active_cell_ids'],['0'])
                    with self.assertRaises(BrokerError):pool.submit(self.call,'0').result()
                finally:self.gate.write_text('release')
                first.result()
        self.assertEqual(len(json.loads((self.root/'budget.json').read_text())['calls']),1)
        self.assertEqual(sum(x['event']=='dispatch_started' for x in self.events()),1)

    def test_back_to_back_requests_from_one_cell_are_not_overlaps(self):
        # 361041: Shepherd workers send the next request milliseconds after a reply; the
        # cell must be released before the reply, not after the handler finishes sending.
        self.kwargs['completion_fn']=completion_with_slow_reply
        self.gate.write_text('release')
        with ConcurrentBrokerProcess(**self.kwargs):
            for _ in range(3):self.call('0')
        self.assertEqual(sum(x['event']=='dispatch_completed' for x in self.events()),3)

    def test_existing_completion_uses_explicit_adapter_and_durable_reservation(self):
        self.kwargs['completion_fn']=actual_completion_with_scripted_http
        with ConcurrentBrokerProcess(**self.kwargs):result=self.call('0')
        self.assertEqual(result['choices'][0]['message']['content'],'native completion')
        ledger=json.loads((self.root/'budget.json').read_text())
        self.assertEqual(ledger['charged'],.000014)
        self.assertEqual(len(ledger['calls']),1)

    def test_orderly_close_awaits_active_handler_before_export(self):
        owner=ConcurrentBrokerProcess(**self.kwargs).start()
        with ThreadPoolExecutor(max_workers=1) as pool:
            pending=pool.submit(self.call,'0');self.wait_markers(1)
            timer=threading.Timer(.15,lambda:self.gate.write_text('release'));timer.start()
            try:owner.close();pending.result()
            finally:timer.join();owner.close()
        self.assertEqual(owner.closure['exitcode'],0)
        self.assertEqual(json.loads((self.root/'budget.json').read_text())['calls'][0]['state'],'settled')

    def test_hung_handler_is_owned_reaped_and_reservation_survives_without_export(self):
        self.cells['0']['config']['behavior']='hang'
        owner=ConcurrentBrokerProcess(**self.kwargs,shutdown_seconds=.1).start()
        with ThreadPoolExecutor(max_workers=1) as pool:
            pending=pool.submit(self.call,'0');self.wait_markers(1);owner.close()
            with self.assertRaises((BrokerError,OSError)):pending.result()
        self.assertTrue(owner.closure['reaped'])
        self.assertFalse((self.root/'budget.json').exists())
        with JournalBudget(self.root/'budget.json',1,policy_sha256=payload_sha256(self.contract),financial_contract=self.contract) as budget:
            self.assertEqual(budget.summary()['charged'],'0.1')
            self.assertEqual(budget.summary()['unsettled_reservations'],1)

    def test_split_financial_authorities_rejected_at_startup(self):
        self.cells['1']['config']['budget_path']=str(self.root/'other-budget.json')
        owner=ConcurrentBrokerProcess(**self.kwargs)
        with self.assertRaises(BrokerError):owner.start()
        self.assertTrue(owner.closure['reaped'])
        self.assertFalse((self.root/'budget.json.events.jsonl').exists())

    def test_cell_journal_cannot_alias_global_financial_events(self):
        self.cells['0']['journal']=str(self.root/'budget.json.events.jsonl')
        owner=ConcurrentBrokerProcess(**self.kwargs)
        with self.assertRaises(BrokerError):owner.start()
        self.assertTrue(owner.closure['reaped'])
        self.assertFalse((self.root/'budget.json.events.jsonl').exists())

    def next_epoch(self,previous):
        self.deadline=time.time()+25
        for cell in self.cells.values():cell['config']['deadline']=self.deadline
        self.kwargs.update(socket_path=str(self.root/'s2'),audit_path=str(self.root/'broker2.jsonl'),
            deadline=self.deadline,epoch_id='allocation-2',previous_closed_epoch_sha256=previous)

    def test_new_allocation_deadline_preserves_money_ids_and_dispatch_counts(self):
        self.gate.write_text('ready')
        with ConcurrentBrokerProcess(**self.kwargs) as first:
            result=self.call('0');self.assertEqual(result['_webprm_broker']['reservation_id'],0)
        snapshot=json.loads((self.root/'budget.json').read_text())
        self.assertIsNone(snapshot['active_epoch_id'])
        self.next_epoch(snapshot['last_closed_epoch_sha256'])
        with ConcurrentBrokerProcess(**self.kwargs) as second:
            result=self.call('1');self.assertEqual(result['_webprm_broker']['reservation_id'],1)
            status=second.status()
            self.assertEqual(status['dispatches'],2)
            self.assertEqual(status['budget']['charged'],'0.02')
            self.assertNotEqual(first.ready['policy_sha256'],second.ready['policy_sha256'])
        snapshot=json.loads((self.root/'budget.json').read_text())
        self.assertEqual(snapshot['dispatches'],2)
        self.assertEqual(len(snapshot['calls']),2)

    def test_prior_before_reserve_failure_consumes_cap_in_successor_epoch(self):
        self.kwargs['max_dispatches']=1
        self.contract.update(max_dispatches=1,max_paid_dispatches=1,max_local_dispatches=1)
        self.cells['0']['config']['behavior']='before_reserve'
        with ConcurrentBrokerProcess(**self.kwargs):
            with self.assertRaises(BrokerError):self.call('0')
        snapshot=json.loads((self.root/'budget.json').read_text())
        self.assertEqual(snapshot['calls'],[])
        self.assertEqual(snapshot['dispatches'],1)
        self.next_epoch(snapshot['last_closed_epoch_sha256'])
        with ConcurrentBrokerProcess(**self.kwargs) as second:
            with self.assertRaises(BrokerError):self.call('1')
            self.assertEqual(second.status()['dispatches'],1)
        self.assertFalse((self.root/'marker-1').exists())

    def test_wrong_predecessor_and_changed_financial_protocol_rejected(self):
        self.gate.write_text('ready')
        with ConcurrentBrokerProcess(**self.kwargs):self.call('0')
        self.next_epoch('0'*64)
        rejected=ConcurrentBrokerProcess(**self.kwargs)
        with self.assertRaises(BrokerError):rejected.start()
        self.assertTrue(rejected.closure['reaped'])
        snapshot=json.loads((self.root/'budget.json').read_text())
        self.kwargs['previous_closed_epoch_sha256']=snapshot['last_closed_epoch_sha256']
        self.contract['protocol_sha256']='f'*64
        rejected=ConcurrentBrokerProcess(**self.kwargs)
        with self.assertRaises(BrokerError):rejected.start()
        self.assertTrue(rejected.closure['reaped'])


    def test_pre_dispatch_rejection_keeps_metadata_without_spending_or_retry(self):
        self.gate.write_text('ready')
        client=make_broker_client(self.kwargs['socket_path'],cell_id='0',identity_sha256=IDENTITY,deadline=self.deadline)
        invalid=dict(self.payload,max_tokens=21,messages=[dict(role='user',content='SECRET request body')])
        with ConcurrentBrokerProcess(**self.kwargs) as owner:
            with self.assertRaises(BrokerError):
                client(self.cells['0']['config'],invalid,'/unused',purpose='policy')
            status=owner.status();self.assertEqual(status['dispatches'],0)
            self.assertEqual(status['budget']['charged'],'0')
            self.assertFalse((self.root/'cell-0.jsonl').exists())
            self.assertFalse((self.root/'marker-0').exists())
            # A distinct subsequent request can succeed; the rejected one is never retried.
            self.call('0');self.assertEqual(owner.status()['dispatches'],1)
        path=self.root/'broker.jsonl.rejections.jsonl';raw=path.read_text()
        self.assertNotIn('SECRET',raw);self.assertNotIn('messages',raw)
        rows=[json.loads(x) for x in raw.splitlines()];self.assertEqual(len(rows),1)
        self.assertEqual(rows[0]['reason_code'],'output_limit')
        self.assertEqual(rows[0]['stage'],'output_limit')
        self.assertFalse(rows[0]['broker_dispatch_recorded'])
        self.assertEqual(len([x for x in self.events() if x['event']=='dispatch_started']),1)

    def test_rejection_metadata_redacts_unregistered_values_and_respects_file_cap(self):
        from openwebrl.webprm_concurrent_broker import rejection_metadata,append_rejection
        request=dict(cell_id='SECRET cell',request_id='SECRET request',purpose='SECRET purpose',
            payload=dict(model='SECRET model',messages='SECRET body'))
        for error in [BrokerError('SECRET error'),OSError(14,'SECRET path')]:
            row=rejection_metadata(error,request,'payload',self.cells,self.kwargs['routes'])
            self.assertNotIn('SECRET',json.dumps(row));self.assertIsNone(row['cell_id'])
            self.assertIsNone(row['model']);self.assertIsNone(row['request_id'])
        self.assertEqual(row['errno'],14)
        path=self.root/'bounded-rejections.jsonl';append_rejection(path,row,limit=1024)
        before=path.read_bytes()
        with self.assertRaisesRegex(BrokerError,'cap exhausted'):
            append_rejection(path,row,limit=len(before))
        self.assertEqual(path.read_bytes(),before)

    def test_rejection_diagnostics_cannot_alias_a_cell_journal(self):
        self.cells['0']['journal']=str(self.root/'broker.jsonl.rejections.jsonl')
        owner=ConcurrentBrokerProcess(**self.kwargs)
        with self.assertRaises(BrokerError):owner.start()
        self.assertTrue(owner.closure['reaped'])
        self.assertFalse((self.root/'budget.json.events.jsonl').exists())



if __name__=='__main__':unittest.main()
