"""Independent, synthetic saved-evidence checks; no provider/browser calls."""
from copy import deepcopy
from decimal import Decimal,localcontext
import hashlib
import io
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from openwebrl.webprm_broker import payload_sha256
from openwebrl.webprm_budget_journal import JournalBudget,VERSION
from openwebrl.webprm_journal_audit import audit_cell_calls
from openwebrl.webprm_worker import input_token_upper_bound

MODEL='synthetic-actor'
CELL='webarena_lite_005/synthetic-actor__no_search'


def append(path,row):
    with path.open('ab') as stream:stream.write(json.dumps(row,sort_keys=True,separators=(',',':')).encode()+b'\n')


def offline_actual_completion(config,payload,journal,**kwargs):
    import openwebrl.webprm_worker as worker
    result=dict(model=payload['model'],choices=[dict(finish_reason='stop',message=dict(content='scripted actual completion'))],
                usage=dict(prompt_tokens=12,completion_tokens=2))
    with patch.object(worker,'load_api_credentials',return_value='offline-only'),patch.object(
            worker,'urlopen',return_value=io.BytesIO(json.dumps(result).encode())):
        return worker.completion(config,payload,journal,**kwargs)


class Fixture:
    def __init__(self):
        self.root=Path(tempfile.mkdtemp(prefix='wja-',dir='/tmp'))
        self.api=self.root/'api.jsonl';self.broker=self.root/'broker.jsonl'
        self.plan=dict(protocol_contract=dict(protocol='synthetic'),runtime='frozen-test',api_usd_cap=1.,
                       prices_usd_per_token={MODEL:[1e-6,1e-6]})
        self.contract=dict(schema_version=1,budget_backend=VERSION,cohort_sha256='b'*64,
            protocol_sha256=payload_sha256(self.plan['protocol_contract']),prices_usd_per_token={MODEL:[1e-6,1e-6]},
            api_usd_cap=1.,max_dispatches=20,max_paid_dispatches=20,max_local_dispatches=20)
        self.policy='d'*64
        self.budget=JournalBudget(self.root/'budget.json',1,policy_sha256=payload_sha256(self.contract),financial_contract=self.contract)
        self.budget.start_epoch(epoch_id='one',broker_policy_sha256=self.policy,deadline=1,
            implementation_sha256={'fixture.py':'a'*64},registered_cells_sha256='c'*64,previous_closed_epoch_sha256=None)
        append(self.broker,dict(event='epoch_started',policy_sha256=self.policy,epoch_id='one'))
        self.calls=0

    def call(self,*,local=False,state='complete',cell=CELL):
        self.calls+=1;ident=f'{self.calls:032x}'
        model='synthetic-local' if local else MODEL
        payload=dict(model=model,messages=[dict(role='user',content=f'offline request {self.calls}')],max_tokens=20,temperature=0)
        meta=dict(schema_version=1,policy_sha256=self.policy,request_id=ident,cell_id=cell,
            identity_sha256=payload_sha256(self.plan),model=model,purpose='policy',local=local,payload_sha256=payload_sha256(payload),
            broker_version='bounded-single-owner-broker-v2')
        self.budget.admit_dispatch(meta);append(self.broker,dict(meta,event='dispatch_started'))
        if state=='before_reserve':
            append(self.broker,dict(meta,event='dispatch_failed',error_type='RuntimeError'));return ident
        adapter=self.budget.for_request(ident,cell)
        amount=2*(input_token_upper_bound(payload['messages'])*Decimal('.000001')+20*Decimal('.000001'))
        reservation=None if local else adapter.reserve(amount,model)
        if state=='other_pending':return ident
        append(self.api,dict(event='request',payload=payload,purpose='policy',local=local,reservation=reservation))
        if state in ('unknown','response_error'):
            if state=='response_error':append(self.api,dict(event='response_error',reservation=reservation,status_code=503,error_body='scripted'))
            append(self.broker,dict(meta,event='dispatch_failed',error_type='HTTPError'));return ident
        response=dict(model=model,choices=[dict(finish_reason='stop',message=dict(content='malformed actor action is still a paid response'))],
            usage=dict(prompt_tokens=12,completion_tokens=2))
        append(self.api,dict(event='response',reservation=reservation,response=response))
        if local:proof=dict(reservation_id=None,reservation_state='local',usage=response['usage'])
        else:
            adapter.settle(reservation,Decimal('.000014'),response['usage']);proof=self.budget.receipt(reservation)
        done=dict(meta,event='dispatch_completed',budget_owner='webprm_worker.completion',
                  journal_sha256=hashlib.sha256(self.api.read_bytes()).hexdigest())
        done.update(proof);append(self.broker,done)
        if state=='send_failure':append(self.broker,dict(meta,event='dispatch_failed',error_type='BrokenPipeError'))
        return ident

    def bindings(self):
        budget=self.budget.event_path.read_bytes();broker=self.broker.read_bytes()
        return dict(plan=self.plan,cell_id=CELL,api_path=self.api,broker_audit_path=self.broker,
            budget_event_path=self.budget.event_path,budget_prefix_bytes=len(budget),budget_prefix_sha256=hashlib.sha256(budget).hexdigest(),
            financial_contract=self.contract,broker_prefix_bytes=len(broker),broker_prefix_sha256=hashlib.sha256(broker).hexdigest())

    def audit(self):return audit_cell_calls(**self.bindings())

    def rewrite_broker(self,mutate):
        rows=[json.loads(x) for x in self.broker.read_text().splitlines()]
        mutate(rows)
        self.broker.write_text(''.join(json.dumps(r,sort_keys=True,separators=(',',':'))+'\n' for r in rows))


class JournalAuditTests(unittest.TestCase):
    def setUp(self):self.fixture=Fixture();self.addCleanup(self.fixture.budget.close)

    def test_paid_local_and_parse_failure_responses_are_all_joined(self):
        f=self.fixture;f.call();f.call(local=True)
        result=f.audit()
        self.assertTrue(result['verified'])
        self.assertEqual(result['broker_policy_sha256s'],[f.policy])
        self.assertEqual((result['paid_calls'],result['local_calls'],result['api_request_count']),(1,1,2))
        self.assertEqual(result['global_charged_usd'],'0.000014')
        self.assertEqual(len(result['pairs']),2)
        self.assertIn('malformed actor',result['pairs'][0]['response']['response']['choices'][0]['message']['content'])

    def test_unknown_provider_failure_and_before_reserve_failure_are_not_dropped(self):
        f=self.fixture;f.call(state='before_reserve');f.call(state='unknown');f.call(state='response_error');f.call()
        result=f.audit()
        self.assertEqual(result['paid_calls'],4)
        self.assertEqual(result['api_request_count'],3)
        self.assertEqual(len(result['broker_only_failures']),1)
        self.assertEqual(result['cell_unknown_reservations'],2)
        self.assertIsNone(result['pairs'][0]['response'])
        self.assertEqual(result['pairs'][1]['error']['status_code'],503)
        self.assertGreater(Decimal(result['global_charged_usd']),Decimal('.000014'))

    def test_delivery_failure_preserves_settled_completion(self):
        self.fixture.call(state='send_failure');result=self.fixture.audit()
        self.assertEqual(result['failed_calls'],1)
        self.assertEqual(result['global_unknown_reservations'],0)
        self.assertEqual(result['pairs'][0]['broker']['event'],'dispatch_completed')
        self.assertEqual(result['pairs'][0]['delivery_failure']['event'],'dispatch_failed')

    def test_other_cell_pending_is_allowed_but_target_pending_is_rejected(self):
        f=self.fixture;f.call();f.call(cell='other-cell',state='other_pending')
        result=f.audit()
        self.assertEqual(result['paid_calls'],1)
        self.assertEqual(result['global_paid_dispatches'],2)
        self.assertEqual(result['global_unknown_reservations'],1)
        f.call(state='other_pending')
        with self.assertRaisesRegex(ValueError,'unfinished'):f.audit()

    def test_node_loss_paid_request_retains_charge_without_inventing_response(self):
        f=self.fixture;ident=f.call(state='other_pending')
        payload=dict(model=MODEL,messages=[dict(role='user',content='offline request 1')],max_tokens=20,temperature=0)
        append(f.api,dict(event='request',payload=payload,purpose='policy',local=False,reservation=0))
        before={p:p.read_bytes() for p in f.root.iterdir() if p.is_file()}
        with self.assertRaisesRegex(ValueError,'unfinished'):f.audit()
        result=audit_cell_calls(**f.bindings(),interrupted_paid_request_ids=[ident])
        self.assertEqual(result['interrupted_paid_dispatches'],[ident])
        self.assertEqual(result['cell_unknown_reservations'],1)
        self.assertIsNone(result['pairs'][0]['response'])
        self.assertIsNone(result['pairs'][0]['broker'])
        self.assertGreater(Decimal(result['global_charged_usd']),0)
        self.assertEqual(before,{p:p.read_bytes() for p in f.root.iterdir() if p.is_file()})
        with self.assertRaisesRegex(ValueError,'exact local or retained paid reservation'):
            audit_cell_calls(**f.bindings(),interrupted_local_request_ids=[ident])

    def test_interruption_opt_in_cannot_hide_settled_completed_or_wrong_calls(self):
        f=self.fixture;ident=f.call()
        with self.assertRaisesRegex(ValueError,'exact local or retained paid reservation'):
            audit_cell_calls(**f.bindings(),interrupted_paid_request_ids=[ident])
        with self.assertRaisesRegex(ValueError,'not in the exact target cell'):
            audit_cell_calls(**f.bindings(),interrupted_paid_request_ids=['f'*32])
        with self.assertRaisesRegex(ValueError,'At most one exact interrupted request'):
            audit_cell_calls(**f.bindings(),interrupted_paid_request_ids=[ident],interrupted_local_request_ids=[ident])

    def test_hash_bound_prefix_ignores_later_other_cell_append(self):
        f=self.fixture;f.call();bindings=f.bindings();f.call(cell='other-cell',state='other_pending')
        result=audit_cell_calls(**bindings)
        self.assertEqual(result['global_dispatches'],1)
        self.assertEqual(result['global_unknown_reservations'],0)

    def test_later_broker_snapshot_allows_unrelated_calls_after_financial_snapshot(self):
        f=self.fixture;f.call();bindings=f.bindings()
        f.call(cell='other-cell',state='other_pending')
        raw=f.broker.read_bytes()
        bindings.update(broker_prefix_bytes=len(raw),broker_prefix_sha256=hashlib.sha256(raw).hexdigest())
        result=audit_cell_calls(**bindings)
        self.assertTrue(result['verified'])
        self.assertEqual(result['global_dispatches'],1)
        self.assertEqual(result['paid_calls'],1)
        self.assertEqual(result['broker_policy_sha256s'],[f.policy])
        # A later target-cell call cannot be hidden by the older budget snapshot.
        f.call(state='other_pending');raw=f.broker.read_bytes()
        bindings.update(broker_prefix_bytes=len(raw),broker_prefix_sha256=hashlib.sha256(raw).hexdigest())
        with self.assertRaisesRegex(ValueError,'dispatch sets differ'):audit_cell_calls(**bindings)

    def test_mixed_epoch_policies_in_any_broker_record_are_rejected(self):
        f=self.fixture;f.call();original=f.broker.read_text()
        previous=f.budget.close_epoch(f.broker)
        f.budget.start_epoch(epoch_id='two',broker_policy_sha256='e'*64,deadline=1,
            implementation_sha256={'fixture.py':'a'*64},registered_cells_sha256='c'*64,previous_closed_epoch_sha256=previous)
        for row in (dict(event='epoch_started',epoch_id='foreign',policy_sha256='e'*64),
                    dict(event='dispatch_started',cell_id='other-cell',request_id='f'*32,policy_sha256='e'*64)):
            f.broker.write_text(original);append(f.broker,row)
            with self.subTest(event=row['event']),self.assertRaisesRegex(ValueError,'Exactly one broker epoch policy'):f.audit()

    def test_read_only_audit_never_opens_writer_or_changes_files(self):
        f=self.fixture;f.call();before={p:p.read_bytes() for p in f.root.iterdir() if p.is_file()}
        with patch.object(JournalBudget,'__init__',side_effect=AssertionError('Auditor must not construct budget owner')):
            self.assertTrue(f.audit()['verified'])
        self.assertEqual(before,{p:p.read_bytes() for p in f.root.iterdir() if p.is_file()})

    def test_worker_price_cap_and_caller_decimal_precision_are_independently_checked(self):
        f=self.fixture;f.call();bindings=f.bindings()
        with localcontext() as context:
            context.prec=2
            self.assertTrue(f.audit()['verified'])
        for change in (dict(api_usd_cap=2),dict(prices_usd_per_token={MODEL:[2e-6,1e-6]})):
            with self.subTest(change=change),self.assertRaisesRegex(ValueError,'worker cap/prices'):
                audit_cell_calls(**dict(bindings,plan=dict(f.plan,**change)))

    def test_wrong_hash_partial_prefix_or_changed_protocol_rejected(self):
        f=self.fixture;f.call();bindings=f.bindings()
        for change in (dict(budget_prefix_sha256='0'*64),dict(broker_prefix_sha256='0'*64),
                       dict(budget_prefix_bytes=bindings['budget_prefix_bytes']-1),dict(broker_prefix_bytes=bindings['broker_prefix_bytes']-1),
                       dict(plan=dict(f.plan,protocol_contract=dict(protocol='changed')))):
            with self.subTest(change=change),self.assertRaises(ValueError):audit_cell_calls(**dict(bindings,**change))

    def test_duplicate_or_unfinished_broker_record_rejected(self):
        f=self.fixture;f.call()
        original=f.broker.read_text()
        f.rewrite_broker(lambda rows:rows.append(deepcopy(rows[-1])))
        with self.assertRaisesRegex(ValueError,'Duplicate'):f.audit()
        f.broker.write_text(original)
        f.rewrite_broker(lambda rows:rows.pop())
        with self.assertRaisesRegex(ValueError,'unfinished'):f.audit()

    def test_receipt_event_or_prefix_mismatch_rejected(self):
        f=self.fixture;f.call();original=f.broker.read_text()
        for field,value in [('settlement_event_sha256','0'*64),('budget_ledger_sha256','0'*64),('journal_sha256','0'*64)]:
            f.broker.write_text(original)
            f.rewrite_broker(lambda rows:rows[-1].update({field:value}))
            with self.subTest(field=field),self.assertRaises(ValueError):f.audit()

    def test_unmatched_or_modified_api_request_rejected(self):
        f=self.fixture;f.call();rows=[json.loads(x) for x in f.api.read_text().splitlines()]
        rows[0]['payload']['messages'][0]['content']='changed request'
        f.api.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        with self.assertRaises(ValueError):f.audit()

    def test_financial_hash_chain_tampering_detected_even_with_updated_outer_hash(self):
        f=self.fixture;f.call()
        rows=[json.loads(x) for x in f.budget.event_path.read_text().splitlines()]
        rows[-1]['actual']='0'
        altered=f.root/'tampered-finance.jsonl';altered.write_text(''.join(json.dumps(r,sort_keys=True,separators=(',',':'))+'\n' for r in rows))
        raw=altered.read_bytes();bindings=f.bindings()
        bindings.update(budget_event_path=altered,budget_prefix_bytes=len(raw),budget_prefix_sha256=hashlib.sha256(raw).hexdigest())
        with self.assertRaisesRegex(ValueError,'hash chain'):audit_cell_calls(**bindings)

    def test_actual_broker_and_completion_artifacts_pass_independent_reader(self):
        from openwebrl.webprm_broker import make_broker_client
        from openwebrl.webprm_concurrent_broker import ConcurrentBrokerProcess
        root=self.fixture.root;deadline=time.time()+10
        plan=dict(protocol_contract=dict(protocol='actual broker offline'),deadline=deadline,budget_path=str(root/'actual-budget.json'),
                  api_usd_cap=1.,prices_usd_per_token={MODEL:[1e-6,1e-6]})
        identity=payload_sha256(plan);config=dict(plan,identity_sha256=identity)
        contract=dict(self.fixture.contract,protocol_sha256=payload_sha256(plan['protocol_contract']))
        kwargs=dict(socket_path=str(root/'s'),deadline=deadline,
            cells={CELL:dict(config=config,journal=str(root/'actual-api.jsonl'),allowed_routes=[[MODEL,'policy']])},
            routes={(MODEL,'policy'):dict(local=False,endpoint=None,max_input_tokens=10000,max_output_tokens=20)},
            max_dispatches=20,max_concurrency=2,audit_path=str(root/'actual-broker.jsonl'),completion_fn=offline_actual_completion,
            financial_contract=contract,epoch_id='actual-offline',previous_closed_epoch_sha256=None)
        with ConcurrentBrokerProcess(**kwargs):
            client=make_broker_client(kwargs['socket_path'],cell_id=CELL,identity_sha256=identity,deadline=deadline)
            client(config,dict(model=MODEL,messages=[dict(role='user',content='offline actual broker')],max_tokens=20),
                   '/unused',purpose='policy')
        budget=(root/'actual-budget.json.events.jsonl').read_bytes();broker=(root/'actual-broker.jsonl').read_bytes()
        result=audit_cell_calls(plan,CELL,root/'actual-api.jsonl',root/'actual-broker.jsonl',root/'actual-budget.json.events.jsonl',
            len(budget),hashlib.sha256(budget).hexdigest(),contract,broker_prefix_bytes=len(broker),broker_prefix_sha256=hashlib.sha256(broker).hexdigest())
        self.assertTrue(result['verified'])
        self.assertEqual(result['paid_calls'],1)
        self.assertEqual(result['global_charged_usd'],'0.000014')


if __name__=='__main__':unittest.main()
