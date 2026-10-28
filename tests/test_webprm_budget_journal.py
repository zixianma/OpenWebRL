"""Offline financial durability/race checks; no provider or allocation calls."""
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal,localcontext
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from openwebrl.webprm_budget_journal import JournalBudget,VERSION
from openwebrl.webprm_broker import payload_sha256

POLICY='a'*64
USAGE=dict(prompt_tokens=3,completion_tokens=2)


class BudgetJournalTests(unittest.TestCase):
    def setUp(self):
        self.root=Path(tempfile.mkdtemp(prefix='wjrn-',dir='/tmp'))
        self.path=self.root/'budget.json'

    def test_exact_accounting_durable_receipt_and_one_final_export(self):
        with JournalBudget(self.path,.3,policy_sha256=POLICY) as budget:
            a=budget.reserve(.1,'actor');b=budget.reserve(.2,'actor')
            self.assertFalse(self.path.exists())
            with self.assertRaises(RuntimeError):budget.reserve('0.000000000000000001','actor')
            budget.settle(a,.03,USAGE)
            proof=budget.receipt(a)
            raw=budget.event_path.read_bytes()
            self.assertEqual(proof['budget_prefix_bytes'],len(raw))
            self.assertEqual(proof['budget_ledger_sha256'],hashlib.sha256(raw).hexdigest())
            event=json.loads(raw.splitlines()[-1])
            self.assertEqual(proof['settlement_event_sha256'],event['event_sha256'])
            self.assertEqual(budget._charged,Decimal('.23'))
            self.assertEqual(budget.snapshot()['calls'][b]['state'],'reserved')
        snapshot=json.loads(self.path.read_text())
        self.assertEqual(snapshot['charged'],.23)
        self.assertEqual(len(snapshot['calls']),2)
        with JournalBudget(self.path,.3,policy_sha256=POLICY) as reopened:
            self.assertEqual(reopened.snapshot(),snapshot)
            self.assertEqual(reopened.reserve(.07,'actor'),2)

    def test_one_lifetime_owner_and_thread_cap_race(self):
        with JournalBudget(self.path,1,policy_sha256=POLICY) as budget:
            with self.assertRaises(BlockingIOError):JournalBudget(self.path,1,policy_sha256=POLICY)
            def reserve(_):
                try:return budget.reserve(.3,'actor')
                except RuntimeError:return None
            with ThreadPoolExecutor(max_workers=8) as pool:ids=list(pool.map(reserve,range(20)))
            self.assertEqual(sorted(x for x in ids if x is not None),[0,1,2])
            self.assertEqual(budget.snapshot()['charged'],.9)

    def test_reservation_rejection_latches_admission_without_charging_or_halting_active_work(self):
        with JournalBudget(self.path,1,policy_sha256=POLICY) as budget:
            ident=budget.reserve('.8','actor')
            before=budget.event_path.read_bytes()
            with self.assertRaisesRegex(RuntimeError,'budget exhausted'):
                budget.reserve('.3','actor')
            status=budget.summary()
            self.assertEqual(status['remaining'],'0.2')
            self.assertEqual(status['reservation_rejection'],dict(amount='0.3',model='actor',remaining='0.2'))
            self.assertIsNone(status['halt_reason'])
            self.assertEqual(budget.event_path.read_bytes(),before)
            # Existing requests retain settlement and reservation authority.
            budget.settle(ident,'.1',USAGE)
            budget.reserve('.5','actor')
            self.assertEqual(budget.summary()['reservation_rejection'],status['reservation_rejection'])
            self.assertEqual(budget.summary()['charged'],'0.6')

    def test_money_accounting_ignores_low_caller_decimal_precision(self):
        with JournalBudget(self.path,1,policy_sha256=POLICY) as budget:
            with localcontext() as context:
                context.prec=2
                budget.reserve('.123456789','actor')
                budget.reserve('.000000001','actor')
                self.assertEqual(budget.summary()['charged'],'0.123456790')

    def test_unknown_cost_survives_restart_and_double_settlement_rejected(self):
        with JournalBudget(self.path,1,policy_sha256=POLICY) as budget:
            budget.reserve(.8,'actor')
        with JournalBudget(self.path,1,policy_sha256=POLICY) as budget:
            with self.assertRaises(RuntimeError):budget.reserve(.3,'actor')
            budget.settle(0,.1,USAGE)
            with self.assertRaises(ValueError):budget.settle(0,0,USAGE)

    def test_overrun_and_explicit_halt_remain_durable(self):
        with JournalBudget(self.path,1,policy_sha256=POLICY) as budget:
            ident=budget.reserve(.1,'actor')
            with self.assertRaises(RuntimeError):budget.settle(ident,.2,USAGE)
            with self.assertRaises(RuntimeError):budget.reserve(0,'actor')
        with JournalBudget(self.path,1,policy_sha256=POLICY) as budget:
            self.assertEqual(budget.snapshot()['charged'],.2)
            self.assertEqual(budget.snapshot()['calls'][0]['state'],'overrun')
            with self.assertRaises(RuntimeError):budget.reserve(.1,'actor')

    def test_incomplete_or_modified_tail_and_changed_policy_fail_closed(self):
        with JournalBudget(self.path,1,policy_sha256=POLICY) as budget:budget.reserve(.4,'actor')
        with self.assertRaises(ValueError):JournalBudget(self.path,2,policy_sha256=POLICY)
        with self.assertRaises(ValueError):JournalBudget(self.path,1,policy_sha256='b'*64)
        event_path=Path(str(self.path)+'.events.jsonl')
        with event_path.open('ab') as stream:stream.write(b'{"event":"reserve"')
        before=event_path.read_bytes()
        with self.assertRaises(ValueError):JournalBudget(self.path,1,policy_sha256=POLICY)
        self.assertEqual(event_path.read_bytes(),before)

    def test_nonfinite_negative_or_invalid_usage_cannot_mutate_ledger(self):
        with JournalBudget(self.path,1,policy_sha256=POLICY) as budget:
            for amount in (-1,float('nan'),float('inf'),True,None):
                with self.subTest(amount=amount),self.assertRaises(ValueError):budget.reserve(amount,'actor')
            ident=budget.reserve(.1,'actor');before=budget.event_path.read_bytes()
            with self.assertRaises(ValueError):budget.settle(ident,.01,dict(prompt_tokens=True,completion_tokens=1))
            self.assertEqual(budget.event_path.read_bytes(),before)

    def test_fsync_failure_poisoned_owner_cannot_dispatch_or_export(self):
        budget=JournalBudget(self.path,1,policy_sha256=POLICY)
        with patch('openwebrl.webprm_budget_journal.os.fsync',side_effect=OSError('scripted durability failure')):
            with self.assertRaises(OSError):budget.reserve(.5,'actor')
        with self.assertRaises(RuntimeError):budget.reserve(.1,'actor')
        budget.close()
        self.assertFalse(self.path.exists())
        # Full bytes may already exist; conservative replay must retain them.
        with JournalBudget(self.path,1,policy_sha256=POLICY) as restarted:
            self.assertEqual(restarted.snapshot()['charged'],.5)

    def test_per_request_path_never_rereads_or_exports_growing_ledger(self):
        with JournalBudget(self.path,10,policy_sha256=POLICY) as budget:
            with patch.object(Path,'read_bytes',side_effect=AssertionError('No full read')),patch.object(
                    budget,'snapshot',side_effect=AssertionError('No per-call snapshot')):
                for _ in range(100):
                    ident=budget.reserve(.1,'actor');budget.settle(ident,.001,USAGE);budget.receipt(ident)
            self.assertEqual(len(budget.snapshot()['calls']),100)

    def test_legacy_migration_and_symlink_paths_rejected(self):
        self.path.write_text('{"cap":1,"charged":0,"calls":[]}')
        with self.assertRaises(ValueError):JournalBudget(self.path,1,policy_sha256=POLICY)
        other=self.root/'other.json';other.symlink_to(self.path)
        with self.assertRaises(ValueError):JournalBudget(other,1,policy_sha256=POLICY)

    def test_epoch_dispatch_caps_include_failure_before_financial_reservation(self):
        contract=dict(schema_version=1,budget_backend=VERSION,cohort_sha256='b'*64,protocol_sha256='c'*64,
            prices_usd_per_token={'actor':[1e-6,1e-6]},api_usd_cap=1.,max_dispatches=2,max_paid_dispatches=1,max_local_dispatches=1)
        policy=payload_sha256(contract)
        def start(budget,ident,previous):
            budget.start_epoch(epoch_id=ident,broker_policy_sha256='d'*64,deadline=1,implementation_sha256={'source.py':'f'*64},
                registered_cells_sha256='e'*64,previous_closed_epoch_sha256=previous)
        def dispatch(name,local=False):
            return dict(request_id=hashlib.sha256(name.encode()).hexdigest()[:32],cell_id='cell',model='actor',local=local,policy_sha256='d'*64)
        audit=self.root/'first-audit.jsonl';audit.write_text('{}\n')
        with JournalBudget(self.path,1,policy_sha256=policy,financial_contract=contract) as budget:
            start(budget,'one',None)
            budget.admit_dispatch(dispatch('first'))
            # No reserve: a credential/preparation failure must still consume a dispatch.
            previous=budget.close_epoch(audit)
        with JournalBudget(self.path,1,policy_sha256=policy,financial_contract=contract) as budget:
            start(budget,'two',previous)
            with self.assertRaises(ValueError):budget.for_request(dispatch('first')['request_id'],'cell').reserve(.1,'actor')
            with self.assertRaises(RuntimeError):budget.admit_dispatch(dispatch('second'))
            budget.admit_dispatch(dispatch('local',True))
            with self.assertRaises(RuntimeError):budget.admit_dispatch(dispatch('extra',True))
            self.assertEqual(budget.summary()['dispatches'],2)
            self.assertEqual(budget.summary()['reservations'],0)
            self.assertEqual(budget.summary()['charged'],'0')

    def test_open_epoch_or_changed_closed_audit_cannot_admit_successor(self):
        contract=dict(schema_version=1,budget_backend=VERSION,cohort_sha256='b'*64,protocol_sha256='c'*64,
            prices_usd_per_token={'actor':[1e-6,1e-6]},api_usd_cap=1.,max_dispatches=2,max_paid_dispatches=2,max_local_dispatches=2)
        policy=payload_sha256(contract)
        args=dict(broker_policy_sha256='d'*64,deadline=1,implementation_sha256={'source.py':'f'*64},registered_cells_sha256='e'*64)
        audit=self.root/'audit.jsonl';audit.write_text('{}\n')
        with JournalBudget(self.path,1,policy_sha256=policy,financial_contract=contract) as budget:
            budget.start_epoch(epoch_id='one',previous_closed_epoch_sha256=None,**args)
        with JournalBudget(self.path,1,policy_sha256=policy,financial_contract=contract) as budget:
            with self.assertRaises(ValueError):budget.start_epoch(epoch_id='two',previous_closed_epoch_sha256=None,**args)
            with self.assertRaises(ValueError):budget.close_epoch(audit)
            closure_path,audit_path=self.recovery_proofs(budget,audit)
            previous=budget.recover_epoch(audit_path=audit,closure_receipt_path=closure_path,audit_receipt_path=audit_path)
        audit.write_text('{"changed":true}\n')
        with JournalBudget(self.path,1,policy_sha256=policy,financial_contract=contract) as budget:
            with self.assertRaises(ValueError):budget.start_epoch(epoch_id='two',previous_closed_epoch_sha256=previous,**args)

    def recovery_proofs(self,budget,audit):
        evidence=self.root/'closure-evidence.json';evidence.write_text('{"offline_fixture":true}')
        binding=dict(path=str(evidence),sha256=hashlib.sha256(evidence.read_bytes()).hexdigest())
        common=dict(schema_version=1,verified=True,epoch_id='one',broker_policy_sha256='d'*64,errors=[])
        closure=dict(common,scheduler_terminal=True,all_owned_processes_reaped=True,scheduler_state='FAILED',job_id='synthetic-only',
            scheduler_evidence=binding,process_closure_evidence=binding)
        summary=budget.summary();raw=audit.read_bytes()
        result=dict(common,audit_path=str(audit),audit_prefix_bytes=len(raw),audit_sha256=hashlib.sha256(raw).hexdigest(),
            budget_prefix_bytes=summary['budget_prefix_bytes'],budget_ledger_sha256=summary['budget_ledger_sha256'],
            dispatches=summary['dispatches'],paid_dispatches=summary['paid_dispatches'],local_dispatches=summary['local_dispatches'],
            unknown_reservations=summary['unsettled_reservations'],unknown_reservations_retained=True)
        closure_path=self.root/'closure-proof.json';closure_path.write_text(json.dumps(closure))
        audit_path=self.root/'audit-proof.json';audit_path.write_text(json.dumps(result))
        return closure_path,audit_path

    def test_recovery_requires_bound_proof_and_preserves_unknown_money_and_caps(self):
        contract=dict(schema_version=1,budget_backend=VERSION,cohort_sha256='b'*64,protocol_sha256='c'*64,
            prices_usd_per_token={'actor':[1e-6,1e-6]},api_usd_cap=1.,max_dispatches=2,max_paid_dispatches=2,max_local_dispatches=2)
        policy=payload_sha256(contract)
        args=dict(broker_policy_sha256='d'*64,deadline=1,implementation_sha256={'source.py':'f'*64},registered_cells_sha256='e'*64)
        audit=self.root/'audit.jsonl';audit.write_text('{}\n')
        meta=dict(request_id='1'*32,cell_id='cell',model='actor',local=False,policy_sha256='d'*64)
        with JournalBudget(self.path,1,policy_sha256=policy,financial_contract=contract) as budget:
            budget.start_epoch(epoch_id='one',previous_closed_epoch_sha256=None,**args)
            budget.admit_dispatch(meta);budget.for_request(meta['request_id'],'cell').reserve(.8,'actor')
        with JournalBudget(self.path,1,policy_sha256=policy,financial_contract=contract) as budget:
            closure_path,audit_path=self.recovery_proofs(budget,audit)
            original=closure_path.read_text();bad=json.loads(original);bad['all_owned_processes_reaped']=False
            closure_path.write_text(json.dumps(bad));before=budget.event_path.read_bytes()
            with self.assertRaises(ValueError):budget.recover_epoch(audit_path=audit,closure_receipt_path=closure_path,audit_receipt_path=audit_path)
            self.assertEqual(budget.event_path.read_bytes(),before)
            closure_path.write_text(original)
            previous=budget.recover_epoch(audit_path=audit,closure_receipt_path=closure_path,audit_receipt_path=audit_path)
            budget.start_epoch(epoch_id='two',previous_closed_epoch_sha256=previous,**args)
            self.assertEqual(budget.summary()['charged'],'0.8')
            self.assertEqual(budget.summary()['unsettled_reservations'],1)
            self.assertEqual(budget.summary()['dispatches'],1)
            with self.assertRaises(ValueError):budget.settle(0,0,USAGE,request_id=meta['request_id'])

    def test_malformed_public_epoch_and_dispatch_do_not_append_or_poison(self):
        contract=dict(schema_version=1,budget_backend=VERSION,cohort_sha256='b'*64,protocol_sha256='c'*64,
            prices_usd_per_token={'actor':[1e-6,1e-6]},api_usd_cap=1.,max_dispatches=2,max_paid_dispatches=2,max_local_dispatches=2)
        args=dict(epoch_id='one',broker_policy_sha256='d'*64,deadline=1,implementation_sha256={'source.py':'f'*64},
                  registered_cells_sha256='e'*64,previous_closed_epoch_sha256=None)
        with JournalBudget(self.path,1,policy_sha256=payload_sha256(contract),financial_contract=contract) as budget:
            before=budget.event_path.read_bytes()
            for override in (dict(deadline=float('nan')),dict(implementation_sha256={}),dict(broker_policy_sha256='bad')):
                with self.assertRaises(ValueError):budget.start_epoch(**dict(args,**override))
                self.assertEqual(budget.event_path.read_bytes(),before)
            budget.start_epoch(**args);before=budget.event_path.read_bytes()
            for meta in ({},dict(request_id='bad',cell_id='cell',model='actor',local=False)):
                with self.assertRaises(ValueError):budget.admit_dispatch(meta)
                self.assertEqual(budget.event_path.read_bytes(),before)
            budget.admit_dispatch(dict(request_id='1'*32,cell_id='cell',model='actor',local=False,policy_sha256='d'*64))
            self.assertEqual(budget.summary()['dispatches'],1)


if __name__=='__main__':unittest.main()
