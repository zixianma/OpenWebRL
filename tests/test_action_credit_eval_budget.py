"""Evaluation retries share campaign caps without entering training inventories."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor

from openwebrl.action_credit_state import CampaignManifest, CampaignStore, BudgetExceeded, StateError, digest
from tests import test_action_credit_state as fixtures


class EvaluationBudgetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="action-credit-eval-budget-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        original = self.artifact("original.json", {"files": []})
        manifest = fixtures.manifest_data()
        manifest['science']['algorithm']['local_actor_requests_cap']=3
        manifest["initialization"]["original_actor_sha256"] = original["sha256"]
        self.store = CampaignStore.create(self.root / "store", CampaignManifest.from_dict(manifest))
        self.ids = [f"held-out-{i}" for i in range(300)]
        self.request = dict(schema_version=1, task_ids=self.ids,
            task_manifest=self.artifact("tasks.json", self.ids), protocol={"fixed": True,"max_turns":30},
            protocol_sha256=digest({"fixed": True,"max_turns":30}), checkpoint_inventory=original,
            policy_sha256=original["sha256"], iteration=0,
            source_inventory=self.artifact("source.json", {"files": {}}),
            accounting_manifest_sha256=self.store.manifest.sha256,
            artifact_schema="native_monitor_full300_v1", diagnostic_only=True)
        self.request["cohort_id"] = digest(self.request)
        self.ref = self.artifact("request.json", self.request)
        self.cohort = self.request["cohort_id"]
        self.store.register_evaluation_cohort(self.cohort, self.ref)

    def artifact(self, name, value):
        path = self.root / name
        data = json.dumps(value).encode()
        path.write_bytes(data)
        return dict(path=str(path), sha256=hashlib.sha256(data).hexdigest())

    def start(self, attempt="eval-a", task=0):
        return self.store.start_evaluation_attempt(self.cohort, self.ids[task], attempt)

    def http(self, name, parent, charge=30):
        return self.store.reserve_subrequest(name, parent_attempt_id=parent,
            payload_sha256=fixtures.sha(name), reserved_cost=dict(gpu_seconds=0,
                api_microdollars=charge, api_requests=1, browser_dispatches=0))

    def settle(self, name, status="succeeded", charge=30):
        return self.store.settle_subrequest(name, status=status,
            actual_cost=dict(gpu_seconds=0, api_microdollars=charge, api_requests=1, browser_dispatches=0),
            response_sha256=fixtures.sha(name) if status=="succeeded" else None,
            reason=None if status=="succeeded" else "transport lost")

    def actor(self,name='actor-a',parent='eval-a',task=0,turn=0,**changes):
        args=dict(parent_attempt_id=parent,task_id=self.ids[task],turn=turn,
            policy_sha256=self.request['policy_sha256'],payload_sha256=fixtures.sha('exact body'))
        args.update(changes)
        return self.store.reserve_evaluation_actor_request(name,**args)

    def test_same_api_and_browser_caps_include_training_evaluation_and_retries(self):
        draw = fixtures.reserve(self.store, policy=self.request["policy_sha256"])["draw_id"]
        self.store.start_attempt(draw, "train-a", reserved_cost=dict(gpu_seconds=0,
            api_microdollars=0, api_requests=0, browser_dispatches=1))
        self.http("train-http", "train-a", 60)
        self.settle("train-http", charge=60)
        self.start()
        self.http("eval-http", "eval-a", 40)
        self.settle("eval-http", "unknown", charge=0)
        self.store.finish_evaluation_attempt("eval-a", status="missing", reason="unknown judge")
        self.start("retry")
        with self.assertRaises(BudgetExceeded): self.http("retry-http", "retry", 1)
        self.store.finish_evaluation_attempt("retry", status="missing", reason="cap exhausted")
        self.start("third", task=1)
        with self.assertRaises(BudgetExceeded): self.start("fourth", task=2)
        state = self.store.snapshot()
        self.assertEqual(state["charged_resources"]["api_microdollars"], 100)
        self.assertEqual(state["charged_resources"]["browser_dispatches"], 4)
        self.assertEqual(len(state["draws"]), 1)
        self.assertEqual(len(state["updates"]["0"]["expected_draws"]), 1)

    def test_crash_reopen_and_valid_failure_cannot_be_reselected(self):
        self.assertTrue(self.start()["created"])
        self.store = CampaignStore(self.root / "store", self.store.manifest)
        self.assertFalse(self.start()["created"])
        with self.assertRaises(StateError): self.start("retry")
        self.http("h", "eval-a")
        result = dict(status="valid", outcome=dict(artifact_sha256=fixtures.sha("verdict"), reward=0))
        with self.assertRaises(StateError): self.store.finish_evaluation_attempt("eval-a", **result)
        self.settle("h")
        self.store.finish_evaluation_attempt("eval-a", **result)
        self.assertFalse(self.store.finish_evaluation_attempt("eval-a", **result)["recorded"])
        with self.assertRaises(StateError): self.start("retry")
        with self.assertRaises(StateError):
            self.store.finish_evaluation_attempt("eval-a", status="valid",
                outcome=dict(artifact_sha256=fixtures.sha("new"), reward=1))
        self.assertEqual(self.store.snapshot()["evaluation_cohorts"][self.cohort]["tasks"][self.ids[0]]["selected_attempt"], "eval-a")

    def test_training_and_evaluation_attempt_identity_cannot_collide(self):
        self.start()
        draw = fixtures.reserve(self.store, policy=self.request["policy_sha256"])["draw_id"]
        with self.assertRaises(StateError):
            self.store.start_attempt(draw, "eval-a", reserved_cost=fixtures.cost(0))
        self.store.start_attempt(draw, "train", reserved_cost=fixtures.cost(0))
        with self.assertRaises(StateError): self.start("train", task=1)

    def test_request_pin_duplicate_and_membership_guards(self):
        self.assertFalse(self.store.register_evaluation_cohort(self.cohort, self.ref)["created"])
        for change in (dict(policy_sha256=fixtures.sha("different")), dict(iteration=10),
                       dict(accounting_manifest_sha256=fixtures.sha("different budget"))):
            request = {k:v for k,v in self.request.items() if k != "cohort_id"}
            request.update(change); request["cohort_id"] = digest(request)
            ref = self.artifact("changed.json", request)
            with self.assertRaises(StateError): self.store.register_evaluation_cohort(request["cohort_id"], ref)
        self.root.joinpath("tasks.json").write_text("[]")
        with self.assertRaises(StateError): self.store.register_evaluation_cohort(self.cohort, self.ref)

    def test_invalid_retry_history_retained_with_no_training_rows(self):
        for index in range(3):
            name = "retry-" + str(index)
            self.start(name)
            self.store.finish_evaluation_attempt(name, status="invalid",
                outcome=dict(artifact_sha256=fixtures.sha(name)), reason="invalid verdict")
        with self.assertRaises(StateError): self.start("too-many")
        state = self.store.snapshot()
        self.assertEqual(len(state["evaluation_attempts"]), 3)
        self.assertEqual(state["attempts"], {})
        self.assertEqual(state["updates"], {})
        self.assertEqual(state["draws"], {})

    def test_actual_http_overrun_persists_and_halts_after_reopen(self):
        self.start()
        self.http('overrun','eval-a',charge=10)
        with self.assertRaises(BudgetExceeded):self.settle('overrun',charge=11)
        self.store=CampaignStore(self.root/'store',self.store.manifest)
        state=self.store.snapshot()
        self.assertEqual(state['charged_resources']['api_microdollars'],11)
        self.assertEqual(state['charged_resources']['api_requests'],1)
        self.assertTrue(state['halt'])
        # Recording a known result remains possible after a halt; new spend does not.
        self.store.finish_evaluation_attempt('eval-a',status='valid',
            outcome=dict(artifact_sha256=fixtures.sha('saved negative verdict'),reward=0))
        with self.assertRaises(StateError):self.start('new-after-overrun',task=1)
        self.assertFalse(self.settle('overrun',charge=11)['recorded'])
        self.assertEqual(self.store.snapshot()['charged_resources']['api_microdollars'],11)

    def test_concurrent_evaluation_attempts_share_last_browser_slot(self):
        for i in range(3):
            self.start('finished-'+str(i),task=i)
            self.store.finish_evaluation_attempt('finished-'+str(i),status='missing',reason='fixture unavailable')
        def admit(index):
            store=CampaignStore(self.root/'store',self.store.manifest)
            try:
                return store.start_evaluation_attempt(self.cohort,self.ids[index],'concurrent-'+str(index))['created']
            except BudgetExceeded:return False
        with ThreadPoolExecutor(max_workers=2) as pool:
            results=list(pool.map(admit,(4,5)))
        self.assertEqual(sorted(results),[False,True])
        self.assertEqual(self.store.snapshot()['charged_resources']['browser_dispatches'],4)

    def test_duplicate_cohort_protocol_and_task_identity_do_not_reset_retry_caps(self):
        self.start('original')
        with self.assertRaises(StateError):self.start('original',task=1)
        changed={k:v for k,v in self.request.items() if k!='cohort_id'}
        changed['protocol']={'different':True};changed['protocol_sha256']=digest(changed['protocol'])
        changed['cohort_id']=digest(changed)
        reference=self.artifact('changed-protocol.json',changed)
        with self.assertRaisesRegex(StateError,'existing cohort'):
            self.store.register_evaluation_cohort(changed['cohort_id'],reference)
        state=self.store.snapshot()
        self.assertEqual(list(state['evaluation_cohorts']),[self.cohort])
        self.assertEqual(list(state['evaluation_attempts']),['original'])

    def test_actor_crash_requires_reconciliation_and_unknown_retry_stays_counted(self):
        self.start();self.assertTrue(self.actor()['created'])
        self.store=CampaignStore(self.root/'store',self.store.manifest)
        self.assertFalse(self.actor()['created'])
        with self.assertRaisesRegex(StateError,'Reconcile'):self.actor('retry')
        with self.assertRaisesRegex(StateError,'actor attempt'):
            self.store.finish_evaluation_attempt('eval-a',status='missing',reason='crashed')
        self.store.settle_evaluation_actor_request('actor-a',status='unknown',reason='lost after dispatch')
        self.assertTrue(self.actor('retry')['created'])
        self.store.settle_evaluation_actor_request('retry',status='received',response_sha256=fixtures.sha('response'))
        self.assertFalse(self.store.settle_evaluation_actor_request('retry',status='received',
            response_sha256=fixtures.sha('response'))['recorded'])
        with self.assertRaisesRegex(StateError,'immutable'):
            self.store.settle_evaluation_actor_request('retry',status='failed',reason='rewrite')
        self.store.finish_evaluation_attempt('eval-a',status='missing',reason='browser stopped')
        self.assertEqual(len(self.store.snapshot()['online_actor_requests']),2)
        with self.assertRaisesRegex(StateError,'Active evaluation parent'):self.actor('after close')

    def test_actor_identity_policy_turn_payload_and_training_namespace_are_bound(self):
        self.start();self.actor()
        for change in (dict(task=1),dict(policy_sha256=fixtures.sha('other policy')),
                       dict(payload_sha256=fixtures.sha('changed body')),dict(turn=1),dict(turn=30),dict(turn=True)):
            with self.subTest(change=change),self.assertRaises(StateError):self.actor(**change)
        # Existing training records share the namespace and are never overwritten.
        with self.store._transaction() as state:
            state['online_actor_requests']['training-key']=dict(attempt_id='train-parent',turn=0,status='failed')
        with self.assertRaises(StateError):self.actor('training-key')
        with self.assertRaises(StateError):
            self.store.settle_evaluation_actor_request('training-key',status='failed',reason='not evaluation')
        self.assertEqual(self.store.snapshot()['online_actor_requests']['training-key']['attempt_id'],'train-parent')

    def test_actor_cap_counts_training_and_concurrent_evaluation_attempts(self):
        with self.store._transaction() as state:
            state['online_actor_requests']={str(i):dict(attempt_id='train',turn=i,status='failed') for i in range(2)}
        self.start('one',task=0);self.start('two',task=1)
        def reserve(index):
            store=CampaignStore(self.root/'store',self.store.manifest)
            try:
                return store.reserve_evaluation_actor_request('actor-'+str(index),parent_attempt_id=('one','two')[index],
                    task_id=self.ids[index],policy_sha256=self.request['policy_sha256'],
                    payload_sha256=fixtures.sha('body'),turn=0)['created']
            except BudgetExceeded:return False
        with ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(reserve,(0,1)))
        self.assertEqual(sorted(results),[False,True])
        self.assertEqual(len(self.store.snapshot()['online_actor_requests']),3)
        # Training's existing NativeCollectionPolicy uses this same len(map) cap.
        from types import SimpleNamespace
        from openwebrl.action_credit_collect import NativeCollectionPolicy,CollectionRejected
        actor=object.__new__(NativeCollectionPolicy);actor.store=self.store;actor.request=SimpleNamespace(attempt_id='new-training')
        with self.assertRaises(CollectionRejected):actor.actor_dispatch(0)

    def test_actor_settlement_survives_halt_but_new_reservation_does_not(self):
        self.start();self.actor()
        with self.store._transaction() as state:state['halt']='external test halt'
        self.store.settle_evaluation_actor_request('actor-a',status='failed',reason='known failure')
        with self.assertRaisesRegex(StateError,'halted'):self.actor('new')
        self.assertEqual(len(self.store.snapshot()['online_actor_requests']),1)


if __name__ == "__main__": unittest.main()
