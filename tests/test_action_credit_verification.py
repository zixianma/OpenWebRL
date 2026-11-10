"""CPU-only purpose, approval, crash and all-attempt accounting tests."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict

from openwebrl.action_credit_state import BudgetExceeded, StateError, digest
from openwebrl.action_credit_verification import (
    PURPOSE, SHARED, VERIFICATION, VerificationManifest, VerificationStore,
    approved_authority, open_accounting_store, prepare_verification_manifest,
)
from openwebrl.action_credit_judge import JudgeHTTPProtocol
from openwebrl.action_credit_evaluation import evaluation_protocol, Full300EvaluationOwner


class VerificationTests(unittest.TestCase):
    def setUp(self):
        # Preserve small diagnostic artifacts; this suite has no deletion step.
        self.root = Path(tempfile.mkdtemp(prefix="action-credit-verification-", dir="/tmp"))
        self.serial = 0
        self.eval_ids = [f"eval-{i}" for i in range(300)]
        self.original = self.artifact("original", {"files": []})
        self.source = self.artifact("sources", {"files": {}})
        self.tasks = self.artifact("eval-tasks", self.eval_ids)
        self.config = self.make_manifest()
        self.manifest = VerificationManifest.from_dict(self.config)
        self.store = VerificationStore.create(self.root / "store", self.manifest)
        self.request = dict(schema_version=1, task_ids=self.eval_ids, task_manifest=self.tasks,
            protocol=evaluation_protocol(), protocol_sha256=digest(evaluation_protocol()),
            checkpoint_inventory=self.original, policy_sha256=self.original["sha256"], iteration=0,
            source_inventory=self.source, accounting_manifest_sha256=self.manifest.sha256,
            artifact_schema="native_monitor_full300_v1", diagnostic_only=True)
        self.request["cohort_id"] = digest(self.request)
        self.cohort = self.request["cohort_id"]
        self.request_ref = self.artifact("request", self.request)
        self.store.register_evaluation_cohort(self.cohort, self.request_ref)

    def artifact(self, name, value):
        self.serial += 1
        path = self.root / f"{name}-{self.serial}.json"
        raw = json.dumps(value, sort_keys=True, allow_nan=False).encode()
        path.write_bytes(raw)
        return dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest())

    def make_manifest(self, *, total=300, shared=100, api_count=6, shared_api_count=2,
                      browsers=5, shared_browsers=2, actor_cap=3, approved=True):
        caps = dict(gpu_seconds=0, api_microdollars=total, api_requests=api_count, browser_dispatches=browsers)
        earmark = dict(gpu_seconds=0, api_microdollars=shared, api_requests=shared_api_count, browser_dispatches=shared_browsers)
        proposal = self.artifact("proposal", dict(validation=dict(separately_proposed_online_verification=dict(
            judge_http_attempts_cap=api_count, judge_usd_cap=total / 1_000_000,
            browser_dispatches_cap=browsers, local_actor_requests_cap=actor_cap,
            includes_shared_initial_full300=True, no_execution_until_api_authority_bound=True,
            shared_initial_full300_reserve=dict(tasks=300, judge_http_attempts=shared_api_count,
                                               usd=shared / 1_000_000, browser_dispatches=shared_browsers)))))
        receipt = dict(status="approved", campaign_id="CPU-verification-only", purpose=PURPOSE, model="gpt-4.1",
            budget_caps=caps, shared_initial_caps=earmark, local_actor_requests_cap=actor_cap,
            proposal_reference=proposal, api_microdollars_cap=total, api_requests_cap=api_count,
            user_authorization_text="CPU TEST FIXTURE: NOT AN ACTUAL USER APPROVAL",
            user_authorization_reference="CPU-only-fixture-no-execution")
        ref = self.artifact("cpu-approval", receipt) if approved else None
        protocol = JudgeHTTPProtocol("gpt-4.1", digest(receipt), digest("CPU input upper bound"), 32, 50, 1_000_000, 1_000_000)
        verification = dict(optimizer_updates=0, policy_sha256=self.original["sha256"],
                            browser="local_process", judge="gpt-4.1/action_history", max_turns=30)
        return dict(schema_version=1, purpose=PURPOSE, campaign_id="CPU-verification-only",
            proposal_reference=proposal, original_actor_inventory=self.original, source_inventory=self.source,
            evaluation_task_manifest=self.tasks, excluded_training_task_ids=["train-0", "train-1", "train-2"],
            verification_task_ids=["train-0", "train-1", "train-2"], verification_protocol=verification,
            verification_protocol_sha256=digest(verification), budget_caps=caps, shared_initial_caps=earmark,
            local_actor_requests_cap=actor_cap, shared_initial_max_attempts_per_task=2,
            verification_max_attempts_per_task=3,
            science=dict(judge=dict(model="gpt-4.1", prompt_sha256=digest("prompt"),
                canonical_reward_source_sha256=digest("canonical source"), http_protocol_sha256=protocol.sha256 if approved else None)),
            approval_reference=ref, judge_http_protocol=asdict(protocol) if approved else None)

    def verification(self, attempt="v", task="train-0"):
        return self.store.start_verification_attempt(task, attempt,
            protocol_sha256=self.manifest.to_dict()["verification_protocol_sha256"])

    def evaluation(self, attempt="e", task=0):
        return self.store.start_evaluation_attempt(self.cohort, self.eval_ids[task], attempt)

    def http(self, key, parent):
        return self.store.reserve_subrequest(key, parent_attempt_id=parent, payload_sha256=digest(key),
            reserved_cost=dict(gpu_seconds=0, api_microdollars=50, api_requests=1, browser_dispatches=0))

    def settle_http(self, key, *, status="unknown", amount=0):
        return self.store.settle_subrequest(key, status=status,
            actual_cost=dict(gpu_seconds=0, api_microdollars=amount, api_requests=1, browser_dispatches=0),
            response_sha256=digest(key) if status == "succeeded" else None,
            reason=None if status == "succeeded" else "CPU simulated transport lost")

    def evidence(self, attempt, status, task="train-0"):
        return self.artifact("verification-evidence", dict(attempt_id=attempt, task_id=task,
            policy_sha256=self.original["sha256"], protocol_sha256=self.config["verification_protocol_sha256"],
            manifest_sha256=self.manifest.sha256, status=status, diagnostic_only=True))

    def finish(self, attempt="v", status="unknown", task="train-0"):
        return self.store.finish_verification_attempt(attempt, status=status, evidence=self.evidence(attempt, status, task),
            reason=None if status == "passed" else "CPU diagnostic")

    def actor(self, name, parent, *, verification=True, task=0, turn=0):
        fn = self.store.reserve_verification_actor_request if verification else self.store.reserve_evaluation_actor_request
        return fn(name, parent_attempt_id=parent, task_id=f"train-{task}" if verification else self.eval_ids[task],
            policy_sha256=self.original["sha256"], payload_sha256=digest(name), turn=turn)

    def test_draft_never_creates_directory_or_receipt(self):
        draft = VerificationManifest.from_dict(self.make_manifest(approved=False))
        path = self.root / "blocked"
        with self.assertRaisesRegex(StateError, "pending"):
            VerificationStore.create(path, draft)
        self.assertFalse(path.exists())
        with self.assertRaises(StateError): open_accounting_store(path, draft.to_dict())

    def test_draft_builder_copies_exact_proposal_and_creates_no_ledger(self):
        m = self.config
        draft = prepare_verification_manifest(campaign_id=m['campaign_id'], proposal_reference=m['proposal_reference'],
            original_actor_inventory=m['original_actor_inventory'], source_inventory=m['source_inventory'],
            evaluation_task_manifest=m['evaluation_task_manifest'], excluded_training_task_ids=m['excluded_training_task_ids'],
            verification_task_ids=m['verification_task_ids'], verification_protocol=m['verification_protocol'],
            shared_initial_max_attempts_per_task=2, verification_max_attempts_per_task=3,
            judge_prompt_sha256=m['science']['judge']['prompt_sha256'],
            canonical_reward_source_sha256=m['science']['judge']['canonical_reward_source_sha256'])
        self.assertEqual(draft.to_dict()['budget_caps'], m['budget_caps'])
        self.assertEqual(draft.to_dict()['shared_initial_caps'], m['shared_initial_caps'])
        self.assertIsNone(draft.to_dict()['approval_reference'])
        with self.assertRaises(StateError): approved_authority(draft)

    def test_receipt_purpose_and_exact_limits_cannot_be_borrowed(self):
        for field, bad in (("purpose", "chain_outcome_grpo"), ("campaign_id", "pilot"),
                           ("api_requests_cap", 10000), ("status", "proposed"),
                           ("local_actor_requests_cap", 20000), ("shared_initial_caps", {})):
            m = copy.deepcopy(self.config)
            receipt = json.loads(Path(m["approval_reference"]["path"]).read_text())
            receipt[field] = bad
            m["approval_reference"] = self.artifact("bad-receipt", receipt)
            m["judge_http_protocol"]["approval_receipt_sha256"] = digest(receipt)
            m["science"]["judge"]["http_protocol_sha256"] = JudgeHTTPProtocol(**m["judge_http_protocol"]).sha256
            with self.subTest(field=field), self.assertRaises(StateError):
                approved_authority(VerificationManifest.from_dict(m))

    def test_approval_change_after_creation_fails_closed_before_dispatch(self):
        path = Path(self.config["approval_reference"]["path"])
        path.write_text("{}")
        with self.assertRaises(StateError): self.verification()
        with self.assertRaises(StateError): self.evaluation()
        with self.assertRaises(StateError): open_accounting_store(self.store.root, self.config)
        self.assertEqual(self.store.snapshot()["charged_resources"]["browser_dispatches"], 0)

    def test_manifest_has_no_training_shortcut_or_training_methods(self):
        for name in ("reserve_draw", "start_attempt", "register_update", "finish_attempt"):
            self.assertFalse(hasattr(self.store, name), name)
        for field in ("arm_id", "initialization", "task_splits"):
            self.assertNotIn(field, self.config)
        with self.assertRaises(StateError):
            VerificationManifest.from_dict(dict(self.config, arm_id="chain_outcome_grpo"))
        with self.assertRaises(StateError):
            open_accounting_store(self.store.root, dict(self.config, purpose="training"))

    def test_shared_initial_and_verification_caps_are_not_transferable(self):
        self.verification(); self.evaluation()
        for index in range(4):
            self.http(f"v{index}", "v"); self.settle_http(f"v{index}")
        with self.assertRaises(BudgetExceeded): self.http("v-extra", "v")
        for index in range(2):
            self.http(f"e{index}", "e"); self.settle_http(f"e{index}")
        with self.assertRaises(BudgetExceeded): self.http("e-extra", "e")
        state = self.store.snapshot()
        self.assertEqual(state["charged_resources"], dict(gpu_seconds=0, api_microdollars=300, api_requests=6, browser_dispatches=2))
        self.assertEqual(self.store._purpose_totals(state, VERIFICATION)["api_microdollars"], 200)
        self.assertEqual(self.store._purpose_totals(state, SHARED)["api_microdollars"], 100)

    def test_unknown_http_retains_reservation_across_reopen_and_retry(self):
        self.verification(); self.http("h", "v")
        self.store = open_accounting_store(self.store.root, self.config)
        self.assertFalse(self.http("h", "v")["created"])
        with self.assertRaises(StateError): self.finish()
        self.settle_http("h"); self.finish()
        self.verification("retry"); self.http("retry-http", "retry")
        self.assertEqual(self.store.snapshot()["charged_resources"]["api_microdollars"], 100)

    def test_actual_overrun_is_committed_and_halts_even_after_reopen(self):
        self.verification(); self.http("h", "v")
        with self.assertRaises(BudgetExceeded): self.settle_http("h", status="succeeded", amount=51)
        self.store = open_accounting_store(self.store.root, self.config)
        self.assertEqual(self.store.snapshot()["charged_resources"]["api_microdollars"], 51)
        self.assertTrue(self.store.snapshot()["halt"])
        with self.assertRaises(StateError): self.evaluation()

    def test_shared_scope_actual_overrun_halts_below_global_total(self):
        self.evaluation(); self.http('h1', 'e'); self.http('h2', 'e')
        self.settle_http('h1')
        with self.assertRaises(BudgetExceeded): self.settle_http('h2', status='succeeded', amount=51)
        self.store = open_accounting_store(self.store.root, self.config)
        state = self.store.snapshot()
        self.assertEqual(state['charged_resources']['api_microdollars'], 101)
        self.assertLess(101, self.config['budget_caps']['api_microdollars'])
        self.assertTrue(state['halt'])
        with self.assertRaises(StateError): self.store.validate_accounting_snapshot(state)

    def test_final_scope_check_rejects_bad_evidence_even_if_total_fits(self):
        self.evaluation(); self.http('h', 'e')
        state = self.store.snapshot()
        state['subrequests']['h']['charged']['api_microdollars'] = 101
        self.assertLess(self.store._totals(state)['api_microdollars'], self.config['budget_caps']['api_microdollars'])
        with self.assertRaises(BudgetExceeded): self.store.validate_accounting_snapshot(state)

    def test_retry_and_idempotency_preserve_every_browser_charge(self):
        self.verification()
        self.assertFalse(self.verification()["created"])
        with self.assertRaises(StateError): self.verification("active-overlap")
        self.finish()
        self.verification("retry"); self.finish("retry")
        self.verification("third"); self.finish("third")
        with self.assertRaises(StateError): self.verification("fourth")
        self.assertEqual(self.store.snapshot()["charged_resources"]["browser_dispatches"], 3)

    def test_shared_browser_earmark_and_attempt_collision_are_enforced(self):
        self.verification()
        with self.assertRaises(StateError): self.evaluation("v")
        self.evaluation()
        with self.assertRaises(StateError): self.verification("e", task="train-1")
        self.evaluation("e2", task=1)
        with self.assertRaises(BudgetExceeded): self.evaluation("e3", task=2)
        self.verification("v2", task="train-1")
        self.assertEqual(self.store.snapshot()["charged_resources"]["browser_dispatches"], 4)

    def test_actor_cap_counts_shared_initial_and_verification_unknowns(self):
        self.verification(); self.evaluation()
        self.actor("va", "v")
        with self.assertRaises(StateError): self.finish()
        self.store.settle_verification_actor_request("va", status="unknown", reason="CPU lost")
        self.actor("ea", "e", verification=False)
        self.store.settle_evaluation_actor_request("ea", status="unknown", reason="CPU lost")
        self.actor("vb", "v", turn=1)
        self.store.settle_verification_actor_request("vb", status="received", response_sha256=digest("response"))
        self.store = open_accounting_store(self.store.root, self.config)
        with self.assertRaises(BudgetExceeded): self.actor("eb", "e", verification=False)
        self.assertEqual(len(self.store.snapshot()["online_actor_requests"]), 3)

    def test_actor_identity_turn_and_concurrent_parent_guard(self):
        self.verification(); self.evaluation()
        self.actor("same", "v")
        self.assertFalse(self.actor("same", "v")["created"])
        with self.assertRaises(StateError): self.actor("same", "e", verification=False)
        with self.assertRaises(StateError): self.actor("next", "v")
        with self.assertRaises(StateError): self.actor("too-late", "v", turn=30)
        with self.assertRaises(StateError):
            self.store.settle_evaluation_actor_request("same", status="unknown", reason="wrong purpose")

    def test_frozen_task_protocol_and_evidence_identity(self):
        with self.assertRaises(StateError): self.verification(task="unlisted")
        with self.assertRaises(StateError):
            self.store.start_verification_attempt("train-0", "v", protocol_sha256=digest("changed"))
        self.verification()
        with self.assertRaises(StateError):
            self.store.finish_verification_attempt("v", status="passed", evidence=self.evidence("different", "passed"))
        evidence = self.evidence("v", "passed")
        self.store.finish_verification_attempt("v", status="passed", evidence=evidence)
        self.assertFalse(self.store.finish_verification_attempt("v", status="passed", evidence=evidence)["recorded"])
        with self.assertRaises(StateError): self.verification("repeat-passed")

    def test_shared_cohort_cannot_charge_iteration10_or_other_source(self):
        for changes in (dict(iteration=10), dict(source_inventory=self.artifact("other-source", {})),
                        dict(protocol={"unapproved": True})):
            request = {k: v for k, v in self.request.items() if k != "cohort_id"}
            request.update(changes); request["protocol_sha256"] = digest(request["protocol"])
            request["cohort_id"] = digest(request)
            with self.assertRaises(StateError):
                self.store.register_evaluation_cohort(request["cohort_id"], self.artifact("changed-request", request))

    def test_concurrent_reservations_cannot_overcommit_shared_cap(self):
        self.evaluation()
        def reserve(index):
            try:
                self.http(f"http-{index}", "e")
                return True
            except BudgetExceeded:
                return False
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(reserve, range(8)))
        self.assertEqual(sum(results), 2)
        self.assertEqual(self.store.snapshot()["charged_resources"]["api_requests"], 2)

    def test_changed_caps_and_missing_state_never_reset_usage(self):
        self.verification()
        changed = copy.deepcopy(self.config); changed["local_actor_requests_cap"] += 1
        with self.assertRaises(StateError): VerificationStore(self.store.root, VerificationManifest.from_dict(changed))
        with self.assertRaises(StateError): VerificationStore(self.root / "missing", self.manifest)
        with self.assertRaises(FileExistsError): VerificationStore.create(self.store.root, self.manifest)
        self.assertEqual(self.store.snapshot()["charged_resources"]["browser_dispatches"], 1)

    def test_symlink_and_forbidden_destination_fail_before_writes(self):
        link = self.root / "alias"; link.symlink_to(self.store.root, target_is_directory=True)
        with self.assertRaises(StateError): VerificationStore(link, self.manifest)
        with self.assertRaises(StateError): VerificationStore.create("/gpfs/home/zixianma/forbidden-cpu-test", self.manifest)


if __name__ == "__main__":
    unittest.main()
