"""Real ledger/pending-file integration with an explicitly CPU-only fake native backend.

These tests execute CPU Adam and file restore, never establish GPU acceptance
and never commit a production optimizer transaction.
"""
from dataclasses import asdict
import copy
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import torch

from openwebrl.action_credit import CollectionContract,digest
from openwebrl.action_credit_binding import token_mask_contract,shared_prefix_contract
from openwebrl.action_credit_campaign import plan_independent_update,draft_independent_update,save_scored_pending_update
from openwebrl.action_credit_native_controller import (NativeUpdateController,validator_source_sha256,
    _write,_json,_trusted_torch,load_scored_pending)
from openwebrl.action_credit_native_gate import state_digest
from openwebrl.action_credit_state import CampaignManifest,CampaignStore
from openwebrl.action_credit_windows import WindowConfig,native_train_collection
import test_action_credit_campaign as campaign_tests
from test_action_credit_state import manifest_data,sha
from test_action_credit_binding import config


class FakeRuntime:
    """One CPU process emulating eight rank records; NOT distributed execution."""
    rank,world_size,tp,dp=0,8,2,4
    verification_kind="cpu_mock"
    def __init__(self):
        self.parameter=torch.nn.Parameter(torch.tensor([-1.5,-1.7],dtype=torch.float64))
        self.optimizer=torch.optim.Adam([self.parameter],lr=.01)
        self.steps=0;self.rng_marker=3;self.train_calls=0;self.backups=0;self.fail_before=None
    def leader(self,fn):return fn()
    def gather(self,value):
        if "path" in value:
            records=[value]
            contents=_trusted_torch(value)
            for rank in range(1,8):
                ref=_write(Path(value["path"]).with_name(f"rank-{rank}.pt"),dict(contents,rank=rank),torch_payload=True)
                records.append(dict(ref,rank=rank))
            return records
        return [dict(value,rank=rank) for rank in range(8)]
    def snapshot(self):
        return dict(rank=0,model=state_digest(self.parameter),optimizer=state_digest(self.optimizer.state_dict()),
                    scheduler=digest(self.steps),rng=digest(self.rng_marker))
    def rng(self):return dict(marker=self.rng_marker)
    def restore_rng(self,value):self.rng_marker=value["marker"]
    def initialize_original(self,path):
        if self.steps or self.optimizer.state:raise ValueError("fake backend is not fresh")
    def save(self,directory,iteration):
        _write(Path(directory)/"fake-native.pt",dict(iteration=iteration,parameter=self.parameter.detach().clone(),
            optimizer=self.optimizer.state_dict(),steps=self.steps,rng=self.rng_marker),torch_payload=True)
    def restore(self,directory,iteration):
        value=torch.load(Path(directory)/"fake-native.pt",map_location="cpu",weights_only=False)
        if value["iteration"]!=iteration:raise ValueError("fake native iteration mismatch")
        with torch.no_grad():self.parameter.copy_(value["parameter"])
        self.optimizer.load_state_dict(value["optimizer"]);self.steps=value["steps"];self.rng_marker=value["rng"]
    def backup_actor(self):self.backups+=1
    def train(self,prepared,schedule,rows,tensors,check_deadline):
        self.train_calls+=1
        def before(window):
            check_deadline()
            if window.plan.ordinal==self.fail_before:raise RuntimeError("injected CPU mock crash")
        def native_window(actor,window,rows,tensors):
            self.optimizer.zero_grad()
            values={e:self.parameter+(0. if "chain00" in e else -.2) for e in {t.edge_id for t in window.plan.terms}}
            window.loss(values).backward();self.optimizer.step();self.steps+=schedule.config.global_batch_size
            self.rng_marker+=1
            return dict(window.accounting,durable_checkpoint_committed=False)
        with patch("openwebrl.action_credit_windows.native_train_window",side_effect=native_window):
            return native_train_collection(None,prepared,schedule,rows,tensors,before_window=before)


class NativeControllerTests(unittest.TestCase):
    write_artifact=campaign_tests.CampaignTests.write_artifact
    complete=campaign_tests.CampaignTests.complete
    scores=campaign_tests.CampaignTests.scores

    def setUp(self):
        temporary=tempfile.TemporaryDirectory(prefix="action-credit-native-cpu-",dir="/tmp")
        self.addCleanup(temporary.cleanup);self.root=Path(temporary.name)
        model_root=self.root/"model"
        source=_write(model_root/"synthetic-model.json",dict(cpu_fake_model=True))
        original=_write(self.root/"original.json",dict(model_root=str(model_root),files=[source]))
        self.config=config("chain_outcome_grpo");self.window_config=WindowConfig(2,256,42)
        self.contract=CollectionContract(0,original["sha256"],sha("tokenizer"),sha("processor"),
            sha("tasks"),sha("judge"),sha("generation"),.8,("task-a",),("task-b",))
        order=dict(source_sha256=sha("dataset"),groups_per_iteration=1,outcome_dependent_replacement=False,
            original_manifest_task_order=["task-a"],iterations=[dict(iteration=1,
                groups=[dict(group_id="iter01-group00",task_id="task-a",source_index=0)])])
        order_ref=_write(self.root/"order.json",order);self.order=Path(order_ref["path"])
        data=manifest_data();data["initialization"]["original_actor_sha256"]=original["sha256"]
        data["science"]["dataset"]["task_order_sha256"]=order_ref["sha256"]
        algorithm=data["science"]["algorithm"]
        algorithm.update(groups_per_iteration=1,chains_per_group=2,objective_config_sha256=digest(asdict(self.config)),
            leaf_selection={"mode":"all_fit_inventory"},ppo_window_config=asdict(self.window_config),training_world_size=8)
        algorithm.update({k:getattr(self.contract,k) for k in ("tokenizer_sha256","processor_sha256", "task_manifest_sha256",
            "judge_protocol_sha256","generation_sha256","likelihood_temperature","likelihood_convention")})
        for section,value in (("token_mask",token_mask_contract()),("shared_prefix_normalization",shared_prefix_contract("response_mean"))):
            data["science"][section]=dict(contract=value,protocol_sha256=digest(value))
        data["science"]["checkpoint"].update(original_actor_path=str(model_root),original_actor_inventory=original,
            native_initial_state_root=str(self.root/"initial"),native_checkpoint_validator_sha256=validator_source_sha256())
        self.store=CampaignStore.create(self.root/"ledger",CampaignManifest.from_dict(data))
        self.plan=plan_independent_update(self.store,self.contract,self.order)
        self.runtime=FakeRuntime()
        evidence=dict(schema_version=1,verification_kind="cpu_mock",passed=True,
            manifest_sha256=self.store.manifest.sha256,validator_source_sha256=validator_source_sha256(),
            topology=dict(world_size=8,tp=2,dp=4),cpu_fixture_not_physical_acceptance=True)
        evidence.update({k:True for k in ("actual_model","full_response_masks","tp_dp_gradient_parity","ppo2_windows",
            "native_checkpoint_restore","per_rank_rng_restore","native_controller_integration")})
        self.evidence=_write(self.root/"cpu-native-acceptance.json",evidence)
        self.controller=NativeUpdateController(self.store,self.runtime,self.evidence,diagnostic_only=True)

    def pending(self):
        initial=self.controller.initialize(self.contract)
        self.rng_artifact=self.controller.restore_and_capture_rng(self.contract,initial,capture_id="capture0")
        episodes=[self.complete(item,float(i==0)) for i,item in enumerate(self.plan["draws"])]
        draft=draft_independent_update(self.store,self.contract,self.plan,episodes)
        pointer,prepared=save_scored_pending_update(self.store,self.config,draft,self.scores(draft),
            scorer_source_sha256=sha("CPU scorer"),window_config=self.window_config,optimizer_rng_artifact=self.rng_artifact)
        accepted=dict(schema_version=1,verification_kind="cpu_mock",passed=True,
            manifest_sha256=self.store.manifest.sha256,policy_sha256=self.contract.policy_sha256,update_id=0,
            pending_file_sha256=pointer["file_sha256"],binding_sha256=pointer["binding_sha256"],
            ppo_schedule_sha256=pointer["ppo_schedule_sha256"],validator_source_sha256=validator_source_sha256())
        accepted.update({k:True for k in ("actor_weight_transfer_verified","saved_episode_artifacts_verified","browser_replay_verified",
            "sample_masks_verified","terminal_judge_artifacts_verified","all_attempt_costs_reconciled")})
        self.online=_write(self.root/"cpu-online-acceptance.json",accepted)
        return pointer,prepared

    def test_private_tensors_use_mmap_without_eager_legacy_fallback(self):
        reference=_write(self.root/"mmap.pt",dict(value=torch.arange(32.)),torch_payload=True)
        with patch("torch.load",wraps=torch.load) as loaded:
            result=_trusted_torch(reference)
        self.assertTrue(loaded.call_args.kwargs["mmap"])
        result["value"][0]=99
        self.assertEqual(float(_trusted_torch(reference)["value"][0]),0.)
        legacy=self.root/"legacy.pt"
        torch.save(dict(value=torch.arange(32.)),legacy,_use_new_zipfile_serialization=False)
        import hashlib
        with self.assertRaises(RuntimeError):
            _trusted_torch(dict(path=str(legacy),sha256=hashlib.sha256(legacy.read_bytes()).hexdigest()))

    def test_complete_cpu_mock_roundtrip_preserves_pending_and_never_commits(self):
        pointer,prepared=self.pending()
        self.assertEqual(load_scored_pending(self.store,0)[0],pointer)
        result=self.controller.execute_pending(0,"cpu-execution1",self.online,deadline_unix=time.time()+60)
        self.assertFalse(result["committed"])
        self.assertTrue(result["diagnostic_only"])
        self.assertEqual(result["receipt"]["verification_kind"],"cpu_mock")
        self.assertEqual(self.runtime.train_calls,1)
        self.assertEqual(self.runtime.steps,512)
        self.assertEqual(self.store.snapshot()["updates"]["0"]["phase"],"training_pending")
        self.assertIsNone(self.store.snapshot()["updates"]["0"]["training_receipt"])
        self.assertEqual(load_scored_pending(self.store,0)[0],pointer)
        review=_json(result["review"])
        inventory=_json(review["checkpoint_inventory"])
        self.assertEqual(inventory["completed_updates"],1)
        self.assertEqual(inventory["native_iteration"],1)
        self.assertEqual(len(inventory["rank_state_sha256"]),8)
        before=self.runtime.snapshot()
        recovered=self.controller.execute_pending(0,"cpu-execution1",self.online,deadline_unix=time.time()+60)
        self.assertEqual(result,recovered)
        self.assertEqual(self.runtime.snapshot(),before)
        self.assertEqual(self.runtime.train_calls,1)

    def test_partial_update_retries_from_prior_checkpoint_keeps_attempts(self):
        pointer,_=self.pending()
        self.runtime.fail_before=1
        with self.assertRaisesRegex(RuntimeError,"injected"):
            self.controller.execute_pending(0,"crashed",self.online,deadline_unix=time.time()+60)
        self.assertEqual(self.runtime.steps,256)
        self.runtime.fail_before=None
        result=self.controller.execute_pending(0,"retry",self.online,deadline_unix=time.time()+60)
        self.assertEqual(self.runtime.steps,512) # Not 768: partial optimizer was discarded by physical restore.
        self.assertEqual(self.runtime.train_calls,2)
        root=self.store.root/"native_updates"/"0"
        self.assertTrue((root/digest("crashed")/"execution.json").is_file())
        self.assertTrue((root/digest("retry")/"execution.json").is_file())
        self.assertEqual(load_scored_pending(self.store,0)[0],pointer)
        self.assertFalse(result["committed"])

    def test_missing_or_changed_physical_collection_proof_fails_before_optimizer(self):
        self.pending()
        for change in (dict(browser_replay_verified=False),dict(policy_sha256=sha("wrong")),
                       dict(pending_file_sha256=sha("changed")),dict(verification_kind="production_runtime")):
            changed=dict(_json(self.online),**change)
            ref=_write(self.root/(digest(change)+".json"),changed)
            with self.assertRaises(ValueError):
                self.controller.execute_pending(0,"rejected",ref,deadline_unix=time.time()+60)
        self.assertEqual(self.runtime.train_calls,0)
        self.assertEqual(self.store.snapshot()["updates"]["0"]["phase"],"sealed")

    def test_cpu_runtime_cannot_enter_production_or_commit_mock_receipt(self):
        with self.assertRaisesRegex(ValueError,"CPU fake"):
            NativeUpdateController(self.store,self.runtime,self.evidence)
        self.pending()
        result=self.controller.execute_pending(0,"mock",self.online,deadline_unix=time.time()+60)
        intent=self.store.snapshot()["updates"]["0"]["training_intent"]
        with self.assertRaisesRegex(ValueError,"CPU checks"):
            self.store.complete_update(0,attempt_id=intent["attempt_id"],receipt=result["receipt"])

    def test_initial_state_reuse_restores_all_states_and_refuses_partial_root(self):
        reference=self.controller.initialize(self.contract)
        self.runtime.parameter.data.add_(9);self.runtime.steps=900;self.runtime.rng_marker=500
        self.assertEqual(self.controller.initialize(self.contract),reference)
        self.assertEqual(self.runtime.steps,0);self.assertEqual(self.runtime.rng_marker,3)
        self.assertTrue(torch.equal(self.runtime.parameter,torch.tensor([-1.5,-1.7],dtype=torch.float64)))

    def test_tampered_rng_artifact_and_expired_deadline_fail_closed(self):
        self.pending()
        with self.assertRaises(TimeoutError):
            self.controller.execute_pending(0,"late",self.online,deadline_unix=time.time()-1)
        rank=_json(self.rng_artifact)["rank_rng_files"][0]
        with Path(rank["path"]).open("ab") as handle:handle.write(b"tamper")
        with self.assertRaisesRegex(ValueError,"SHA mismatch"):
            self.controller.execute_pending(0,"bad-rng",self.online,deadline_unix=time.time()+60)
        self.assertEqual(self.runtime.train_calls,0)

    def test_diagnostic_bootstrap_does_not_require_its_own_future_acceptance(self):
        native=dict(_json(self.evidence),native_controller_integration=False)
        reference=_write(self.root/"native-bootstrap.json",native)
        self.controller=NativeUpdateController(self.store,self.runtime,reference,diagnostic_only=True)
        self.pending()
        fixture=dict(_json(self.online),synthetic_collection_provenance=True,saved_fixture_artifact_verified=True,
            browser_replay_verified=False,actor_weight_transfer_verified=False,terminal_judge_artifacts_verified=False)
        fixture_ref=_write(self.root/"fixture-bootstrap.json",fixture)
        result=self.controller.execute_pending(0,"fixture",fixture_ref,deadline_unix=time.time()+60)
        self.assertFalse(result["committed"])
        self.assertTrue(_json(result["review"])["diagnostic_only"])
        self.assertTrue(_json(_json(result["review"])["checkpoint_inventory"])["diagnostic_only"])

    def test_foreign_inventory_root_and_concurrent_native_writer_rejected(self):
        self.pending()
        sidecar=_json(self.rng_artifact)
        original=_json(sidecar["previous_checkpoint_inventory"])
        changed=dict(original,native_checkpoint_root="/tmp/outside-this-inventory")
        reference=_write(self.root/"wrong-inventory.json",changed)
        with self.assertRaisesRegex(ValueError,"escaped"):
            self.controller._restore_inventory(reference)
        import fcntl
        with (self.store.root/"native-update.lock").open("a+b") as lock:
            fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            with self.assertRaises(BlockingIOError):
                self.controller.execute_pending(0,"concurrent",self.online,deadline_unix=time.time()+60)
        self.assertEqual(self.runtime.train_calls,0)


if __name__=="__main__":unittest.main()
