"""CPU orchestration fixtures; no production browser/API/GPU acceptance."""
import asyncio
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import unittest

from openwebrl.action_credit import digest
from openwebrl.action_credit_binding import q_inventory_contract
from openwebrl.action_credit_collect import ReconcileRequired, physical_verifier_sha256
from openwebrl.action_credit_online import (online_collection_protocol, plan_online_update,
    collect_online_update, materialize_tree_episode, _all)
from openwebrl.action_credit_selection import TreeSelectionPolicy, TreeUpdateSelectionPolicy, TreeSchedulePin
from openwebrl.action_credit_slot_selection import slot_selection_policy
from openwebrl.action_credit_state import CampaignManifest, CampaignStore
import test_action_credit_campaign as campaign_fixture
import test_action_credit_collect as collection_fixture
from test_action_credit_binding import config
from test_action_credit_state import sha, cost


def order_fixture(root, dataset_sha):
    order = dict(source_sha256=dataset_sha, groups_per_iteration=48,
        outcome_dependent_replacement=False, original_manifest_task_order=["task-a"],
        iterations=[dict(iteration=i+1, groups=[dict(group_id=f"iter{i+1:02d}-group{g:02d}",
            task_id="task-a", source_index=0) for g in range(48)]) for i in range(10)])
    path = root/"online-task-order.json"
    path.write_text(json.dumps(order))
    return path, order


def tree_policy(order, order_sha):
    return TreeSelectionPolicy(order_sha, tuple(TreeUpdateSelectionPolicy(i,
        tuple(TreeSchedulePin(g["task_id"],g["group_id"],2,2,2,4,42)
              for g in entry["groups"])) for i,entry in enumerate(order["iterations"])))


class OnlinePlanningTests(unittest.TestCase):
    def setUp(self):
        self.fixture = campaign_fixture.CampaignTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.contract = self.fixture.contract
        self.root = self.fixture.root
        self.data = self.fixture.store.manifest.to_dict()
        self.order_path, self.order = order_fixture(self.root, self.data["science"]["dataset"]["source_sha256"])
        tasks=tuple(f"task-{i:02d}" for i in range(48))
        self.contract=replace(self.contract,train_task_ids=tasks)
        self.data["task_splits"]={**{task:"train" for task in tasks},"task-b":"holdout"}
        self.data["split_sha256"]=digest(self.data["task_splits"])
        self.order["original_manifest_task_order"]=list(tasks)
        for entry in self.order["iterations"]:
            for index,group in enumerate(entry["groups"]):
                group.update(task_id=tasks[index],source_index=index)
        self.order_path.write_text(json.dumps(self.order))
        self.order_sha = hashlib.sha256(self.order_path.read_bytes()).hexdigest()
        self.data["science"]["dataset"]["task_order_sha256"] = self.order_sha
        self.data["science"]["algorithm"]["online_collection_protocol"] = online_collection_protocol(42)

    def store(self, arm):
        data = json.loads(json.dumps(self.data))
        data["arm_id"] = arm
        data["science"]["algorithm"]["objective_config_sha256"] = digest(asdict(config(arm)))
        policy = None
        if arm == "continuation_q":
            data["science"]["algorithm"]["q_inventory_protocol"] = q_inventory_contract()
        if arm.startswith("tree_"):
            policy = tree_policy(self.order, self.order_sha)
            data["science"]["algorithm"]["leaf_selection"] = slot_selection_policy(policy)
        return CampaignStore.create(self.root/arm, CampaignManifest.from_dict(data)), policy

    def test_all_five_preserve_exact_inventory_before_dispatch(self):
        for arm in ("chain_outcome_grpo", "gigpo", "tree_outcome_sampling", "tree_grpo", "continuation_q"):
            store, policy = self.store(arm)
            plan, schedules = plan_online_update(store, self.contract, self.order_path, tree_policy=policy)
            expected = 704 if arm == "continuation_q" else 480 if schedules else 384
            self.assertEqual(len(plan["draws"]), expected)
            self.assertEqual(len(store.snapshot()["updates"]["0"]["expected_draws"]), expected)
            self.assertEqual(store.snapshot()["attempts"], {})
            self.assertEqual(len({x["leaf_id"] for x in plan["draws"]}), expected)
            if arm == "continuation_q":
                self.assertEqual(sum(x["spec"]["inventory_role"]=="base" for x in plan["draws"]),384)
                self.assertEqual(sum(x["spec"]["inventory_role"]=="q_fit" for x in plan["draws"]),240)
                self.assertEqual(sum(x["spec"]["inventory_role"]=="q_holdout" for x in plan["draws"]),80)
            replay, _ = plan_online_update(store,self.contract,self.order_path,tree_policy=policy)
            self.assertEqual(replay,plan)

    def test_changed_task_or_policy_never_resets_plan(self):
        store, _ = self.store("chain_outcome_grpo")
        plan_online_update(store,self.contract,self.order_path)
        changed = self.order_path.with_name("changed.json")
        changed.write_text(self.order_path.read_text()+"\n")
        with self.assertRaisesRegex(ValueError,"artifact mismatch"):
            plan_online_update(store,self.contract,changed)
        with self.assertRaises(ValueError):
            plan_online_update(store,replace(self.contract,policy_sha256=sha("different policy")),self.order_path)

    def test_scientific_counts_not_user_inferred_defaults(self):
        self.data["science"]["algorithm"].pop("online_collection_protocol")
        store, _ = self.store("chain_outcome_grpo")
        with self.assertRaises(ValueError):
            plan_online_update(store,self.contract,self.order_path)

    def test_pending_payload_preserves_exact_q_panels_and_tree_selection(self):
        import torch
        from test_action_credit import graph_fixture
        from test_action_credit_objective import with_q_base
        from test_action_credit_binding import collection_fixture, panel
        from test_action_credit_selection import selection_fixture, graph_receipts
        from openwebrl.action_credit_megatron import BackendRow, FrozenScoreReceipt, tensor_payload_sha256
        from openwebrl.action_credit_campaign import save_scored_pending_update
        qgraph=with_q_base(graph_fixture())
        tree,selection=selection_fixture()
        for graph,arm,panels,selected in ((qgraph,"continuation_q",(panel(),),None),
                                          (tree,"tree_outcome_sampling",(),selection)):
            cfg=config(arm)
            receipts=graph_receipts(graph)
            graph=replace(graph,edges=tuple(replace(e,replay_evidence_sha256=receipts[e.edge_id].sha256) for e in graph.edges))
            def manifest_change(data):
                data["science"]["algorithm"].update(training_world_size=8,
                    ppo_window_config=asdict(self.fixture.window_config))
                data["science"]["checkpoint"].update(native_initial_state_root=str(self.root/"native-initial-state"),
                    native_checkpoint_validator_sha256=sha("synthetic CPU validator; no physical proof"))
            with collection_fixture(graph,cfg,panels,selection=selected,manifest_change=manifest_change) as (store,mapping):
                self.fixture.store,self.fixture.contract=store,graph.contract
                rng=self.fixture.rng_fixture()
                tensors={e.edge_id:({"pixel_values":torch.zeros((1,))} if e.media_sha256 else {}) for e in graph.edges}
                from openwebrl.action_credit_tensor_refs import create_tensor_blob
                # The numerical fixture above is already terminal/sealed. Open
                # only a synthetic owner-write interval and restore its state
                # exactly; these CPU placeholders confer no physical authority.
                for index, edge in enumerate(graph.edges):
                    if not tensors[edge.edge_id]:
                        continue
                    leaf = next(leaf for leaf in graph.leaves if edge.edge_id in leaf.edge_ids)
                    attempt = "attempt-" + leaf.leaf_id
                    status = store.snapshot()["attempts"][attempt]["status"]
                    with store._transaction() as state:
                        state["attempts"][attempt]["status"] = "active"
                    tensors[edge.edge_id] = create_tensor_blob(store, attempt, index, tensors[edge.edge_id])
                    with store._transaction() as state:
                        state["attempts"][attempt]["status"] = status
                rows={e.edge_id:BackendRow(e.edge_id,e.input_ids,e.response_ids,e.sampled_mask,e.media_sha256,
                    graph.contract.processor_sha256,tensor_payload_sha256(tensors[e.edge_id])) for e in graph.edges}
                scores={e.edge_id:FrozenScoreReceipt(e.edge_id,graph.contract.sha256,graph.contract.policy_sha256,
                    e.context_sha256,digest(dict(response_ids=e.response_ids,sampled_mask=e.sampled_mask)),
                    rows[e.edge_id].tensor_payload_sha256,sha("CPU scorer"),.8,e.old_log_probs) for e in graph.edges}
                draft=dict(graph=graph,rows=rows,tensor_payloads=tensors,
                           replay_receipts=receipts,leaf_to_draw=mapping)
                pointer,prepared=save_scored_pending_update(store,cfg,draft,scores,
                    scorer_source_sha256=sha("CPU scorer"),window_config=self.fixture.window_config,
                    optimizer_rng_artifact=rng,panels=panels,selection=selected)
                saved=torch.load(pointer["path"],map_location="cpu",weights_only=False)
                self.assertEqual(saved["panels"],panels)
                self.assertEqual(saved["selection"],selected)
                self.assertTrue(any(t.component=="local" for t in prepared.plan.terms) if panels else
                                saved["selection"].selected_leaf_ids)


class OnlineDispatchTests(unittest.IsolatedAsyncioTestCase):
    async def test_waits_for_all_owned_workers_before_propagating_failure(self):
        completed=[]
        async def bad():
            raise ValueError("failure")
        async def late():
            await asyncio.sleep(.01)
            completed.append(True)
        with self.assertRaisesRegex(ValueError,"failure"):
            await _all(bad(),late())
        self.assertEqual(completed,[True])

    async def test_actual_native_fixture_reload_and_all_missing_slots_preserved(self):
        f=collection_fixture.CollectionTests()
        f.setUp(); self.addCleanup(f.doCleanups)
        data=f.store.manifest.to_dict()
        path, order=order_fixture(Path(f.temp.name),data["science"]["dataset"]["source_sha256"])
        data["science"]["dataset"]["task_order_sha256"]=hashlib.sha256(path.read_bytes()).hexdigest()
        data["science"]["algorithm"]["online_collection_protocol"]=online_collection_protocol(42)
        f.store=CampaignStore.create(Path(f.temp.name)/"online-ledger",CampaignManifest.from_dict(data))
        plan,_=plan_online_update(f.store,f.contract,path)
        # Bulk CPU fixture setup; the public reserve/mark-missing transitions
        # have their own tests. The production driver only reads these slots.
        with f.store._transaction() as state:
            for item in plan["draws"][1:]:
                rid=digest([f.store.manifest.sha256,f.contract.sha256,item["leaf_id"],"episode"])
                identity=f.store._draw_identity(0,f.contract.policy_sha256,item["spec"])
                draw_id=digest(identity)
                state["requests"][rid]=draw_id
                state["draws"][draw_id]=dict(identity,request_id=rid,status="missing",attempts=[],
                    selected_attempt=None,missing_reason="CPU fixture missing-slot evidence")
        fake=collection_fixture.NativeFake()
        base_request=f.request(fake)
        pins={k:v for k,v in asdict(base_request).items() if k in (
            "task_data_sha256","tools_sha256","environment_policy_sha256","browser_config_sha256",
            "browser_manifest_sha256","generator_protocol_sha256")}
        async def dispatch(request,prefix):
            episode=await f.run_episode(fake,request,plan=False,prefix=prefix)
            pointer=f.store.snapshot()["pending_episodes"][request.attempt_id]
            f.store.finish_attempt(request.attempt_id,status="valid",actual_cost=cost(2),
                outcome=dict(artifact_sha256=pointer["sha256"],reward=1.))
            return episode
        draft=await collect_online_update(f.store,f.contract,f.actor,path,dispatch=dispatch,
            request_pins=lambda task:pins,attempt_namespace="cpu-fixture",verifier_sha256=physical_verifier_sha256())
        self.assertEqual(len(draft["graph"].leaves),384)
        self.assertEqual(sum(x.status=="missing" for x in draft["graph"].leaves),383)
        self.assertEqual(len(draft["graph"].edges),1)
        self.assertEqual(fake.created,1)
        # Restart must reload the existing episode, never call the native generator again.
        resolved=[]
        def actor_for_request(request):
            resolved.append(request.leaf_id)
            return f.actor
        replay=await collect_online_update(f.store,f.contract,actor_for_request,path,dispatch=dispatch,
            request_pins=lambda task:pins,attempt_namespace="new-controller",verifier_sha256=physical_verifier_sha256())
        self.assertEqual(draft["graph"],replay["graph"])
        self.assertEqual(len(resolved),384)
        self.assertEqual(fake.created,1)
        episode=next(iter(f.store.snapshot()["pending_episodes"]))
        from openwebrl.action_credit_episode import load_pending_episode
        saved=load_pending_episode(f.store,f.contract,f.actor,episode,verifier_sha256=physical_verifier_sha256())
        nodes,receipts,prefixes=materialize_tree_episode(saved,f.contract,round_index=0)
        self.assertEqual(len(nodes),2)
        self.assertFalse(nodes[0].terminal); self.assertTrue(nodes[-1].terminal)
        self.assertEqual(nodes[-1].remaining_turns,nodes[0].remaining_turns-1)
        self.assertNotEqual(nodes[-1].context_sha256,nodes[0].context_sha256)
        self.assertEqual(len(prefixes[nodes[-1].node_id]),1)
        self.assertEqual(set(receipts),{n.node_id for n in nodes})
