"""CPU orchestration integration; fake Ray/native/browser endpoints, no paid work.

Controller math/physical restore have separate tests and acceptance. These tests
exercise all-eight RPC ownership, phase ordering, durable draft/restart behavior
and fail-closed admission, not actual distributed execution or browser truth.
"""
import asyncio
from dataclasses import asdict, replace
import json
from pathlib import Path
from types import SimpleNamespace
import time
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, patch

import torch

from openwebrl.action_credit import digest
from openwebrl.action_credit_driver import DriverLaunch, PilotDriver, _owned, _call_all, storage_boundary, storage_guarded
from openwebrl.action_credit_native_controller import _write, _json, _trusted_torch
import test_action_credit_native_controller as native_fixture


class Remote:
    def __init__(self, call): self.call = call
    def remote(self, *args): return self.call(*args)


class Collective:
    """Eight awaited coroutines share one CPU fixture-side effect per RPC."""
    def __init__(self, owner):
        self.owner, self.rounds = owner, {}
        self.handles = tuple(SimpleNamespace(**{name: Remote(lambda *args, rank=rank, name=name:
            self.call(rank, name, args)) for name in ("action_credit_operation", "action_credit_driver_operation", "update_weights")})
            for rank in range(8))
    async def call(self, rank, method, args):
        key = (method, digest(args))
        value = self.rounds.setdefault(key, dict(ranks=set(), future=asyncio.get_running_loop().create_future()))
        value["ranks"].add(rank)
        self.owner.rpc_seen.append((method, rank))
        if len(value["ranks"]) == 8:
            try: result = self.owner.operation(method, args)
            except BaseException as error: value["future"].set_exception(error)
            else: value["future"].set_result(result)
            del self.rounds[key]
        return await asyncio.shield(value["future"])


class DriverTests(IsolatedAsyncioTestCase):
    def setUp(self):
        self.fixture = native_fixture.NativeControllerTests()
        self.fixture.setUp(); self.addCleanup(self.fixture.doCleanups)
        self.root, self.store = self.fixture.root, self.fixture.store
        manifest = _write(self.root/"driver-manifest.json", self.store.manifest.to_dict())
        self.launch = DriverLaunch(1, manifest, str(self.store.root),
            dict(path=str(self.fixture.order), sha256=self.store.manifest.to_dict()["science"]["dataset"]["task_order_sha256"]),
            asdict(self.fixture.contract), asdict(self.fixture.config), self.fixture.evidence,
            "CPU-driver-execution", time.time()+300, None)
        self.args = SimpleNamespace(offload_train=False, offload_rollout=False, debug_train_only=False,
                                    debug_rollout_only=False, use_critic=False)
        self.trace, self.rpc_seen, self.pools = [], [], []
        self.storage_calls = []
        self.version, self.collect_calls, self.fail_score = "0", 0, False
        self.collective = Collective(self)
        async def url1(): return "http://127.0.0.1:11001"
        async def url2(): return "http://127.0.0.1:11002"
        async def url3(): return "http://127.0.0.1:11003"
        async def url4(): return "http://127.0.0.1:11004"
        engines = [SimpleNamespace(get_url=Remote(fn)) for fn in (url2,url1,url3,url4)]
        async def engines_rpc(): return engines, None, 0, [2]*4, [0,2,4,6]
        self.manager = SimpleNamespace(get_updatable_engines_and_lock=Remote(engines_rpc))
        for target, value in (("_real_handlers", lambda model:self.collective.handles),
                              ("served_weight_version", self.served), ("collect_online_update", self.collect)):
            mock = patch("openwebrl.action_credit_driver."+target, value)
            mock.start(); self.addCleanup(mock.stop)

    async def served(self, url): return self.version

    def operation(self, method, args):
        command = args[0] if args else {}
        name = command.get("operation", "transfer")
        self.trace.append(name)
        if name == "initialize":
            path = self.root/"driver-fake-initial.json"
            return dict(path=str(path), sha256=digest("init"))
        if name == "restore_capture":
            return _write(self.root/(digest(command["capture_id"])+"-rng.json"), {"cpu_fake_rng":True})
        if name == "restore_for_transfer":
            self.version = str(command["contract"]["update_id"]+1)
            return {"restored":True}
        if name == "transfer": return None
        if name == "score_save":
            if self.fail_score:
                self.fail_score = False
                raise RuntimeError("CPU injected scorer failure")
            payload = _trusted_torch(command["draft_artifact"])
            contract = payload["draft"]["graph"].contract
            pointer = dict(file_sha256=command["draft_artifact"]["sha256"], update_id=contract.update_id,
                           policy_sha256=contract.policy_sha256)
            with self.store._transaction() as state:
                state.setdefault("pending_batches", {})[str(contract.update_id)] = pointer
                state["updates"][str(contract.update_id)] = dict(phase="sealed")
            return pointer
        if name == "update":
            update = command["update_id"]
            inventory = _write(self.root/f"fake-completed-{update}.json", {"cpu_fake_completed":update+1})
            review = _write(self.root/f"fake-review-{update}.json", {"cpu_fake_review":True})
            receipt = dict(new_checkpoint_sha256=inventory["sha256"], new_policy_sha256=digest(["CPU policy",update+1]))
            with self.store._transaction() as state:
                state["updates"][str(update)] = dict(phase="consumed",training_receipt=receipt)
                state.setdefault("native_completed_updates",{})[str(update)] = dict(checkpoint_inventory=inventory,review=review)
            return dict(committed=True,receipt=receipt,review=review)
        raise AssertionError(name)

    def pool(self, store, contract, resolve, endpoint):
        owner = self
        class Pool:
            concurrency = 64
            async def collect(self, request, prefix, actor):
                owner.trace.append("browser")
                self.url = endpoint(actor)
                return None
            async def resume_pending(self, episode, actor): raise AssertionError("unused")
            async def close(self):
                owner.trace.append("close_pool")
                self.closed = True
        pool = Pool(); self.pools.append(pool)
        return pool

    async def collect(self, store, contract, actor, task_order_path, *, dispatch, **kwargs):
        self.collect_calls += 1
        request = SimpleNamespace(request_id=digest(["CPU request",contract.update_id]))
        await dispatch(request, ())
        from openwebrl.action_credit import Leaf, ResponseGraph
        from openwebrl.action_credit_online import _persist
        draws=self.fixture.plan['draws']
        result=store.plan_update(contract.update_id,policy_sha256=contract.policy_sha256,draws=[d['spec'] for d in draws])
        plan=dict(update_id=contract.update_id,policy_sha256=contract.policy_sha256,task_order_sha256=self.launch.task_order['sha256'],graph_contract_sha256=contract.sha256,plan_sha256=result['plan_sha256'],draws=draws)
        _persist(store,'online_collection_plans',str(contract.update_id),plan)
        leaves=[];mapping={}
        with store._transaction() as state:
            for item in draws:
                spec=item['spec'];identity=store._draw_identity(contract.update_id,contract.policy_sha256,spec);draw_id=digest(identity)
                state['draws'][draw_id]=dict(identity,request_id=digest([draw_id,'CPU']),status='missing',attempts=[],selected_attempt=None,missing_reason='CPU driver fixture, no browser provenance')
                leaves.append(Leaf(item['leaf_id'],spec['task_id'],spec['tree_id'],(),'missing',None,spec['draw_role'],spec['group_id']));mapping[item['leaf_id']]=draw_id
        return dict(graph=ResponseGraph(contract,(),tuple(leaves),tuple(mapping)),rows={},tensor_payloads={},replay_receipts={},leaf_to_draw=mapping,panels=(),selection_builder_data=None)

    async def acceptance(self, store, contract, pointer):
        self.trace.append("physical_acceptance")
        return _write(self.root/f"cpu-acceptance-{contract.update_id}.json",dict(cpu_fake_not_physical=True))

    def storage(self, store, stage):
        self.storage_calls.append(stage)
        return _write(self.root/f'cpu-storage-{len(self.storage_calls)}.json',dict(
            cpu_fixture_not_physical=True,passed=True,manifest_sha256=store.manifest.sha256,stage=stage,
            reservation=dict(manifest_sha256=store.manifest.sha256,store_root=str(store.root),epoch=time.time())))

    def boundaries(self):
        return sorted((json.loads(path.read_text()) for path in
            (self.store.root/'storage_boundaries').glob('*/*.json')),key=lambda value:value['started_unix'])

    def driver(self, **changes):
        kwargs = dict(worker_pool_factory=self.pool, request_pins=lambda _: {}, collection_acceptance=self.acceptance,
                      storage_admission=self.storage)
        kwargs.update(changes)
        return PilotDriver(self.args, object(), self.manager, self.launch, **kwargs)

    async def test_ten_updates_own_all_rpcs_close_workers_and_commit_before_next_policy(self):
        result = await self.driver().run()
        self.assertTrue(result["training_complete"])
        self.assertFalse(result["evaluation_complete_verified_by_driver"])
        self.assertEqual(self.collect_calls,10)
        self.assertTrue(all(p.closed for p in self.pools))
        self.assertEqual(len(self.rpc_seen),8*(1+10*5))
        for index, item in enumerate(self.trace):
            if item == "score_save": self.assertEqual(self.trace[index-1],"close_pool")
            if item == "update": self.assertEqual(self.trace[index-1],"physical_acceptance")
        self.assertEqual(len(self.store.snapshot()["driver_drafts"]),10)
        phases=[(b['phase'],b['position']) for b in self.boundaries() if b['stage']=='optimizer-update-0']
        self.assertEqual(phases,[(phase,position) for phase in
            ('native-initialize','restore-rng','collection','draft','score-save','native-update-checkpoint')
            for position in ('before','after')])
        self.assertTrue(all(b['hard_byte_cap_enforced'] is False and b['additional_storage_reserved_bytes']==0
                            for b in self.boundaries()))

    async def test_required_storage_hook_prevents_native_or_browser_work(self):
        with self.assertRaisesRegex(ValueError,'storage-admission'):
            self.driver(storage_admission=None)
        self.assertEqual(self.rpc_seen,[])
        self.assertEqual(self.collect_calls,0)

    async def test_native_post_commit_refusal_retains_commit_and_resume_checks_reuse(self):
        armed=True
        def provider(store,stage):
            nonlocal armed
            if armed and store.snapshot()['updates'].get('0',{}).get('phase')=='consumed':
                armed=False
                raise ValueError('CPU post-commit storage refusal')
            return self.storage(store,stage)
        with self.assertRaisesRegex(ValueError,'post-commit storage refusal'):
            await self.driver(storage_admission=provider).run()
        self.assertEqual(self.store.snapshot()['updates']['0']['phase'],'consumed')
        self.assertEqual(self.trace.count('update'),1)
        await self.driver().run()
        self.assertEqual(self.trace.count('update'),10)
        self.assertTrue(any(b['reused'] and b['phase']=='native-update-checkpoint' and
                            b['stage']=='optimizer-update-0' for b in self.boundaries()))

    async def test_storage_refusals_prevent_specific_writer_stages(self):
        # Exact boundaries are selected here; the real helper still validates
        # the provider and records each refusal before any corresponding write.
        for target,forbidden in (('collection','browser'),('draft','score_save'),
                                 ('score-save','score_save'),('native-update-checkpoint','update')):
            with self.subTest(target=target):
                context={}
                async def observed(store,provider,stage,phase,position,execution_id,**kwargs):
                    context.update(phase=phase,position=position)
                    return await storage_boundary(store,provider,stage,phase,position,execution_id,**kwargs)
                def provider(store,stage):
                    if context==dict(phase=target,position='before'):
                        raise ValueError('CPU refused '+target)
                    return self.storage(store,stage)
                before=self.trace.count(forbidden)
                with patch('openwebrl.action_credit_driver.storage_boundary',observed):
                    with self.assertRaisesRegex(ValueError,'CPU refused'):
                        await self.driver(storage_admission=provider).run()
                self.assertEqual(self.trace.count(forbidden),before)
                self.assertTrue(any(b['phase']==target and b['position']=='before' and b['passed'] is False
                                    for b in self.boundaries()))
                if target=='draft':
                    self.assertTrue(self.pools[-1].closed)
                    self.assertNotIn('0',self.store.snapshot().get('driver_drafts',{}))

    async def test_collector_failure_survives_pool_cleanup_and_storage_failure(self):
        async def broken(*args,**kwargs):raise RuntimeError('CPU primary collection failure')
        class Pool:
            concurrency=64
            async def close(self):raise RuntimeError('CPU cleanup failure')
        failed=False
        async def collecting(*args,**kwargs):
            nonlocal failed
            failed=True
            await broken()
        def provider(store,stage):
            if failed:raise ValueError('CPU reconciliation failure')
            return self.storage(store,stage)
        with patch('openwebrl.action_credit_driver.collect_online_update',collecting):
            with self.assertRaisesRegex(RuntimeError,'primary collection') as result:
                await self.driver(storage_admission=provider,worker_pool_factory=lambda *args:Pool()).run()
        self.assertTrue(any('cleanup failure' in note for note in result.exception.__notes__))
        self.assertTrue(any('reconciliation failure' in note for note in result.exception.__notes__))

    async def test_storage_evidence_stale_foreign_or_failed_is_rejected(self):
        for mutation in ({'stage':'initial'},{'manifest_sha256':digest('foreign')},{'passed':False},
                         {'reservation':dict(manifest_sha256=self.store.manifest.sha256,
                              store_root=str(self.store.root),epoch=1.)}):
            def provider(store,stage):
                proof=_json(self.storage(store,stage));proof.update(mutation)
                return _write(self.root/f'wrong-storage-{len(self.storage_calls)}.json',proof)
            called=[]
            with self.assertRaisesRegex(ValueError,'storage admission'):
                await storage_guarded(self.store,provider,'optimizer-update-0','CPU invalid proof',
                    self.launch.execution_id,lambda:called.append(True))
            self.assertEqual(called,[])

    async def test_restart_reuses_streamed_draft_and_rng_after_scorer_failure(self):
        self.fail_score = True
        with self.assertRaisesRegex(RuntimeError,"scorer failure"):
            await self.driver().run()
        self.assertEqual(self.collect_calls,1)
        saved = self.store.snapshot()["driver_drafts"]["0"]
        self.launch = replace(self.launch,execution_id="CPU-replacement-attempt")
        result = await self.driver().run()
        self.assertTrue(result["training_complete"])
        self.assertEqual(self.collect_calls,10)
        self.assertEqual(self.store.snapshot()["driver_drafts"]["0"],saved)

    async def test_physical_acceptance_failure_blocks_optimizer_and_keeps_pending(self):
        async def missing(*args): raise ValueError("missing physical proof")
        with self.assertRaisesRegex(ValueError,"physical proof"):
            await self.driver(collection_acceptance=missing).run()
        self.assertNotIn("update",self.trace)
        self.assertIn("0",self.store.snapshot()["pending_batches"])
        self.assertTrue(self.pools[0].closed)

    async def test_failed_completion_hook_retries_without_retraining_committed_update(self):
        seen = []
        async def hook(store, completed, result):
            seen.append(completed)
            if completed == 10 and seen.count(10) == 1: raise RuntimeError("CPU eval handoff failed")
            return {"cpu_hook":completed}
        with self.assertRaisesRegex(RuntimeError,"handoff failed"):
            await self.driver(completion_hook=hook).run()
        count = self.trace.count("update")
        result = await self.driver(completion_hook=hook).run()
        self.assertTrue(result["training_complete"])
        self.assertEqual(self.trace.count("update"),count)
        self.assertEqual(seen,list(range(1,11))+[10])
        await self.driver(completion_hook=hook).run()
        self.assertEqual(seen,list(range(1,11))+[10])
        self.assertTrue(any(b['phase']=='iteration10-evaluation' and b['reused'] for b in self.boundaries()))

    async def test_wrong_served_version_blocks_browser_and_close_on_collector_failure(self):
        with patch("openwebrl.action_credit_driver.served_weight_version",AsyncMock(return_value="wrong")):
            with self.assertRaisesRegex(ValueError,"policy version"):
                await self.driver().run()
        self.assertEqual(self.collect_calls,0)
        async def broken(*args,**kwargs): raise RuntimeError("CPU browser adapter failed")
        self.launch = replace(self.launch,execution_id="CPU-replacement")
        with patch("openwebrl.action_credit_driver.collect_online_update",broken):
            with self.assertRaisesRegex(RuntimeError,"browser adapter"):
                await self.driver().run()
        self.assertTrue(self.pools[-1].closed)

    async def test_exclusive_owner_and_expired_deadline_reject_dispatch(self):
        import fcntl
        with (self.store.root/"driver-owner.lock").open("a+b") as handle:
            fcntl.flock(handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            with self.assertRaises(BlockingIOError): await self.driver().run()
        self.launch = replace(self.launch,deadline_unix=time.time()-1)
        with self.assertRaises(TimeoutError): await self.driver().run()
        self.assertEqual(self.rpc_seen,[])

    async def test_saved_episode_keeps_original_actor_envelope_after_endpoint_restart(self):
        driver=self.driver();contract=self.fixture.contract
        inventory={"path":str(self.root/"cpu-inventory.json"),"sha256":digest("CPU inventory")}
        rng=_write(self.root/"cpu-freeze-rng.json",{"cpu_fake":True})
        resolve,endpoint=await driver.freeze(contract,inventory,rng)
        request=SimpleNamespace(request_id=digest("saved CPU request"))
        old=resolve(request);old_url=endpoint(old)
        with self.store._transaction() as state:
            state.setdefault("online_requests",{})[request.request_id]={"actor":asdict(old)}
        async def moved_url():return "http://127.0.0.1:12003"
        async def moved_url2():return "http://127.0.0.1:12004"
        async def moved_url3():return "http://127.0.0.1:12005"
        async def moved_url4():return "http://127.0.0.1:12006"
        async def moved_engines():return [SimpleNamespace(get_url=Remote(fn)) for fn in (moved_url,moved_url2,moved_url3,moved_url4)],None,0,[2]*4,[0,2,4,6]
        driver.rollout_manager=SimpleNamespace(get_updatable_engines_and_lock=Remote(moved_engines))
        resolve,endpoint=await driver.freeze(contract,inventory,rng)
        self.assertEqual(resolve(request),old)
        self.assertEqual(endpoint(old),old_url)
        fresh=resolve(SimpleNamespace(request_id=digest("new CPU request")))
        self.assertIn(endpoint(fresh),["http://127.0.0.1:"+str(port)+"/generate" for port in (12003,12004,12005,12006)])
        self.assertEqual(fresh.weight_version,old.weight_version)

    async def test_waiting_worker_rechecks_deadline_after_capacity_admission(self):
        count=0;expired=False;released=asyncio.Event()
        class Pool:
            concurrency=64
            async def collect(self,request,prefix,actor):
                nonlocal count,expired
                count+=1
                if count==64:expired=True;released.set()
                await released.wait()
            async def close(self):self.closed=True
        pool=Pool()
        driver=self.driver(worker_pool_factory=lambda *args:pool)
        def deadline():
            if expired:raise TimeoutError("CPU deadline expired while queued")
        driver.deadline=deadline
        async def full_queue(*args,dispatch,**kwargs):
            await _owned([dispatch(SimpleNamespace(request_id=digest(i)),()) for i in range(65)])
        rng=_write(self.root/"queued-rng.json",{"cpu_fake":True})
        with patch("openwebrl.action_credit_driver.collect_online_update",full_queue):
            with self.assertRaisesRegex(TimeoutError,"queued"):
                await driver.collect(self.fixture.contract,{"sha256":digest("inventory")},rng)
        self.assertEqual(count,64)
        self.assertTrue(pool.closed)


class ChildOwnershipTests(IsolatedAsyncioTestCase):
    async def test_failure_and_parent_cancellation_drain_every_started_child(self):
        finished = []
        async def child(i):
            await asyncio.sleep(.01*i)
            finished.append(i)
            if i == 0: raise RuntimeError("CPU failed rank")
            return i
        with self.assertRaisesRegex(RuntimeError,"failed rank"):
            await _owned([child(i) for i in range(8)])
        self.assertEqual(set(finished),set(range(8)))
        finished.clear()
        async def slow(i):
            await asyncio.sleep(.02);finished.append(i)
        task=asyncio.create_task(_owned([slow(i) for i in range(8)]))
        await asyncio.sleep(.005);task.cancel()
        with self.assertRaises(asyncio.CancelledError):await task
        self.assertEqual(set(finished),set(range(8)))
