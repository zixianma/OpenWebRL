"""Draft crash recovery and fixed engine ownership; CPU fixtures, no API or GPU calls."""
from pathlib import Path
from dataclasses import asdict, replace
import copy
import json
import os
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch, AsyncMock
import torch
from openwebrl.action_credit import digest
from openwebrl.action_credit_driver import _stream_draft, PilotDriver, _draft_value, _draft_identity
import openwebrl.action_credit_driver as driver
from openwebrl.action_credit_collect import physical_verifier_sha256
from openwebrl.action_credit_online import collect_online_update, plan_online_update, online_collection_protocol, draft_online_update
from openwebrl.action_credit_state import CampaignStore, CampaignManifest
from openwebrl.action_credit_windows import WindowConfig
from openwebrl.action_credit_native_controller import _trusted_torch, _hash
import test_action_credit_collect as collection
from test_action_credit_online import order_fixture, tree_policy
from test_action_credit_binding import config, panel
from test_action_credit_state import cost
from test_action_credit_selection import selection_fixture
from test_action_credit import graph_fixture

class RecoveryTests(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        self.f = collection.CollectionTests()
        self.f.setUp()
        self.addCleanup(self.f.doCleanups)
        f = self.f
        data = f.store.manifest.to_dict()
        self.config = config('chain_outcome_grpo')
        (path, _) = order_fixture(Path(f.temp.name), data['science']['dataset']['source_sha256'])
        self.path = path
        data['science']['dataset']['task_order_sha256'] = _hash(path)
        data['science']['algorithm'].update(online_collection_protocol=online_collection_protocol(42), objective_config_sha256=digest(asdict(self.config)), ppo_window_config=asdict(WindowConfig(2, 256, 42)))
        f.store = CampaignStore.create(Path(f.temp.name) / 'online-ledger', CampaignManifest.from_dict(data))
        (plan, _) = plan_online_update(f.store, f.contract, path)
        with f.store._transaction() as state:
            for item in plan['draws'][1:]:
                rid = digest([f.store.manifest.sha256, f.contract.sha256, item['leaf_id'], 'episode'])
                identity = f.store._draw_identity(0, f.contract.policy_sha256, item['spec'])
                draw_id = digest(identity)
                state['requests'][rid] = draw_id
                state['draws'][draw_id] = dict(identity, request_id=rid, status='missing', attempts=[], selected_attempt=None, missing_reason='CPU fixture closed missing slot')
        self.fake = collection.NativeFake()
        req = f.request(self.fake)
        self.pins = {k: v for (k, v) in asdict(req).items() if k in ('task_data_sha256', 'tools_sha256', 'environment_policy_sha256', 'browser_config_sha256', 'browser_manifest_sha256', 'generator_protocol_sha256')}
        self.calls = 0

        async def dispatch(request, prefix):
            self.calls += 1
            ep = await f.run_episode(self.fake, request, plan=False, prefix=prefix)
            pointer = f.store.snapshot()['pending_episodes'][request.attempt_id]
            f.store.finish_attempt(request.attempt_id, status='valid', actual_cost=cost(2), outcome=dict(artifact_sha256=pointer['sha256'], reward=1.0))
            return ep
        self.dispatch = dispatch
        self.draft = await self.rebuild('initial')
        self.before = self.accounting()

    async def rebuild(self, namespace):
        return await collect_online_update(self.f.store, self.f.contract, self.f.actor, self.path, dispatch=self.dispatch, request_pins=lambda task: self.pins, attempt_namespace=namespace, verifier_sha256=physical_verifier_sha256())

    def accounting(self):
        state = self.f.store.snapshot()
        return {key: value for (key, value) in state.items() if key not in ('driver_drafts', 'updated_unix', 'revision')}

    def files(self):
        return sorted((self.f.store.root / 'driver_drafts' / '0').glob('*.pt'))

    def write(self, draft=None, contract=None, config_value=None):
        return _stream_draft(self.f.store, contract or self.f.contract, draft or self.draft, config_value or self.config)

    async def assert_recovered(self, orphans):
        old = {str(p): _hash(p) for p in orphans}
        rebuilt = await self.rebuild('after-crash')
        ref = self.write(rebuilt)
        self.assertEqual(self.calls, 1)
        self.assertEqual(self.fake.created, 1)
        self.assertEqual(self.accounting(), self.before)
        self.assertEqual({p: _hash(p) for p in map(Path, old)}, {Path(p): h for (p, h) in old.items()})
        count = len(self.files())
        self.assertEqual(self.write(await self.rebuild('second-restart')), ref)
        self.assertEqual(len(self.files()), count)
        self.assertEqual(self.calls, 1)
        self.assertEqual(self.accounting(), self.before)
        self.assertEqual(_trusted_torch(ref)['draft']['graph'], self.draft['graph'])

    async def test_partial_torch_write_preserved_and_rebuilt(self):

        def fail(payload, handle):
            handle.write(b'partial-uncommitted')
            handle.flush()
            raise OSError('injected partial write')
        with patch('torch.save', fail), self.assertRaisesRegex(OSError, 'partial'):
            self.write()
        self.assertNotIn('0', self.f.store.snapshot().get('driver_drafts', {}))
        await self.assert_recovered(self.files())

    async def test_complete_file_before_pointer_crash(self):
        with patch.object(driver, '_persist', side_effect=RuntimeError('injected before pointer')), self.assertRaisesRegex(RuntimeError, 'before pointer'):
            self.write()
        await self.assert_recovered(self.files())

    async def test_directory_fsync_failure_retained(self):
        original = os.fsync
        calls = []

        def fail(fd):
            calls.append(fd)
            if len(calls) == 2:
                raise OSError('directory fsync fault')
            return original(fd)
        with patch.object(driver.os, 'fsync', fail), self.assertRaisesRegex(OSError, 'directory'):
            self.write()
        await self.assert_recovered(self.files())

    async def test_pointer_committed_then_caller_crashes_no_duplicate(self):
        original = driver._persist

        def fail(*args, **kw):
            original(*args, **kw)
            raise RuntimeError('injected after pointer')
        with patch.object(driver, '_persist', fail), self.assertRaisesRegex(RuntimeError, 'after pointer'):
            self.write()
        ref = self.f.store.snapshot()['driver_drafts']['0']
        count = len(self.files())
        self.assertEqual(self.write(await self.rebuild('restart')), ref)
        self.assertEqual(len(self.files()), count)
        self.assertEqual(self.calls, 1)
        self.assertEqual(self.accounting(), self.before)

    async def test_conflicting_batch_is_rejected_without_files_or_cursor(self):
        ref = self.write()
        changed = dict(self.draft)
        changed['panels'] = ('conflicting batch',)
        with self.assertRaisesRegex(ValueError, 'conflicts'):
            self.write(changed)
        self.assertEqual(self.f.store.snapshot()['driver_drafts']['0'], ref)
        self.assertEqual(len(self.files()), 1)
        self.assertEqual(self.accounting(), self.before)

    async def test_wrong_update_policy_and_objective_rejected(self):
        for changed in (replace(self.f.contract, update_id=1), replace(self.f.contract, policy_sha256=digest('wrong'))):
            with self.assertRaises(ValueError):
                self.write(contract=changed)
        with self.assertRaises(ValueError):
            self.write(config_value=config('gigpo'))
        self.assertEqual(self.files(), [])
        self.assertEqual(self.accounting(), self.before)

    async def test_fixed_slots_and_plan_hash_checked(self):
        changed = dict(self.draft, leaf_to_draw={})
        with self.assertRaisesRegex(ValueError, 'fixed draw slots'):
            self.write(changed)
        with self.f.store._transaction() as state:
            state['online_collection_plans']['0']['plan_sha256'] = digest('wrong')
        with self.assertRaisesRegex(ValueError, 'collection plan'):
            self.write()
        self.assertEqual(self.files(), [])

    async def test_checksum_failure_does_not_rebuild_over_committed_pointer(self):
        ref = self.write()
        p = Path(ref['path'])
        p.write_bytes(p.read_bytes() + b'corruption-retained')
        with self.assertRaises(ValueError):
            self.write()
        self.assertEqual(len(self.files()), 1)
        self.assertEqual(self.f.store.snapshot()['driver_drafts']['0'], ref)

    async def test_source_identity_and_window_change_refused(self):
        ref = self.write()
        data = self.f.store.manifest.to_dict()
        for changed in ('source', 'window'):
            altered = copy.deepcopy(data)
            if changed == 'source':
                altered['code_revision_sha256'] = digest('another-source')
            else:
                altered['science']['algorithm']['ppo_window_config']['seed'] += 1
            fake_store = SimpleNamespace(manifest=CampaignManifest.from_dict(altered), root=self.f.store.root, snapshot=self.f.store.snapshot, _draw_identity=self.f.store._draw_identity)
            with self.assertRaises(ValueError):
                driver._checked_draft_reference(fake_store, self.f.contract, self.config, ref)

    async def test_legacy_orphan_never_read_or_adopted(self):
        d = self.f.store.root / 'driver_drafts' / '0'
        d.mkdir(parents=True)
        p = d / 'draft.pt'
        p.write_bytes(b'never unpickle this orphan')
        await self.assert_recovered([p])

    async def test_no_cursor_advance_from_retry_or_draft_commit(self):
        self.write()
        await self.assert_recovered([])
        self.assertEqual(self.f.store.snapshot()['updates']['0']['phase'], 'collecting')
        self.assertIsNone(self.f.store.snapshot()['updates']['0']['training_receipt'])
        with self.assertRaises(ValueError):
            self.f.store.plan_update(1, policy_sha256=digest('future'), draws=[self.f.request(self.fake).spec()])

    async def test_later_update_exact_prior_consumed_cursor(self):
        f = self.f
        new_policy = digest('CPU synthetic update1 policy')
        with f.store._transaction() as state:
            state['updates']['0'].update(phase='consumed', training_receipt=dict(new_policy_sha256=new_policy, new_checkpoint_sha256=digest('CPU synthetic checkpoint')))
        f.contract = replace(f.contract, update_id=1, policy_sha256=new_policy)
        (plan, _) = plan_online_update(f.store, f.contract, self.path)
        with f.store._transaction() as state:
            for item in plan['draws']:
                identity = f.store._draw_identity(1, new_policy, item['spec'])
                state['draws'][digest(identity)] = dict(identity, request_id=digest([1, item['leaf_id']]), status='missing', attempts=[], selected_attempt=None, missing_reason='CPU closed slot')
        draft = draft_online_update(f.store, f.contract, plan, ())
        ref = _stream_draft(f.store, f.contract, draft, self.config)
        self.assertEqual(_trusted_torch(ref)['identity']['previous_consumed_receipt_sha256'], digest(f.store.snapshot()['updates']['0']['training_receipt']))
        original = f.store.snapshot()['updates']['0']
        for change in ('phase', 'policy'):
            with f.store._transaction() as state:
                if change == 'phase':
                    state['updates']['0']['phase'] = 'sealed'
                else:
                    state['updates']['0']['training_receipt']['new_policy_sha256'] = digest('wrong prior cursor')
            with self.assertRaisesRegex(ValueError, 'previous consumed cursor'):
                _draft_identity(f.store, f.contract, draft, self.config)
            with f.store._transaction() as state:
                state['updates']['0'] = json.loads(json.dumps(original))
        self.assertEqual(f.store.snapshot()['updates']['1']['phase'], 'collecting')

    async def test_competing_committed_pointer_not_overwritten(self):
        original = driver._persist

        def competing(store, section, key, value):
            altered = dict(value, batch_identity_sha256=digest('competing identity'))
            original(store, section, key, altered)
            return original(store, section, key, value)
        with patch.object(driver, '_persist', competing), self.assertRaisesRegex(ValueError, 'immutable online decision changed'):
            self.write()
        recorded = self.f.store.snapshot()['driver_drafts']['0']
        before = {str(p): _hash(p) for p in self.files()}
        with self.assertRaisesRegex(ValueError, 'conflicts'):
            self.write()
        self.assertEqual(recorded, self.f.store.snapshot()['driver_drafts']['0'])
        self.assertEqual(before, {str(p): _hash(p) for p in self.files()})
        self.assertEqual(self.calls, 1)

class EngineTests(unittest.IsolatedAsyncioTestCase):

    async def freeze(self, count=4, counts=None, offsets=None, urls=None):
        f = collection.CollectionTests()
        f.setUp()
        self.addCleanup(f.doCleanups)
        root = Path(f.temp.name)
        d = object.__new__(PilotDriver)
        d.store = f.store
        d.handles = tuple(range(8))
        d.launch = SimpleNamespace(deadline_unix=time.time() + 60)
        d.common = {}
        actual_urls = urls if urls is not None else ['http://127.0.0.1:' + str(12000 + i) for i in range(count)]

        class Remote:

            def __init__(self, fn):
                self.fn = fn

            def remote(self, *a, **kw):
                return self.fn(*a, **kw)
        engines = [SimpleNamespace(get_url=Remote(AsyncMock(return_value=url))) for url in actual_urls]
        reply = (engines, None, 0, counts if counts is not None else [2] * count, offsets if offsets is not None else list(range(0, count * 2, 2)))
        d.rollout_manager = SimpleNamespace(get_updatable_engines_and_lock=Remote(AsyncMock(return_value=reply)))
        with patch.object(driver, '_call_all', AsyncMock(return_value=[{}] * 8)), patch.object(driver, 'served_weight_version', AsyncMock(return_value='1')):
            return await d.freeze(f.contract, dict(path=str(root / 'unused'), sha256=digest('checkpoint')), dict(sha256=digest('rng')))

    async def test_exact_four_engines_succeed(self):
        await self.freeze()

    async def test_three_engines_rejected(self):
        with self.assertRaisesRegex(ValueError, 'four owned'):
            await self.freeze(count=3)

    async def test_missing_endpoint_rejected(self):
        urls = ['http://127.0.0.1:12000', 'http://127.0.0.1:12001', 'http://127.0.0.1:12002', None]
        with self.assertRaisesRegex(ValueError, 'all four'):
            await self.freeze(urls=urls)

    async def test_wrong_topology_rejected(self):
        for kwargs in (dict(counts=[1, 1, 1, 1]), dict(offsets=[0, 1, 2, 3])):
            with self.assertRaisesRegex(ValueError, 'four owned'):
                await self.freeze(**kwargs)

    async def test_duplicate_urls_rejected(self):
        with self.assertRaises(ValueError):
            await self.freeze(urls=['http://127.0.0.1:12000'] * 4)

class Metadata(unittest.TestCase):

    def test_all_five_arm_metadata_types_stable(self):
        import io
        with tempfile.TemporaryDirectory(prefix='action-credit-draft-metadata-') as directory:
            (_, order) = order_fixture(Path(directory), digest('fixture dataset'))
        (graph, selection) = selection_fixture()
        policy = tree_policy(order, digest('fixture order'))
        for arm in ('chain_outcome_grpo', 'gigpo', 'tree_grpo', 'tree_outcome_sampling', 'continuation_q'):
            value = dict(config=config(arm), graph=graph if arm.startswith('tree') else graph_fixture(), panels=(panel(),) if arm == 'continuation_q' else (), selection_builder_data=dict(policy=policy, selection=selection, partitions=()) if arm.startswith('tree') else None)
            expected = digest(_draft_value(value))
            f = io.BytesIO()
            torch.save(value, f)
            f.seek(0)
            self.assertEqual(expected, digest(_draft_value(torch.load(f, weights_only=False))))
if __name__ == '__main__':
    unittest.main(verbosity=2)
