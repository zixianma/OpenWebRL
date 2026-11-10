"""CPU-only real tensor/persistence checks, not physical GPU acceptance."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import io
import os
from pathlib import Path
import pickle
import tempfile
import unittest
import zipfile

import torch

from openwebrl.action_credit import digest, clipped_loss, leaf_loss_plan, append_local_terms
from openwebrl.action_credit_megatron import tensor_payload_sha256, EdgeIterator
from openwebrl.action_credit_state import CampaignManifest, CampaignStore
from openwebrl.action_credit_tensor_refs import (TensorBlobMapping, create_tensor_blob,
    tensor_inventory, bind_tensor_inventory, episode_tensor_mappings)
from test_action_credit_state import manifest_data, reserve, cost
from test_action_credit_megatron import fixture
import test_action_credit_collect as collection


def serialized(value):
    stream = io.BytesIO()
    torch.save(value, stream)
    return stream.getvalue()


def tensor_storage_bytes(raw):
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        return sum(item.file_size for item in archive.infolist()
                   if '/data/' in item.filename)


class TensorReferenceTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix='action-credit-tensor-refs-', dir='/tmp')
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.store = CampaignStore.create(self.root/'ledger', CampaignManifest.from_dict(manifest_data()))
        self.draw = reserve(self.store)['draw_id']
        self.store.start_attempt(self.draw, 'attempt', reserved_cost=cost())

    def create(self, tensors=None, turn=0):
        tensors = tensors if tensors is not None else {
            'pixel_values': torch.arange(48, dtype=torch.bfloat16).reshape(3, 4, 4),
            'grid': torch.tensor([[1, 2, 3]], dtype=torch.int64)}
        return create_tensor_blob(self.store, 'attempt', turn, tensors)

    def restore(self, mapping):
        owners = tensor_inventory(self.store, (mapping,))
        restored = pickle.loads(pickle.dumps(mapping))
        with self.assertRaisesRegex(ValueError, 'unbound'):
            restored[next(iter(restored))]
        bind_tensor_inventory(self.store, (restored,), owners)
        return restored

    def test_real_dtype_shape_stride_storage_digest_and_descriptor_only_reload(self):
        source = {'pixels': torch.arange(120, dtype=torch.float32).reshape(10, 12)[1:9:2, ::2],
                  'bf16': torch.arange(6, dtype=torch.bfloat16), 'grid': torch.tensor([1, 2, 3])}
        mapping = self.create(source)
        restored = self.restore(mapping)
        self.assertEqual(tensor_payload_sha256(source), tensor_payload_sha256(restored))
        for key in source:
            torch.testing.assert_close(source[key], restored[key], rtol=0, atol=0)
            self.assertEqual((source[key].shape, source[key].stride(), source[key].storage_offset()),
                             (restored[key].shape, restored[key].stride(), restored[key].storage_offset()))
        # Deepcopy and native Sample.to_dict use the same descriptor boundary.
        from slime.utils.types import Sample
        native = Sample(multimodal_train_inputs=mapping)
        for value in (mapping, {'captured': mapping, 'native': native}, native.to_dict(),
                      deepcopy(native.to_dict())):
            self.assertEqual(tensor_storage_bytes(serialized(value)), 0)

    def test_five_retained_tensor_copies_reduce_to_one_owner_even_after_warm_cache(self):
        source = {'pixels': torch.arange(256*1024, dtype=torch.float32)}
        mapping = self.create(source)
        mapping['pixels']  # Ensure materialization cannot leak into pickle.
        old = [serialized(source), serialized({'captured': source,
                'native': {k: v.clone() for k, v in source.items()}}),
               serialized({'draft': source}), serialized({'scored': source})]
        new = [serialized({'captured': mapping, 'native': mapping}),
               serialized({'prefix': (mapping,)}), serialized({'draft': mapping}),
               serialized({'scored': mapping})]
        self.assertEqual(sum(tensor_storage_bytes(raw) for raw in old), 5*1024*1024)
        self.assertEqual(sum(tensor_storage_bytes(raw) for raw in new), 0)
        self.assertEqual(mapping.ref.tensor_schema[0][-1], 1024*1024)
        self.assertLess(mapping.ref.file_bytes + sum(map(len, new)), 1.1*1024*1024)

    def test_root_manifest_source_path_and_owner_attempt_binding(self):
        mapping = self.create()
        other = CampaignStore.create(self.root/'foreign', self.store.manifest)
        with self.assertRaises(ValueError):
            tensor_inventory(other, (mapping,))
        for changes in ({'manifest_sha256': digest('foreign')}, {'root_sha256': digest('foreign')},
                        {'source_sha256': digest('changed source')}, {'relative_path': '../escaped.pt'},
                        {'relative_path': '/tmp/escaped.pt'}, {'attempt_id': 'absent'},
                        {'draw_id': digest('different charged draw')}, {'file_sha256': digest('tampered')},
                        {'tensor_payload_sha256': digest('tampered values')}, {'file_bytes': 1},
                        {'tensor_schema': (), 'tensor_payload_sha256': tensor_payload_sha256({})},
                        {'tensor_schema': (('pixel_values', 'torch.float32', (1,), (1,), 0, 4),)}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                tensor_inventory(self.store, (TensorBlobMapping(replace(mapping.ref, **changes)),))

    def test_missing_foreign_symlink_hardlink_or_public_owner_fails_closed(self):
        mapping = self.create()
        path = self.store.root/mapping.ref.relative_path
        ref = replace(mapping.ref, turn=1, relative_path=str(Path(mapping.ref.relative_path).with_name('0001-processor.pt')))
        missing = TensorBlobMapping(ref)
        with self.assertRaises(FileNotFoundError):
            tensor_inventory(self.store, (missing,))
        link = self.store.root/ref.relative_path
        link.symlink_to(path)
        with self.assertRaises(ValueError):
            tensor_inventory(self.store, (missing,))
        os.link(path, self.root/'unexpected-hardlink.pt')
        with self.assertRaisesRegex(ValueError, 'single-link'):
            mapping['pixel_values']
        independent = self.create(turn=2)
        (self.store.root/independent.ref.relative_path).chmod(0o644)
        with self.assertRaisesRegex(ValueError, 'private'):
            independent['pixel_values']

    def test_complete_inventory_checks_unselected_owner_and_rejects_extra_inline_values(self):
        first, second = self.create(), self.create(turn=1)
        owners = tensor_inventory(self.store, (first, second, first))
        self.assertEqual(len(owners), 2)
        with self.assertRaisesRegex(ValueError, 'inventory differs'):
            bind_tensor_inventory(self.store, (first, second), owners[:1])
        with self.assertRaisesRegex(ValueError, 'lossless tensor reference'):
            tensor_inventory(self.store, (first, {'pixels': torch.ones(1)}))
        path = self.store.root/second.ref.relative_path
        with path.open('ab') as handle:
            handle.write(b'corrupt otherwise-unselected owner')
        with self.assertRaisesRegex(ValueError, 'size/hash'):
            bind_tensor_inventory(self.store, (first, second), owners)

    def test_mutated_cache_does_not_write_owner_or_get_erased_by_rebinding(self):
        mapping = self.create()
        path = self.store.root/mapping.ref.relative_path
        original_file = hashlib.sha256(path.read_bytes()).hexdigest()
        mapping['pixel_values'].add_(1)
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), original_file)
        with self.assertRaisesRegex(ValueError, 'payload changed'):
            tensor_inventory(self.store, (mapping,))
        self.assertNotEqual(tensor_payload_sha256(mapping), mapping.ref.tensor_payload_sha256)
        reopened = TensorBlobMapping(mapping.ref)
        tensor_inventory(self.store, (reopened,))
        self.assertEqual(tensor_payload_sha256(reopened), mapping.ref.tensor_payload_sha256)

    def test_owner_change_rejected_with_warm_cache_and_collective_identity_rechecked(self):
        mapping = self.create()
        mapping['pixel_values']
        path = self.store.root/mapping.ref.relative_path
        with path.open('ab') as handle:
            handle.write(b'x')
        with self.assertRaisesRegex(ValueError, 'changed after'):
            mapping['pixel_values']
        clean = self.create(turn=1)
        inventory = [clean.ref.to_dict()]
        def changed_after_leader(fn):
            result = fn()
            with (self.store.root/clean.ref.relative_path).open('ab') as handle:
                handle.write(b'x')
            return result
        with self.assertRaisesRegex(ValueError, 'collective verification'):
            bind_tensor_inventory(self.store, (clean,), inventory, leader=changed_after_leader)

    def test_exclusive_crash_and_unknown_attempt_never_overwrite_adopt_or_redispatch(self):
        mapping = self.create()
        before = self.store.snapshot()
        with self.assertRaisesRegex(ValueError, 'crash reconciliation'):
            self.create()
        self.assertEqual(self.store.snapshot(), before)
        # Marking an interrupted attempt unknown does not erase its retained
        # owner or permit another creation. The reference remains readable.
        with self.store._transaction() as state:
            state['attempts']['attempt']['status'] = 'unknown'
        self.assertEqual(tensor_payload_sha256(self.restore(mapping)), mapping.ref.tensor_payload_sha256)
        with self.assertRaisesRegex(ValueError, 'active retained attempt'):
            self.create(turn=1)

    def test_real_tensor_loss_gradient_reload_and_each_window_hash_equivalence(self):
        graph, rows, raw = fixture()
        refs = {key: self.create(value, turn=i) for i, (key, value) in enumerate(raw.items())}
        refs = {key: self.restore(value) for key, value in refs.items()}
        for reduction in ('response_mean', 'trajectory_mean', 'token_mean'):
            plan = leaf_loss_plan(graph, {'a': 1., 'b': -1.}, arm_id='tree_outcome_sampling', reduction=reduction)
            plan = append_local_terms(replace(plan, arm_id='continuation_q'), [{'good': .25, 'bad': -.25}], coefficient=.5)
            def evaluate(payloads):
                parameter = torch.tensor([-.02, .015], dtype=torch.float64, requires_grad=True)
                values = {key: -2 + parameter * payload['pixel_values'].double().sum()/100
                          for key, payload in payloads.items()}
                loss = clipped_loss(plan, values, clip_low=.2, clip_high=.2)
                return loss.detach(), torch.autograd.grad(loss, parameter)[0]
            expected, actual = evaluate(raw), evaluate(refs)
            for left, right in zip(expected, actual):
                torch.testing.assert_close(left, right, atol=0, rtol=0)
        for _ in range(2):
            iterator = EdgeIterator(graph, rows, refs, ('good', None), device='cpu')
            result = iterator.get_next(('tokens', 'loss_masks', 'multimodal_train_inputs'))
            self.assertEqual(result['tokens'][0].tolist(), [0, 1, 2, 3])
        refs['good']['pixel_values'].add_(1)
        with self.assertRaisesRegex(ValueError, 'payload changed'):
            EdgeIterator(graph, rows, refs, ('good',), device='cpu')

    def test_all_five_objectives_have_bit_identical_real_tensor_losses_and_gradients(self):
        from test_action_credit_objective import build_objective, config, chains, ContinuationPanel, graph_fixture
        source = {'pixel_values': torch.arange(48, dtype=torch.float32).reshape(3, 4, 4)}
        restored = self.restore(self.create(source))
        for arm in ('chain_outcome_grpo', 'tree_outcome_sampling', 'tree_grpo', 'continuation_q', 'gigpo'):
            graph = chains() if arm in ('chain_outcome_grpo', 'gigpo') else graph_fixture()
            prepared = build_objective(graph, config(arm), panels=(ContinuationPanel(
                's', ('good', 'bad'), (('a',), ('b',))),) if arm == 'continuation_q' else ())
            def evaluate(payload):
                parameter = torch.tensor([-.02, .015], dtype=torch.float64, requires_grad=True)
                values = {key: -2 + parameter * payload['pixel_values'].double().sum()/1000
                          for key in {term.edge_id for term in prepared.plan.terms}}
                loss = prepared.loss(values)
                return loss.detach(), torch.autograd.grad(loss, parameter)[0]
            with self.subTest(arm=arm):
                for left, right in zip(evaluate(source), evaluate(restored)):
                    torch.testing.assert_close(left, right, atol=0, rtol=0)


class NativeTensorOwnerTests(unittest.IsolatedAsyncioTestCase):
    setUp = collection.CollectionTests.setUp
    request = collection.CollectionTests.request
    actor_call = collection.CollectionTests.actor_call
    version = collection.CollectionTests.version
    run_episode = collection.CollectionTests.run_episode

    async def test_native_and_captured_share_owner_and_episode_contains_no_tensor_bytes(self):
        episode = await self.run_episode(collection.NativeFake())
        self.assertIs(episode.turns[0].tensors, episode.native_samples[0].multimodal_train_inputs)
        self.assertEqual(len(tensor_inventory(self.store, episode_tensor_mappings(episode))), 1)
        pointer = self.store.snapshot()['pending_episodes']['attempt1']
        self.assertEqual(tensor_storage_bytes(Path(pointer['path']).read_bytes()), 0)

    async def test_wrong_native_values_rejected_before_committing_pending_episode(self):
        fake = collection.NativeFake()
        generate = fake.generate_turn_sample
        async def changed(*args, **kwargs):
            samples = await generate(*args, **kwargs)
            samples[0].multimodal_train_inputs['pixel_values'].add_(1)
            return samples
        fake.generate_turn_sample = changed
        with self.assertRaisesRegex(ValueError, 'native processor tensors differ'):
            await self.run_episode(fake)
        state = self.store.snapshot()
        self.assertFalse(state.get('pending_episodes'))
        self.assertEqual(state['attempts']['attempt1']['status'], 'active')
        self.assertEqual(self.calls, 1)

    async def test_prefix_reuses_original_owner_and_reload_checks_transitive_old_attempt(self):
        from openwebrl.action_credit_episode import load_pending_episode
        from openwebrl.action_credit_collect import physical_verifier_sha256
        fake = collection.NativeFake()
        first = self.request(fake)
        self.store.plan_update(0, policy_sha256=self.contract.policy_sha256,
            draws=[first.spec(), dict(first.spec(), draw_slot=1)])
        original = await self.run_episode(fake, first, plan=False)
        second = self.request(fake, attempt='attempt2', leaf='leaf2', slot=1, prefix=original.turns)
        replay = await self.run_episode(fake, second, plan=False, prefix=original.turns)
        self.assertEqual(self.calls, 1)
        self.assertEqual(replay.turns[0].tensors.ref, original.turns[0].tensors.ref)
        self.assertFalse((Path(replay.artifact_directory)/'0000-processor.pt').exists())
        self.assertTrue((Path(replay.artifact_directory)/'0000-processor-reference.json').is_file())
        kwargs = dict(verifier_sha256=physical_verifier_sha256())
        loaded = load_pending_episode(self.store, self.contract, self.actor, 'attempt2', **kwargs)
        self.assertEqual(tensor_payload_sha256(loaded.turns[0].tensors), original.turns[0].row.tensor_payload_sha256)
        owner = self.store.root/original.turns[0].tensors.ref.relative_path
        with owner.open('ab') as handle:
            handle.write(b'changed original prefix owner')
        with self.assertRaisesRegex(ValueError, 'size/hash'):
            load_pending_episode(self.store, self.contract, self.actor, 'attempt2', **kwargs)


class ScoredTensorOwnerTests(unittest.TestCase):
    def test_real_owner_survives_draft_scored_batch_and_native_reload_with_same_science(self):
        import test_action_credit_campaign as campaign
        from openwebrl.action_credit_campaign import draft_independent_update, save_scored_pending_update
        from openwebrl.action_credit_driver import _stream_draft
        from openwebrl.action_credit_native_controller import _trusted_torch, load_scored_pending
        fixture = campaign.CampaignTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        episodes = []
        for index, item in enumerate(fixture.plan['draws']):
            episode = fixture.complete(item, reward=float(index == 0))
            # This CPU fixture originally creates an already-terminal empty-MM
            # episode. Recreate only its owner-write interval, then restore the
            # original ledger verbatim; no production authority is fabricated.
            attempt = episode.request.attempt_id
            original = fixture.store.snapshot()['attempts'][attempt]['status']
            with fixture.store._transaction() as state:
                state['attempts'][attempt]['status'] = 'active'
            mapping = create_tensor_blob(fixture.store, attempt, 0,
                {'pixel_values': torch.arange(48, dtype=torch.float32).reshape(3, 4, 4)})
            with fixture.store._transaction() as state:
                state['attempts'][attempt]['status'] = original
            turn = episode.turns[0]
            media = (digest('CPU original image'),)
            receipt = replace(turn.replay_receipt,
                context_sha256=digest(dict(input_ids=turn.edge.input_ids, media_sha256=media)))
            edge = replace(turn.edge, media_sha256=media, replay_evidence_sha256=receipt.sha256)
            row = replace(turn.row, media_sha256=media, tensor_payload_sha256=tensor_payload_sha256(mapping))
            episodes.append(replace(episode, turns=(replace(turn, edge=edge, row=row,
                replay_receipt=receipt, tensors=mapping),)))
        draft = draft_independent_update(fixture.store, fixture.contract, fixture.plan, episodes)
        # This owner-serialization fixture has synthetic committed episodes;
        # make its already-frozen plan and artifact pointers explicit. It does
        # not claim the physical acceptance exercised by the online tests.
        from openwebrl.action_credit_native_controller import _write
        refs={episode.request.attempt_id:_write(fixture.root/('cpu-episode-'+str(i)+'.pt'),
            dict(cpu_fixture_not_physical=True,episode=episode),torch_payload=True)
            for i,episode in enumerate(episodes)}
        with fixture.store._transaction() as state:
            state.setdefault('online_collection_plans',{})['0']=fixture.plan
            state.setdefault('pending_episodes',{}).update(refs)
        draft_pointer = _stream_draft(fixture.store, fixture.contract, draft, fixture.config)
        self.assertEqual(tensor_storage_bytes(Path(draft_pointer['path']).read_bytes()), 0)
        restored = _trusted_torch(draft_pointer)
        bind_tensor_inventory(fixture.store, restored['draft']['tensor_payloads'].values(), restored['tensor_owners'])
        pointer, prepared = save_scored_pending_update(fixture.store, fixture.config, restored['draft'],
            fixture.scores(restored['draft']), scorer_source_sha256=campaign.sha('CPU scorer'),
            window_config=fixture.window_config, optimizer_rng_artifact=fixture.rng_artifact)
        self.assertEqual(tensor_storage_bytes(Path(pointer['path']).read_bytes()), 0)
        new_pointer, payload, reloaded = load_scored_pending(fixture.store, 0)
        self.assertEqual(new_pointer, pointer)
        self.assertEqual(reloaded.plan, prepared.plan)
        self.assertEqual(payload['optimizer_rng_artifact'], pointer['optimizer_rng_artifact'])
        for edge in payload['graph'].edges:
            payload['rows'][edge.edge_id].validate(edge, payload['graph'].contract,
                payload['tensor_payloads'][edge.edge_id])
        # Full inventory validation covers every retained owner before selection.
        last = episodes[-1].turns[0].tensors
        with (fixture.store.root/last.ref.relative_path).open('ab') as handle:
            handle.write(b'changed unselected tensor owner')
        with self.assertRaisesRegex(ValueError, 'size/hash'):
            load_scored_pending(fixture.store, 0)


if __name__ == '__main__':
    unittest.main()
