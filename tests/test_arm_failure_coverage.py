"""Deferred failure labels: state fidelity, independent knobs and real PPO scaling."""
import asyncio
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from test_arm_failure_bonus import failed_group
from test_arm_turn_bonus_pipeline import fixture
from test_arm_turn_bonus import PROMPT, output
from openwebrl import arm_failure_coverage as coverage
from openwebrl import arm_failure_additive as add
from openwebrl import arm_failure_aux as aux
from openwebrl import arm_failure_recipe as recipe
from openwebrl import arm_turn_bonus as bonus


class Coverage(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.rows, self.current, self.args, _, _ = fixture(self.temp.name, training=True)
        self.config = self.current['config']
        self.config.update(admit_all_failure_groups=True, additive_failure_groups=True,
            failure_group_cap=8, failure_loss_coefficient=1/6, rollout_id=0,
            failure_ablation='coverage', failure_turn_budget=4, failure_beta=.5,
            selector_endpoint='unused')
        self.args.sglang_router_ip, self.args.sglang_router_port = 'localhost', 1
        self.patch = patch.object(bonus, '_STATE', self.current)
        self.patch.start(); self.addCleanup(self.patch.stop)

    def capture(self, trajectory, reverse=False):
        reservoir = coverage.TurnReservoir(self.config, trajectory[0].metadata['trajectory_id'])
        records = {}
        for s in reversed(trajectory) if reverse else trajectory:
            s.response = output(0)[0]
            identity = (self.config['seed'], self.config.get('sampling_policy_id', self.config['policy_id']),
                s.metadata['task_id'], str(s.metadata['trajectory_id']), s.metadata['turn_index'])
            s.metadata['arm_sampling_identity'] = list(identity)
            record = dict(context=dict(parent_sample_index=s.index, group_index=s.group_index))
            records[s.index] = record
            state = dict(prompt=PROMPT, params=dict(temperature=.8), images=['encoded-image'],
                observation=dict(screenshot=b'image', active_tab_url='https://example.test'),
                history=['old reasoning/action'], task='training task', original=output(0))
            reservoir.capture(identity, record, state, 30)
            state['history'].append('mutation after browser action')
        selected = set(reservoir.items)
        reservoir.persist()
        reservoir.persist()  # completion is idempotent
        for s in trajectory:
            if 'failure_coverage_state' in records[s.index]:
                s.metadata['arm_failure_coverage_state'] = records[s.index]['failure_coverage_state']
        return selected

    def group(self, labels=False):
        group = failed_group(self.rows)
        for trajectory in group:
            if not labels:
                for s in trajectory: s.metadata['arm_turn_bonus'] = {}
            self.capture(trajectory)
        return group

    def test_separate_knobs_and_legacy_defaults(self):
        self.assertEqual(recipe.failure_beta({}), .5)
        self.assertEqual(recipe.coverage_budget({}), 0)
        for variant, beta, count in [('control', .5, 0), ('coverage', .5, 4), ('weight', 1., 0)]:
            c = dict(self.config, failure_ablation=variant, failure_beta=beta, failure_turn_budget=count)
            recipe.validate_recipe(c)
            for key, value in [('beta', 1), ('scored_fraction', .4), ('candidate_gate', 'min2'),
                               ('failure_beta', 1. if beta == .5 else .5)]:
                with self.assertRaises(ValueError): recipe.validate_recipe(dict(c, **{key:value}))

    def test_reservoir_order_and_exact_immutable_state(self):
        trajectory = failed_group(self.rows)[0]
        selected = self.capture(trajectory)
        self.assertEqual(selected, self.capture(deepcopy(trajectory), reverse=True))
        self.assertEqual(len(selected), 4)
        for s in trajectory:
            if not s.metadata.get('arm_failure_coverage_state'): continue
            body = coverage.load_state(s, self.config)
            self.assertEqual(body['state']['history'], ['old reasoning/action'])
            self.assertEqual(body['state']['observation']['screenshot'], b'image')
            with self.assertRaisesRegex(ValueError, 'provenance'):
                coverage.load_state(s, dict(self.config, checkpoint='different actor'))
        short = deepcopy(trajectory[:2])
        for s in short: s.metadata.pop('arm_failure_coverage_state', None)
        self.assertEqual(len(self.capture(short)), 2)

    def test_unlabeled_failures_retained_without_consuming_mixed_quota(self):
        group = self.group()
        result = add.filter_groups(self.args, group)
        self.assertFalse(result.keep)
        self.assertEqual(result.reason, 'zero_std_0.0')
        self.assertEqual(len(self.current['failure_additive_pool']), 1)
        self.assertNotIn('failure_coverage_control', self.current)
        mixed = [self.rows[i*10:(i+1)*10] for i in range(5)]
        before = deepcopy(mixed)
        self.assertTrue(add.filter_groups(self.args, mixed).keep)
        self.assertEqual([s.metadata for t in mixed for s in t], [s.metadata for t in before for s in t])
        invalid = self.group(); invalid[0][-1].metadata['reward']['judge_timeout'] = True
        add.filter_groups(self.args, invalid)
        self.assertEqual(self.current['failure_coverage_rejections'], {'judge_timeout': 1})

    def test_finalize_real_candidate_generation_and_reuse_rejections(self):
        group = self.group(labels=True)
        add.filter_groups(self.args, group)
        chosen = next(s for t in group for s in t if s.metadata.get('arm_failure_coverage_state'))
        self.current['records'].append(dict(policy_id='p', trajectory_id=str(chosen.metadata['trajectory_id']),
            turn=chosen.metadata['turn_index'], context=dict(parent_sample_index=chosen.index, group_index=0),
            candidate_requests=4, candidate_response_tokens=4, selector_requests=0,
            eligible=False, sampled=True, executed_index=0, reason='label_unavailable'))
        calls, votes = [], []
        async def infer(url, prompt, params, images, timeout_secs):
            calls.append((prompt, params, images)); return output(len(calls) % 4 + 1)
        async def request(endpoint, payload, timeout, unused):
            votes.append(payload); return dict(raw=json.dumps({'selection':1}))
        original_init = bonus.ShadowSelector.__init__
        def init(selector, config, trajectory_id):
            original_init(selector, config, trajectory_id, request=request)
        with patch.object(bonus.ShadowSelector, '__init__', init):
            asyncio.run(coverage.finalize(self.args, 0, infer=infer))
        report = self.current['failure_coverage_report']
        self.assertEqual(report['selected_turns'], 20)
        self.assertEqual(report['reused'], 1)
        self.assertEqual(report['new_candidate_requests'], 76)
        self.assertEqual(len(calls), 76)
        self.assertEqual(len(votes), 19)
        self.assertEqual(report['four_turn_coverage']['labels'], 19)
        self.assertEqual(report['historical_admission_control']['labels'], 5)
        self.assertEqual(report['four_turn_coverage']['turns'], 50)
        self.assertTrue(all(payload['screenshot'] == 'aW1hZ2U=' for payload in votes))
        for t in group:
            for s in t:
                self.assertTrue(s.metadata['arm_all_failure_group'])
                if not s.metadata.get('arm_failure_coverage_state'):
                    self.assertEqual(bonus.unit_bonus(s, 'p'), 0)
        with self.assertRaisesRegex(ValueError, 'Repeated'):
            asyncio.run(coverage.finalize(self.args, 0, infer=infer))

    def test_tamper_or_missing_state_aborts_before_requests(self):
        group = self.group(); add.filter_groups(self.args, group)
        selected = next(s for t in group for s in t if s.metadata.get('arm_failure_coverage_state'))
        path = Path(selected.metadata['arm_failure_coverage_state']['path'])
        path.write_bytes(path.read_bytes() + b' ')
        async def forbidden(*args, **kwargs): raise AssertionError('No request before validation')
        with self.assertRaisesRegex(ValueError, 'changed after collection'):
            asyncio.run(coverage.finalize(self.args, 0, infer=forbidden))
        self.assertFalse(self.current.get('failure_coverage_complete'))
        selected.metadata.pop('arm_failure_coverage_state')
        with self.assertRaisesRegex(ValueError, 'Missing uniformly'):
            asyncio.run(coverage.finalize(self.args, 0, infer=forbidden))

    def test_empty_pool_is_valid_zero_signal_and_serialization_requires_completion(self):
        with self.assertRaisesRegex(ValueError, 'must finish'):
            add.write_auxiliary(self.args, self.rows, self.current, {})
        async def forbidden(*args, **kwargs): raise AssertionError('Empty pool has no requests')
        asyncio.run(coverage.finalize(self.args, 0, infer=forbidden))
        self.assertEqual(self.current['failure_coverage_report']['selected_turns'], 0)
        self.assertEqual(self.current['failure_coverage_report']['four_turn_coverage']['coefficient'], 0)

    def test_failure_beta_doubles_actual_clipped_loss_and_gradients(self):
        module = ModuleType('slime.backends.megatron_utils.loss')
        module.get_log_probs_and_entropy = lambda logits, **kw: (None, {'log_probs':[logits]})
        module.policy_loss_function = lambda *a: None
        args = SimpleNamespace(eps_clip=.2, eps_clip_high=.28)
        def evaluate(advantage, scale, value):
            batch = dict(arm_source=['arm_failure'], unconcat_tokens=[], total_lengths=[],
                response_lengths=[], arm_old_log_probs=[torch.zeros(2)],
                arm_advantage=[advantage], arm_scale=[scale])
            logits = torch.full((2,), value, requires_grad=True)
            loss, _ = aux.loss(args, batch, logits, torch.mean)
            loss.backward(); return loss.detach(), logits.grad
        with patch.dict(sys.modules, {'slime.backends.megatron_utils.loss':module}):
            for unit in (.8, -.2):
                for value in (-1., 0., 1.):
                    control = evaluate(.5*unit, 2., value)
                    beta = evaluate(unit, 2., value)
                    scale = evaluate(.5*unit, 4., value)
                    for a, b, c in zip(control, beta, scale):
                        torch.testing.assert_close(b, 2*a)
                        torch.testing.assert_close(b, c)

    def test_weight_manifest_preserves_labels_population_and_mixed_samples(self):
        self.config.update(failure_ablation='weight', failure_turn_budget=0, failure_beta=1.)
        group = failed_group(self.rows)
        add.filter_groups(self.args, group)
        mixed = []
        for i in range(48):
            sample = deepcopy(self.rows[0]); sample.group_index = i
            mixed.extend([sample]*16)
        original = [deepcopy(s.metadata) for s in mixed]
        add.write_auxiliary(self.args, mixed, self.current, {})
        manifest = json.loads((Path(self.temp.name)/'failure_auxiliary.json').read_text())
        self.assertEqual(manifest['beta'], 1.)
        self.assertEqual(manifest['coefficient'], 1/48)
        self.assertEqual(manifest['total_failure_rows'], 50)
        self.assertEqual(len(manifest['records']), 5)
        self.assertEqual([s.metadata for s in mixed], original)
        for row in manifest['records']:
            self.assertIn(row['advantage'], (.8, -.2))
            aux.validate_auxiliary_advantage(row, self.config)

    def test_forty_percent_is_nested_bernoulli_and_preserves_admission(self):
        self.config.update(failure_turn_budget=0, failure_scored_fraction=.4)
        group = self.group(labels=False)
        self.assertFalse(add.filter_groups(self.args, group).keep)
        self.assertFalse(self.current.get('failure_additive_pool'))
        ids = [(42, 'p', 'task', str(i), t) for i in range(100) for t in range(15)]
        low = {i for i in ids if coverage.sampled_at(i,.2)}
        high = {i for i in ids if coverage.sampled_at(i,.4)}
        self.assertTrue(low < high)
        self.assertTrue(.35 < len(high)/len(ids) < .45)
        selected = [s for t in group for s in t if s.metadata.get('arm_failure_coverage_state')]
        for t in group:
            for s in t:
                identity = (42,'p',s.metadata['task_id'],str(s.metadata['trajectory_id']),s.metadata['turn_index'])
                self.assertEqual(s in selected, coverage.sampled_at(identity,.4))
        # Admit via an existing q=.2 usable label, independent of the new labels.
        existing = next(s for s in selected if coverage.sampled_at((42,'p',s.metadata['task_id'],str(s.metadata['trajectory_id']),s.metadata['turn_index']),.2))
        label = dict(policy_id='p',trajectory_id=str(existing.metadata['trajectory_id']),
            turn=existing.metadata['turn_index'],context=dict(parent_sample_index=existing.index,group_index=0),
            sampled=True,eligible=True,executed_index=0,selected_index=0,unit_bonus=.8,
            reason='admitted',candidate_requests=4,candidate_response_tokens=4,selector_requests=1)
        existing.metadata['arm_turn_bonus'] = dict(label)
        self.current['records'].append(label)
        self.assertFalse(add.filter_groups(self.args,group).keep)
        self.assertEqual(len(self.current['failure_additive_pool']),1)
        async def fake_label(selector,infer,url,state,identity,record,timeout):
            record.update(reason='admitted',eligible=True,selected_index=1,unit_bonus=-.2)
        with patch.object(bonus.ShadowSelector,'_label',fake_label):
            asyncio.run(coverage.finalize(self.args,0,infer=lambda:None))
        report=self.current['failure_coverage_report']
        self.assertEqual(report['selected_turns'],len(selected))
        self.assertEqual(report['reused'],1)
        self.assertEqual(report['failure_coverage']['labels'],len(selected))
        self.assertEqual(report['failure_q'],.4)
        self.assertEqual(bonus.unit_bonus(existing,'p'),.8)
        self.assertIsNone(report['four_turn_coverage'])

    def test_dense_serial_accumulation_keeps_loss_normalization(self):
        with self.assertRaises(ValueError): add.window_schedule([{}]*240,600,[256]*5,256,1/6)
        schedule=add.window_schedule([{}]*240,600,[256]*5,256,1/6,128)
        self.assertEqual(sum(len(w) for w in schedule),240)
        self.assertAlmostEqual(schedule[0][0][1]/256/5,(1/6)/600)

    def test_deferred_sampling_preserves_request_identity_across_browser_alias(self):
        self.config.update(failure_turn_budget=0, failure_scored_fraction=.4,
                           sampling_policy_id='shared-from-zero:rollout0')
        group = failed_group(self.rows)
        for trajectory in group:
            for s in trajectory: s.metadata['task_id'] = 'webvoyager/61350'
            self.capture(trajectory)
            for s in trajectory: s.metadata['task_id'] = '61350'
        chosen = [s for t in group for s in t if s.metadata.get('arm_failure_coverage_state')]
        self.assertTrue(chosen)
        # Prove that recomputing the seed from the resolved ID changes membership.
        self.assertTrue(any(coverage.sampled_at(s.metadata['arm_sampling_identity'], .4) !=
            coverage.sampled_at([self.config['seed'], self.config['sampling_policy_id'], '61350',
                str(s.metadata['trajectory_id']), s.metadata['turn_index']], .4)
            for t in group for s in t))
        self.assertFalse(add.filter_groups(self.args, group).keep)
        self.assertEqual(len(self.current['failure_additive_pool']), 1)
        async def fake_label(selector, infer, url, state, identity, record, timeout):
            self.assertEqual(identity[2], 'webvoyager/61350')
            record.update(reason='admitted', eligible=True, selected_index=1, unit_bonus=-.2)
        with patch.object(bonus.ShadowSelector, '_label', fake_label):
            asyncio.run(coverage.finalize(self.args, 0, infer=lambda: None))
        self.assertEqual(self.current['failure_coverage_report']['selected_turns'], len(chosen))
        tampered = deepcopy(chosen[0])
        tampered.metadata['arm_sampling_identity'][3] = 'different-trajectory'
        with self.assertRaisesRegex(ValueError, 'sampling identity'):
            coverage.load_state(tampered, self.config)

    def test_generate_attaches_exact_sampling_identity_to_every_turn(self):
        self.config.update(failure_turn_budget=0, failure_scored_fraction=.4,
                           sampling_policy_id='shared-from-zero:rollout0')
        trajectory = failed_group(self.rows)[0]
        parent = deepcopy(trajectory[0])
        parent.index = trajectory[0].metadata['trajectory_id']
        parent.metadata['task_id'] = 'webvoyager/61350'
        browser = ModuleType('openwebrl.generate_browser')
        async def infer(*args, **kwargs): return output(0)
        async def fake_generate(args, sample, params):
            selector = args.browser_action_selector
            for s in trajectory:
                s.metadata['task_id'] = '61350'
                s.response = output(0)[0]
                selector.set_training_context(dict(parent_sample_index=s.index, group_index=s.group_index))
                await selector(infer=infer, url='unused', input_text=PROMPT,
                    sampling_params=params, images=['encoded-image'],
                    observation=dict(screenshot=b'image', active_tab_url='https://example.test'),
                    history=[], task='training task', task_id=sample.metadata['task_id'],
                    turn=s.metadata['turn_index'], timeout=30)
            return trajectory
        browser.generate_turn_sample = fake_generate
        async def fake_label(selector, infer, url, state, identity, record, timeout):
            record.update(reason='label_unavailable')
        with patch.dict(sys.modules, {'openwebrl.generate_browser': browser}), \
                patch.object(bonus.ShadowSelector, '_label', fake_label):
            turns = asyncio.run(bonus.generate(self.args, parent, dict(temperature=.8)))
        for s in turns:
            identity = coverage.sampling_identity(s, self.config)
            self.assertEqual(identity, [42, self.config['sampling_policy_id'], 'webvoyager/61350',
                str(parent.index), s.metadata['turn_index']])
            self.assertEqual(bool(s.metadata.get('arm_failure_coverage_state')),
                             coverage.sampled_at(identity, .4))

    def test_cancellation_cannot_mark_deferred_phase_complete(self):
        add.filter_groups(self.args, self.group())
        async def cancel(*args, **kwargs): raise asyncio.CancelledError()
        with patch.object(bonus.ShadowSelector, '_label', cancel):
            with self.assertRaises(asyncio.CancelledError):
                asyncio.run(coverage.finalize(self.args, 0, infer=cancel))
        self.assertFalse(self.current.get('failure_coverage_complete'))


if __name__ == '__main__': unittest.main()
