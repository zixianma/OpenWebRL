import asyncio
import copy
import math
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

from openwebrl.arm_rl import (LossRow, admit_rescue, calibration, component_losses,
    group_advantages, local_advantages, optimizer_windows, outcome_reward,
    rescue_eligibility, stable_seed)
from openwebrl.arm_rl_collection import LocalPreferenceSelector
from openwebrl.eval_benchmark import configure, SAMPLING


class Objectives(unittest.TestCase):
    def row(self, values=(-1., -1.), adv=1., mask=(1, 1), old=None):
        x = torch.tensor(values, dtype=torch.float64, requires_grad=True)
        return LossRow(x, mask, x.detach().clone() if old is None else torch.tensor(old), adv,
                       'frozen_actor_teacher_forced')

    def test_reward_and_std_match_reference_recipe(self):
        self.assertEqual(outcome_reward(format_error_failed=True, format_valid=True, judge_success=True), -1)
        self.assertEqual(outcome_reward(format_error_failed=False, format_valid=False, judge_success=True), 0)
        self.assertEqual(outcome_reward(format_error_failed=False, format_valid=True, judge_success=True), 1)
        self.assertIsNone(outcome_reward(format_error_failed=False, format_valid=True, judge_success=True, valid=False))
        a = group_advantages([1., 0., 0., 0., 0.])
        self.assertAlmostEqual(a[0], .8 / (math.sqrt(.2) + 1e-6))
        self.assertEqual(group_advantages([0.] * 5), [0.] * 5)
        with self.assertRaises(ValueError): group_advantages([0., None])

    def test_zero_aux_recovers_exact_loss_and_gradient_even_with_aux_records(self):
        a, b = self.row(), self.row()
        x = component_losses([a])['total']
        y = component_losses([b], demo_rows=[object()], local_groups=[[object()]])['total']
        x.backward(); y.backward()
        self.assertTrue(torch.equal(x, y))
        self.assertTrue(torch.equal(a.logp.grad, b.logp.grad))

    def test_positive_and_negative_clipping(self):
        winner = self.row(values=(math.log(1.5),), old=(0.,), mask=(1,))
        loser = self.row(values=(math.log(.5),), old=(0.,), adv=-1., mask=(1,))
        loss = component_losses([winner, loser])['total']
        self.assertAlmostEqual(loss.item(), (-1.28 + .8) / 2)
        loss.backward()
        self.assertEqual(winner.logp.grad.item(), 0)
        self.assertEqual(loser.logp.grad.item(), 0)

    def test_reasoning_mask_and_gradient_signs(self):
        winner, loser = self.row(mask=(0, 1)), self.row(mask=(0, 1), adv=-1.)
        losses = component_losses([self.row()], local_groups=[[winner, loser]], lam=.3)
        losses['total'].backward()
        self.assertEqual(winner.logp.grad[0], 0)
        self.assertEqual(loser.logp.grad[0], 0)
        self.assertLess(winner.logp.grad[1], 0)
        self.assertGreater(loser.logp.grad[1], 0)

    def test_aux_normalization_independent_of_candidate_and_demo_counts(self):
        base, demo = self.row(), self.row(values=(-2., -2.))
        r = component_losses([base], demo_rows=[demo], eta=.2)
        repeated = component_losses([base], demo_rows=[demo] * 32, eta=.2)
        self.assertEqual(r['total'].item(), repeated['total'].item())
        pair = [self.row(values=(-.8, -.8), old=(-1., -1.)), self.row(adv=-1)]
        five = [pair[0], *[self.row(adv=-.25) for _ in range(4)]]
        a = component_losses([base], local_groups=[pair], lam=.1)
        b = component_losses([base], local_groups=[five], lam=.1)
        self.assertAlmostEqual(a['local'].item(), b['local'].item())

    def test_action_duplicate_classes(self):
        self.assertEqual(local_advantages(['a', 'a', 'b', 'b', 'c'], 1), [0, 1, -.5, 0, -.5])
        self.assertIsNone(local_advantages(['a'] * 5, 2))
        with self.assertRaises(ValueError): local_advantages(['a', ''], 0)
        with self.assertRaises(ValueError): local_advantages(['a', 'b'], None)

    def test_synthetic_old_logps_rejected(self):
        w, l = self.row(), self.row(adv=-1.)
        w.old_logp_provenance = 'generator_appended_zero'
        with self.assertRaisesRegex(ValueError, 'Recompute old'):
            component_losses([self.row()], local_groups=[[w, l]], lam=.1)

    def test_bad_masks_and_nonfinite_logps_fail(self):
        for row in [self.row(mask=(0, 0)), self.row(mask=(1, 2)), self.row(values=(float('nan'), -1.))]:
            with self.assertRaises(ValueError): component_losses([row])

    def test_no_auxiliary_optimizer_steps(self):
        a = optimizer_windows(list(range(600)))
        b = optimizer_windows(list(range(600)), demos=list(range(100)))
        self.assertEqual([w['outcome'] for w in a], [w['outcome'] for w in b])
        self.assertEqual([len(w['outcome']) for w in b], [256, 256, 88])
        self.assertEqual([len(w['demo']) for w in b], [32, 32, 32])
        self.assertEqual(optimizer_windows([], demos=[1]), [])

    def test_calibration_and_no_zero_norm_shortcuts(self):
        r = calibration(torch.tensor([3., 4.]), torch.tensor([0., 2.]))
        self.assertAlmostEqual(r['coefficient'], .5)
        self.assertAlmostEqual(r['cosine'], .8)
        with self.assertRaises(ValueError): calibration(torch.zeros(2), torch.ones(2))


class Eligibility(unittest.TestCase):
    def setUp(self):
        self.group = dict(task_id='train', intent='Find the price', policy_id='p', judge_id='j',
            trajectories=[dict(trajectory_id=str(i), policy_id='p', judge_id='j', valid=True, reward=0.) for i in range(5)])
        self.identity = dict(policy_id='p', judge_id='j')

    def test_five_valid_failures(self):
        self.assertIsNone(rescue_eligibility(self.group, **self.identity))
        for key, value in [('valid', False), ('reward', 1.), ('reward', -1.), ('policy_id', 'old')]:
            g = copy.deepcopy(self.group); g['trajectories'][0][key] = value
            self.assertIsNotNone(rescue_eligibility(g, **self.identity))

    def test_no_eval_leakage_or_unversioned_archive(self):
        self.assertEqual(rescue_eligibility(self.group, **self.identity, excluded_task_ids=['train']), 'evaluation_task')
        self.assertEqual(rescue_eligibility(self.group, **self.identity, excluded_intents=['find the price']), 'missing_or_evaluation_intent')
        g = copy.deepcopy(self.group); g.pop('policy_id')
        self.assertEqual(rescue_eligibility(g, **self.identity), 'policy_mismatch')

    def test_only_executed_successful_teacher_responses(self):
        r = dict(task_id='train', policy_id='p', judge_id='j', valid=True, reward=1.,
                 turns=[dict(executed=True, fallback=None, selected_index=2)])
        self.assertIsNone(admit_rescue(self.group, r, **self.identity))
        r['turns'][0]['fallback'] = 'parse error'
        self.assertEqual(admit_rescue(self.group, r, **self.identity), 'missing_execution_or_selection')

    def test_seed_covers_policy_and_trajectory(self):
        self.assertNotEqual(stable_seed(42, 'p', 'a', 0), stable_seed(42, 'p', 'b', 0))
        self.assertNotEqual(stable_seed(42, 'p', 'a', 0), stable_seed(42, 'q', 'a', 0))


class Collection(unittest.IsolatedAsyncioTestCase):
    async def test_actor_executes_before_selector_finishes_and_state_is_frozen(self):
        finished = asyncio.Event(); records = []; payloads = []
        async def request(endpoint, payload, timeout, connect_timeout):
            payloads.append(payload); await finished.wait(); return dict(raw='{"selection": 2}')
        async def infer(url, text, params, images, timeout_secs):
            return ('reason</think><tool_call>{}</tool_call><|im_end|>\n', [1, 2], [-1., 0.], 'stop')
        async def sink(record): records.append(record)
        selector = LocalPreferenceSelector(policy_id='p', trajectory_id='t', endpoint='unused',
            sink=sink, request=request, scored_fraction=1.)
        obs = dict(screenshot=b'before', active_tab_url='before')
        history = ['before']
        original, meta = await asyncio.wait_for(selector(infer=infer, url='actor', input_text='prefix',
            sampling_params={'temperature': .8}, images=['before'], observation=obs, history=history,
            task='training', task_id='train', turn=0, timeout=1), 1)
        self.assertEqual(meta['executed_index'], 0)
        self.assertEqual(records, [])
        obs['screenshot'] = b'AFTER'; history.append('AFTER')
        finished.set(); await selector.drain()
        self.assertEqual(len(records), 1)
        self.assertEqual(payloads[0]['screenshot'], 'YmVmb3Jl')
        self.assertEqual(len(payloads[0]['history']), 1)
        self.assertEqual(records[0]['selected_index'], records[0]['permutation'][1])
        self.assertIsNone(records[0]['terminal_rewards'])
        self.assertEqual(records[0]['outputs'][0], original)

    async def test_selector_timeout_or_malformed_label_drops_only_auxiliary(self):
        for mode in ['timeout', 'malformed']:
            records = []
            async def request(*args):
                if mode == 'timeout': raise TimeoutError('mock')
                return dict(raw='invalid')
            async def infer(*args, **kwargs): return ('ordinary', [1], [-1.], 'stop')
            async def sink(record): records.append(record)
            s = LocalPreferenceSelector(policy_id='p', trajectory_id='t', endpoint='unused',
                sink=sink, request=request, scored_fraction=1.)
            original, _ = await s(infer=infer, url='', input_text='', sampling_params={}, images=[],
                observation={'screenshot': b'x'}, history=[], task='t', task_id='t', turn=0, timeout=1)
            await s.drain()
            self.assertEqual(original[0], 'ordinary')
            self.assertEqual(records, [])
            self.assertEqual(s.audit[-1]['reason'], 'auxiliary_dropped')


class Benchmark(unittest.TestCase):
    def test_partial_results_keep_success_and_missing_verdict_distinct(self):
        import json
        import tempfile
        from unittest.mock import patch
        from openwebrl.eval_benchmark import save_completed_task
        with tempfile.TemporaryDirectory() as directory, patch.dict(
                'os.environ', OPENWEBRL_BENCHMARK_RESULTS_DIR=directory):
            for score in [1., None]:
                turn = SimpleNamespace(
                    status='completed', reward=score if score is not None else 0.,
                    remove_sample=score is None, response='reasoning and action',
                    metadata={'is_last_turn': True, 'task_id': str(score),
                              'reward': {'judge': score, 'combined': score},
                              'judge_invalid': score is None})
                save_completed_task(SimpleNamespace(judge_api_model='o4-mini'), [turn])
            records = [json.loads(p.read_text()) for p in Path(directory).glob('*.json')]
            self.assertEqual(len(records), 2)
            self.assertFalse(list(Path(directory).glob('*.tmp')))
            success = next(r for r in records if not r['remove_sample'])
            invalid = next(r for r in records if r['remove_sample'])
            self.assertEqual(success['reward']['judge'], 1.)
            self.assertIsNone(invalid['reward']['judge'])
            self.assertEqual(invalid['transport_reward'], 0.)
            self.assertEqual(success['responses'], ['reasoning and action'])

    def test_missing_agenttrek_verdict_cannot_trigger_generic_rejudging(self):
        from types import ModuleType
        from unittest.mock import patch
        import sys
        from openwebrl.eval_benchmark import generate
        turn=SimpleNamespace(status='completed',reward=None,metadata={},remove_sample=False)
        async def rollout(*args):return [turn]
        async def reward(*args):return [None]
        modules={name:ModuleType(name) for name in ['openwebrl.generate_browser',
            'openwebrl.eval.reward_online_mind2web','slime.utils.types']}
        modules['openwebrl.generate_browser'].generate_turn_sample=rollout
        modules['openwebrl.eval.reward_online_mind2web'].reward_func=reward
        modules['slime.utils.types'].Sample=SimpleNamespace(Status=SimpleNamespace(ABORTED='aborted'))
        with patch.dict(sys.modules,modules),patch.dict('os.environ',{},clear=True):
            result=asyncio.run(generate(SimpleNamespace(),None,{},evaluation=True))
        self.assertEqual(result[0].reward,0.)
        self.assertTrue(result[0].remove_sample)
        self.assertIsNone(result[0].metadata['benchmark_judge_reward'])

    def test_paper_protocol_is_explicit_and_does_not_mutate_training(self):
        args = SimpleNamespace(judge_api_model='gpt-4.1', max_steps=15)
        result, sampling = configure(args, dict(temperature=0, top_k=1, max_new_tokens=1024))
        self.assertEqual(sampling, SAMPLING)
        self.assertEqual((result.judge_api_model, result.max_steps), ('o4-mini', 30))
        self.assertEqual((args.judge_api_model, args.max_steps), ('gpt-4.1', 15))
        self.assertIn('GPT-4.1', Path('openwebrl/online_mind2web_monitor.yaml').read_text())

    def test_benchmark_disallows_actor_selector(self):
        with self.assertRaises(ValueError): configure(SimpleNamespace(browser_action_selector=object()), {})


if __name__ == '__main__':
    unittest.main()
