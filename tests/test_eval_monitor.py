import asyncio
import gc
import os
import tempfile
import weakref
from pathlib import Path
import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import patch
from openwebrl.eval_monitor import generate


class EvalMonitorTest(unittest.TestCase):
    def test_eval_overrides_are_isolated_and_rewards_propagate(self):
        args = SimpleNamespace(max_steps=15, inference_step_timeout_secs=30, judge_timeout_secs=60)
        turns = [SimpleNamespace(status='completed', reward=None) for _ in range(2)]
        async def rollout(local, sample, sampling):
            self.assertEqual((local.max_steps, local.inference_step_timeout_secs), (30, 120))
            self.assertEqual(sampling['max_new_tokens'], 4096)
            return turns
        async def reward(local, samples):
            self.assertEqual(local.judge_timeout_secs, 120)
            return [1., 1.]
        modules = {}
        for name in ['openwebrl.generate_browser', 'openwebrl.reward_browser', 'slime.utils.types']:
            modules[name] = ModuleType(name)
        modules['openwebrl.generate_browser'].generate_turn_sample = rollout
        modules['openwebrl.reward_browser'].reward_func = reward
        modules['slime.utils.types'].Sample = SimpleNamespace(Status=SimpleNamespace(ABORTED='aborted'))
        with patch.dict(sys.modules, modules):
            result = asyncio.run(generate(args, None, {'max_new_tokens':4096}, evaluation=True))
        self.assertEqual([s.reward for s in result], [1.,1.])
        self.assertEqual((args.max_steps, args.judge_timeout_secs), (15,60))

    def test_refuses_training_use(self):
        with self.assertRaises(ValueError):
            asyncio.run(generate(None, None, {}))

class EvalImageStorageTest(unittest.TestCase):
    def test_completed_and_aborted_trajectories_release_images_losslessly(self):
        import torch
        for status in ('completed', 'aborted'):
            with self.subTest(status=status):
                directory = tempfile.mkdtemp(prefix='openwebrl-eval-storage-')
                turn = SimpleNamespace(status=status, reward=None, tokens=[1, 2],
                                       metadata={'task': 'test'},
                                       multimodal_train_inputs={'pixels': torch.arange(24, dtype=torch.bfloat16)})
                original = weakref.ref(turn.multimodal_train_inputs['pixels'])
                turns = [turn]
                reward_calls = []
                async def rollout(local, sample, sampling):
                    return turns
                async def reward(local, samples):
                    reward_calls.append(samples)
                    return [0.75]
                modules = {name: ModuleType(name) for name in (
                    'openwebrl.generate_browser', 'openwebrl.reward_browser', 'slime.utils.types')}
                modules['openwebrl.generate_browser'].generate_turn_sample = rollout
                modules['openwebrl.reward_browser'].reward_func = reward
                modules['slime.utils.types'].Sample = SimpleNamespace(Status=SimpleNamespace(ABORTED='aborted'))
                args = SimpleNamespace(max_steps=15)
                with patch.dict(sys.modules, modules), patch.dict(os.environ, OPENWEBRL_MULTIMODAL_STORAGE_DIR=directory):
                    result = asyncio.run(generate(args, None, {}, evaluation=True))
                gc.collect()
                self.assertIs(result, turns)
                self.assertIs(result[0], turn)
                self.assertIsNone(original())
                self.assertTrue(torch.equal(turn.multimodal_train_inputs['pixels'], torch.arange(24, dtype=torch.bfloat16)))
                self.assertEqual(turn.reward, 0.75 if status == 'completed' else None)
                self.assertEqual(len(reward_calls), int(status == 'completed'))
                self.assertEqual(turn.tokens, [1, 2])
                self.assertEqual(turn.metadata, {'task': 'test'})
                self.assertEqual(args.max_steps, 15)
                self.assertEqual(len(list(Path(directory).glob('*.bin'))), 1)

if __name__ == '__main__':
    unittest.main()
