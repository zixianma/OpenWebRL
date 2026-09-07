import asyncio
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

if __name__ == '__main__':
    unittest.main()
