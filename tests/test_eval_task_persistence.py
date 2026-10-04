import asyncio
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from slime.utils.types import Sample
from openwebrl.eval_monitor import generate


class TaskPersistence(unittest.TestCase):
    def run_case(self, root, status, judge_error=False, empty=False):
        turn = Sample(index=1, prompt='fixture', metadata=dict(task_id='task-one', turn_index=0,
            reward=dict(judge_text='SUCCESS', judge_prompt_variant='action_history')))
        turn.status = status
        turn.multimodal_train_inputs = {'pixels': torch.arange(12, dtype=torch.bfloat16)}
        turns = [] if empty else [turn]
        async def rollout(*_):
            return turns
        async def reward(*_):
            if judge_error:
                raise TimeoutError('fixture judge error')
            return [1.]
        modules = {name: ModuleType(name) for name in ('openwebrl.generate_browser', 'openwebrl.reward_browser')}
        modules['openwebrl.generate_browser'].generate_turn_sample = rollout
        modules['openwebrl.reward_browser'].reward_func = reward
        args = SimpleNamespace(max_steps=15, reward_key=None, judge_api_model='gpt-4.1',
                               judge_prompt_variant='action_history')
        with tempfile.TemporaryDirectory() as mapped, patch.dict(sys.modules, modules), patch.dict(os.environ,
                OPENWEBRL_EVAL_ROLLOUT_DIR=str(root), OPENWEBRL_MULTIMODAL_STORAGE_DIR=mapped):
            if judge_error:
                with self.assertRaises(TimeoutError):
                    asyncio.run(generate(args, turn, {}, evaluation=True))
            else:
                self.assertIs(asyncio.run(generate(args, turn, {}, evaluation=True)), turns)
        # Read the archive after its temporary image mapping has disappeared.
        path = root/(hashlib.sha256(b'task-one').hexdigest()+'.pt')
        payload = torch.load(path, map_location='cpu', weights_only=False)
        if not empty:
            self.assertTrue(torch.equal(payload['turns'][0]['multimodal_train_inputs']['pixels'],
                                        torch.arange(12, dtype=torch.bfloat16)))
        return payload, json.loads(path.with_suffix('.json').read_text())

    def test_success_preserves_pixels_and_terminal_verdict(self):
        with tempfile.TemporaryDirectory() as tmp:
            payload, record = self.run_case(Path(tmp), Sample.Status.COMPLETED)
            self.assertEqual(payload['turns'][0]['reward'], 1.)
            self.assertEqual(record['metrics']['successes'], 1)
            self.assertEqual(record['reward_metadata']['judge_text'], 'SUCCESS')

    def test_aborted_and_empty_attempts_remain_invalid_and_persisted(self):
        for empty in [False, True]:
            with self.subTest(empty=empty), tempfile.TemporaryDirectory() as tmp:
                _, record = self.run_case(Path(tmp), Sample.Status.ABORTED, empty=empty)
                self.assertEqual(record['metrics']['invalid_trajectories'], 1)
                self.assertEqual(record['metrics']['successes'], 0)

    def test_judge_exception_preserves_browser_trajectory_for_rejudging(self):
        with tempfile.TemporaryDirectory() as tmp:
            payload, record = self.run_case(Path(tmp), Sample.Status.COMPLETED, judge_error=True)
            self.assertEqual(payload['error_type'], 'TimeoutError')
            self.assertEqual(len(payload['turns']), 1)
            self.assertEqual(record['metrics']['valid_trajectories'], 0)


if __name__ == '__main__':
    unittest.main()
