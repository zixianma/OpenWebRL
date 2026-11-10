"""The prospective stopping rule must match replay and avoid unused actor calls."""
import asyncio
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from openwebrl.adaptive_selection import AdaptiveScorePolicy, select_adaptively


class AdaptiveSelection(unittest.TestCase):
    def test_replay_and_live_callbacks_agree(self):
        values = [-7.0, -6.0, -4.0, -3.0, -2.0]
        policy = AdaptiveScorePolicy(-5.0)
        calls = []

        async def propose(index):
            calls.append(index)
            return {'index': index}

        async def score(index, proposal):
            self.assertEqual(index, proposal['index'])
            return values[index]

        result = asyncio.run(select_adaptively(propose, score, policy))
        self.assertEqual((result['generated_candidates'], result['selected_index']), policy.replay(values))
        self.assertEqual(calls, [0, 1, 2])
        self.assertEqual(result['scores'], values[:3])

    def test_falls_back_to_five_and_ties_choose_first(self):
        policy = AdaptiveScorePolicy(10.0)
        self.assertEqual(policy.replay([1., 2., 2., 0., 1.]), (5, 1))

    def test_rejects_nonfinite_score_before_next_proposal(self):
        policy = AdaptiveScorePolicy(0.0)
        calls = []

        async def propose(index):
            calls.append(index)
            return index

        async def score(index, proposal):
            return float('nan')

        with self.assertRaisesRegex(ValueError, 'nonfinite'):
            asyncio.run(select_adaptively(propose, score, policy))
        self.assertEqual(calls, [0])

    def test_frozen_policy_rejects_another_checkpoint(self):
        with tempfile.TemporaryDirectory() as root:
            head = Path(root) / 'head.pt'
            head.write_bytes(b'original-head')
            policy_path = Path(root) / 'policy.json'
            policy_path.write_text(json.dumps({
                'revision': 'adaptive-selection-score-threshold-v1',
                'head_sha256': hashlib.sha256(head.read_bytes()).hexdigest(),
                'threshold': -5.0, 'min_candidates': 1, 'max_candidates': 5}))
            self.assertEqual(AdaptiveScorePolicy.from_file(policy_path, head).threshold, -5.0)
            head.write_bytes(b'another-head')
            with self.assertRaisesRegex(ValueError, 'identity'):
                AdaptiveScorePolicy.from_file(policy_path, head)


if __name__ == '__main__':
    unittest.main()
