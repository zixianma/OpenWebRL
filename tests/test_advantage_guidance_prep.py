import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from prepare_advantage_guidance_pilot import action_only_prompt, action_suffix, pooled_rates


class AdvantageGuidancePrepTests(unittest.TestCase):
    def test_duplicate_action_keys_share_valid_only_rate(self):
        keys = ["a", "a", "b", "c", "d"]
        evidence = [
            {"successes": 1, "valid": 1}, {"successes": 0, "valid": 3},
            {"successes": 2, "valid": 2}, {"successes": 0, "valid": 2},
            {"successes": 1, "valid": 2}]
        self.assertEqual(pooled_rates(keys, evidence), [.25, .25, 1., 0., .5])

    def test_shared_empty_reasoning_and_terminated_action(self):
        self.assertEqual(action_only_prompt("prompt<think>\n"), "prompt<think>\n</think>")
        value = "reason</think>\n<tool_call>{}</tool_call><|im_end|>\n"
        self.assertEqual(action_suffix(value), "\n<tool_call>{}</tool_call><|im_end|>")


if __name__ == "__main__":
    unittest.main()
