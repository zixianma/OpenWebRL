import json
from pathlib import Path
import tempfile
import unittest

from scripts.summarize_arm_policy_behavior import behavior


class PolicyBehaviorTest(unittest.TestCase):
    def test_reports_collapse_indicators(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            results = root / "results"
            results.mkdir()
            rows = [
                {
                    "valid": True,
                    "reward": 1,
                    "total_steps": 2,
                    "metadata": {"messages": [
                        {"role": "assistant", "content": '<tool_call>{"name":"scroll"}</tool_call>'},
                        {"role": "assistant", "content": '<tool_call>{"name":"done"}</tool_call>'},
                    ]},
                },
                {
                    "valid": True,
                    "reward": 0,
                    "total_steps": 30,
                    "metadata": {"messages": [
                        {"role": "assistant", "content": '<tool_call>{"name":"scroll"}</tool_call>'},
                        {"role": "assistant", "content": '<tool_call>{"name":"scroll"}</tool_call>'},
                    ]},
                },
                {"valid": False, "reward": None},
            ]
            for index, row in enumerate(rows):
                (results / f"{index}.json").write_text(json.dumps(row))
            report = behavior(root)
            self.assertEqual((report["attempted"], report["valid"], report["successes"]), (3, 2, 1))
            self.assertEqual(report["termination_rate"], 0.5)
            self.assertEqual(report["step_cap_30_rate"], 0.5)
            self.assertEqual(report["scroll_call_fraction"], 0.75)
            self.assertEqual(report["repeated_primary_action_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
