"""Ensure diagnostic categories don't conflate syntax and action selection."""
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from audit_arm_serial_failures import diagnose
from openwebrl.serial_actions import serialize


class SerialFailureAuditTest(unittest.TestCase):
    def setUp(self):
        self.tools = {'wait': {'type': 'object', 'properties': {'seconds': {'type': 'integer'}}}}
        candidates = [dict(reasoning=f'Wait {i} seconds.', actions=[{'name': 'wait', 'arguments': {'seconds': i}}])
                      for i in range(1, 6)]
        self.text, _ = serialize(candidates, list(range(5)), 1)

    def test_valid_selection(self):
        r = diagnose(self.text, 'all5', self.tools)
        self.assertTrue(r['protocol_valid'])
        self.assertTrue(r['final_schema_valid'])
        self.assertTrue(r['final_matches_selected'])

    def test_early_think_close_keeps_alternatives_count(self):
        r = diagnose('Ordinary reasoning.\n</think>\n' + self.text, 'all5', self.tools)
        self.assertFalse(r['protocol_valid'])
        self.assertEqual(r['alternative_openings'], 5)
        self.assertEqual(r['think_closings'], 2)
        self.assertTrue(r['serial_suffix_valid_for_diagnosis_only'])
        self.assertTrue(r['final_matches_selected'])

    def test_final_schema_error_without_selection(self):
        r = diagnose('Reasoning.\n</think>\n<tool_call>{"arguments":{"action":"wait"}}</tool_call>', 'all5', self.tools)
        self.assertEqual(r['alternative_openings'], 0)
        self.assertFalse(r['final_schema_valid'])
        self.assertIsNone(r['final_matches_selected'])

    def test_real_selected_final_mismatch(self):
        text = self.text.replace('<tool_call>{"arguments":{"seconds":2}', '<tool_call>{"arguments":{"seconds":5}')
        r = diagnose(text, 'all5', self.tools)
        self.assertFalse(r['protocol_valid'])
        self.assertTrue(r['final_schema_valid'])
        self.assertFalse(r['final_matches_selected'])


if __name__ == '__main__': unittest.main()
