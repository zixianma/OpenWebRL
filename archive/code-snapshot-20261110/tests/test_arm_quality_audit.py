import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from prepare_arm_quality_audit import choose_attempt, inspect_candidates


class QualityAudit(unittest.TestCase):
    def test_panel_assignment_is_invariant_to_outcome_and_input_order(self):
        rows = [dict(task_id='a', attempt=i, valid=True, reward=0) for i in range(5)]
        expected = choose_attempt(rows)['attempt']
        changed = [dict(r, valid=False, reward=1) for r in reversed(rows)]
        self.assertEqual(choose_attempt(changed)['attempt'], expected)

    def test_distinct_answers_are_not_distinct_tool_sequences(self):
        tool = {'function': {'name': 'done', 'parameters': {'type': 'object',
                'properties': {'response': {'type': 'string'}}, 'required': ['response']}}}
        prompt = '<tools>\n' + json.dumps(tool) + '\n</tools>'
        candidates = [{'action': '<tool_call>' + json.dumps({'name': 'done',
                       'arguments': {'response': str(i)}}) + '</tool_call>'} for i in range(5)]
        result = inspect_candidates(candidates, prompt, 0, ['stop'] * 5)
        self.assertEqual(result['unique_valid_actions'], 5)
        self.assertEqual(result['unique_valid_tool_sequences'], 1)
        self.assertTrue(result['all_five_are_done'])
        candidates[0]['action'] = 'not a tool call'
        result = inspect_candidates(candidates, prompt, 0, ['stop'] * 5)
        self.assertFalse(result['selected_action_valid'])
        self.assertEqual(result['valid_candidates'], 4)
        self.assertIsNone(result['selected_action_multiplicity'])


if __name__ == '__main__':
    unittest.main()
