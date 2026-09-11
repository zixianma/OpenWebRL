"""Structural checks that distinguish response variation from action variation."""
import json
import unittest

from scripts.audit_arm_preference_pairs import coordinate_distance, parse_action, schemas


class PreferenceAuditTests(unittest.TestCase):
    def setUp(self):
        self.tools = {
            'click': {'type': 'object', 'required': ['point_2d'], 'properties': {
                'point_2d': {'type': 'array', 'items': {'type': 'number'}, 'minItems': 2, 'maxItems': 2},
                'button': {'type': 'string', 'enum': ['left', 'right'], 'default': 'left'}}},
            'write': {'type': 'object', 'required': ['message'], 'properties': {'message': {'type': 'string'}}}}

    def action(self, name, arguments, reasoning=''):
        return parse_action(reasoning + '<tool_call>' + json.dumps({'name': name, 'arguments': arguments}) + '</tool_call>', self.tools)[0]

    def test_defaults_and_reasoning_do_not_create_different_actions(self):
        a = self.action('click', {'point_2d': [1, 2]})
        b = self.action('click', {'button': 'left', 'point_2d': [1.0, 2]}, '<think>different reasoning</think>')
        self.assertEqual(a, b)
        self.assertEqual(coordinate_distance(a, b), 0)

    def test_case_and_call_order_are_semantics(self):
        a = self.action('write', {'message': 'TokenA'})
        b = self.action('write', {'message': 'tokena'})
        self.assertIsNone(coordinate_distance(a, b))
        c = self.action('click', {'point_2d': [1, 2]})
        self.assertIsNone(coordinate_distance(a + c, c + a))

    def test_coordinate_only_distance_preserves_other_arguments(self):
        a = self.action('click', {'point_2d': [1, 2]})
        b = self.action('click', {'point_2d': [4, 6]})
        self.assertEqual(coordinate_distance(a, b), 5)
        c = self.action('click', {'point_2d': [1, 2], 'button': 'right'})
        self.assertIsNone(coordinate_distance(a, c))

    def test_invalid_or_extra_blocks_are_excluded(self):
        with self.assertRaises(ValueError): self.action('click', {'point_2d': [1]})
        with self.assertRaises(ValueError): self.action('click', {'point_2d': [True, 2]})
        with self.assertRaises(ValueError): self.action('write', {})
        with self.assertRaises(ValueError): parse_action('<tool_call>broken</tool_call>', self.tools)

    def test_prompt_tag_mention_is_not_schema(self):
        prompt = 'Use <tools></tools>.\n<tools>\n' + json.dumps({'function': {'name': 'click', 'parameters': self.tools['click']}}) + '\n</tools>'
        self.assertEqual(schemas(prompt), {'click': self.tools['click']})


if __name__ == '__main__': unittest.main()
