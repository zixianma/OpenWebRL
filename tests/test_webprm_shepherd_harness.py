import unittest
from types import SimpleNamespace

try:
    from browsergym.core.action.highlevel import HighLevelActionSet
except ImportError:
    HighLevelActionSet = None

from openwebrl.webprm_shepherd_harness import (
    SHEPHERD_ACTION_SUBSETS, classify_shepherd_choices, parse_shepherd_response,
)


@unittest.skipIf(HighLevelActionSet is None, 'BrowserGym is required for native parser parity')
class ShepherdNativeParsing(unittest.TestCase):
    def setUp(self):
        self.actions = HighLevelActionSet(subsets=list(SHEPHERD_ACTION_SUBSETS),
                                          strict=False, multiaction=False)

    def test_canonical_actions_have_native_execution_equivalence(self):
        for raw in (
            'I will select the button.\n```click("12")```',
            '```python\nfill("4", "O\\\'Brien")\n```',
            'click("12", button="right")',
            'select_option("8", ["one", "two"])',
            '# click("ignored")\nclick("3")',
        ):
            with self.subTest(raw=raw):
                parsed = parse_shepherd_response(raw, self.actions)
                self.assertEqual(parsed['raw_response'], raw)
                self.assertEqual(self.actions.to_python_code(raw),
                                 self.actions.to_python_code(parsed['action']))

    def test_ambiguous_prose_remains_native_error(self):
        for raw in (
            'I will fill location("Dallas, TX").\n```click("12")```',
            'First click("12") then ```click("13")```',
            'No action available.',
            '```unsupported("12")```',
        ):
            with self.subTest(raw=raw):
                with self.assertRaises((ValueError, NameError)) as original:
                    self.actions.to_python_code(raw)
                with self.assertRaises(type(original.exception)) as wrapper:
                    parse_shepherd_response(raw, self.actions)
                self.assertEqual(str(original.exception), str(wrapper.exception))

    def test_bad_samples_do_not_discard_valid_fixed_draws(self):
        choices = [
            dict(index=8, finish_reason='stop', message=dict(content='click("12")')),
            dict(index=9, finish_reason='stop', message=dict(content='click("12") click("13")')),
            dict(index=10, finish_reason='length', message=dict(content='click("14")')),
            dict(index=11, finish_reason='stop', message=dict(content='click("12")')),
        ]
        accepted, rejected = classify_shepherd_choices(choices, self.actions)
        self.assertEqual([row['sample_index'] for row in accepted], [0, 3])
        self.assertEqual([row['choice_index'] for row in accepted], [8, 11])
        self.assertEqual([row['sample_index'] for row in rejected], [1, 2])
        self.assertEqual(rejected[0]['raw_response'], choices[1]['message']['content'])
        self.assertEqual(len(accepted) + len(rejected), len(choices))

    def test_all_invalid_samples_return_explicit_empty_pool(self):
        accepted, rejected = classify_shepherd_choices([
            dict(finish_reason='stop', message=dict(content='No action.')),
            dict(finish_reason='stop', message=dict(content=None)),
        ], self.actions)
        self.assertEqual(accepted, [])
        self.assertEqual(len(rejected), 2)

    def test_incompatible_action_set_is_rejected(self):
        for kwargs in (dict(strict=True, multiaction=False),
                       dict(strict=False, multiaction=True)):
            with self.assertRaisesRegex(ValueError, 'single-action'):
                parse_shepherd_response('click("1")', HighLevelActionSet(**kwargs))

    def test_native_environment_reports_parse_error_as_observation(self):
        # A native malformed action consumes a step and is returned as feedback;
        # it is not an exception that terminates the complete task episode.
        from browsergym.core.env import BrowserEnv
        env = SimpleNamespace(action_mapping=self.actions.to_python_code,
                              pre_step=lambda: ({}, None, None))
        env.post_step = lambda info: dict(last_action_error=env.last_action_error)
        observation = BrowserEnv.step(env, 'click("12") click("13")')
        self.assertIn('multi-action', observation['last_action_error'])
        self.assertEqual(env.last_action, 'click("12") click("13")')


if __name__ == '__main__':
    unittest.main()
