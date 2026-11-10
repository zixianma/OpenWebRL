import asyncio
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from openwebrl import arm_loop_credit as loop
from openwebrl.arm_outcome_reweight import transform, validate_config


def call(name, **arguments):
    return '<tool_call>' + json.dumps(dict(name=name, arguments=arguments)) + '</tool_call>'


def row(tid, selected=None, repeat=False, mixed=False, masked=False):
    label = {} if selected is None else dict(eligible=True, policy_id='p', executed_index=0,
        selected_index=selected, unit_bonus=float(selected == 0)-.2,
        candidate_gate='min2', credit_assignment='response_index', action_class_ids=[0, 0, 1, 1, 1])
    meta = dict(trajectory_id=tid, arm_turn_bonus=label,
                arm_loop=dict(executed_repeat=repeat, mixed_candidates=mixed, loop_relevant=repeat or mixed))
    return SimpleNamespace(group_index=1, metadata=meta, remove_sample=False, loss_mask=[0] if masked else [1])


class Credit(unittest.TestCase):
    def test_loop_bonus_keeps_arm_bonus_only_on_loop_relevant_turns(self):
        samples = [row('a', 1, repeat=True), row('a', 0, mixed=True), row('a', 0), row('a', repeat=True)]
        values, report = transform(samples, [1]*4, [1]*4, 'p', mode='loop_bonus')
        # rejected repeat -0.1; picked mixed +0.4; picked but not loop-relevant 0; unlabelled repeat 0
        self.assertEqual(values, [.9, 1.4, 1., 1.])
        self.assertEqual((report['nonzero_unit_rows'], report['executed_repeat_rows'], report['loop_relevant_rows']),
                         (2, 2, 3))

    def test_repeat_penalty_ignores_arm(self):
        samples = [row('a', 0, repeat=True), row('a', 1), row('a', repeat=True), row('a', repeat=True, masked=True)]
        values, _ = transform(samples, [0]*4, [-1]*4, 'p', mode='repeat_penalty')
        self.assertEqual(values, [-1.1, -1., -1.1, -1.])

    def test_modes_validate(self):
        for mode in loop.LOOP_MODES:
            validate_config(dict(advantage_mode=mode, k=5, beta=.5, scored_fraction=.2,
                                 reweight_lambda=.5, candidate_gate='min2'))


class Flags(unittest.TestCase):
    def test_repeat_definition_matches_label_analysis(self):
        prior = loop.prior_calls(['I click.\n' + call('click', x=1, y=2), call('wait'), 'no call'])
        self.assertEqual(prior, {loop.norm(dict(name='click', arguments=dict(x=1, y=2)))})
        self.assertIsNone(loop.first_call(call('done', answer='x')))
        actions = [json.dumps([dict(name='click', arguments=dict(x=1, y=2))]),
                   json.dumps([dict(name='scroll', arguments=dict(direction='down'))])] * 2 + ['[]']
        flags = loop.loop_flags(dict(executed_repeat=False, prior_calls=sorted(prior), actions=actions))
        self.assertEqual(flags, dict(executed_repeat=False, mixed_candidates=True, loop_relevant=True))
        flags = loop.loop_flags(dict(executed_repeat=True, prior_calls=[]))
        self.assertEqual(flags, dict(executed_repeat=True, mixed_candidates=False, loop_relevant=True))


class Selector(unittest.TestCase):
    """An executed repeat is always sampled; other turns keep the parent's q."""
    def run_turn(self, executed, history, q=0.):
        sampled = {}

        class Parent:
            def __init__(self, config, trajectory_id, **kw):
                self.config, self.trajectory_id, self.records = config, str(trajectory_id), {}

            async def __call__(self, *, infer, turn, **kw):
                out = await infer('u', 'prompt', {}, [], timeout_secs=1)
                sampled['value'] = self.config['scored_fraction'] >= 1.
                self.records[turn] = dict(sampled=sampled['value'])
                return out, {}

        async def infer(*a, **kw):
            return (executed, [1], [0.], 'stop')

        with patch.dict('sys.modules', {'openwebrl.arm_turn_bonus': SimpleNamespace(ShadowSelector=Parent)}):
            cls = loop.selector_class()
        selector = cls(dict(scored_fraction=q), 9)
        asyncio.run(selector(infer=infer, url='u', input_text='p', sampling_params={}, images=[], observation={},
                             history=history, task='t', task_id='1', turn=3, timeout=1))
        loop._REGISTRY.pop('9', None)
        self.assertEqual(selector.config['scored_fraction'], q)
        return sampled['value'], selector.records[3]['executed_repeat']

    def test_repeat_forced(self):
        self.assertEqual(self.run_turn(call('click', x=1), [call('click', x=1)]), (True, True))

    def test_new_action_not_forced(self):
        self.assertEqual(self.run_turn(call('click', x=2), [call('click', x=1)]), (False, False))


if __name__ == '__main__':
    unittest.main()
