import asyncio
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from openwebrl import arm_rescue_rl as rescue
from slime.utils.types import Sample


def turn(gid, tid, reward, turn_index=0, status=Sample.Status.COMPLETED, removed=False, excluded=False):
    s = Sample(group_index=gid, index=tid*100+turn_index, status=status, reward=reward, remove_sample=removed)
    s.metadata = dict(task_id='t1', trajectory_id=tid, turn_index=turn_index)
    if excluded:
        s.metadata['arm_calibration_excluded'] = True
    return s


def group(rewards, gid=7, **kw):
    return [[turn(gid, i, r, 0, **kw), turn(gid, i, r, 1, **kw)] for i, r in enumerate(rewards)]


ARGS = SimpleNamespace(n_samples_per_prompt=5, reward_key=None)


class Eligibility(unittest.TestCase):
    def test_only_five_valid_zero_reward_trajectories(self):
        self.assertTrue(rescue.all_valid_failures(ARGS, group([0, 0, 0, 0, 0])))
        self.assertFalse(rescue.all_valid_failures(ARGS, group([0, 0, 0, 0, 1])))
        self.assertFalse(rescue.all_valid_failures(ARGS, group([1, 1, 1, 1, 1])))
        self.assertFalse(rescue.all_valid_failures(ARGS, group([0, 0, 0, 0])))
        self.assertFalse(rescue.all_valid_failures(ARGS, group([0]*5, status=Sample.Status.ABORTED)))
        self.assertFalse(rescue.all_valid_failures(ARGS, group([0]*5, removed=True)))
        self.assertFalse(rescue.all_valid_failures(ARGS, group([0]*5, excluded=True)))
        self.assertFalse(rescue.all_valid_failures(ARGS, group([-1, 0, 0, 0, 0])))

    def test_drop_index_is_deterministic_and_in_range(self):
        self.assertEqual(rescue.drop_index('p', 't', 5), rescue.drop_index('p', 't', 5))
        self.assertTrue(all(0 <= rescue.drop_index('p', str(i), 5) < 5 for i in range(50)))


class Flow(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        cfg = dict(output=self.tmp.name, selector_endpoint='http://x', policy_id='p', seed=42,
                   max_rescues_per_rollout=2, wait_seconds=5)
        path = Path(self.tmp.name)/'cfg.json'
        path.write_text(json.dumps(cfg))
        self.env = patch.dict(os.environ, OPENWEBRL_ARM_RESCUE_CONFIG=str(path))
        self.env.start()
        rescue._ROUND = None
        rescue._ORIGINALS.clear()

    def tearDown(self):
        self.env.stop()
        self.tmp.cleanup()

    def set_cfg(self, **kw):
        path = Path(os.environ['OPENWEBRL_ARM_RESCUE_CONFIG'])
        path.write_text(json.dumps(dict(json.loads(path.read_text()), **kw)))

    def run_round(self, groups, outcome, delay=0.):
        """outcome: one reward for every attempt, or a {(mode, slot): reward} map."""
        async def fake(args, original, params, c, rid, mode='selection', slot=7):
            await asyncio.sleep(delay)
            r = outcome.get((mode, slot), 0) if isinstance(outcome, dict) else outcome
            return [turn(99, 50, r, 0), turn(99, 50, r, 1)]

        async def main():
            args = SimpleNamespace(**vars(ARGS), _arm_rescue_rollout_id=3)
            for g in groups:
                rescue._ORIGINALS[g[0][0].group_index] = (Sample(index=1), {})
            native = SimpleNamespace(keep=False, reason='zero_std_0.0')
            with patch('slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonempty_nonzero_std',
                       return_value=native), patch.object(rescue, 'guided_trajectory', fake):
                for g in groups:
                    self.assertFalse(rescue.filter_groups(args, g).keep)
                return await rescue.collect(args, 3)
        return asyncio.run(main())

    def test_success_replaces_one_failure_and_keeps_group_shape(self):
        out = self.run_round([group([0]*5, gid=7)], outcome=1)
        self.assertEqual(len(out), 1)
        g = out[0]
        self.assertEqual(len(g), 5)
        rewards = [t[-1].reward for t in g]
        self.assertEqual(sorted(rewards), [0, 0, 0, 0, 1])
        guided = [t for t in g if t[0].metadata.get('arm_rescue')]
        self.assertEqual(len(guided), 1)
        self.assertTrue(all(x.group_index == 7 for x in guided[0]))
        report = json.loads((Path(self.tmp.name)/'rescue-reports/0003.json').read_text())
        self.assertEqual((report['attempted'], report['rescued']), (1, 1))

    def test_failed_rescue_adds_nothing(self):
        self.assertEqual(self.run_round([group([0]*5)], outcome=0), [])

    def test_cap_limits_attempts(self):
        out = self.run_round([group([0]*5, gid=g) for g in (1, 2, 3)], outcome=1)
        self.assertEqual(len(out), 2)

    def test_wait_bound_cancels_slow_rescues(self):
        self.set_cfg(wait_seconds=.1)
        self.assertEqual(self.run_round([group([0]*5)], outcome=1, delay=2.), [])
        report = json.loads((Path(self.tmp.name)/'rescue-reports/0003.json').read_text())
        self.assertTrue(report['records'][0].get('cancelled'))

    def test_mixed_group_does_not_trigger(self):
        out = self.run_round([group([0, 0, 0, 0, 1])], outcome=1)
        self.assertEqual(out, [])

    def report(self):
        return json.loads((Path(self.tmp.name)/'rescue-reports/0003.json').read_text())

    def test_control_success_is_telemetry_only(self):
        self.set_cfg(control_attempts=2)
        self.assertEqual(self.run_round([group([0]*5)], outcome={('baseline', 1): 1}), [])
        arms = self.report()['arms']
        self.assertEqual((arms['guided']['attempts'], arms['control']['attempts']), (1, 2))
        self.assertEqual((arms['guided']['successes'], arms['control']['successes']), (0, 1))
        attempts = self.report()['records'][0]['attempts']
        self.assertEqual(sorted((a['arm'], a['slot']) for a in attempts), [('control', 1), ('control', 2), ('guided', 7)])

    def test_later_guided_success_is_inserted_once(self):
        self.set_cfg(guided_attempts=3)
        out = self.run_round([group([0]*5)], outcome={('selection', 8): 1, ('selection', 9): 1})
        self.assertEqual(len(out), 1)
        self.assertEqual(sorted(t[-1].reward for t in out[0]), [0, 0, 0, 0, 1])
        rec = self.report()['records'][0]
        self.assertEqual((rec['rescued_slot'], rec['reward']), (8, 0))  # summary fields stay on the first attempt
        self.assertEqual(self.report()['arms']['guided']['groups_with_success'], 1)

    def test_attempt_ranges_validated(self):
        self.set_cfg(guided_attempts=4)
        with self.assertRaises(ValueError):
            rescue.config()
        self.set_cfg(guided_attempts=1, control_attempts=4)
        with self.assertRaises(ValueError):
            rescue.config()

    def test_outcome_diagnostics(self):
        t0, t1 = turn(1, 1, 0, 0), turn(1, 1, 0, 1)
        t0.response = '<tool_call>{"name": "click", "arguments": {}}</tool_call>'
        t1.response = '<tool_call>{"name": "done", "arguments": {"answer": "x"}}</tool_call>'
        t1.metadata.update(terminate_reason='task_completed', reward=dict(judge_text='NOT SUCCESS'))
        d = rescue.outcome([t1, t0])
        self.assertEqual((d['actions'], d['terminate_reason'], d['judge_text']), (['click', 'done'], 'task_completed', 'NOT SUCCESS'))


class Endpoint(unittest.TestCase):
    def test_live_selector_endpoint_overrides_planned_port(self):
        with tempfile.TemporaryDirectory() as d:
            cfg, live = Path(d)/'c.json', Path(d)/'arm.json'
            cfg.write_text(json.dumps(dict(output=d, selector_endpoint='http://127.0.0.1:1', policy_id='p', seed=1,
                                           max_rescues_per_rollout=2, wait_seconds=5)))
            live.write_text(json.dumps(dict(selector_endpoint='http://127.0.0.1:22042')))
            with patch.dict(os.environ, OPENWEBRL_ARM_RESCUE_CONFIG=str(cfg), OPENWEBRL_ARM_TURN_BONUS_CONFIG=str(live)):
                self.assertEqual(rescue.config()['selector_endpoint'], 'http://127.0.0.1:22042')


class Advantages(unittest.TestCase):
    def test_outcome_only_advantages_returned(self):
        args = SimpleNamespace(dynamic_sampling_filter_path='openwebrl.arm_rescue_rl.filter_groups')
        fake_parent = SimpleNamespace(state=lambda a: dict(config={}),
                                      post_process_rewards=lambda a, s, r, n: (r, [x+.4 for x in n]))
        with patch.dict('sys.modules', {'openwebrl.arm_turn_bonus': fake_parent}), \
                patch('openwebrl.arm_turn_bonus', fake_parent, create=True):
            raw, adv = rescue.post_process_rewards(args, [], [1, 0], [1., -1.])
        self.assertEqual((raw, adv), ([1, 0], [1., -1.]))

    def test_rejects_failure_auxiliary(self):
        args = SimpleNamespace(dynamic_sampling_filter_path='openwebrl.arm_rescue_rl.filter_groups')
        fake_parent = SimpleNamespace(state=lambda a: dict(config=dict(additive_failure_groups=True)),
                                      post_process_rewards=None)
        with patch.dict('sys.modules', {'openwebrl.arm_turn_bonus': fake_parent}), \
                patch('openwebrl.arm_turn_bonus', fake_parent, create=True):
            with self.assertRaises(ValueError):
                rescue.post_process_rewards(args, [], [], [])



class LabelsOff(unittest.TestCase):
    """v2: no ARM label requests; evaluation-overlap tasks keep the exclusion mark (never rescued)."""
    def run_generate(self, task_id):
        with tempfile.TemporaryDirectory() as d:
            cfg = Path(d)/'c.json'
            cfg.write_text(json.dumps(dict(output=d, selector_endpoint='http://x', policy_id='p', seed=1,
                                           max_rescues_per_rollout=2, wait_seconds=5, telemetry_labels=False)))
            seen = {}

            async def fake_turns(args, sample, params):
                seen['selector'] = args.browser_action_selector
                return [turn(3, 0, 0, 0), turn(3, 0, 0, 1)]

            fake_parent = SimpleNamespace(
                state=lambda a: dict(excluded_ids={'eval-task'}, excluded_intents=set()),
                task_identity=lambda m: (m.get('task_id'), m.get('intent')),
                generate=None)
            sample = Sample(group_index=3, index=1)
            sample.metadata = dict(task_id=task_id)
            args = SimpleNamespace(browser_action_selector='frozen-label-hook')
            fake_browser = SimpleNamespace(generate_turn_sample=fake_turns)
            with patch.dict(os.environ, OPENWEBRL_ARM_RESCUE_CONFIG=str(cfg)), \
                    patch.dict('sys.modules', {'openwebrl.arm_turn_bonus': fake_parent,
                                               'openwebrl.generate_browser': fake_browser}), \
                    patch('openwebrl.arm_turn_bonus', fake_parent, create=True), \
                    patch('openwebrl.generate_browser', fake_browser, create=True):
                rescue._ORIGINALS.clear()
                turns = asyncio.run(rescue.generate(args, sample, {}))
            self.assertIsNone(seen['selector'])
            self.assertIn(3, rescue._ORIGINALS)
            return turns

    def test_no_labels_and_training_task_stays_rescuable(self):
        turns = self.run_generate('train-task')
        self.assertFalse(any((t.metadata or {}).get('arm_calibration_excluded') for t in turns))

    def test_evaluation_overlap_is_marked(self):
        turns = self.run_generate('eval-task')
        self.assertTrue(all(t.metadata['arm_calibration_excluded'] for t in turns))


class TrainingGate(unittest.TestCase):
    """The ARM calibration gate does not apply; only the frozen collection checks and the time budget."""
    def gate(self, rows, deadline_in):
        import time
        with tempfile.TemporaryDirectory() as d:
            cfg = Path(d)/'arm.json'
            cfg.write_text(json.dumps(dict(output=d, seconds_per_optimizer_update=90,
                                           deadline_epoch_seconds=time.time()+deadline_in)))
            # A failed calibration (too few labels) must not block rescue training.
            (Path(d)/'calibration.json').write_text(json.dumps(dict(batch_rows=rows, decision=dict(passed=False))))
            writes = {}
            fake_parent = SimpleNamespace(verify_shadow_complete=lambda a, r: writes.setdefault('verified', r),
                                          write_json=lambda path, data: writes.__setitem__(Path(path).name, data))
            args = SimpleNamespace(global_batch_size=256, ppo_epochs=2)
            with patch.dict(os.environ, OPENWEBRL_ARM_TURN_BONUS_CONFIG=str(cfg)), \
                    patch.dict('sys.modules', {'openwebrl.arm_turn_bonus': fake_parent}), \
                    patch('openwebrl.arm_turn_bonus', fake_parent, create=True):
                stop = rescue.before_optimizer(args, 4)
            self.assertEqual(writes['verified'], 4)
            return stop, writes['training_gate.json']

    def test_trains_despite_failed_calibration(self):
        stop, g = self.gate(rows=2048, deadline_in=7200)
        self.assertFalse(stop)
        self.assertEqual(g['expected_optimizer_updates'], 16)
        self.assertFalse(g['calibration']['applies'])

    def test_stops_without_time(self):
        stop, g = self.gate(rows=2048, deadline_in=1000)
        self.assertTrue(stop)
        self.assertFalse(g['ready'])

if __name__ == '__main__':
    unittest.main()
