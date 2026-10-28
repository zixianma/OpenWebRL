"""Exercise actual pinned upstream validators and history helpers, offline."""
from pathlib import Path
from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from openwebrl.webprm_viewer_parity import load_viewer_helpers, history_from_saved_observations
from openwebrl import webprm_viewer_parity as parity
from openwebrl import webprm_reproduction_rollout as stopping
from openwebrl.webprm_webarena_worker import ViewerActor

SOURCE = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-baselines-20261009/sources/arbiter')


class ViewerParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.helpers = load_viewer_helpers(SOURCE)

    def test_missing_click_type_rejected_but_hover_not_covered_upstream(self):
        with patch('socket.socket.connect', side_effect=AssertionError('Offline only')):
            for action, expected in [('click [9]', False), ('type [9] text', False),
                                     ('hover [9]', True), ('click [1]', True),
                                     ('type [1] text', True)]:
                block = SimpleNamespace(thought='Act', action=action, confidence=.5)
                self.assertEqual(self.helpers.validate_candidate(block, ['1']), expected)

    def test_upstream_confidence_and_empty_thought_checks(self):
        for thought, confidence, expected in [('', .5, False), ('Act', 1.1, False),
                                              ('Act', None, True)]:
            block = SimpleNamespace(thought=thought, action='click [1]', confidence=confidence)
            self.assertEqual(self.helpers.validate_candidate(block, ['1']), expected)

    def test_action_time_labels_survive_id_reuse_and_do_not_change_action(self):
        history = [dict(thought='Edit bio', action='click [1]'),
                   dict(thought='Save', action='click [1]')]
        observations = [dict(text="[1] link 'Edit biography'"), dict(text="[1] button 'Save'")]
        actual = history_from_saved_observations(self.helpers, history, observations)
        self.assertEqual(actual, 'Step 0:\nTHOUGHT: Edit bio\nACTION: click [1] "link \'Edit biography\'"'
                         '\nStep 1:\nTHOUGHT: Save\nACTION: click [1] "button \'Save\'"')
        self.assertEqual(history[0]['action'], 'click [1]')

    def test_missing_action_time_observation_refused(self):
        with self.assertRaises(ValueError):
            history_from_saved_observations(self.helpers, [dict(thought='', action='click [1]')], [])

    def actor(self, condition):
        cfg = dict(arbiter_source=str(SOURCE), selector='no_search', actor_model=stopping.MINI,
                   reproduction_diagnostic=stopping.contract(stopping.CONDITIONS[1]),
                   viewer_parity_diagnostic=parity.contract(condition, SOURCE))
        cfg['code_sha256'] = {str(Path(m.__file__).resolve()):
            hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest() for m in (parity, stopping)}
        requests = []
        def request(_, payload, journal, **kwargs):
            requests.append(deepcopy(dict(payload=payload, **kwargs)))
            text = (json.dumps(dict(thought='Continue', action=None, confidence=.5))
                    if kwargs['purpose'] == 'terminal' else 'Changed' if kwargs['purpose'] == 'state_diff'
                    else json.dumps(dict(BLOCKS=[dict(thought='Edit bio', action='click [9]', confidence=.5)])))
            return dict(choices=[dict(message=dict(content=text), finish_reason='stop')])
        return ViewerActor(cfg, '/tmp/unused-parity-journal', request=request), requests, cfg

    def test_filter_changes_acceptance_and_keeps_requests_identical_without_resampling(self):
        view = dict(intent='Save bio', start_url='http://127.0.0.1')
        observation = dict(text="[1] link 'Edit bio'", url=view['start_url'])
        control, requests, _ = self.actor('unchanged')
        treatment, other_requests, _ = self.actor('filter_only')
        before, after = (actor.propose(view, observation, []) for actor in (control, treatment))
        self.assertEqual(requests, other_requests)
        self.assertEqual(len(before['candidates']), 1)
        self.assertEqual(after['candidates'], [])
        self.assertEqual(after['raw_response'], before['raw_response'])
        self.assertEqual(after['requested_raw_slots'], 1)
        self.assertEqual(len(after['rejected_slots']), 1)
        self.assertEqual([r['purpose'] for r in requests], ['terminal', 'policy'])

    def test_history_factor_changes_only_prompt_history_and_uses_past_labels(self):
        view = dict(intent='Save bio', start_url='http://127.0.0.1')
        actor, requests, _ = self.actor('history_only')
        actor.propose(view, dict(text="[9] link 'Edit biography'", url=view['start_url']), [])
        proposal = actor.propose(view, dict(text="[9] button 'Save'", url=view['start_url']),
                                 [dict(thought='Edit bio', action='click [9]')])
        self.assertIn('"link \'Edit biography\'"', proposal['context']['trajectory'])
        self.assertNotIn('button', proposal['context']['trajectory'])
        self.assertEqual(len(proposal['candidates']), 1)
        self.assertEqual([r['purpose'] for r in requests], ['terminal', 'policy', 'state_diff', 'terminal', 'policy'])

    def test_changed_or_unfrozen_scientific_contract_refused(self):
        _, _, cfg = self.actor('unchanged')
        cfg['viewer_parity_diagnostic']['filter_policy_candidates'] = True
        with self.assertRaises(ValueError): parity.validate(cfg)
        _, _, cfg = self.actor('unchanged'); cfg['code_sha256'] = {}
        with self.assertRaises(ValueError): parity.validate(cfg)
        _, _, cfg = self.actor('unchanged')
        cfg['reproduction_diagnostic'] = stopping.contract(stopping.CONDITIONS[0])
        with self.assertRaises(ValueError): parity.validate(cfg)

    def test_actor_replay_binds_actual_history_filter_and_every_paid_payload(self):
        from scripts.audit_webprm_reproduction_rollouts import audit_parity_actor
        actor, requests, cfg = self.actor('filter_and_history')
        view = dict(intent='Save bio', start_url='http://127.0.0.1')
        proposal = actor.propose(view, dict(text="[1] link 'Edit bio'", url=view['start_url']), [])
        responses = [dict(choices=[dict(message=dict(content=json.dumps(dict(thought='Continue', action=None, confidence=.5))),
                                       finish_reason='stop')]),
                     dict(choices=[dict(message=dict(content=json.dumps(dict(BLOCKS=[dict(thought='Edit bio', action='click [9]', confidence=.5)]))),
                                       finish_reason='stop')])]
        calls = dict(pairs=[dict(request=dict(payload=r['payload'], purpose=r['purpose']),
                                response=dict(response=response)) for r, response in zip(requests, responses)])
        rollout = [dict(proposal=proposal, selected_index=None, execution_error='No admissible native action')]
        audit_parity_actor(cfg, view, rollout, calls)
        altered = deepcopy(calls); altered['pairs'][1]['request']['payload']['temperature'] = 0
        with self.assertRaisesRegex(ValueError, 'paid request'): audit_parity_actor(cfg, view, rollout, altered)
        altered = deepcopy(rollout); altered[0]['proposal']['candidates'] = [dict(action='click [9]', thought='Edit bio', raw_slot=0)]
        with self.assertRaisesRegex(ValueError, 'proposal differs'): audit_parity_actor(cfg, view, altered, calls)

    def test_replicated_prospective_grid_and_nonadditive_caps(self):
        from scripts import launch_webprm_reproduction_rollouts as launcher
        with patch.dict(launcher.__dict__):
            launcher.configure_study('viewer_parity_8_v1')
            rows = launcher.cells()
            self.assertEqual(len(rows), 8)
            self.assertEqual(len({(r['condition'],r['replicate']) for r in rows}), 8)
            self.assertEqual([r['condition'] for r in rows[:4]], list(parity.CONDITIONS))
            self.assertEqual([r['condition'] for r in rows[4:]], list(parity.CONDITIONS[::-1]))
            self.assertEqual({r['actor'] for r in rows}, {stopping.MINI})
            prior = launcher.prior_limits()
            self.assertLessEqual(prior['charged_usd']+launcher.LIMITS['api_usd'], 15)
            self.assertLessEqual(prior['paid_requests']+8*90, 1800)
            self.assertLessEqual(prior['scheduler_seconds']+launcher.RESOURCES['total_seconds'], 14400)
            self.assertEqual(launcher.RESOURCES['maximum_allocations'], 1)
            with patch.object(launcher.subprocess, 'check_output', side_effect=AssertionError('No scheduler')):
                with self.assertRaises(ValueError): launcher.execute({}, {})
            self.assertIn('openwebrl/webprm_viewer_parity.py', launcher.REQUIRED_ENTRYPOINTS)

    def test_four_cell_factorial_preserves_original_cumulative_caps(self):
        from scripts import launch_webprm_reproduction_rollouts as launcher
        with patch.dict(launcher.__dict__):
            launcher.configure_study('viewer_parity_4_v1')
            rows = launcher.cells()
            self.assertEqual(len(rows), 4)
            self.assertEqual({r['condition'] for r in rows}, set(parity.CONDITIONS))
            self.assertEqual({r['replicate'] for r in rows}, {0})
            self.assertEqual({r['task_id'] for r in rows}, {97})
            self.assertEqual(launcher.LIMITS['scratch_bytes'], 1280*1024**3)
            self.assertEqual(launcher.LIMITS['paid_requests'], 360)
            self.assertEqual(launcher.LIMITS['api_usd'], 6.5)
            self.assertEqual(launcher.RESOURCES['gpus'], 0)
            self.assertEqual(launcher.RESOURCES['total_seconds'], 7200)
            prior = launcher.prior_limits()
            required = (len(rows)+1)*50*1024**3 + 32*1024**3
            self.assertLess(prior['scratch_bytes']+required, launcher.LIMITS['scratch_bytes'])
            self.assertLessEqual(prior['paid_requests']+360, 1800)
            self.assertLessEqual(prior['charged_usd']+6.5, 15)
            self.assertLessEqual(prior['scheduler_seconds']+7200, 14400)
            with self.assertRaises(ValueError):
                launcher.configure_study('viewer_parity_8_v1')
            with patch.object(launcher.subprocess, 'check_output', side_effect=AssertionError('No scheduler')):
                with self.assertRaises(ValueError): launcher.execute({}, {})


if __name__ == '__main__': unittest.main()
