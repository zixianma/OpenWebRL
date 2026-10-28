"""Offline differential and authorization tests; no browser or provider calls."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from openwebrl import webprm_reproduction_rollout as adapter
from openwebrl.webprm_webarena_worker import ViewerActor, validate_protocol_contract
from scripts import launch_webprm_reproduction_rollouts as launch

SOURCE = '/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-baselines-20261009/sources/arbiter'


def config(condition):
    path = Path(adapter.__file__).resolve()
    return dict(arbiter_source=SOURCE, selector='no_search', selectors=['no_search'],
                actor_model=launch.ACTORS[1], reproduction_diagnostic=adapter.contract(condition),
                code_sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest()})


def collect(condition):
    requests = []
    def request(_, payload, journal, **kwargs):
        requests.append(dict(payload=deepcopy(payload), **kwargs))
        purpose = kwargs['purpose']
        text = (json.dumps(dict(thought='Continue', action=None, confidence=.5)) if purpose == 'terminal'
                else 'Visible page changed' if purpose == 'state_diff'
                else json.dumps({'BLOCKS': [dict(thought='Click visible link', action='click [1]', confidence=.5)]}))
        return dict(choices=[dict(message=dict(content=text), finish_reason='stop')])
    actor = ViewerActor(config(condition), '/tmp/unused-diagnostic-journal', request=request)
    view = dict(intent='Save the requested change', start_url='http://127.0.0.1')
    actor.propose(view, dict(text='Form', url=view['start_url']), [])
    actor.propose(view, dict(text='Filled form', url=view['start_url']),
                  [dict(thought='Click visible link', action='click [1]')])
    return requests


class ReproductionRolloutTests(unittest.TestCase):
    def setUp(self):
        forbidden = patch('socket.socket.connect', side_effect=AssertionError('Offline tests only'))
        forbidden.start(); self.addCleanup(forbidden.stop)

    def test_actual_actor_changes_only_terminal_suffix_between_conditions(self):
        control, treatment = (collect(c) for c in adapter.CONDITIONS)
        self.assertEqual([x['purpose'] for x in control], ['terminal', 'policy', 'state_diff', 'terminal', 'policy'])
        for before, after in zip(control, treatment):
            expected = deepcopy(before)
            if before['purpose'] == 'terminal':
                expected['payload']['messages'][0]['content'] += adapter.WRITE_RULE
            self.assertEqual(after, expected)
            model = launch.ACTORS[1] if before['purpose'] == 'policy' else adapter.MINI
            self.assertEqual(before['payload']['model'], model)

    def test_adapter_preserves_original_payload_and_nonhelper_requests(self):
        records = []
        wrapped = adapter.adapt_request(config(adapter.CONDITIONS[1]), lambda _, p, *a, **kw: records.append(p))
        payload = dict(model=launch.ACTORS[1], temperature=1., top_p=.9,
                       messages=[dict(role='system', content='Original'), dict(role='user', content='Current state')])
        original = deepcopy(payload)
        wrapped({}, payload, 'unused', purpose='terminal')
        self.assertEqual(payload, original)
        wrapped({}, payload, 'unused', purpose='policy')
        self.assertIs(records[1], payload)

    def test_legacy_actor_is_unwrapped(self):
        request = object()
        self.assertIs(adapter.adapt_request({}, request), request)

    def test_broker_directory_is_private_under_normal_umask_and_not_reusable(self):
        with tempfile.TemporaryDirectory(dir='/tmp') as folder:
            root = Path(folder)
            before = os.umask(0o022)
            try:
                with patch.object(launch, 'SCRATCH', root):
                    path = launch.broker_directory(root, 0)
                    self.assertEqual(path.stat().st_mode & 0o777, 0o700)
                    with self.assertRaises(FileExistsError): launch.broker_directory(root, 0)
            finally: os.umask(before)

    def test_every_subprocess_entrypoint_is_present_and_hash_bound(self):
        with tempfile.TemporaryDirectory(dir='/tmp') as folder:
            root=Path(folder);proposal=dict(source=str(root),sha256={})
            for name in launch.REQUIRED_ENTRYPOINTS:
                path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('pass\n')
                proposal['sha256'][str(path)]=launch.sha(path)
            launch.validate_entrypoints(proposal)
            path=root/'openwebrl/webprm_browser_preflight.py'
            path.write_text('changed\n')
            with self.assertRaisesRegex(ValueError,'entrypoint'):launch.validate_entrypoints(proposal)
            path.write_text('pass\n');del proposal['sha256'][str(path)]
            with self.assertRaisesRegex(ValueError,'entrypoint'):launch.validate_entrypoints(proposal)

    def test_replacement_stays_below_remaining_original_time(self):
        self.assertLessEqual(5+launch.RESOURCES['total_seconds'],14400)
        self.assertEqual(launch.RESOURCES['total_seconds'],239*60)
        self.assertIn('openwebrl/webprm_browser_preflight.py',launch.OWN_FILES)

    def test_mutated_or_unfrozen_contract_is_rejected(self):
        for key, value in [('auxiliary_temperature', 0), ('terminal_system_suffix', 'Different'),
                           ('auxiliary_model', launch.ACTORS[1])]:
            cfg = config(adapter.CONDITIONS[0]); cfg['reproduction_diagnostic'][key] = value
            with self.assertRaises(ValueError): adapter.validate(cfg)
        cfg = config(adapter.CONDITIONS[0]); cfg['code_sha256'] = {}
        with self.assertRaisesRegex(ValueError, 'frozen'): adapter.validate(cfg)
        cfg = config(adapter.CONDITIONS[0]); cfg['selector'] = 'arbiter7b'
        with self.assertRaises(ValueError): adapter.validate(cfg)

    def test_single_factor_rule_matches_previously_tested_diagnostic(self):
        from scripts.probe_webprm_reproduction import WRITE_RULE
        self.assertEqual(adapter.WRITE_RULE, WRITE_RULE)

    def test_twenty_unique_fixed_cells_and_balanced_order(self):
        rows = launch.cells()
        self.assertEqual(len(rows), 20)
        self.assertEqual(len({(r['task_id'], r['actor'], r['condition']) for r in rows}), 20)
        self.assertEqual({r['task_id'] for r in rows}, set(launch.TASKS))
        first = [rows[i]['condition'] for i in range(0, 20, 2)]
        self.assertEqual(first.count(adapter.CONDITIONS[0]), 5)
        self.assertEqual(first.count(adapter.CONDITIONS[1]), 5)
        self.assertEqual(len(rows)*launch.LIMITS['per_episode_requests'], launch.LIMITS['paid_requests'])

    def test_missing_or_broadened_approval_refused_before_scheduler_or_spawn(self):
        proposal = dict(resources=launch.RESOURCES, limits=launch.LIMITS)
        approval = dict(status='approved', proposal_sha256=launch.digest(proposal), resources=launch.RESOURCES,
                        limits=launch.LIMITS, provider_payload_approved=True, user_approval='Exact request accepted')
        with patch.object(launch.subprocess, 'check_output', side_effect=AssertionError('No scheduler')):
            with self.assertRaises(ValueError): launch.execute(proposal, {})
        launch.approval_check(proposal, approval)
        for key in ('resources', 'limits', 'provider_payload_approved', 'user_approval'):
            changed = deepcopy(approval); changed[key] = None
            with self.assertRaises(ValueError): launch.approval_check(proposal, changed)

    def test_scheduler_rejects_gpus_wrong_account_and_extra_time(self):
        job = '123'
        snapshot = dict(errors=[], jobs=[dict(job_id=123, job_state=['RUNNING'], user_id=os.getuid(), restart_cnt=0,
            **{k:dict(set=True, infinite=False, number=v) for k,v in
               dict(start_time=1000, end_time=15340, time_limit=239).items()})])
        text = 'Account=vlm Partition=gpu-h200 NumCPUs=8 NumNodes=1 AllocTRES=cpu=8,mem=32G,node=1,billing=8'
        self.assertEqual(launch.scheduler_contract(text, json.dumps(snapshot), job)['limit_seconds'], 14340)
        for bad in (text.replace('vlm', 'zixianma'), text+',gres/gpu=1', text.replace('32G','64G')):
            with self.assertRaises(ValueError): launch.scheduler_contract(bad, json.dumps(snapshot), job)
        snapshot['jobs'][0]['time_limit']['number'] = 241
        with self.assertRaises(ValueError): launch.scheduler_contract(text, json.dumps(snapshot), job)


if __name__ == '__main__': unittest.main()
