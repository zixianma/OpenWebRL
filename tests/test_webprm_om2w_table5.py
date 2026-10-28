"""OM2W Table 5-style harness contracts with all network connections blocked."""
import base64
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from openwebrl import webprm_om2w_table5 as om
from openwebrl.webprm_webarena import EvaluationNotReady, digest
from openwebrl.webprm_webarena_worker import ACTOR_SNAPSHOTS, PRIMARY_SELECTORS, ViewerActor, ViewerSelector, cell_name, run_loop

REPO = Path(__file__).resolve().parents[1]
SOURCE = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-baselines-20261009/sources/arbiter')
TASKS = REPO/om.TASK_FILE
COHORTS = REPO/om.COHORTS_FILE


def png(width=1280, height=720):
    from PIL import Image
    stream = io.BytesIO(); Image.new('RGB', (width, height), 'white').save(stream, format='PNG')
    return stream.getvalue()


class Scripted:
    """Provider-free responses for every purpose a cell can request."""
    def __init__(self, terminal_after=1, count=5, verdict='Status: success'):
        self.requests, self.terminals = [], 0
        self.terminal_after, self.count, self.verdict = terminal_after, count, verdict

    def __call__(self, config, payload, journal, **kwargs):
        self.requests.append(dict(payload=deepcopy(payload), **kwargs))
        purpose = kwargs.get('purpose')
        if purpose == 'terminal':
            stop = self.terminals >= self.terminal_after; self.terminals += 1
            text = json.dumps(dict(thought='Done', action='send_msg_to_user [Plans compared]' if stop else None, confidence=.9))
        elif purpose == 'state_diff':
            text = 'The page changed.'
        elif purpose == 'policy':
            text = json.dumps({'BLOCKS': [dict(thought=f'Candidate {i}', action=f'goto [https://example.org/p{i}]',
                                               confidence=.5) for i in range(self.count)]})
        elif purpose == 'selector_pair':
            text = '<Answer>Response 1</Answer>'
        elif purpose == 'selector_checklist':
            text = '<answer>Open the plans page.</answer>'
        elif purpose == 'selector_pointwise':
            text = '<answer>Yes</answer>'
        elif purpose == 'selector_piotr':
            text = '{"selection":4}'
        elif purpose == om.JUDGE_PURPOSE:
            text = self.verdict
        else:
            raise AssertionError('Unexpected purpose: '+str(purpose))
        choice = dict(message={'content': text}, finish_reason='stop')
        if purpose == 'selector_pointwise':
            choice['logprobs'] = {'content': [dict(token='<answer>', top_logprobs=[]),
                dict(token='Yes', top_logprobs=[dict(token='Yes', logprob=-.1), dict(token='No', logprob=-3.),
                                                dict(token='In', logprob=-4.)])]}
        return dict(model=payload['model'], choices=[choice], usage=dict(prompt_tokens=10, completion_tokens=5))


class FakeEnv:
    def __init__(self, output):
        self.output, self.actions, self.messages = output, [], []
        image = png()
        self.page = SimpleNamespace(url='https://www.flightaware.com/', is_closed=lambda: False,
            screenshot=lambda **kw: b'\xff\xd8\xff\xe0fakejpeg' if kw.get('type') == 'jpeg' else image)

    def observation(self, error=None):
        import numpy as np
        return dict(url=self.page.url, chat_messages=deepcopy(self.messages), last_action_error=error,
                    screenshot=np.zeros((2, 2, 3), dtype=np.uint8))

    def reset(self, seed):
        return self.observation(), {}

    def step(self, action):
        self.actions.append(action)
        if action.startswith('send_msg_to_user'):
            self.messages.append(dict(role='assistant', message=action.split('[', 1)[1][:-1]))
            return self.observation(), 0, True, False, {}
        self.page.url = action.split('[', 1)[1][:-1]
        return self.observation(), 0, False, False, {}

    def close(self):
        pass


class CohortTests(unittest.TestCase):
    def test_focused_controls_keep100_tasks_and_only_four_methods(self):
        cohort = om.cohort(TASKS, COHORTS, [ACTOR_SNAPSHOTS[1]], om.FOCUSED_CONTROL_SELECTORS)
        cells = [c for c in cohort['cells'] if c['cell_id'] in set(om.candidate_control_ids(cohort, 100, focused=True))]
        self.assertEqual(len(cells), 400)
        self.assertEqual({c['task_index'] for c in cells}, set(range(108, 208)))
        self.assertEqual({c['selector'] for c in cells}, {'no_search', 'random5', 'arbiter7b', 'piotr_selectionarm'})
        limits = om.candidate_control_envelope(100, focused=True)['limits']
        self.assertEqual((limits['paid_requests'], limits['local_requests'], limits['total_requests']), (36400, 15022, 51422))
        with self.assertRaises(ValueError):
            om.candidate_control_spec(8, focused=True)

    def test_larger_control_is_disjoint_and_complete_with_cumulative_caps(self):
        cohort = om.cohort(TASKS, COHORTS, [ACTOR_SNAPSHOTS[1]], om.CONTROL_SELECTORS)
        small = set(om.candidate_control_ids(cohort))
        large = set(om.candidate_control_ids(cohort, 100))
        self.assertEqual(len(large), 600)
        self.assertFalse(small & large)
        cells = [c for c in cohort['cells'] if c['cell_id'] in large]
        self.assertEqual({c['task_index'] for c in cells}, set(range(108, 208)))
        for index in range(108, 208):
            self.assertEqual({c['selector'] for c in cells if c['task_index'] == index}, set(om.CONTROL_SELECTORS))
        limits = om.candidate_control_envelope(100)
        self.assertEqual(limits['resources']['total_seconds'], 48*3600)
        self.assertEqual(limits['limits']['maximum_attempts'], 2)
        self.assertEqual(limits['limits']['paid_requests'], 66600)
        self.assertEqual(limits['limits']['local_requests'], 33022)
        self.assertEqual(limits['limits']['total_requests'], 99622)
        self.assertEqual(om.candidate_control_envelope()['limits']['total_requests'], 7979)

    def test_arbitrary_control_sizes_cannot_redefine_frozen_membership(self):
        for count in (0, 9, 90, 101, 100.0, True, '100'):
            with self.subTest(count=count), self.assertRaises(ValueError):
                om.candidate_control_spec(count)

    def test_candidate_controls_are_complete_outcome_independent_holdout_blocks(self):
        cohort = om.cohort(TASKS, COHORTS, [ACTOR_SNAPSHOTS[1]], om.CONTROL_SELECTORS)
        wanted = om.candidate_control_ids(cohort)
        cells = [c for c in cohort['cells'] if c['cell_id'] in wanted]
        self.assertEqual(len(cells), 48)
        self.assertNotIn('first5', om.CONTROL_SELECTORS)
        self.assertEqual({c['task_index'] for c in cells}, set(range(100, 108)))
        self.assertEqual({c['stage'] for c in cells}, {'rest200'})
        for index in range(100, 108):
            self.assertEqual({c['selector'] for c in cells if c['task_index'] == index}, set(om.CONTROL_SELECTORS))
        baseline = {'actor': {'no_search': [['actor', 'policy'], ['actor', 'terminal'], ['actor', 'state_diff']]}}
        _, routed = om.routes([], baseline, ['actor'], candidate_controls=True)
        self.assertNotIn('first5', routed['actor'])
        self.assertEqual(routed['actor']['random5'], routed['actor']['no_search'])
        self.assertNotIn('first5', baseline['actor'])

    def test_prior_fixture_time_is_bound_and_scheduler_verified(self):
        from scripts.launch_webprm_om2w_table5 import preproduction_seconds, sha
        with tempfile.TemporaryDirectory(dir='/tmp') as tmp:
            path = Path(tmp)/'ledger.json'
            rows = [dict(job_id='1', elapsed_seconds=10), dict(job_id='2', elapsed_seconds=649)]
            path.write_text(json.dumps(dict(allocations=rows)))
            proposal = dict(preproduction_ledger=str(path), source_sha256={str(path): sha(path)})
            self.assertEqual(preproduction_seconds(proposal, history='1|COMPLETED|10|\n2|FAILED|649|'), 659)
            for history in ('1|COMPLETED|10|', '1|COMPLETED|10|\n2|RUNNING|649|',
                            '1|COMPLETED|10|\n2|FAILED|650|', '1|COMPLETED|10|\n1|FAILED|649|'):
                with self.assertRaises(ValueError):
                    preproduction_seconds(proposal, history=history)
            path.write_text(json.dumps(dict(allocations=[])))
            with self.assertRaises(ValueError):
                preproduction_seconds(proposal, history='')

    def test_raw_browsergym_metadata_preserves_executor_evidence_without_duplicate_extraction(self):
        from browsergym.core.env import BrowserEnv
        page = SimpleNamespace(url='https://example.org', title=lambda: 'Fixture')
        env = BrowserEnv.__new__(BrowserEnv)
        env.use_raw_page_output = True
        env.page = page
        env.context = SimpleNamespace(pages=[page])
        env.chat = SimpleNamespace(messages=[dict(role='assistant', message='Actual answer')])
        env.goal_object = [dict(type='text', text='Fixed intent')]
        env.last_action = 'send_msg_to_user [Actual answer]'
        env.last_action_error = 'preserved error'
        env.start_time = time.time()
        with patch('browsergym.core.env._pre_extract', side_effect=AssertionError('Unused DOM extraction')), \
             patch('browsergym.core.env.extract_screenshot', side_effect=AssertionError('Unused screenshot')):
            obs = env._get_obs()
        self.assertIs(obs['page'], page)
        self.assertEqual(obs['url'], page.url)
        self.assertEqual(obs['chat_messages'], tuple(env.chat.messages))
        self.assertEqual(obs['last_action_error'], env.last_action_error)
        self.assertNotIn('axtree_object', obs)
        self.assertNotIn('screenshot', obs)

    def test_fixed_task_major_cohort_partitions_300_tasks_into_three_disjoint_stages(self):
        first = om.cohort(TASKS, COHORTS, ACTOR_SNAPSHOTS, PRIMARY_SELECTORS)
        again = om.cohort(TASKS, COHORTS, ACTOR_SNAPSHOTS, PRIMARY_SELECTORS)
        self.assertEqual(first, again)
        cells = first['cells']
        self.assertEqual(len(cells), 3000); self.assertEqual(len({c['cell_id'] for c in cells}), 3000)
        self.assertEqual([sum(c['stage'] == s for c in cells) for s in om.STAGES], [100, 900, 2000])
        for index in range(300):  # each task's ten conditions are contiguous
            block = cells[10*index:10*index+10]
            self.assertEqual({c['task_index'] for c in block}, {index})
            self.assertEqual({(c['actor'], c['selector']) for c in block},
                             {(a, s) for a in ACTOR_SNAPSHOTS for s in PRIMARY_SELECTORS})
        cohorts = json.loads(COHORTS.read_text())['cohorts']
        self.assertEqual(first['task_order'][:100], cohorts['checkpoint_selection_100']['task_ids'])

    def test_changed_task_file_is_rejected(self):
        with tempfile.NamedTemporaryFile('w', suffix='.jsonl', dir='/tmp') as handle:
            handle.write(TASKS.read_text()[:-2]); handle.flush()
            with self.assertRaises(ValueError):
                om.load_tasks(handle.name)

    def test_actor_view_exposes_only_intent_and_start_url(self):
        record = next(iter(om.load_tasks(TASKS).values()))
        task = om.OM2WTask.from_record(dict(record, evaluator_reference=['NEVER_SEND_THIS']))
        self.assertEqual(set(task.actor_view), {'intent', 'start_url', 'sites'})
        self.assertNotIn('NEVER_SEND_THIS', json.dumps(dict(task.actor_view, sites=list(task.actor_view['sites']))))
        self.assertNotIn('NEVER_SEND_THIS', json.dumps(task.native_config()))
        for bad in ({'require_login': True}, {'storage_state': 'x'}, {'task_id': '../x'}, {'start_url': 'ftp://x'}):
            with self.assertRaises(ValueError):
                om.OM2WTask.from_record(dict(record, metadata=dict(record['metadata'], **bad)))

    def test_routes_replace_fuzzy_with_one_judge_route_per_cell(self):
        reference = json.loads(Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-webarena-20261011/next-stage-preparation/'
                                    'cms-table5-350-v1/proposal.json').read_text())
        routes, cells = om.routes(reference['broker_routes'], reference['cell_routes'], ACTOR_SNAPSHOTS)
        self.assertNotIn('native_fuzzy', [r['purpose'] for r in routes])
        self.assertEqual([r for r in routes if r['purpose'] == om.JUDGE_PURPOSE],
                         [dict(local=False, model=om.JUDGE, purpose=om.JUDGE_PURPOSE, max_input_tokens=120000, max_output_tokens=4096)])
        for actor in ACTOR_SNAPSHOTS:
            for selector, rows in cells[actor].items():
                self.assertEqual(rows[-1], [om.JUDGE, om.JUDGE_PURPOSE])
                self.assertFalse(any(purpose == 'native_fuzzy' for _, purpose in rows))
                self.assertTrue(all(model == actor for model, purpose in rows if purpose in ('policy', 'state_diff', 'terminal')))

    def test_judge_trajectory_contains_only_executed_selected_actions(self):
        rollout = [dict(selected_index=1, proposal=dict(candidates=[dict(thought='a', action='click [1]'),
                                                                   dict(thought='b', action='click [2]')])),
                   dict(selected_index=None, execution_error='No admissible action')]
        text = om.judge_trajectory(rollout, dict(status='executed', action='send_msg_to_user [N/A]'), 'turn_limit')
        self.assertIn('Action 1: click [2]', text); self.assertNotIn('click [1]', text)
        self.assertIn('Action 3: send_msg_to_user [N/A]', text)
        self.assertTrue(text.endswith('Agent stopped: turn_limit (this is not a success verdict)'))


@unittest.skipUnless(SOURCE.exists(), 'Pinned source unavailable')
class EpisodeTests(unittest.TestCase):
    def setUp(self):
        self.network = patch('socket.socket.connect', side_effect=AssertionError('Network forbidden'))
        self.network.start(); self.addCleanup(self.network.stop)
        self.temporary = tempfile.TemporaryDirectory(prefix='om2w-t5-', dir='/tmp'); self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)
        self.task = om.OM2WTask.from_record(next(iter(om.load_tasks(TASKS).values())))

    def config(self, selector, turns=30, actor=ACTOR_SNAPSHOTS[0]):
        from openwebrl.webprm_webarena_piotr import CHECKPOINT, MODEL, REVISION, SOURCE as PIOTR_SOURCE
        return dict(arbiter_source=str(SOURCE), shepherd_source=str(SOURCE.parent/'shepherd'), actor_model=actor,
            selector=selector, seed=42, max_turns=turns, deadline=time.time()+120, reset_id=om.LIVE_RESET_ID,
            identity_sha256='synthetic', cell_name=cell_name(actor, selector), no_search_prompt_version='single_blocks_envelope_v2',
            turn_limit_protocol='released_viewer_na_fallback_v1', piotr_representation='native_operation_descriptor_preaction_status_v2',
            arbiter_model='ZYao720/WebArbiter-7B', arbiter_endpoint='http://critic.invalid/v1',
            shepherd_model='LangAGI-Lab/WebShepherd_8B', shepherd_endpoint='http://critic.invalid/v1',
            piotr_model=MODEL, piotr_revision=REVISION, piotr_checkpoint=str(CHECKPOINT), piotr_source=str(PIOTR_SOURCE),
            piotr_endpoint='http://critic.invalid/v1')

    def run_cell(self, selector, *, request=None, turns=30, actor=ACTOR_SNAPSHOTS[0]):
        folder = self.folder/selector; folder.mkdir()
        config = self.config(selector, turns, actor); request = request or Scripted(count=1 if selector == 'no_search' else 5)
        env = FakeEnv(folder)
        observe = lambda e, o: dict(text='[1] link Plans', url=e.page.url, last_action_error=o.get('last_action_error'),
                                    viewport={'width': 1280, 'height': 720}, element_bounds={})
        result = run_loop(self.task, selector, config, env, ViewerActor(config, folder/'r.jsonl', request=request),
                          ViewerSelector(selector, config, folder/'r.jsonl', request=request), observe,
                          om.om2w_evaluator(self.task, config, request, folder), folder)
        return result, env, request, folder

    def test_all_five_selectors_execute_selected_action_and_are_judged_once(self):
        # Native queue knockout with constant 'Response 1' answers yields original slot 2.
        expected = {'no_search': 'p0', 'gpt4omini': 'p2', 'shepherd8b': 'p0', 'arbiter7b': 'p2', 'piotr_selectionarm': 'p3'}
        for selector in PRIMARY_SELECTORS:
            with self.subTest(selector=selector):
                result, env, request, folder = self.run_cell(selector)
                self.assertTrue(result['valid']); self.assertEqual(result['success'], 1)
                self.assertEqual(env.actions, ['goto [https://example.org/%s]' % expected[selector], 'send_msg_to_user [Plans compared]'])
                judged = [r for r in request.requests if r['purpose'] == om.JUDGE_PURPOSE]
                self.assertEqual(len(judged), 1)
                payload = judged[0]['payload']
                self.assertEqual((payload['model'], payload['max_completion_tokens']), (om.JUDGE, 4096))
                self.assertEqual(set(payload), {'model', 'messages', 'max_completion_tokens'})
                text = payload['messages'][1]['content'][0]['text']
                self.assertIn(self.task.actor_view['intent'], text)
                self.assertIn('send_msg_to_user [Plans compared]', text)
                self.assertNotIn('Candidate 4', text if selector != 'piotr_selectionarm' else '')
                self.assertEqual(payload['messages'][1]['content'][1]['image_url']['detail'], 'high')
                saved = json.loads((folder/'om2w-judge.json').read_text())
                self.assertEqual(saved['parsed'], 1.0)
                self.assertTrue((folder/'native-evaluation/native-verdict.json').exists())
                local = {r['purpose'] for r in request.requests if r.get('local')}
                self.assertEqual(local, {'no_search': set(), 'gpt4omini': set(), 'arbiter7b': {'selector_pair'},
                    'shepherd8b': {'selector_checklist', 'selector_pointwise'},
                    'piotr_selectionarm': {'selector_piotr'}}[selector])

    def test_candidate_controls_use_guided_generation_and_no_critic_calls(self):
        policies = {}
        for selector in ('no_search', 'first5', 'random5', 'arbiter7b', 'piotr_selectionarm'):
            result, env, request, folder = self.run_cell(selector, actor=ACTOR_SNAPSHOTS[1])
            self.assertTrue(result['valid'])
            policies[selector] = [r['payload'] for r in request.requests if r['purpose'] == 'policy']
            if selector in ('first5', 'random5'):
                self.assertFalse(any(r['purpose'].startswith('selector_') for r in request.requests))
                self.assertEqual(sum(r['purpose'] == om.JUDGE_PURPOSE for r in request.requests), 1)
                rollout = json.loads((folder/'rollout.json').read_text())
                self.assertEqual(rollout[0]['proposal']['requested_raw_slots'], 5)
                self.assertEqual(len(rollout[0]['proposal']['candidates']), 5)
        for selector in ('random5', 'arbiter7b', 'piotr_selectionarm'):
            self.assertEqual(policies['first5'], policies[selector])
        self.assertNotEqual(policies['no_search'], policies['first5'])

    def test_failure_verdict_is_valid_zero_and_unparseable_is_invalid(self):
        result, *_ = self.run_cell('no_search', request=Scripted(count=1, verdict='Thoughts: no.\nStatus: failure'))
        self.assertTrue(result['valid']); self.assertEqual(result['success'], 0)
        result, *_ = self.run_cell('arbiter7b', request=Scripted(verdict='I cannot tell.'))
        self.assertFalse(result['valid']); self.assertEqual(result['success'], 0)
        self.assertEqual(result['error']['stage'] if isinstance(result.get('error'), dict) else None, 'native_evaluation')

    def test_turn_limit_executes_released_na_fallback_before_judging(self):
        result, env, request, folder = self.run_cell('no_search', request=Scripted(terminal_after=99, count=1), turns=2)
        self.assertEqual(env.actions[-1], 'send_msg_to_user [N/A]')
        judged = [r for r in request.requests if r['purpose'] == om.JUDGE_PURPOSE][0]['payload']
        self.assertIn('send_msg_to_user [N/A]', judged['messages'][1]['content'][0]['text'])
        self.assertTrue(result['valid'])

    def test_second_actor_snapshot_drives_all_actor_calls_and_mini_drives_gpt_selector(self):
        _, _, request, _ = self.run_cell('gpt4omini', actor=ACTOR_SNAPSHOTS[1])
        for row in request.requests:
            if row['purpose'] in ('policy', 'state_diff', 'terminal'):
                self.assertEqual(row['payload']['model'], ACTOR_SNAPSHOTS[1])
            if row['purpose'] == 'selector_pair':
                self.assertEqual(row['payload']['model'], ACTOR_SNAPSHOTS[0]); self.assertFalse(row.get('local'))

    def test_execution_config_requires_approval_protocol_and_prices(self):
        config = dict(self.config('no_search'), authorization_status='pending')
        with self.assertRaises(EvaluationNotReady):
            om.validate_execution_config(config)
        with patch.dict('os.environ', {'SLURM_JOB_ID': '7'}):
            config = dict(self.config('no_search'), authorization_status='approved', job_id='7', api_usd_cap=1.,
                          protocol='other', protocol_contract={})
            with self.assertRaisesRegex(EvaluationNotReady, 'protocol'):
                om.validate_execution_config(config)
            config.update(protocol=om.PROTOCOL, protocol_contract=om.protocol_contract(str(SOURCE)),
                          prices_usd_per_token={ACTOR_SNAPSHOTS[0]: [1e-7, 1e-7]})
            with self.assertRaisesRegex(EvaluationNotReady, 'prices'):
                om.validate_execution_config(config)


class ControllerTests(unittest.TestCase):
    def setUp(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location('om2w_launch', REPO/'scripts/launch_webprm_om2w_table5.py')
        self.launch = importlib.util.module_from_spec(spec); spec.loader.exec_module(self.launch)
        spec = importlib.util.spec_from_file_location('om2w_collect', REPO/'scripts/collect_webprm_om2w_table5.py')
        self.collect = importlib.util.module_from_spec(spec); spec.loader.exec_module(self.collect)

    def test_frozen_watchdog_exception_is_counted_only_as_closed_invalid(self):
        cell=dict(task_id='t1',actor='actor',selector='arbiter7b')
        plan=dict(protocol='frozen')
        watchdog=dict(reason='phase_timeout',remaining_owned_processes={},
            last_phase=dict(identity_sha256=digest(plan)))
        result=dict(task_id='om2w_t1',arm='actor__arbiter7b',identity_sha256=digest(plan),
            attempted=True,valid=False,success=0,error_type='EpisodeWatchdogTimeout',watchdog=watchdog)
        with tempfile.TemporaryDirectory(dir='/tmp') as root:
            path=Path(root)/'result.json';wd=Path(root)/'watchdog.json'
            path.write_text(json.dumps(result));wd.write_text(json.dumps(watchdog))
            self.assertEqual(self.collect.verified_watchdog_exception(path,cell,plan),result)
            for change in (dict(valid=True),dict(success=1),dict(success=False),dict(actor_model='actor'),
                           dict(selector='arbiter7b'),dict(arm='other__arbiter7b'),dict(identity_sha256='wrong')):
                path.write_text(json.dumps(dict(result,**change)))
                with self.subTest(change=change),self.assertRaises(ValueError):
                    self.collect.verified_watchdog_exception(path,cell,plan)
            changed=dict(watchdog,remaining_owned_processes={'1':{}})
            path.write_text(json.dumps(dict(result,watchdog=changed)));wd.write_text(json.dumps(changed))
            with self.assertRaises(ValueError):self.collect.verified_watchdog_exception(path,cell,plan)

    def test_previous_epoch_requires_closed_chain(self):
        with tempfile.TemporaryDirectory(dir='/tmp') as root:
            path = Path(root)/'events.jsonl'
            self.assertIsNone(self.launch.previous_closed_epoch(path))
            def event(sequence, previous, kind):
                core = dict(sequence=sequence, previous_sha256=previous, event=kind)
                return dict(core, event_sha256=digest(core))
            a = event(0, '0'*64, 'header'); b = event(1, a['event_sha256'], 'epoch_started')
            path.write_text(json.dumps(a)+'\n')
            self.assertIsNone(self.launch.previous_closed_epoch(path))
            path.write_text(json.dumps(a)+'\n'+json.dumps(b)+'\n')
            with self.assertRaisesRegex(ValueError, 'not closed'):
                self.launch.previous_closed_epoch(path)
            c = event(2, b['event_sha256'], 'epoch_closed')
            path.write_text(json.dumps(a)+'\n'+json.dumps(b)+'\n'+json.dumps(c)+'\n')
            self.assertEqual(self.launch.previous_closed_epoch(path), c['event_sha256'])
            path.write_text(json.dumps(a)+'\n'+json.dumps(dict(b, sequence=5))+'\n')
            with self.assertRaisesRegex(ValueError, 'chain'):
                self.launch.previous_closed_epoch(path)

    def test_approval_must_name_cap_stage_resources_and_quote(self):
        proposal = dict(stage='pilot10', resources=dict(gpus=1), limits=dict(api_usd=40.))
        approval = dict(status='approved', proposal_sha256=digest(proposal), api_usd_cap=40., resources=proposal['resources'],
                        limits=proposal['limits'], stage='pilot10', user_approval_quote='yes $40')
        self.launch.validate_approval(proposal, approval)
        for change in (dict(api_usd_cap=41.), dict(stage='next90'), dict(user_approval_quote=' '), dict(status='pending')):
            with self.assertRaises(ValueError):
                self.launch.validate_approval(proposal, dict(approval, **change))

    def test_collector_withholds_rates_until_complete_and_counts_unresolved_claims(self):
        cell = lambda t, s: dict(cell_id=f'om2w_{t}/a__{s}', actor='a', selector=s, task_id=t)
        rows = [(cell('t1', 'no_search'), dict(attempted=True, valid=True, success=0), 'recorded'),
                (cell('t1', 'arbiter7b'), dict(attempted=True, valid=True, success=1), 'recorded'),
                (cell('t2', 'no_search'), dict(attempted=True, valid=False, success=0, error_type='RuntimeError'), 'recorded'),
                (cell('t2', 'arbiter7b'), None, 'pending')]
        value = self.collect.summarize(rows)
        arb = next(r for r in value['by_condition'] if r['selector'] == 'arbiter7b')
        self.assertEqual((arb['planned'], arb['recorded'], arb['valid'], arb['successes']), (2, 1, 1, 1))
        self.assertEqual((arb['overall_success_pct'], arb['valid_only_success_pct']), (None, None))
        self.assertTrue(value['rates_withheld'])
        self.assertEqual(value['paired_vs_no_search'], [])
        unresolved = list(rows)
        unresolved[-1] = (rows[-1][0], None, 'attempted_without_outcome')
        unresolved_row = next(r for r in self.collect.summarize(unresolved)['by_condition'] if r['selector'] == 'arbiter7b')
        self.assertEqual(unresolved_row['attempted'], 2)
        base = next(r for r in value['by_condition'] if r['selector'] == 'no_search')
        self.assertEqual(base['invalid_reasons'], {'RuntimeError': 1})
        rows[-1] = (rows[-1][0], dict(attempted=True, valid=True, success=0), 'recorded')
        complete = self.collect.summarize(rows)
        self.assertFalse(complete['rates_withheld'])
        arb = next(r for r in complete['by_condition'] if r['selector'] == 'arbiter7b')
        self.assertEqual((arb['overall_success_pct'], arb['valid_only_success_pct']), (50., 50.))
        paired = complete['paired_vs_no_search'][0]
        self.assertEqual((paired['paired_tasks'], paired['difference_pp']), (2, 50.))



class ScreenshotTest(unittest.TestCase):
    def test_viewport_screenshots_use_cdp_capture_without_font_wait(self):
        from playwright.sync_api import Page
        sent, original_calls = [], []

        class Session:
            def send(self, method, params):
                sent.append((method, params)); return dict(data=base64.b64encode(b'img').decode())

            def detach(self):
                sent.append('detached')
        page = SimpleNamespace(context=SimpleNamespace(new_cdp_session=lambda p: Session()))
        with patch.object(Page, 'screenshot', lambda self, **kw: original_calls.append(kw)):
            om.use_cdp_screenshots(); om.use_cdp_screenshots()  # idempotent
            self.assertEqual(Page.screenshot(page, type='jpeg'), b'img')
            Page.screenshot(page, type='png', full_page=True)
        self.assertEqual(sent, [('Page.captureScreenshot', dict(format='jpeg')), 'detached'])
        self.assertEqual(original_calls, [dict(type='png', path=None, quality=None, full_page=True)])


if __name__ == '__main__':
    unittest.main()


class NavigationTitleTest(unittest.TestCase):
    class Page:
        def __init__(self, failures, message='Page.title: Execution context was destroyed, most likely because of a navigation'):
            self.failures, self.message, self.waits = failures, message, 0
        def title(self):
            if self.failures:
                self.failures -= 1
                raise RuntimeError(self.message)
            return 'Loaded'
        def wait_for_load_state(self, state, timeout):
            self.waits += 1

    def test_title_read_retries_through_navigation(self):
        page = self.Page(2)
        with patch.object(om.time, 'sleep'):
            self.assertEqual(om._stable_title(page), 'Loaded')
        self.assertEqual(page.waits, 2)

    def test_persistent_navigation_yields_empty_title_but_other_errors_raise(self):
        with patch.object(om.time, 'sleep'):
            self.assertEqual(om._stable_title(self.Page(99)), '')
            with self.assertRaisesRegex(RuntimeError, 'boom'):
                om._stable_title(self.Page(1, 'boom'))


class InitialNavigationPolicyTest(unittest.TestCase):
    class Page:
        url = 'https://example.test/'
        def __init__(self, status):
            self.status = status
        def goto(self, url, wait_until, timeout):
            return SimpleNamespace(status=self.status)

    def test_om2w_proceeds_on_http_errors_but_default_still_rejects(self):
        from openwebrl.webprm_runtime import live_task_type
        task_type = live_task_type()
        live = task_type(1, 'https://example.test/', 'goal', deadline=time.time() + 60, reject_http_errors=False)
        goal, info = live.setup(self.Page(403))
        self.assertEqual(info['initial_navigation']['status'], 403)
        strict = task_type(1, 'https://example.test/', 'goal', deadline=time.time() + 60)
        with self.assertRaisesRegex(RuntimeError, 'HTTP status 403'):
            strict.setup(self.Page(403))
