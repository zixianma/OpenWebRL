"""Native computation replay on offline values; no browser/provider/site calls."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from openwebrl.webprm_evaluator_transcript import (EvaluationRecorder,
    TRANSCRIPT_CONFIGURATION, encoded, replay_native_evaluation)
from openwebrl.webprm_webarena import EvaluationNotReady, load_native_evaluator
from openwebrl.webprm_webarena_helpers import NativeHelpers, OwnedShoppingRequests, HELPER_SHA256
from openwebrl.webprm_webarena_worker import protocol_definition, validate_protocol_contract
from tests.test_webprm_webarena_helpers import SOURCE, ORIGIN, ACCOUNTS, TRAJECTORY, Page, task


class TranscriptTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory(prefix='webprm-native-transcript-', dir='/tmp')
        self.addCleanup(self.tmp.cleanup); self.root = Path(self.tmp.name)
        for name, options in [('socket.socket.connect', {'side_effect': AssertionError('No network')}),
                              ('time.sleep', {'return_value': None})]:
            wrapped = patch(name, **options); wrapped.start(); self.addCleanup(wrapped.stop)

    def record(self, config, page=None, helpers=None):
        recorder = EvaluationRecorder(self.root / 'transcript.json')
        evaluator = load_native_evaluator(SOURCE, observation_recorder=recorder, deployment_helpers=helpers)
        verdict = evaluator.evaluate(config, TRAJECTORY, page, termination='stop')
        document = json.loads((self.root / 'transcript.json').read_text())
        self.assertEqual(verdict['observation_transcript']['sha256'],
            hashlib.sha256((self.root / 'transcript.json').read_bytes()).hexdigest())
        return document, verdict

    def replay(self, config, document, verdict):
        with patch('urllib.request.build_opener', side_effect=AssertionError('Replay must not construct HTTP transport')):
            return replay_native_evaluation(SOURCE, config, TRAJECTORY, document, verdict)

    def rebound(self, document, verdict):
        value = deepcopy(verdict)
        value['observation_transcript']['sha256'] = hashlib.sha256(encoded(document) + b'\n').hexdigest()
        value['observation_transcript']['bytes'] = len(encoded(document) + b'\n')
        return value

    def test_plain_string_has_empty_complete_transcript_and_identical_default_verdict(self):
        config = {'intent':'Fixture', 'eval':{'eval_types':['string_match'],
                  'reference_answers':{'exact_match':'done'}}}
        document, verdict = self.record(config)
        self.assertEqual(document['events'], [])
        proof = self.replay(config, document, verdict)
        legacy = load_native_evaluator(SOURCE).evaluate(config, TRAJECTORY, termination='stop')
        self.assertEqual(proof['result'], legacy)
        self.assertNotIn('observation_transcript', legacy)
        self.assertFalse(proof['website_truth_independently_verified'])

    def test_navigation_content_and_url_values_replay_without_live_page(self):
        page = Page()
        def goto(url):
            page.url = url
            return SimpleNamespace(status=200, url=url, headers={'secret':'must never serialize'})
        page.goto = goto
        config = task(url=ORIGIN + '/target')
        document, verdict = self.record(config, page)
        self.assertTrue(self.replay(config, document, verdict)['verified'])
        serialized = json.dumps(document)
        self.assertNotIn('must never serialize', serialized)
        event = next(e for e in document['events'] if e['operation'] == 'page.goto')
        self.assertEqual(event['value'], dict(kind='navigation_response',status=200,url=ORIGIN+'/target'))
        self.assertIn('page.content', [e['operation'] for e in document['events']])

    def test_javascript_values_inputs_and_kwargs_are_exact_replay_contract(self):
        config = task(locator="document.querySelector('h1').textContent")
        document, verdict = self.record(config, Page())
        self.assertTrue(self.replay(config, document, verdict)['verified'])
        modified = deepcopy(document)
        event = next(e for e in modified['events'] if e['operation'] == 'page.evaluate')
        event['inputs']['args'][0] += ' /* changed */'
        with self.assertRaisesRegex(EvaluationNotReady, 'native result discarded'):
            self.replay(config, modified, self.rebound(modified, verdict))

    def test_reddit_helper_recomputes_pinned_body_from_nested_url_observation(self):
        config = task(url="func:reddit_get_post_url('__last_url__')")
        document, verdict = self.record(config, Page(), NativeHelpers(SOURCE))
        event = next(e for e in document['events'] if e['operation'] == 'helper')
        self.assertEqual(event['inputs']['native_source_sha256'], HELPER_SHA256)
        self.assertEqual(event['inputs']['function'], 'reddit_get_post_url')
        self.assertTrue(event['events'])
        self.assertTrue(self.replay(config, document, verdict)['verified'])
        changed = deepcopy(document)
        next(e for e in changed['events'] if e['operation'] == 'helper')['value'] = ORIGIN + '/changed'
        with self.assertRaisesRegex(EvaluationNotReady, 'helper recomputation differs'):
            self.replay(config, changed, self.rebound(changed, verdict))

    def test_gitlab_helper_recomputes_from_recorded_evaluate_values(self):
        config = task(locator="func:gitlab_get_project_memeber_role(__page__, 'fixture')",expected='Owner')
        page = Page()
        replies = iter([0, 'Owner'])
        page.evaluate = lambda _: next(replies)
        document, verdict = self.record(config, page, NativeHelpers(SOURCE))
        event = next(e for e in document['events'] if e['operation'] == 'helper')
        self.assertTrue(verdict['success'])
        self.assertEqual([e['operation'] for e in event['events']], ['page.evaluate','page.evaluate'])
        self.assertTrue(self.replay(config, document, verdict)['verified'])

    def test_shopping_replay_is_explicit_return_boundary_and_never_copies_auth_secrets(self):
        config = task(url='func:shopping_get_latest_order_url()')
        helper = NativeHelpers(SOURCE,shopping_origin=ORIGIN,shopping_accounts=ACCOUNTS)
        responses = [SimpleNamespace(status_code=200,json=lambda:'private-auth-token'),
                     SimpleNamespace(status_code=200,json=lambda:{'items':[{'increment_id':'23'}]})]
        with patch.object(OwnedShoppingRequests,'request',side_effect=responses):
            document, verdict = self.record(config, Page(), helper)
        event = next(e for e in document['events'] if e['operation'] == 'helper')
        self.assertEqual(event['events'], [])
        text = json.dumps(document)
        for private in ('private-auth-token',ACCOUNTS['shopping_site_admin']['password'],
                        ACCOUNTS['shopping_site_admin']['username']):
            self.assertNotIn(private, text)
        proof = self.replay(config, document, verdict)
        self.assertTrue(proof['verified'])
        self.assertIn('shopping_return_boundary',proof['replay_scope'])

    def test_native_js_missing_element_error_preserves_completed_zero_replay(self):
        error = type('Error',(Exception,),{'__module__':'playwright._impl._errors'})
        page = Page()
        def evaluate(*args): raise error('TypeError: missing synthetic node')
        page.evaluate = evaluate
        config = task(locator="document.querySelector('h1').textContent")
        document, verdict = self.record(config,page)
        self.assertFalse(verdict['success'])
        self.assertTrue(self.replay(config,document,verdict)['verified'])

    def test_generic_transport_error_is_failed_evidence_not_valid_zero(self):
        page = Page()
        def evaluate(*args): raise RuntimeError('synthetic transport disconnect')
        page.evaluate = evaluate
        recorder = EvaluationRecorder(self.root/'failed.json')
        evaluator = load_native_evaluator(SOURCE,observation_recorder=recorder)
        with self.assertRaises(EvaluationNotReady):
            evaluator.evaluate(task(locator='document.title'),TRAJECTORY,page)
        document = json.loads((self.root/'failed.json').read_text())
        self.assertEqual(document['status'],'failed')
        self.assertIsNone(document['outcome_sha256'])

    def test_oversize_dom_fails_closed_even_when_native_catches_locator_errors(self):
        page = Page(value='x'*(TRANSCRIPT_CONFIGURATION['max_value_bytes']+1))
        recorder = EvaluationRecorder(self.root/'oversize.json')
        evaluator = load_native_evaluator(SOURCE,observation_recorder=recorder)
        with self.assertRaises(EvaluationNotReady):
            evaluator.evaluate(task(locator='document.title'),TRAJECTORY,page)
        self.assertEqual(json.loads((self.root/'oversize.json').read_text())['status'],'failed')
        self.assertLess((self.root/'oversize.json').stat().st_size,10000)

    def test_non_json_dom_value_and_excess_event_count_are_not_silently_truncated(self):
        recorder = EvaluationRecorder(None);recorder.start({'fixture':True})
        with self.assertRaises(EvaluationNotReady):recorder.call('test',{},lambda:object())
        with self.assertRaises(EvaluationNotReady):recorder.finish(outcome={'score':0})
        recorder.finish(error=ValueError())
        recorder = EvaluationRecorder(None);recorder.start({'fixture':True})
        for _ in range(TRANSCRIPT_CONFIGURATION['max_events']):recorder.call('test',{},lambda:None)
        with self.assertRaises(EvaluationNotReady):recorder.call('test',{},lambda:None)

    def test_missing_extra_or_changed_observations_and_task_verdict_binding_fail(self):
        config = task();document, verdict = self.record(config,Page())
        for mutate in (lambda d:d['events'].pop(),lambda d:d['events'].append(deepcopy(d['events'][-1])),
                       lambda d:d['binding'].update(task_sha256='0'*64),
                       lambda d:d.update(outcome_sha256='0'*64)):
            changed = deepcopy(document);mutate(changed)
            with self.assertRaises(EvaluationNotReady):
                self.replay(config,changed,self.rebound(changed,verdict))
        with self.assertRaises(EvaluationNotReady):self.replay(config,document,{})

    def test_recording_protocol_is_versioned_without_changing_default(self):
        before = protocol_definition(SOURCE)
        after = protocol_definition(SOURCE,observation_configuration=TRANSCRIPT_CONFIGURATION)
        self.assertNotIn('evaluator_observations',before)
        config = dict(arbiter_source=str(SOURCE),protocol=after['identity'],protocol_contract=after,
                      evaluator_observations=deepcopy(TRANSCRIPT_CONFIGURATION))
        self.assertEqual(validate_protocol_contract(config),after)
        with self.assertRaises(EvaluationNotReady):
            validate_protocol_contract(dict(config,evaluator_observations=None))

    def test_guarded_fuzzy_replay_retains_raw_positive_and_effective_zero(self):
        from openwebrl.webprm_webarena import ABSTENTION_EVALUATOR_PROTOCOL
        class Judge:
            enabled=True
            manifest={'synthetic_only':True}
            def __init__(self):self.provider_calls=0
            def fuzzy_match(self,*args):self.provider_calls+=1;return 1.
            def ua_match(self,*args):raise AssertionError('Unexpected unachievable route')
        config={'intent':'Synthetic numeric fixture','eval':{'eval_types':['string_match'],
            'reference_answers':{'fuzzy_match':['0']}}}
        trajectory=deepcopy(TRAJECTORY);trajectory[-1]['answer']='N/A'
        recorder=EvaluationRecorder(self.root/'fuzzy.json')
        evaluator=load_native_evaluator(SOURCE,observation_recorder=recorder,
            scoring_protocol=ABSTENTION_EVALUATOR_PROTOCOL,fuzzy_judge=Judge())
        verdict=evaluator.evaluate(config,trajectory,termination='stop')
        self.assertFalse(verdict['success']);self.assertTrue(verdict['native_success'])
        proof=replay_native_evaluation(SOURCE,config,trajectory,recorder.document,verdict,fuzzy_judge=Judge())
        self.assertTrue(proof['verified']);self.assertFalse(proof['result']['success'])
        self.assertTrue(proof['result']['native_success'])

    def test_worker_evaluator_emits_durable_transcript_when_explicitly_enabled(self):
        from openwebrl.webprm_webarena_episode import TaskEnvelope
        from openwebrl.webprm_webarena_worker import episode_evaluator
        native=dict(task_id=999,sites=['map'],intent='Synthetic worker fixture',start_url='__MAP__',
            storage_state=None,eval={'eval_types':['string_match'],'reference_answers':{'exact_match':'done'}})
        wrapped=TaskEnvelope.from_release(native,{'map':ORIGIN})
        config=dict(arbiter_source=str(SOURCE),evaluator_observations=deepcopy(TRANSCRIPT_CONFIGURATION))
        evaluator=episode_evaluator(wrapped,config,None,self.root)
        verdict=evaluator.evaluate(wrapped.native_config(),TRAJECTORY,termination='stop')
        path=self.root/'native-evaluator-transcript.json'
        self.assertTrue(path.is_file())
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),verdict['observation_transcript']['sha256'])


if __name__=='__main__':unittest.main()
