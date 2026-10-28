"""Offline Piotr/WebArena causal grounding; no model loads or network requests."""
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import time
import unittest
from unittest.mock import patch

from PIL import Image

from openwebrl.webprm_webarena import EvaluationNotReady
from openwebrl.webprm_webarena_piotr import (
    CHECKPOINT, MODEL, REVISION, SOURCE, PiotrSelectionARMSelector,
    ground_native_action, validate_piotr_selectionarm,
)

ARBITER = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-baselines-20261009/sources/arbiter')


def config():
    return dict(piotr_model=MODEL, piotr_revision=REVISION, piotr_checkpoint=str(CHECKPOINT),
                piotr_source=str(SOURCE), arbiter_source=str(ARBITER),
                piotr_endpoint='http://127.0.0.1:8197/v1', deadline=time.time() + 60)


def screenshot(width=1280, height=720):
    stream = io.BytesIO()
    Image.new('RGB', (width, height), color='white').save(stream, format='PNG')
    return stream.getvalue()


def fixture():
    actions = ['click [1]', 'type [1] café', 'clear [1]', 'hover [1]', 'scroll [down]']
    proposal = dict(terminal=False, exact_five_raw_slots=True, raw_slot_count=5,
        candidates=[dict(raw_slot=i, thought='Original GPT thought ' + str(i), action=action)
                    for i, action in enumerate(actions)],
        context=dict(intent='Synthetic visible task', current_url='http://reddit.invalid',
                     text_observation='AX1 input', trajectory='', state_diff=''))
    evidence = dict(screenshot=screenshot(), viewport={'width': 1280, 'height': 720},
                    element_bounds={'1': [600, 320, 80, 80]}, history=[])
    return proposal, evidence


@unittest.skipUnless(SOURCE.exists() and ARBITER.exists() and CHECKPOINT.exists(), 'Pinned local inputs unavailable')
class PiotrTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.verified = validate_piotr_selectionarm(config())

    def setUp(self):
        self.network = patch('socket.socket.connect', side_effect=AssertionError('No network allowed'))
        self.network.start(); self.addCleanup(self.network.stop)
        self.requests = []
        self.response = dict(model=MODEL, choices=[dict(finish_reason='stop', message={'content': '{"selection":2}'})])
        # Verify actual source/checkpoint once; tests never load or rehash tensors.
        self.validation = patch('openwebrl.webprm_webarena_piotr.validate_piotr_selectionarm', return_value=deepcopy(self.verified))
        self.validation.start(); self.addCleanup(self.validation.stop)
        self.proposal, self.evidence = fixture()

    def request(self, cfg, payload, journal, **kwargs):
        self.requests.append(dict(payload=deepcopy(payload), kwargs=kwargs))
        return deepcopy(self.response)

    def selector(self, request=None):
        return PiotrSelectionARMSelector(config(), '/tmp/piotr-offline-unused-journal',
                                         request=self.request if request is None else request)

    def test_frozen_identity_records_exact_metadata_and_unverified_runtime(self):
        self.assertEqual(self.verified['revision'], REVISION)
        self.assertEqual(sum(row['bytes'] for row in self.verified['weights']), 8875719328)
        self.assertEqual(len(self.verified['weights'][0]['cached_lfs_sha256']), 64)
        self.assertFalse(self.verified['weights_fresh_sha256_verified'])
        self.assertFalse(self.verified['runtime_model_loading_verified'])
        self.assertTrue(self.verified['action_representation_adapted'])

    def test_canonical_prompt_schema_and_original_selected_index(self):
        original = deepcopy(self.proposal)
        index, trace = self.selector().select(self.proposal, 0, evidence=self.evidence)
        self.assertEqual(index, 1)
        self.assertEqual(self.proposal, original)
        self.assertEqual(trace['selected_raw_slot'], 1)
        self.assertEqual(json.loads(trace['selected_grounded_action'])['native_action'], original['candidates'][1]['action'])
        call = self.requests[0]
        self.assertEqual(call['kwargs']['purpose'], 'selector_piotr')
        self.assertTrue(call['kwargs']['local'])
        self.assertEqual(call['payload']['model'], MODEL)
        self.assertEqual(call['payload']['temperature'], 0)
        schema = call['payload']['response_format']['json_schema']
        self.assertTrue(schema['strict'])
        self.assertEqual(schema['name'], 'vision_selection_nocot')
        self.assertEqual(schema['schema']['properties']['selection']['maximum'], 5)
        blocks = call['payload']['messages'][1]['content']
        self.assertEqual([b['type'] for b in blocks], ['text', 'image_url', 'text'])
        self.assertIn('Original GPT thought 1', blocks[2]['text'])
        self.assertNotIn('image/png;base64', json.dumps(trace))

    def test_trace_representation_matches_actual_v1_and_v2_descriptors(self):
        from openwebrl.webprm_webarena_piotr import REPRESENTATION, REPRESENTATION_V2
        original = deepcopy(self.proposal)
        for version in (REPRESENTATION, REPRESENTATION_V2):
            with self.subTest(representation=version), patch(
                    'openwebrl.webprm_webarena_piotr.validate_piotr_selectionarm',
                    return_value=dict(self.verified, representation=version)):
                options = config()
                if version == REPRESENTATION_V2:
                    options['piotr_representation'] = version
                selector = PiotrSelectionARMSelector(options, '/tmp/piotr-offline-unused-journal', request=self.request)
                index, trace = selector.select(self.proposal, 0, evidence=self.evidence)
                self.assertEqual(trace['representation'], selector.manifest['representation'])
                descriptor = json.loads(trace['selected_grounded_action'])
                self.assertEqual('target_status' in descriptor, version == REPRESENTATION_V2)
                self.assertIn(trace['selected_grounded_action'], self.requests[-1]['payload']['messages'][1]['content'][2]['text'])
                self.assertEqual((index, trace['selected_raw_slot']), (1, 1))
                self.assertEqual(descriptor['native_action'], original['candidates'][1]['action'])
                self.assertEqual(self.proposal, original)

    def test_raw_center_and_normalized_coordinates_match_exact_native_dispatch(self):
        _, trace = self.selector().select(self.proposal, 0, evidence=self.evidence)
        for row in trace['grounded_candidates'][:4]:
            self.assertEqual(row['grounding']['native_center_css_pixels'], [640, 360])
            self.assertEqual(row['grounding']['normalized_point_2d'], [500, 500])
            self.assertEqual(row['descriptor']['point_2d'], [500, 500])
        typed = trace['grounded_candidates'][1]['descriptor']
        self.assertTrue(typed['clear_existing'])
        self.assertTrue(typed['press_enter'])
        self.assertEqual(typed['text'], 'café')
        scrolled = trace['grounded_candidates'][4]['descriptor']
        self.assertEqual(scrolled['amount'], 'one window.innerHeight')
        self.assertEqual(scrolled['mechanism'], 'assign scrollTop')

    def test_public_normalizer_cannot_rewrite_typed_click_literal(self):
        self.proposal['candidates'][1]['action'] = 'type [1] literal click(640,360)'
        _, trace = self.selector().select(self.proposal, 0, evidence=self.evidence)
        display = trace['grounded_candidates'][1]['action']
        self.assertEqual(json.loads(display)['text'], 'literal click(640,360)')
        prompt = self.requests[0]['payload']['messages'][1]['content'][2]['text']
        self.assertIn(display, prompt)
        self.assertNotIn('click(500, 500)', prompt)

    def test_saved_history_keeps_earlier_grounding_and_last_five_window(self):
        selector = self.selector()
        _, first = selector.select(self.proposal, 0, evidence=self.evidence)
        row = dict(action=self.proposal['candidates'][1]['action'], thought='Original past thought',
                   selectionarm_action=first['selected_grounded_action'])
        self.evidence['history'] = [dict(row, thought='History ' + str(i)) for i in range(7)]
        self.evidence['element_bounds']['1'] = [0, 0, 80, 80]
        _, current = selector.select(self.proposal, 7, evidence=self.evidence)
        history_text = self.requests[-1]['payload']['messages'][1]['content'][0]['text']
        self.assertIn(first['selected_grounded_action'], history_text)
        self.assertNotEqual(first['grounded_candidates'][1]['grounding'], current['grounded_candidates'][1]['grounding'])
        self.assertNotIn('History 0', history_text)
        self.assertNotIn('History 1', history_text)
        self.assertIn('History 6', history_text)

    def test_unknown_or_wrong_historical_grounding_rejected(self):
        for row in [dict(action='click [1]', thought='previous'),
                    dict(action='click [1]', thought='previous', selectionarm_action='{"native_action":"click [2]"}')]:
            self.evidence['history'] = [row]
            with self.assertRaises(EvaluationNotReady):
                self.selector().select(self.proposal, 1, evidence=self.evidence)
        self.assertEqual(self.requests, [])

    def test_screenshot_and_grounding_fail_before_request(self):
        changes = [dict(screenshot=screenshot(640, 360)), dict(screenshot=b'not PNG'),
                   dict(viewport={'width': 640, 'height': 360}), dict(element_bounds={}),
                   dict(element_bounds={'1': [1280, 0, 80, 80]}),
                   dict(element_bounds={'1': [0, 0, float('nan'), 80]})]
        for change in changes:
            with self.subTest(change=list(change)), self.assertRaises(EvaluationNotReady):
                self.selector().select(self.proposal, 0, evidence=dict(self.evidence, **change))
        self.assertEqual(self.requests, [])

    def test_tabs_forward_and_missing_raw_slots_are_explicitly_unsupported(self):
        for action in ['go_forward', 'new_tab', 'tab_focus [1]', 'close_tab', 'click [named button]']:
            self.proposal['candidates'][0]['action'] = action
            with self.subTest(action=action), self.assertRaises(EvaluationNotReady):
                self.selector().select(self.proposal, 0, evidence=self.evidence)
        self.proposal, self.evidence = fixture()
        self.proposal['candidates'].pop()
        with self.assertRaisesRegex(EvaluationNotReady, 'five'):
            self.selector().select(self.proposal, 0, evidence=self.evidence)
        self.assertEqual(self.requests, [])

    def test_page_navigation_and_native_key_mapping_are_described_without_execution(self):
        selector = self.selector()
        for action, kind in [('goto [https://example.invalid/a]', 'goto_url'), ('go_back', 'go_back'), ('press [CTRL+A]', 'key_press')]:
            result = ground_native_action(action, parse_action=selector.harness.parse_action,
                bounds={}, viewport=self.evidence['viewport'], key_mapping=selector.harness.key_mapping)
            self.assertEqual(result['descriptor']['operation'], kind)
            self.assertEqual(result['descriptor']['native_action'], action)
        self.assertEqual(self.requests, [])

    def test_no_evaluator_or_post_action_fields_reach_model(self):
        self.proposal['eval'] = {'reference_answers': 'SECRET_REFERENCE'}
        self.proposal['context'].update(reference_answer='SECRET_REFERENCE', outcome='SECRET_OUTCOME',
                                        state_diff='SECRET_POST_ACTION', text_observation='UNUSED_AX_TEXT')
        self.selector().select(self.proposal, 0, evidence=self.evidence)
        payload = json.dumps(self.requests[0]['payload'])
        for forbidden in ['SECRET_REFERENCE', 'SECRET_OUTCOME', 'SECRET_POST_ACTION', 'UNUSED_AX_TEXT']:
            self.assertNotIn(forbidden, payload)
        with self.assertRaisesRegex(EvaluationNotReady, 'Only explicit'):
            self.selector().select(self.proposal, 0, evidence=dict(self.evidence, outcome='SECRET'))

    def test_global_environment_cannot_change_canonical_prompt_or_be_mutated(self):
        with patch.dict(os.environ, {'VISION_ABLATE_IMAGE': '1', 'VISION_ABLATE_CAND_THOUGHTS': '1'}):
            before = dict(os.environ)
            self.selector().select(self.proposal, 0, evidence=self.evidence)
            self.assertEqual(dict(os.environ), before)
        blocks = self.requests[-1]['payload']['messages'][1]['content']
        self.assertEqual(blocks[1]['type'], 'image_url')
        self.assertIn('Original GPT thought', blocks[2]['text'])

    def test_bad_responses_never_fall_back_or_change_original_candidate(self):
        selector = self.selector()
        for raw in ['garbage', '{"selection":0}', '{"selection":6}', '{"selection":true}',
                    '{"selection":2,"thought":"extra"}', 'prefix {"selection":1}']:
            self.response['choices'][0]['message']['content'] = raw
            with self.subTest(raw=raw), self.assertRaises(EvaluationNotReady):
                selector.select(self.proposal, 0, evidence=self.evidence)
        self.response['choices'][0]['message']['content'] = '{"selection":1}'
        self.response['choices'][0]['finish_reason'] = 'length'
        with self.assertRaises(EvaluationNotReady):
            selector.select(self.proposal, 0, evidence=self.evidence)
        self.response['choices'][0]['finish_reason'] = 'stop'
        self.response['model'] = 'replacement'
        with self.assertRaises(EvaluationNotReady):
            selector.select(self.proposal, 0, evidence=self.evidence)

    def selector_v2(self):
        from openwebrl.webprm_webarena_piotr import REPRESENTATION_V2
        cfg=dict(config(),piotr_representation=REPRESENTATION_V2)
        with patch('openwebrl.webprm_webarena_piotr.validate_piotr_selectionarm',
                   return_value=dict(self.verified,representation=REPRESENTATION_V2)):
            return PiotrSelectionARMSelector(cfg,'/tmp/piotr-v2-unused',request=self.request)

    def test_v2_manifest_is_explicit_and_v1_remains_default(self):
        from openwebrl.webprm_webarena_piotr import REPRESENTATION,REPRESENTATION_V2
        self.assertEqual(validate_piotr_selectionarm(config())['representation'],REPRESENTATION)
        self.assertEqual(validate_piotr_selectionarm(dict(config(),piotr_representation=REPRESENTATION_V2))['representation'],REPRESENTATION_V2)
        for value in ['',True,'guessed']:
            with self.assertRaises(EvaluationNotReady):validate_piotr_selectionarm(dict(config(),piotr_representation=value))

    def test_v2_missing_targets_keep_five_slots_without_invented_points(self):
        self.evidence['element_bounds']={};before=deepcopy(self.proposal)
        selected,trace=self.selector_v2().select(self.proposal,0,evidence=self.evidence)
        self.assertEqual(selected,1);self.assertEqual(self.proposal,before);self.assertEqual(len(self.requests),1)
        self.assertEqual(len(trace['grounded_candidates']),5)
        for row in trace['grounded_candidates'][:4]:
            self.assertEqual(row['descriptor']['target_status'],'unavailable')
            self.assertIsNone(row['descriptor']['point_2d']);self.assertIsNone(row['grounding']['native_center_css_pixels'])
        self.assertEqual(trace['selected_raw_slot'],1)
        self.assertEqual(json.loads(trace['selected_grounded_action'])['native_action'],before['candidates'][1]['action'])

    def test_v2_offscreen_geometry_is_measured_not_clamped_or_predicted(self):
        self.evidence['element_bounds']['1']=[1300,50,100,20]
        _,trace=self.selector_v2().select(self.proposal,0,evidence=self.evidence)
        desc=trace['grounded_candidates'][0]['descriptor']
        self.assertEqual(desc['target_status'],'center_outside_viewport');self.assertIsNone(desc['point_2d'])
        self.assertEqual(desc['observed_center_css_pixels'],[1350,60])
        self.assertEqual(desc['observed_bounds_css_pixels'],[1300,50,100,20])
        self.assertNotIn('success',desc);self.assertNotIn('will_fail',desc)
        self.assertNotIn('coordinate_space',desc)

    def test_v2_tab_forward_and_pool_terminal_keep_native_operations(self):
        cases=[('go_forward','go_forward'),('new_tab','new_tab'),('tab_focus [2]','page_focus'),
               ('close_tab','page_close'),('send_msg_to_user [literal answer]','send_message_to_user')]
        for action,kind in cases:
            self.proposal['candidates'][0]['action']=action
            _,trace=self.selector_v2().select(self.proposal,0,evidence=self.evidence)
            desc=trace['grounded_candidates'][0]['descriptor']
            self.assertEqual(desc['native_action'],action);self.assertEqual(desc['operation'],kind)
            self.assertNotIn('point_2d',desc)
            if kind=='page_focus':self.assertEqual(desc['tab_index'],2)
            if kind=='send_message_to_user':self.assertEqual(desc['message'],'literal answer')
            else:self.assertEqual(desc['destination_state'],'not_observed_pre_action')
        self.assertEqual(len(self.requests),5)

    def test_v2_corrupt_geometry_screenshot_and_grammar_still_fail_closed(self):
        for rectangle in [None,[1,2,3],[0,0,float('nan'),10],[0,0,True,10],[0,0,0,10],[0,0,-1,10]]:
            with self.subTest(rectangle=rectangle),self.assertRaises(EvaluationNotReady):
                self.selector_v2().select(self.proposal,0,evidence=dict(self.evidence,element_bounds={'1':rectangle}))
        with self.assertRaises(EvaluationNotReady):
            self.selector_v2().select(self.proposal,0,evidence=dict(self.evidence,screenshot=b'bad'))
        self.proposal['candidates'][0]['action']='this_is_not_native_grammar'
        with self.assertRaises(ValueError):self.selector_v2().select(self.proposal,0,evidence=self.evidence)
        self.assertEqual(self.requests,[])

    def test_v2_history_retains_saved_unavailable_status_after_target_appears(self):
        selector=self.selector_v2();self.evidence['element_bounds']={}
        _,first=selector.select(self.proposal,0,evidence=self.evidence)
        self.evidence['history']=[dict(action=self.proposal['candidates'][1]['action'],thought='Past original thought',
            selectionarm_action=first['selected_grounded_action'])]
        self.evidence['element_bounds']={'1':[600,320,80,80]}
        _,second=selector.select(self.proposal,1,evidence=self.evidence)
        self.assertEqual(first['grounded_candidates'][1]['descriptor']['target_status'],'unavailable')
        self.assertEqual(second['grounded_candidates'][1]['descriptor']['target_status'],'in_viewport')
        history=self.requests[-1]['payload']['messages'][1]['content'][0]['text']
        self.assertIn(first['selected_grounded_action'],history)

    def test_v2_missing_raw_slot_is_not_padded_or_resampled(self):
        self.proposal['candidates'].pop()
        with self.assertRaisesRegex(EvaluationNotReady,'five'):
            self.selector_v2().select(self.proposal,0,evidence=self.evidence)
        self.assertEqual(self.requests,[])

    def test_absent_transport_is_blocked(self):
        selector = PiotrSelectionARMSelector(config(), '/tmp/piotr-unused')
        with self.assertRaisesRegex(EvaluationNotReady, 'transport'):
            selector.select(self.proposal, 0, evidence=self.evidence)
        self.assertEqual(self.requests, [])


if __name__ == '__main__':
    unittest.main()
