"""Offline startup-fixture guards; never launch Chromium or a model."""
import json
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from openwebrl import webprm_browser_preflight as fixture


class Page:
    viewport_size = dict(fixture.VIEWPORT)
    clicked = None
    def evaluate(self, expression):
        return self.clicked
    def screenshot(self, **kwargs):
        return b'\x89PNG\r\n\x1a\nsynthetic-offline-bytes'


class Environment:
    def __init__(self):
        self.page=Page();self.task=SimpleNamespace(viewport=dict(fixture.VIEWPORT));self.actions=[]
        self.observation=dict(screenshot=SimpleNamespace(shape=(720,1280,3),dtype='uint8'),last_action_error='')
        self.error=None
    def reset(self, seed):
        assert seed==42
        return self.observation,{}
    def step(self, action):
        self.actions.append((action,self.action_mapping(action)))
        self.page.clicked='yes'
        return dict(self.observation,last_action_error=self.error),0,False,False,{}


class Processor:
    last_geometry=dict(screenWidth=1280,screenHeight=720,innerWidth=1280,innerHeight=720,devicePixelRatio=1)
    manifest=dict(scope='offline-test')
    target='[42] button \'Synthetic native click\''
    box=[1050,610,190,60]
    def observe(self, page):
        return self.target+ ('\nSynthetic click observed' if page.clicked=='yes' else ''),{'42':self.box}


class PreflightTests(unittest.TestCase):
    def setUp(self):
        self.root=Path(tempfile.mkdtemp(prefix='wprm-preflight-test-',dir='/tmp'))
        self.env=Environment();self.processor=Processor()
        self.native=SimpleNamespace(parse_action=lambda action:dict(original=action),key_mapping={},manifest={})

    def exercise(self):
        with patch.object(fixture,'native_execution_code',return_value='native click code') as native:
            result=fixture._exercise(self.env,self.processor,self.native,'unused',self.root,deadline=time.time()+30)
        self.assertEqual(native.call_args.args[1],{'original':'click [42]'})
        return result

    def test_exact_native_target_action_and_array_evidence(self):
        result=self.exercise()
        self.assertTrue(result['native_click_observed'])
        self.assertEqual(self.env.actions,[('click [42]','native click code')])
        before=json.loads((self.root/'preaction.json').read_text())
        after=json.loads((self.root/'postaction.json').read_text())
        self.assertEqual(before['element_bounds']['42'],[1050,610,190,60])
        self.assertEqual(after['screenshot_shape'],[720,1280,3])
        with self.assertRaisesRegex(ValueError,'prespecified'):
            self.env.action_mapping('goto [https://example.org]')

    def test_fixed_literal_answer_runs_positive_negative_native_checks(self):
        original=self.env.step
        def step(action):
            result=list(original(action))
            if action=='send_msg_to_user [0]':
                result[0]['chat_messages']=[dict(role='assistant',message='0',timestamp=1)]
            return tuple(result)
        self.env.step=step
        source='/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-baselines-20261009/sources/arbiter'
        with patch.object(fixture,'native_execution_code',return_value='fixed native code'),\
                patch('socket.socket.connect',side_effect=AssertionError('network forbidden')):
            result=fixture._exercise(self.env,self.processor,self.native,source,self.root,
                deadline=time.time()+30,terminal_answer_check=True)
        self.assertTrue(result['normal_final_answer_verified'])
        self.assertEqual([row[0] for row in self.env.actions],['click [42]','send_msg_to_user [0]'])
        record=json.loads((self.root/'normal-final-answer.json').read_text())
        self.assertEqual([x['verdict']['score'] for x in record['controls']],[1,0])

    def test_synthetic_final_answer_cannot_be_fabricated_from_successful_step(self):
        with patch.object(fixture,'native_execution_code',return_value='fixed native code'):
            with self.assertRaisesRegex(RuntimeError,'append its exact literal'):
                fixture._exercise(self.env,self.processor,self.native,'unused',self.root,
                    deadline=time.time()+30,terminal_answer_check=True)

    def test_missing_target_does_not_execute(self):
        self.processor.target='[42] button \'Another target\''
        with self.assertRaisesRegex(RuntimeError,'unique'):self.exercise()
        self.assertEqual(self.env.actions,[])

    def test_changed_viewport_does_not_execute(self):
        self.env.page.viewport_size=dict(width=800,height=600)
        with self.assertRaisesRegex(RuntimeError,'viewport'):self.exercise()
        self.assertEqual(self.env.actions,[])

    def test_action_failure_retains_preaction_and_cannot_pass(self):
        self.env.error='synthetic dispatch failure'
        with self.assertRaisesRegex(RuntimeError,'error'):self.exercise()
        self.assertTrue((self.root/'preaction.json').is_file())
        self.assertFalse((self.root/'postaction.json').exists())

    def test_bad_screenshot_shape_and_invalid_bounds_fail_before_action(self):
        self.env.observation['screenshot'].shape=(720,1280,4)
        with self.assertRaisesRegex(RuntimeError,'screenshot'):self.exercise()
        self.env.observation['screenshot'].shape=(720,1280,3)
        self.processor.box=[1050,610,float('nan'),60]
        with self.assertRaisesRegex(RuntimeError,'geometry'):self.exercise()
        self.assertEqual(self.env.actions,[])

    def test_unsafe_paths_deadline_and_host_network_rejected_without_spawn(self):
        protected=self.root/'protected-fixture';protected.mkdir()
        (self.root/'redirect').symlink_to(protected)
        with self.assertRaises(ValueError):fixture.safe_path(self.root/'redirect'/'child')
        with self.assertRaises(ValueError):fixture.safe_path('/gpfs/home/zixianma/forbidden')
        self.assertEqual(list(protected.iterdir()),[])
        with patch.object(fixture,'run_supervised_episode',side_effect=AssertionError('spawn forbidden')):
            for deadline in [float('nan'),float('inf'),0,True]:
                with self.assertRaises(ValueError):
                    fixture.run_preflight('unused',self.root/'unused',deadline=deadline,host_netns='net:[2]')
            with patch.object(fixture.os,'readlink',return_value='net:[2]'),self.assertRaisesRegex(ValueError,'separate'):
                fixture.run_preflight('unused',self.root/'unused',deadline=time.time()+30,host_netns='net:[2]')
        self.assertFalse((self.root/'unused').exists())

    def test_native_pass_requires_complete_owned_cleanup(self):
        for suffix,overrides,expected in [('clean',{},'passed'),('orphan',{'remaining_owned_processes':{'10':{}}},'failed'),
                                          ('killed',{'child_exitcode':-15},'failed'),
                                          ('timeout',{'reason':'cleanup_timeout'},'failed')]:
            result=dict(valid=True,native_click_observed=True,cleanup_status='completed',
                watchdog=dict(remaining_owned_processes={},child_exitcode=0,reason=None,**{}))
            result['watchdog'].update(overrides)
            with patch.object(fixture.os,'readlink',return_value='net:[2]'), \
                    patch.dict(os.environ,WEBPRM_BROWSER_TMP_ROOT=str(self.root)), \
                    patch.object(fixture,'load_arbiter_harness',return_value=self.native), \
                    patch.object(fixture,'run_supervised_episode',return_value=result):
                receipt=fixture.run_preflight('unused',self.root/suffix,deadline=time.time()+30,host_netns='net:[1]')
            self.assertEqual(receipt['status'],expected)
            self.assertEqual(receipt['provider_calls'],0)
            self.assertEqual(receipt['benchmark_episodes'],0)


if __name__=='__main__':unittest.main()
