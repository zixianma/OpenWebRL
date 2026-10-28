"""Saved CMS interruption verification and adversarial evidence controls."""
from copy import deepcopy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from openwebrl import webprm_preauth_audit as module
from scripts import audit_webprm_interrupted_cell as legacy
from scripts.collect_webprm_table5_campaign import Evidence, load_preauth_implementation, preauth_binding

PACKAGE=Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-webarena-20261011/next-stage-preparation/cms-preauth-interruption-review-v5')


class PreauthAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (PACKAGE/'preparation.json').exists():raise unittest.SkipTest('Saved CMS evidence unavailable')
        cls.prep=json.loads((PACKAGE/'preparation.json').read_text())
        cls.cohort=json.loads(Path(cls.prep['cohort']['path']).read_text())['cohort']
        cls.module=load_preauth_implementation(Evidence(120),cls.prep['implementation'])

    def setUp(self):
        guard=patch('socket.socket.connect',side_effect=AssertionError('No network permitted'))
        guard.start();self.addCleanup(guard.stop)
        self.inputs=self.case(self.prep['cases'][0])

    def case(self,case):
        inputs=json.loads(Path(case['input']['path']).read_text())
        return inputs

    def verify(self,evidence=None,inputs=None):
        inputs=deepcopy(self.inputs if inputs is None else inputs)
        return self.module.verify(evidence or Evidence(120),inputs,inputs['cell'],self.cohort,
            Path(self.prep['campaign_root']),Path(inputs['cell_folder']))

    def altered(self,name,update):
        ref=(self.inputs['artifacts'].get(name) or self.inputs.get(name));target=Path(ref['path'])
        class Changed(Evidence):
            def read(self,path):
                result=super().read(path)
                if Path(path)==target:update(result)
                return result
        return Changed(120)

    def test_both_saved_failures_remain_invalid_with_absent_verdicts_and_zero_calls(self):
        for case in self.prep['cases']:
            inputs=self.case(case);before={r['path']:legacy.sha(r['path']) for r in inputs['artifacts'].values()}
            result=self.verify(inputs=inputs)
            self.assertTrue(result['record']['attempted']);self.assertFalse(result['record']['valid'])
            self.assertIsNone(result['record']['success']);self.assertIsNone(result['record']['native_success'])
            self.assertFalse(result['authentication_verified']);self.assertFalse(result['complete_for_disposal'])
            self.assertEqual(result['call_audit']['paid_calls'],0);self.assertEqual(result['call_audit']['local_calls'],0)
            self.assertEqual(before,{p:legacy.sha(p) for p in before})

    def test_success_native_verdict_or_later_turn_cannot_be_inserted(self):
        for change in [dict(valid=True),dict(success=1),dict(native_success=False),dict(reset_id='forged'),dict(turns=1)]:
            with self.subTest(change=change),self.assertRaisesRegex(ValueError,'supervisor failure'):
                self.verify(self.altered('result',lambda value:value.update(change)))

    def test_another_queue_owner_cannot_close_the_claim(self):
        with self.assertRaisesRegex(ValueError,'queue failure/owner'):
            self.verify(self.altered('queue_outcome',lambda value:value.update(owner='another-owner')))

    def test_other_state_or_actor_lease_is_rejected(self):
        for name,change in [('claim',dict(state_path='/tmp/wrong')),('lease',dict(actor_model='other')),
                            ('instance',dict(job_id='other'))]:
            with self.subTest(name=name),self.assertRaises(ValueError):
                self.verify(self.altered(name,lambda value:value.update(change)))

    def test_live_process_or_nonterminal_scheduler_is_rejected(self):
        for name in ['watchdog','site_closure','outer_closure']:
            with self.subTest(name=name),self.assertRaises(ValueError):
                self.verify(self.altered(name,lambda value:value.update(remaining_owned_processes={'1':{}})))
        closure=json.loads(Path(self.inputs['owner_closure']['path']).read_text())
        target=Path(closure['scheduler_evidence']['path'])
        class Changed(Evidence):
            def read(self,path):
                value=super().read(path)
                if Path(path)==target:value['state']='RUNNING'
                return value
        with self.assertRaisesRegex(ValueError,'scheduler'):self.verify(Changed(120))

    def test_later_phase_or_api_journal_prevents_preauth_classification(self):
        with self.assertRaises(ValueError):
            self.verify(self.altered('phase',lambda value:value.update(phase='actor_request')))
        episode=Path(self.inputs['artifacts']['result']['path']).parent;original=Path.iterdir
        with patch.object(Path,'iterdir',lambda path:iter([*original(path),episode/'api.jsonl']) if path==episode else original(path)):
            with self.assertRaisesRegex(ValueError,'later worker phase'):self.verify()

    def test_changed_source_and_financial_recovery_are_rejected(self):
        inputs=deepcopy(self.inputs);inputs['dependency_sources']['openwebrl/webprm_preauth_audit.py']['sha256']='0'*64
        with self.assertRaises(ValueError):self.verify(inputs=inputs)
        with self.assertRaisesRegex(ValueError,'recovery'):
            self.verify(self.altered('recovery',lambda value:value.update(new_provider_calls=1)))

    def test_nonzero_or_unknown_calls_cannot_be_classified_as_zero_call_interruption(self):
        calls=self.verify()['call_audit']
        for changes in [dict(paid_calls=1),dict(local_calls=1),dict(cell_unknown_reservations=1)]:
            with self.subTest(changes=changes),patch.object(self.module,'audit_cell_calls',return_value=dict(calls,**changes)):
                with self.assertRaisesRegex(ValueError,'zero-call classification'):self.verify()

    def test_collector_requires_independent_review_and_preserves_null_verdict(self):
        case=self.prep['cases'][0];pointer=PACKAGE/(Path(case['folder']).name+'.pointer-v2.json')
        value=json.loads(pointer.read_text())
        def verify(target=None,update=None):
            class Changed(Evidence):
                def read(self,path):
                    result=super().read(path)
                    if target is not None and Path(path)==target:update(result)
                    return result
            return preauth_binding(Changed(120),pointer,Path(case['folder']),self.inputs['cell'],
                self.cohort,Path(self.prep['campaign_root']))
        record,_=verify();self.assertFalse(record['valid']);self.assertIsNone(record['native_success'])
        changes=[(pointer,lambda v:v.update(raw_success=1)),
            (Path(value['activation_review']['path']),lambda v:v.update(verified=False)),
            (Path(value['record']['path']),lambda v:v.update(valid=True)),
            (pointer,lambda v:v.update(independent_audit=v['audit']))]
        for target,update in changes:
            with self.subTest(target=target),self.assertRaises(ValueError):verify(target,update)
