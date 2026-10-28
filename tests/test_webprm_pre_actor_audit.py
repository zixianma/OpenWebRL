"""Read-only integration/negative controls on the eight closed startup failures."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from scripts.collect_webprm_table5_campaign import Evidence, pre_actor_binding

PACKAGE = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-webarena-20261011/next-stage-preparation/pre-actor-readiness-review-v2')


class PreActorAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (PACKAGE/'preparation.json').exists():
            raise unittest.SkipTest('Closed local eight-cell evidence is unavailable')
        cls.prep = json.loads((PACKAGE/'preparation.json').read_text())
        cls.cohort = json.loads(Path(cls.prep['cohort']).read_text())['cohort']
        spec = importlib.util.spec_from_file_location('frozen_pre_actor_test', cls.prep['implementation']['path'])
        cls.module = importlib.util.module_from_spec(spec); spec.loader.exec_module(cls.module)
        cls.case = cls.prep['cases'][0]
        cls.inputs = json.loads(Path(cls.case['input']).read_text())

    def verify(self, evidence=None, inputs=None):
        inputs = deepcopy(self.inputs if inputs is None else inputs)
        return self.module.verify(evidence or Evidence(120), inputs, inputs['cell'], self.cohort,
                                  Path(self.prep['campaign_root']), Path(inputs['cell_folder']))

    def altered(self, artifact, update):
        target = Path(self.inputs['artifacts'][artifact]['path'])
        class Changed(Evidence):
            def read(self, path):
                value = super().read(path)
                if Path(path) == target: update(value)
                return value
        return Changed(120)

    def test_all_eight_real_failures_reconstruct_without_native_verdict_or_disposal(self):
        evidence = Evidence(120)
        self.assertEqual(len(self.prep['cases']), 8)
        for case in self.prep['cases']:
            with self.subTest(cell=case['cell_id']):
                result = self.verify(evidence, json.loads(Path(case['input']).read_text()))
                self.assertFalse(result['record']['valid'])
                self.assertIsNone(result['record']['native_success'])
                self.assertIsNone(result['record']['success'])
                self.assertFalse(result['complete_for_disposal'])
                self.assertEqual(result['call_audit']['paid_calls'], 0)

    def test_another_queue_owner_cannot_close_the_claim(self):
        evidence = self.altered('queue_outcome', lambda value: value.update(owner='another-owner'))
        with self.assertRaisesRegex(ValueError, 'queue failure/owner'): self.verify(evidence)

    def test_live_site_processes_prevent_terminal_reconstruction(self):
        evidence = self.altered('site_closure', lambda value: value.update(remaining_owned_processes={'123': {}}))
        with self.assertRaisesRegex(ValueError, 'process closure'): self.verify(evidence)

    def test_a_nonterminal_scheduler_snapshot_is_rejected(self):
        evidence = self.altered('scheduler', lambda value: value.update(state='RUNNING'))
        with self.assertRaisesRegex(ValueError, 'terminal scheduler'): self.verify(evidence)

    def test_an_actor_launch_cannot_be_reclassified_as_pre_actor(self):
        launch = Path(self.inputs['cell_folder'])/'worker-launch.json'; original = Path.exists
        with patch.object(Path, 'exists', lambda path: path == launch or original(path)):
            with self.assertRaisesRegex(ValueError, 'launched worker'): self.verify()

    def test_any_dispatched_request_prevents_zero_call_failure_classification(self):
        calls = dict(verified=True, paid_calls=1, local_calls=0, failed_calls=0, global_unknown_reservations=0)
        with patch.object(self.module, 'load_financial_verifier', return_value=lambda *args, **kwargs: calls):
            with self.assertRaisesRegex(ValueError, 'dispatched or unresolved'): self.verify()

    def test_an_open_financial_epoch_cannot_support_a_terminal_disposition(self):
        inputs = deepcopy(self.inputs); prefix = b''
        with Path(inputs['budget']['path']).open('rb') as stream:
            for raw in stream:
                prefix += raw; event = json.loads(raw)
                if event.get('event') == 'epoch_started' and event.get('epoch_id') == 'job-364924': break
        inputs['budget'].update(bytes=len(prefix), sha256=hashlib.sha256(prefix).hexdigest())
        with self.assertRaisesRegex(ValueError, 'does not close this allocation'): self.verify(inputs=inputs)

    def test_collector_independently_rejects_a_fabricated_positive_native_verdict(self):
        pointer = Path(self.case['candidate_pointer']); value = json.loads(pointer.read_text())
        record_path = Path(value['record']['path'])
        class Forged(Evidence):
            def read(self, path):
                result = super().read(path)
                if Path(path) == record_path: result.update(success=1, native_success=1)
                return result
        with self.assertRaisesRegex(ValueError, 'fabricates a verdict'):
            pre_actor_binding(Forged(120), pointer, Path(self.case['folder']), self.inputs['cell'],
                              self.cohort, Path(self.prep['campaign_root']))

    def test_unactivated_candidate_cannot_enter_the_table(self):
        with self.assertRaisesRegex(ValueError, 'lacks activation review'):
            pre_actor_binding(Evidence(120), Path(self.case['candidate_pointer']), Path(self.case['folder']),
                              self.inputs['cell'], self.cohort, Path(self.prep['campaign_root']))
