import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/prepare_harmful_native_pilot.py'
spec = importlib.util.spec_from_file_location('prepare_native_pilot', SCRIPT)
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


def profiles():
    result = {}
    for axis in ('compute', 'browser'):
        result[axis] = dict(caps={k: 100000 for k in prepare.METERS},
            primary_meter='modeled_flops' if axis == 'compute' else 'browser_operations',
            flops_model='Fixture upper bound includes actor and selector forwards')
    return result


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.actor, self.head = self.root / 'actor', self.root / 'head'
        self.actor.mkdir(); self.head.mkdir()
        (self.actor / 'config.json').write_text('{"model_type":"fixture"}')
        (self.actor / 'model.safetensors').write_bytes(b'fixture, not real weights')
        (self.head / 'head.pt').write_bytes(b'fixture, no torch load')
        (self.head / 'metrics.json').write_text(json.dumps(dict(args=dict(pointwise='True', layers='18,27,36',
            pools='thought_mean,action_mean,last'), best=dict(epoch=13, validation=dict(panels=3759)))))
        self.admission = self.root / 'admission.json'
        self.admission.write_text(json.dumps(dict(status='fixed_panel_recovery_partially_verified',
            complete_task_ids=[470,471,473], selected_panel=dict(candidate_tasks=9), establishes_policy_safety_benefit=False)))
        stub = patch.object(prepare, 'verify_admission', return_value={'status': 'unit_fixture_only'})
        stub.start(); self.addCleanup(stub.stop)

    def build(self, **kwargs):
        return prepare.prepare(self.root / 'proposal', self.admission, profiles(),
                               actor=self.actor, head=self.head, **kwargs)

    def test_package_is_explicitly_unapproved_and_not_deployable(self):
        plan = self.build()
        self.assertFalse(plan['launch_ready'])
        self.assertIsNone(plan['resource_approval'])
        self.assertTrue(plan['selector']['trained_checkpoint_available'])
        self.assertFalse(plan['selector']['native_live_serving_validated'])
        self.assertIsNone(plan['selector']['automatic_substitute'])
        self.assertTrue(plan['actor']['weight_content_reverified'])
        self.assertEqual(plan['work_performed'], dict(model_calls=0, browser_calls=0, api_calls=0, allocations=0))
        self.assertEqual(len(plan['schedule']), 18)
        self.assertEqual(plan['absolute_workload_caps']['actor_proposals'], 864)
        self.assertEqual(plan['absolute_workload_caps']['evaluation_actor_proposals'], 792)
        self.assertEqual(plan['absolute_workload_caps']['evaluation_max_attempts'], 42)
        self.assertEqual(self.build(), plan)

    def test_sources_are_small_immutable_copies(self):
        plan = self.build()
        for name, identity in plan['sources'].items():
            self.assertEqual(prepare.sha(self.root/'proposal/source'/name), identity['sha256'])
        (self.root/'proposal/source'/next(iter(plan['sources']))).write_text('changed')
        with self.assertRaisesRegex(ValueError, 'frozen source differs'):
            self.build()

    def test_wrong_task_admission_rejected(self):
        data = json.loads(self.admission.read_text())
        data['complete_task_ids'] = [470, 471, 472]
        self.admission.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, 'task identities'):
            self.build()

    def test_missing_or_untrained_head_has_no_silent_fallback(self):
        (self.head/'metrics.json').write_text('{"args":{"pointwise":"False"}}')
        with self.assertRaisesRegex(ValueError, 'trained teacher-label'):
            self.build()

    def test_wrong_axis_and_omitted_caps_rejected(self):
        bad = profiles(); bad['compute']['primary_meter'] = 'browser_operations'
        with self.assertRaises(ValueError):
            prepare.validate_profiles(bad)
        bad = profiles(); del bad['browser']['caps']['modeled_flops']
        with self.assertRaises(ValueError):
            prepare.validate_profiles(bad)

    def test_no_home_output_before_any_writes(self):
        with self.assertRaisesRegex(ValueError, 'approved'):
            prepare.prepare('/gpfs/home/zixianma/forbidden-pilot', self.admission, profiles(),
                            actor=self.actor, head=self.head)


if __name__ == '__main__':
    unittest.main()
