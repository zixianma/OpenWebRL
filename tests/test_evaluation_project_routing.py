"""Separate eval workers must never create runs in the training project."""
import copy
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import evaluate_baseline_checkpoint as evaluator
import resume_arm_mixed_to60 as mixed
import resume_arm_gate_b90 as gate90
import resume_arm_gate_b100 as gate100
import run_arm_iteration80_eval as base
import prepare_arm_gate_to60 as gate_base


def legacy_plan():
    return dict(job_id='routing-test', output='/tmp/eval-routing-preview',
        wandb_run_id='separate-eval', wandb_project='openwebrl',
        parent_training_run='actual-training-run',
        environment=dict(WANDB_PROJECT='openwebrl', NUM_ROLLOUT='0'),
        command=['launch', '--wandb-project', 'openwebrl', '--wandb-team',
                 'zixianma', '--wandb-group', 'checkpoint-eval'])


class RoutingTests(unittest.TestCase):
    def assert_eval_project(self, p):
        self.assertEqual(p['wandb_project'], 'openwebrl-evals')
        self.assertEqual(p['environment']['WANDB_PROJECT'], 'openwebrl-evals')
        i = p['command'].index('--wandb-project')
        self.assertEqual(p['command'][i + 1], 'openwebrl-evals')
        self.assertIn('/openwebrl-evals/runs/', p['wandb_url'])

    def test_explicit_legacy_training_project_is_redirected(self):
        for requested in ['openwebrl', ' OpenWebRL ']:
            with self.subTest(requested=requested):
                p = legacy_plan()
                before = copy.deepcopy(p)
                evaluator.configure_evaluation_tracking(p, requested)
                self.assert_eval_project(p)
                self.assertEqual(p['wandb_run_id'], before['wandb_run_id'])
                self.assertEqual(p['parent_training_run'], before['parent_training_run'])
                self.assertEqual(p['environment']['NUM_ROLLOUT'], '0')
                self.assertEqual(p['wandb_project_redirected_from'], requested.strip())

    def test_stale_saved_manifest_is_routed_at_worker_entry(self):
        p = legacy_plan()
        with patch.dict(os.environ, {'SLURM_JOB_ID': 'not-a-real-allocation'}):
            with self.assertRaisesRegex(ValueError, 'explicitly authorized Slurm'):
                evaluator.run(p, Path('/unused.env'))
        self.assert_eval_project(p)

    def test_mixed_controllers_override_even_explicit_legacy_project(self):
        for variant in ['bonus', 'reweight']:
            with self.subTest(variant=variant), patch.object(base, 'plan', return_value=legacy_plan()):
                p = mixed.evaluation_plan('preview', variant, '/unused', 20)
            self.assert_eval_project(p)
            self.assertEqual(p['parent_training_run'], mixed.RUN_IDS[variant])
            self.assertEqual(p['environment']['NUM_GPUS'], '8')

    def test_gate_b90_and_b100_separate_evals_use_eval_project(self):
        for controller, target in [(gate90, 90), (gate100, 100)]:
            with self.subTest(target=target), patch.object(gate_base, 'evaluation_plan', return_value=legacy_plan()):
                p = controller.evaluation_plan('preview', '/unused', target)
            self.assert_eval_project(p)

    def test_explicit_nontraining_project_still_supported(self):
        p = legacy_plan()
        evaluator.configure_evaluation_tracking(p, 'openwebrl-evals-sandbox')
        self.assertEqual(p['wandb_project'], 'openwebrl-evals-sandbox')
        self.assertNotIn('wandb_project_redirected_from', p)


if __name__ == '__main__':
    unittest.main()
