import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import run_arm_milestone_queue as queue


class MilestoneQueue(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.training = self.root/'training'
        self.request = dict(job_id='123', variant='B', use_existing_allocation=True,
            evaluations=[dict(iteration=30, training_root=str(self.training))])
        (self.root/'123.json').write_text(json.dumps(self.request))

    def run_queue(self, receipt=False, ready=True, skip_unavailable=False):
        plan = dict(output=str(self.root/'output'), source=str(self.root/'source'))
        if receipt:
            (self.root/'123-iteration30-audit.json').write_text('{}')
        with patch.object(queue, 'CONTROL', self.root), patch.dict(os.environ, SLURM_JOB_ID='123'), \
                patch('run_arm_gate_checkpoint_eval.checkpoint_ready', return_value='checkpoint' if ready else None) as checkpoint, \
                patch('prepare_arm_gate_to60.evaluation_plan', return_value=plan) as planner, \
                patch('evaluate_baseline_checkpoint.run') as execute, \
                patch('run_stage1_to100.audit_rollouts', return_value=dict(tasks=300)) as audit:
            queue.run_due_evaluations('B', self.training, 40, '123', skip_unavailable=skip_unavailable)
            return checkpoint, planner, execute, audit

    def test_exact_intermediate_checkpoint_is_awaited_and_audited(self):
        checkpoint, planner, execute, audit = self.run_queue()
        checkpoint.assert_called_once_with('B', self.training, 30)
        planner.assert_called_once_with('B', self.training, 30, '123')
        execute.assert_called_once()
        audit.assert_called_once()
        self.assertEqual(json.loads((self.root/'123.json').read_text())['evaluations'][0]['state'], 'complete')

    def test_completed_evaluation_is_reaudited_without_gpu_rerun(self):
        _, _, execute, audit = self.run_queue(receipt=True)
        execute.assert_not_called()
        audit.assert_called_once()

    def test_missing_checkpoint_never_substitutes_a_later_one(self):
        with self.assertRaisesRegex(ValueError, 'not durable'):
            self.run_queue(ready=False)

    def test_partial_training_evaluates_the_durable_milestone(self):
        _, _, execute, audit = self.run_queue(skip_unavailable=True)
        execute.assert_called_once()
        audit.assert_called_once()

    def test_partial_training_without_milestone_does_not_launch_evaluation(self):
        _, planner, execute, audit = self.run_queue(ready=False, skip_unavailable=True)
        planner.assert_not_called()
        execute.assert_not_called()
        audit.assert_not_called()
        self.assertEqual(json.loads((self.root/'123.json').read_text())['evaluations'][0]['state'],
                         'waiting_for_durable_checkpoint')

    def test_another_allocation_cannot_run_the_queue(self):
        with patch.object(queue, 'CONTROL', self.root), patch.dict(os.environ, SLURM_JOB_ID='456'):
            with self.assertRaisesRegex(ValueError, 'owning allocation'):
                queue.run_due_evaluations('B', self.training, 40, '123')

    def test_training_worker_awaits_milestone_before_returning_to_parent(self):
        import prepare_arm_gate_dp as controller
        calls = []
        with patch.object(sys, 'argv', ['worker', '--variant', 'C', '--job-id', '123',
                '--worker', 'train', '--target', '60', '--resume-from', str(self.training)]), \
                patch.dict(os.environ, SLURM_JOB_ID='123'), \
                patch.object(controller, 'request', return_value=dict(stage='released', remaining_seconds=7200)), \
                patch.object(controller, 'plan', return_value={}), \
                patch('resume_arm_failure_variants.execute', side_effect=lambda p: calls.append('train')), \
                patch.object(queue, 'run_due_evaluations', side_effect=lambda *a, **k: calls.append('eval')) as evaluate:
            controller.main()
        self.assertEqual(calls, ['train', 'eval'])
        self.assertEqual(evaluate.call_args.kwargs, dict(skip_unavailable=True))

    def test_failed_training_worker_does_not_start_evaluation(self):
        import prepare_arm_gate_dp as controller
        with patch.object(sys, 'argv', ['worker', '--variant', 'C', '--job-id', '123',
                '--worker', 'train', '--target', '60', '--resume-from', str(self.training)]), \
                patch.dict(os.environ, SLURM_JOB_ID='123'), \
                patch.object(controller, 'request', return_value=dict(stage='released', remaining_seconds=7200)), \
                patch.object(controller, 'plan', return_value={}), \
                patch('resume_arm_failure_variants.execute', side_effect=RuntimeError('training failed')), \
                patch.object(queue, 'run_due_evaluations') as evaluate:
            with self.assertRaisesRegex(RuntimeError, 'training failed'):
                controller.main()
        evaluate.assert_not_called()

    def test_ablation_tenth_iteration_uses_its_own_lineage(self):
        self.request.update(variant='coverage', evaluations=[dict(iteration=10, training_root=str(self.training))])
        (self.root/'123.json').write_text(json.dumps(self.request))
        self.training.mkdir()
        (self.training/'launch_manifest.json').write_text(json.dumps(dict(
            failure_ablation_training='coverage', wandb_run_id='arm-failure-coverage-fromzero-123')))
        plan = dict(output=str(self.root/'output'), source=str(self.root/'source'))
        with patch.object(queue, 'CONTROL', self.root), patch.dict(os.environ, SLURM_JOB_ID='123'), \
                patch('run_arm_iteration80_eval.checkpoint_ready', return_value='checkpoint') as checkpoint, \
                patch('prepare_arm_failure_ablations.evaluation_plan', return_value=plan) as planner, \
                patch('evaluate_baseline_checkpoint.run') as execute, \
                patch('run_stage1_to100.audit_rollouts', return_value=dict(tasks=300)):
            queue.run_due_evaluations('coverage', self.training, 20, '123')
        checkpoint.assert_called_once_with('additive', 10, self.training)
        planner.assert_called_once_with('coverage', self.training, '123', 10)
        execute.assert_called_once()


if __name__ == '__main__':
    unittest.main()
