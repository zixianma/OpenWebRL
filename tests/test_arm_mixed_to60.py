"""Milestone ordering and budget/lineage regression checks for long continuations."""
import sys
from pathlib import Path
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import resume_arm_mixed_to60 as runner


class MilestoneTests(unittest.TestCase):
    def test_evaluate_before_next_training(self):
        self.assertEqual(runner.next_stage(9, set()), ('train', 10))
        self.assertEqual(runner.next_stage(10, set()), ('eval', 10))
        self.assertEqual(runner.next_stage(10, {10}), ('train', 20))
        self.assertEqual(runner.next_stage(23, {10, 20}), ('train', 30))

    def test_cannot_claim_completion_with_missing_cohort(self):
        self.assertEqual(runner.next_stage(60, {10, 20, 30, 50, 60}), ('eval', 40))
        self.assertEqual(runner.next_stage(60, set(runner.MILESTONES)), ('complete', 60))

    def test_retry_reservation_counts_against_original_cap(self):
        ledger = dict(variant='bonus', attempts=[dict(job_id='1', attempt_seconds=86400),
            dict(job_id='2', attempt_seconds=86400), dict(job_id='3', attempt_seconds=600)])
        with patch.object(runner, 'approval', return_value=dict(additional_seconds_per_variant=172800,max_attempt_seconds=86400)), patch.object(runner, 'read', return_value=ledger):
            with self.assertRaisesRegex(ValueError, 'exceed'):
                runner.attempt('3', 'bonus')
            ledger['attempts'][0]['charged_seconds'] = 80000
            self.assertEqual(runner.attempt('3', 'bonus')['attempt_seconds'], 600)

    def test_wrong_lineage_and_skipped_milestone_rejected(self):
        o = dict(manifest=dict(mixed_reweight_variant='bonus',wandb_run_id=runner.RUN_IDS['bonus']),
                 checkpoint_report=dict(iteration=7))
        with patch.object(runner, 'origin', return_value=o):
            with self.assertRaisesRegex(ValueError, 'lineage'):
                runner.training_plan('PREPARE', 'reweight', '/unused', 10)
            with self.assertRaisesRegex(ValueError, 'milestone'):
                runner.training_plan('PREPARE', 'bonus', '/unused', 20)

    def test_resume_preserves_science_and_places_restore_in_parent(self):
        parent = '/gpfs/scrubbed/zixianma/openwebrl-runtime/evaluations/arm-mixed-fixture'
        old = dict(source=str(runner.SOURCE), output=parent, mixed_reweight_variant='reweight',
            wandb_run_id=runner.RUN_IDS['reweight'],
            command=['launch', '--dynamic-sampling-filter-path', runner.previous.original.FILTER],
            environment=dict(NUM_GPUS='8', TP_SIZE='2', GLOBAL_BATCH_SIZE='256', BROWSER_CONCURRENCY='64',
                OPENWEBRL_REPLAY_FIRST_BATCH='old-batch', WANDB_RUN_ID=runner.RUN_IDS['reweight']),
            arm_config=dict(advantage_mode='outcome_reweight', beta=.5, scored_fraction=.2,
                reweight_lambda=.5, candidate_gate='min2', credit_assignment='response_index',
                failure_group_cap=0, failure_loss_coefficient=0., additive_failure_groups=False,
                admit_all_failure_groups=False, deadline_epoch_seconds=123),
            requested_resources=dict(gpus=8, cpus=64, memory_gib=960), multimodal_storage={})
        report=dict(iteration=29, checkpoint=parent+'/runtime/iter_0000029',completed_optimizer_updates=450)
        o=dict(manifest=old,checkpoint_report=report)
        with patch.object(runner, 'origin', return_value=o), patch.object(runner.previous, 'replay_provenance', return_value=None):
            p=runner.training_plan('PREPARE', 'reweight', parent, 40)
        self.assertEqual(p['start_rollout_id'],30)
        self.assertEqual(p['requested_iterations'],10)
        self.assertEqual(p['initial_optimizer_updates'],450)
        self.assertFalse(p['fresh_optimizer'])
        self.assertEqual(p['environment']['NUM_ROLLOUT'],'40')
        self.assertEqual(p['environment']['WANDB_RESUME'],'must')
        self.assertNotIn('deadline_epoch_seconds',p['arm_config'])
        self.assertNotIn('OPENWEBRL_REPLAY_FIRST_BATCH',p['environment'])
        self.assertTrue(Path(p['gpu_restore_output']).is_relative_to(parent))
        self.assertFalse(Path(p['gpu_restore_output']).is_relative_to(parent+'/runtime'))
        self.assertIn('--use-checkpoint-opt-param-scheduler',p['command'])
        self.assertEqual(p['arm_config']['advantage_mode'],'outcome_reweight')


if __name__ == '__main__':
    unittest.main()
