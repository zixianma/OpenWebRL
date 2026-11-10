"""Milestone ordering and budget/lineage regression checks for long continuations."""
import sys
from pathlib import Path
import unittest
import tempfile
import json
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import resume_arm_mixed_to90 as runner


class MilestoneTests(unittest.TestCase):
    def check_recovery_origin(self, *, latest_durable=True, wrong_lineage=False):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            control, parent, supervisor = [base/name for name in ('control', 'parent', 'supervisor')]
            for folder in (control, parent, supervisor):
                folder.mkdir()
            old, latest, pending = [base/name for name in ('old', 'latest', 'pending')]
            for folder, index, durable in ((old, 86, True), (latest, 88, latest_durable), (pending, 89, False)):
                (folder/'runtime').mkdir(parents=True)
                (folder/'runtime/latest_checkpointed_iteration.txt').write_text(str(index))
                if durable:
                    proof = folder/f'iterations/{index:04d}/checkpoint-validation.json'
                    proof.parent.mkdir(parents=True)
                    proof.write_text('{}')
            (control/'proposal.json').write_text(json.dumps({'variants': {'reweight': {'resume_from': str(old)}}}))
            (control/'reweight-progress.json').write_text(json.dumps({'training_roots': [str(old)]}))
            (supervisor/'registry.json').write_text(json.dumps({'jobs': [{
                'key': 'mixed-reweight', 'training_root': str(pending),
                'supervised_attempts': [{'training_root': str(latest)}]}]}))
            def checked_origin(path):
                run = 'bonus' if wrong_lineage and path == latest else 'reweight'
                return {'manifest': {'wandb_run_id': runner.RUN_IDS[run]}}
            with patch.object(runner, 'CONTROL', control), patch.object(runner, 'PARENT_CONTROL', parent), \
                    patch.object(runner, 'SUPERVISOR', supervisor), patch.object(runner, 'origin', side_effect=checked_origin):
                if wrong_lineage:
                    with self.assertRaisesRegex(ValueError, 'another run'):
                        runner.choose_origin('reweight')
                else:
                    self.assertEqual(runner.choose_origin('reweight'), latest if latest_durable else old)

    def test_failed_worker_checkpoint_discovered_from_supervised_attempt(self):
        self.check_recovery_origin()

    def test_resume_marker_without_own_checkpoint_not_selected(self):
        self.check_recovery_origin(latest_durable=False)

    def test_recovery_registry_cannot_cross_lineage(self):
        self.check_recovery_origin(wrong_lineage=True)

    def test_evaluate_before_next_training(self):
        self.assertEqual(runner.next_stage(9, set()), ('train', 10))
        self.assertEqual(runner.next_stage(10, set()), ('eval', 10))
        self.assertEqual(runner.next_stage(10, {10}), ('train', 20))
        self.assertEqual(runner.next_stage(23, {10, 20}), ('train', 30))

    def test_cannot_claim_completion_with_missing_cohort(self):
        self.assertEqual(runner.next_stage(60, {10, 20, 30, 50, 60}), ('eval', 40))
        self.assertEqual(runner.next_stage(90, set(runner.MILESTONES)), ('complete', 90))

    def test_extension_boundaries_and_partial_parent(self):
        prior=set(range(10,61,10))
        self.assertEqual(runner.next_stage(58,prior-{60}),('train',60))
        self.assertEqual(runner.next_stage(60,prior),('train',70))
        self.assertEqual(runner.next_stage(70,prior),('eval',70))
        self.assertEqual(runner.next_stage(80,prior|{70,80}),('train',90))
        self.assertEqual(runner.next_stage(90,set(runner.MILESTONES)-{80}),('eval',80))

    def test_new_resource_approval_required(self):
        with patch.object(runner,'read',return_value={'approved':False}):
            with self.assertRaisesRegex(ValueError,'approval missing'):
                runner.approval()

    def test_old_controller_lock_blocks_overlapping_writer(self):
        import tempfile,fcntl
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            with (root/'bonus-execution.lock').open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                with patch.object(runner,'PARENT_CONTROL',root),patch.object(runner,'refresh_accounting') as refresh:
                    with self.assertRaises(BlockingIOError):runner.execute('123','bonus')
                    refresh.assert_not_called()

    def test_inherited_evaluation_cannot_cross_lineage(self):
        state={'training_roots':['/old'],'evaluations':{'60':{'plan':'/plan','root':'/eval'}}}
        def read(path):
            return state if str(path).endswith('progress.json') else {'parent_training_run':runner.RUN_IDS['reweight']}
        with patch.object(runner,'read',side_effect=read),patch.object(runner,'audit_evaluation') as audit:
            with self.assertRaisesRegex(ValueError,'another lineage'):runner.inherited_progress('bonus')
            audit.assert_not_called()

    def test_retry_reservation_counts_against_original_cap(self):
        ledger = dict(variant='bonus', attempts=[dict(job_id='1', attempt_seconds=86400),
            dict(job_id='2', attempt_seconds=86400), dict(job_id='3', attempt_seconds=600)])
        with patch.object(runner, 'approval', return_value=dict(additional_seconds_by_variant=dict(bonus=172800,reweight=86400),max_attempt_seconds=86400)), patch.object(runner, 'read', return_value=ledger):
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
