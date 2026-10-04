"""Controller completion must still wake an agent for the independent audit."""
from pathlib import Path
import os
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import arm_job_supervisor as supervisor


def snapshot(*, complete=True, reviewed=False, verified=True, code=0):
    return dict(scheduler='335699|RUNNING|g015',
                controller={'stage': 'training', 'iteration': 30},
                training={'stage': 'collection'}, progress_age_seconds=1,
                evaluations=[dict(iteration=20, verified=verified,
                    agent_reviewed=reviewed,
                    status={'complete': complete, 'returncode': code},
                    last_record_age_seconds=1)])


class EvaluationNotificationTests(unittest.TestCase):
    def test_native_generate_phase_is_observed_as_collection(self):
        status = {'stage': 'training'}
        for phase in ['generate', 'generate_rollout']:
            self.assertEqual(supervisor.observed_training_status(status,
                f'[TrainProgress] rollout=1/60 phase={phase}')['stage'], 'collection')
        self.assertEqual(status['stage'], 'training')
        self.assertEqual(supervisor.observed_training_status({'stage':'evaluation'},
            '[TrainProgress] rollout=1/60 phase=generate')['stage'], 'evaluation')

    def test_final_controller_completion_keeps_agent_review_active(self):
        job = dict(verified_complete=True, require_agent_completion_review=True)
        self.assertEqual(supervisor.active_jobs({'jobs': [job]}), [job])

    def test_agent_final_audit_closes_supervision(self):
        job = dict(verified_complete=True, require_agent_completion_review=True,
                   agent_completion_reviewed=True)
        self.assertEqual(supervisor.active_jobs({'jobs': [job]}), [])

    def test_real_approval_blocker_is_not_reopened_by_completion_review(self):
        job = dict(verified_complete=True, require_agent_completion_review=True,
                   requires_user=True)
        self.assertEqual(supervisor.active_jobs({'jobs': [job]}), [])

    def test_final_review_notifies_even_after_all_evaluations_were_reviewed(self):
        row = snapshot(reviewed=True)
        row['agent_completion_pending'] = True
        self.assertEqual(supervisor.urgent_issue(row), 'final_artifacts_ready')

    def test_controller_verified_result_still_requests_agent_audit(self):
        self.assertEqual(supervisor.urgent_issue(snapshot()),
                         'evaluation_20_artifacts_ready')

    def test_reviewed_result_does_not_repeat_notification(self):
        self.assertIsNone(supervisor.urgent_issue(snapshot(reviewed=True)))

    def test_unfinished_result_does_not_claim_completion(self):
        self.assertIsNone(supervisor.urgent_issue(
            snapshot(complete=False, verified=False, code=None)))

    def test_worker_failure_overrides_stale_verification_flag(self):
        self.assertEqual(supervisor.urgent_issue(snapshot(code=1)),
                         'evaluation_20_failed')

    def test_repeated_checkpoint_cohorts_have_distinct_completion_events(self):
        first, second = snapshot(), snapshot()
        first['evaluations'][0]['label'] = 'baseline-r1'
        second['evaluations'][0]['label'] = 'baseline-r2'
        self.assertNotEqual(supervisor.urgent_issue(first), supervisor.urgent_issue(second))


class StartupAgeTests(unittest.TestCase):
    def test_milestone_restart_uses_current_stage_not_original_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, timestamp in [('launch_manifest.json', 100), ('status.json', 4900)]:
                path = root/name; path.write_text('{}')
                os.utime(path, (timestamp, timestamp))
            row = snapshot(reviewed=True)
            row['training'] = {'stage':'actor-startup'}
            row['startup_age_seconds'] = supervisor.startup_age_seconds(root, row['training'], 5000)
            self.assertEqual(row['startup_age_seconds'], 100)
            self.assertIsNone(supervisor.urgent_issue(row))
            row['startup_age_seconds'] = supervisor.startup_age_seconds(root, row['training'], 6000)
            self.assertEqual(supervisor.urgent_issue(row), 'actor_startup_stale')

    def test_explicit_stage_start_survives_status_heartbeat_updates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root/'status.json').write_text('{}')
            status = {'stage':'actor-startup', 'stage_started_epoch_seconds':100}
            self.assertEqual(supervisor.startup_age_seconds(root, status, 1100), 1000)

    def test_legacy_manifest_fallback_when_status_file_is_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); path = root/'launch_manifest.json'; path.write_text('{}')
            os.utime(path, (100, 100))
            self.assertEqual(supervisor.startup_age_seconds(root, {'stage':'actor-startup'}, 1100), 1000)


class CompletionDeduplicationTests(unittest.TestCase):
    @staticmethod
    def event(reason, key='eval', job='123'):
        return [key, job, 'RUNNING', 'evaluation', 'evaluation', 90,
                None, None, None, reason]

    def due(self, previous, current, *, now=120, queued=100):
        return supervisor.notification_due(
            {'last_review_signature': previous, 'last_queued_epoch': queued},
            current, reviewed=110, now=now, cadence=3600,
            urgent=any(row[-1] for row in current), notify_on_change=False)

    def test_artifact_audit_then_slurm_completion_is_one_event(self):
        self.assertFalse(self.due(
            [self.event('evaluation_baseline-r2_artifacts_ready')],
            [self.event('scheduler:COMPLETED')]))

    def test_slurm_completion_then_final_review_is_one_event(self):
        self.assertFalse(self.due(
            [self.event('scheduler:COMPLETED')],
            [self.event('final_artifacts_ready')]))

    def test_new_evaluation_milestone_still_wakes_agent(self):
        self.assertTrue(self.due(
            [self.event('evaluation_20_artifacts_ready')],
            [self.event('evaluation_30_artifacts_ready')]))

    def test_failure_during_completion_review_still_wakes_agent(self):
        self.assertTrue(self.due(
            [self.event('evaluation_20_artifacts_ready')],
            [self.event('scheduler:FAILED')]))

    def test_different_job_completion_is_not_suppressed(self):
        self.assertTrue(self.due(
            [self.event('scheduler:COMPLETED')],
            [self.event('scheduler:COMPLETED', key='other', job='456')]))

    def test_outstanding_notification_is_not_duplicated(self):
        self.assertFalse(self.due([], [self.event('scheduler:COMPLETED')],
                                  queued=115))

    def test_hourly_followup_remains_enabled(self):
        event = [self.event('scheduler:COMPLETED')]
        self.assertFalse(self.due(event, event))
        self.assertTrue(self.due(event, event, now=3710))


if __name__ == '__main__':
    unittest.main()
