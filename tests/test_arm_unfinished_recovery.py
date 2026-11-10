"""Prevent duplicate evaluation work and quota-blind GPU release."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import recover_arm_unfinished as recovery
import watch_arm_storage_release as watcher


class RecoveryTests(unittest.TestCase):
    def test_coverage_preserves_experiment_and_fresh_initialization(self):
        p=recovery.coverage_plan('999999')
        self.assertEqual(p['wandb_run_id'],'arm-failure-coverage-fromzero-318950')
        self.assertEqual(p['start_rollout_id'],0)
        self.assertEqual(p['initial_optimizer_updates'],0)
        self.assertTrue(p['fresh_optimizer'])
        self.assertNotIn('resume_from',p)
        self.assertEqual(p['environment']['TP_SIZE'],'2')
        self.assertEqual(p['arm_config']['failure_scored_fraction'],.4)
        self.assertEqual(p['arm_config']['scored_fraction'],.2)
        self.assertIn('999999',p['output'])
        self.assertNotIn('deadline_epoch_seconds',p['arm_config'])

    def test_beta_retries_only_missing_tasks(self):
        p=recovery.evaluation_plan('weight','TEST_ONLY',10,recovery.BETA_ROOT,2)
        original=recovery.read(recovery.CONTROL/'beta10-recovery.json')
        self.assertEqual(p['expected_task_count'],44)
        self.assertEqual(len(set(p['expected_rollout_task_ids'])),44)
        self.assertFalse(set(p['expected_rollout_task_ids'])&set(original['verdict_sha256']))
        self.assertEqual(set(p['expected_rollout_task_ids'])|set(original['verdict_sha256']),set(original['expected_ids']))
        self.assertEqual(p['gpus'],2)

    def test_bad_archive_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'rollouts').mkdir()
            (root/'rollouts/a.pt').write_bytes(b'broken quota write')
            (root/'rollouts/a.json').write_text(json.dumps(dict(task_id='a',rollout_file='a.pt')))
            with self.assertRaises(zipfile.BadZipFile):recovery.records(root)

    def release_case(self,available,released=False,active_grace=False):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            jobs=[dict(job_id='1',storage_reserve_bytes=256*1024**3,controller_root=str(root/'beta')),
                  dict(job_id='2',storage_reserve_bytes=2*1024**4,controller_root=str(root/'B'),released=released),
                  dict(job_id='3',storage_reserve_bytes=2*1024**4,controller_root=str(root/'q'))]
            (root/'submissions.json').write_text(json.dumps(jobs))
            states={str(i):dict(state='PENDING',reason='(JobHeldUser)') for i in [1,2,3]}
            if released:states['2']=dict(state='RUNNING',reason='g000')
            quota={'safe_headroom_bytes':0 if active_grace else int(available*1024**4),
                   'launch_headroom_bytes':int(available*1024**4)}
            with patch.object(watcher,'CONTROL',root),patch.object(watcher,'live_jobs',return_value=states),\
                 patch.object(watcher,'check_storage_quota',return_value=quota),\
                 patch.object(watcher.subprocess,'run') as run:
                watcher.tick()
            return [call.args[0][-1] for call in run.call_args_list]

    def test_no_quota_no_release(self):
        self.assertEqual(self.release_case(0),[])

    def test_eval_fits_before_training(self):
        self.assertEqual(self.release_case(1),['1'])

    def test_no_double_booking_headroom(self):
        self.assertEqual(self.release_case(2.25),['1','2'])

    def test_all_release_with_enough_headroom(self):
        self.assertEqual(self.release_case(4.25),['1','2','3'])

    def test_released_job_reserves_space(self):
        self.assertEqual(self.release_case(3,released=True),['1'])

    def test_active_grace_uses_admissible_headroom(self):
        self.assertEqual(self.release_case(3.55,active_grace=True),['1','2'])


if __name__=='__main__':unittest.main()
