"""Later stages must not consume quota reservations before prerequisites succeed."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import watch_arm_storage_release as watcher


class StorageReleaseTests(unittest.TestCase):
    def test_administrator_hold_is_never_treated_as_released(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'submissions.json').write_text(json.dumps([dict(job_id='123',kind='weight-to60',
                controller_root=str(root/'controller'),released=True,storage_reserve_bytes=2)]))
            with patch.object(watcher,'CONTROL',root),patch.object(watcher,'live_jobs',return_value={
                    '123':dict(state='PENDING',reason='(JobHeldAdmin)')}),patch.object(
                    watcher,'check_storage_quota',return_value=dict(safe_headroom_bytes=100)),patch.object(
                    watcher.subprocess,'run') as release:
                self.assertFalse(watcher.tick())
                release.assert_not_called()
            state=json.loads((root/'watch-status.json').read_text())['jobs']['123']
            self.assertFalse(state['released'])
            self.assertIn('Administrator hold',state['hold_reason'])

    def test_running_or_unregistered_parent_blocks_later_stage(self):
        item=dict(dependency='afterok:123')
        self.assertFalse(watcher.prerequisite_complete(item,[],{'123':dict(state='RUNNING')}))
        self.assertFalse(watcher.prerequisite_complete(item,[],{}))

    def test_both_controller_and_slurm_must_succeed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path=root/'status.json'
            jobs=[dict(job_id='123',controller_root=str(root))]
            item=dict(dependency='afterok:123')
            path.write_text(json.dumps(dict(complete=False)))
            self.assertFalse(watcher.prerequisite_complete(item,jobs,{}))
            path.write_text(json.dumps(dict(complete=True)))
            for status,expected in [('COMPLETED',True),('FAILED',False),('TIMEOUT',False),('RUNNING',False)]:
                with patch.object(watcher.subprocess,'check_output',return_value=f'123|{status}|\n'):
                    self.assertEqual(watcher.prerequisite_complete(item,jobs,{}),expected)

    def test_start_dependency_leaves_scheduling_to_slurm(self):
        self.assertTrue(watcher.prerequisite_complete(dict(dependency='after:331778'),[],{}))
        self.assertTrue(watcher.prerequisite_complete({},[],{}))


if __name__=='__main__':unittest.main()
