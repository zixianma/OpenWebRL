import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('quota',Path(__file__).resolve().parents[1]/'scripts/check_storage_quota.py')
quota=importlib.util.module_from_spec(spec);spec.loader.exec_module(quota)
HEADER='mmlsquota:user:HEADER:version:reserved:reserved:filesystemName:quotaType:id:name:blockUsage:blockQuota:blockLimit:blockInDoubt:blockGrace:filesUsage:filesQuota:filesLimit:filesInDoubt:filesGrace:remarks:fid:filesetname:\n'

class QuotaTests(unittest.TestCase):
    def report(self,used=90*1024**3,doubt=0,grace='none'):
        return HEADER+f'mmlsquota:user:0:1:::mmfs1:USR:1:test:{used}:{100*1024**3}:{110*1024**3}:{doubt}:{grace}:1:0:0:0:none::10:scrubbed:\n'
    def test_kib_units_and_soft_limit(self):
        report=quota.parse_quota(self.report(),'test')
        self.assertEqual(report['safe_headroom_bytes'],10*1024**4)
        self.assertEqual(report['hard_headroom_bytes'],20*1024**4)
    def test_reserves_in_doubt_blocks(self):
        report=quota.parse_quota(self.report(doubt=1024**3),'test')
        self.assertEqual(report['safe_headroom_bytes'],9*1024**4)
    def test_wrong_user_fails_closed(self):
        with self.assertRaises(ValueError):quota.parse_quota(self.report(),'someone-else')
    def test_hard_limit_blocks_launch(self):
        with patch.object(quota.getpass,'getuser',return_value='test'),patch.object(quota.subprocess,'check_output',return_value=self.report(110*1024**3)):
            with self.assertRaises(OSError):quota.check_storage_quota()
    def check(self,used=106*1024**3,doubt=0,grace='7 days'):
        with patch.object(quota.getpass,'getuser',return_value='test'),patch.object(quota.subprocess,'check_output',return_value=self.report(used,doubt,grace)):
            return quota.check_storage_quota()
    def test_long_active_grace_allows_hard_headroom(self):
        report=self.check()
        self.assertEqual(report['safe_headroom_bytes'],0)
        self.assertEqual(report['launch_headroom_bytes'],4*1024**4)
        self.assertEqual(report['launch_quota_basis'],'hard_limit_with_active_grace')
    def test_grace_preserves_in_doubt_reservation(self):
        self.assertEqual(self.check(doubt=1024**3)['launch_headroom_bytes'],3*1024**4)
    def test_grace_never_overrides_hard_limit(self):
        with self.assertRaises(OSError):self.check(used=110*1024**3)
    def test_expired_or_unknown_grace_blocks(self):
        for grace in ['expired','none','unknown','unlimited']:
            with self.subTest(grace=grace),self.assertRaises(OSError):self.check(grace=grace)
    def test_short_or_rounded_grace_blocks(self):
        for grace in ['1 day','2 days','12 hours','48 hours','30 minutes']:
            with self.subTest(grace=grace),self.assertRaises(OSError):self.check(grace=grace)
    def test_below_soft_limit_uses_hard_headroom(self):
        # Crossing the soft limit starts a fresh grace period, so the hard limit is the basis.
        report=self.check(used=95*1024**3,grace='none')
        self.assertEqual(report['launch_headroom_bytes'],report['hard_limit_bytes']-95*1024**4)
        self.assertEqual(report['launch_quota_basis'],'hard_limit_with_active_grace')

if __name__=='__main__':unittest.main()
