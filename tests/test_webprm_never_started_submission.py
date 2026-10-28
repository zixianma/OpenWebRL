import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from scripts.launch_webprm_table5_campaign import closed_attempt_storage


class NeverStartedTests(unittest.TestCase):
    def setUp(self):
        tmp=TemporaryDirectory(dir='/tmp');self.addCleanup(tmp.cleanup);self.root=Path(tmp.name)
        self.proof=dict(job_id='42',status='cancelled_held_before_any_allocation',
            elapsed_seconds=0,start_time='None',allocated_tres='',controller_launched=False,
            campaign_attempt_exists=False,primary_claims=0,model_calls=0)

    def proposal(self):
        path=self.root/'review.json';path.write_text(json.dumps(self.proof))
        return dict(campaign_root=str(self.root),never_started_attempts={
            '42':dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())})

    def test_zero_runtime_never_allocated_cancellation_needs_no_fake_closure(self):
        with patch('scripts.launch_webprm_table5_campaign.subprocess.check_output',
                   return_value='42|CANCELLED by 123|0|None|\n'):
            self.assertIsNone(closed_attempt_storage(self.proposal(),'42'))
        self.assertFalse((self.root/'attempt-42').exists())

    def test_allocated_started_or_positive_runtime_is_rejected(self):
        for row in ['42|CANCELLED|1|None|','42|CANCELLED|0|2026-10-18|',
                    '42|CANCELLED|0|None|cpu=8','42|PENDING|0|None|']:
            with self.subTest(row=row),patch('scripts.launch_webprm_table5_campaign.subprocess.check_output',return_value=row):
                with self.assertRaisesRegex(ValueError,'Scheduler'):
                    closed_attempt_storage(self.proposal(),'42')

    def test_missing_review_and_any_execution_evidence_fail_closed(self):
        with self.assertRaises(ValueError):closed_attempt_storage(dict(campaign_root=str(self.root)),'42')
        self.proof['model_calls']=1
        with self.assertRaises(ValueError):closed_attempt_storage(self.proposal(),'42')
        self.proof['model_calls']=0
        (self.root/'attempt-42').mkdir()
        with self.assertRaises(FileNotFoundError):closed_attempt_storage(self.proposal(),'42')


if __name__=='__main__':unittest.main()
