"""Fresh-process tests of unchanged actual frozen packages and strict denial guard."""
import json
from pathlib import Path
import subprocess
import sys
import unittest

REPO=Path(__file__).resolve().parents[1]
ADAPTER=REPO/'scripts/retry_webprm_hosting_after_denial.py'
CHILD=r'''
import importlib.util,json,sys,tempfile
from pathlib import Path
from unittest.mock import patch
spec=importlib.util.spec_from_file_location('unfrozen_retry_adapter',sys.argv[1])
adapter=importlib.util.module_from_spec(spec);spec.loader.exec_module(adapter)
submitter,proposal,approval,metadata=adapter.load(sys.argv[2])
campaign=Path(submitter.h.read(proposal)['campaign_root'])
before=adapter.sha(proposal)
with patch.object(submitter.subprocess,'check_output',side_effect=AssertionError('No scheduler calls in empty-history preview')):
    preview=submitter.preview(proposal,approval)[2]
if sys.argv[3]=='missing':
    real=submitter.h.read
    def read(path,*args,**kwargs):
        value=real(path,*args,**kwargs)
        if Path(path).name.startswith('submission-denial-'):
            value=dict(value,reconciliation=dict(path='/tmp/webprm-deliberately-missing-denial-evidence.json',sha256='a'*64))
        return value
    with patch.object(submitter.h,'read',side_effect=read):
        try:submitter.submission_lineage(campaign,[])
        except submitter.h.Closed:pass
        else:raise AssertionError('Missing denial evidence was accepted')
elif sys.argv[3]=='ambiguous':
    with tempfile.TemporaryDirectory(dir='/tmp') as folder:
        root=Path(folder);submitter.h.put(root/('submission-intent-'+'a'*24+'.json'),dict(nonce='a'*24))
        try:submitter.submission_lineage(root,[])
        except submitter.h.Closed:pass
        else:raise AssertionError('Unreconciled intent was accepted')
assert before==adapter.sha(proposal)
print(json.dumps(dict(site=sys.argv[2],mode=sys.argv[3],preview=preview,proposal_sha256=metadata['proposal_sha256'])))
'''


class RetryAdapterTests(unittest.TestCase):
    def child(self,site,mode):
        result=subprocess.run([sys.executable,'-I','-B','-c',CHILD,str(ADAPTER),site,mode],capture_output=True,text=True,timeout=30)
        self.assertEqual(result.returncode,0,result.stderr)
        row=json.loads(result.stdout)
        self.assertEqual(row['preview']['allocations_submitted'],0)
        self.assertEqual(row['preview']['prior_job_ids'],[])
        return row

    def test_actual_frozen_packages_and_reconciled_sidecars_preview_without_scheduler(self):
        for site,minutes in (('gitlab',120),('map',240)):
            with self.subTest(site=site):self.assertEqual(self.child(site,'valid')['preview']['requested_minutes'],minutes)

    def test_actual_denial_with_missing_bound_evidence_rejects_without_mutation(self):
        for site in ('gitlab','map'):
            with self.subTest(site=site):self.child(site,'missing')

    def test_ambiguous_unresolved_intent_remains_blocked(self):
        self.child('map','ambiguous')


if __name__=='__main__':unittest.main()
