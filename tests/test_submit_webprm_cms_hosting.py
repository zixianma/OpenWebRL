import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch,Mock

from scripts import submit_webprm_cms_hosting as submit


class CMSSubmissionTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir='/tmp');self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.campaign=self.root/'campaign'
        self.approval=self.root/'approval.json';self.approval.write_text('{}')
        self.proposal=self.root/'proposal.json';self.proposal.write_text('{}')
        self.request=dict(prior_job_ids=[],terminal_rows={},budget=dict(remaining_seconds=7200),
                          requested_minutes=120,batch_script=str(self.root/'batch'),allocations_submitted=0)
        self.p=dict(campaign_root=str(self.campaign))

    def test_held_receipt_and_registration_precede_release(self):
        events=[];original=submit.h.put
        def record(path,value):
            if Path(path).name=='submission-42.json':events.append('receipt')
            return original(path,value)
        def command(argv,**_):
            if argv[0]=='sbatch':events.append('held');return '42\n'
            if argv[1]=='--json':return '{}'
            events.append('release');return ''
        with patch.object(submit,'preview',return_value=(self.p,{},self.request)),\
             patch.object(submit.subprocess,'check_output',side_effect=command),\
             patch.object(submit.h,'register',side_effect=lambda *_:events.append('registered')),\
             patch.object(submit.h,'put',side_effect=record):
            result=submit.submit(self.proposal,self.approval)
        self.assertEqual(events,['held','receipt','registered','release'])
        self.assertEqual(result['job_id'],'42')

    def test_both_receipt_writes_failing_still_cancels_known_held_job(self):
        original=submit.h.put
        def record(path,value):
            if Path(path).name in ('submission-42.json','submission-failure-42.json'):raise OSError('fixture')
            return original(path,value)
        with patch.object(submit,'preview',return_value=(self.p,{},self.request)),\
             patch.object(submit.subprocess,'check_output',return_value='42'),\
             patch.object(submit.subprocess,'run',return_value=Mock(returncode=0)) as cancel,\
             patch.object(submit.h,'register') as register,patch.object(submit.h,'put',side_effect=record):
            with self.assertRaisesRegex(submit.h.Closed,'job 42 failed'):
                submit.submit(self.proposal,self.approval)
            cancel.assert_called_once();self.assertEqual(cancel.call_args.args[0],['scancel','42'])
            register.assert_not_called()

    def test_registration_failure_never_releases(self):
        with patch.object(submit,'preview',return_value=(self.p,{},self.request)),\
             patch.object(submit.subprocess,'check_output',side_effect=['42','{}']) as commands,\
             patch.object(submit.subprocess,'run',return_value=Mock(returncode=0)),\
             patch.object(submit.h,'register',side_effect=ValueError('fixture')):
            with self.assertRaisesRegex(submit.h.Closed,'job 42 failed'):
                submit.submit(self.proposal,self.approval)
            self.assertEqual(commands.call_count,2)

    def test_preview_requires_prior_terminal_state(self):
        rows=[dict(job_id='41',sequence=1,requested_seconds=7200)]
        with patch.object(submit.h,'read',side_effect=[self.p,{}]),patch.object(submit.h,'validate'),\
             patch.object(submit.h,'history',return_value=rows),\
             patch.object(submit.owner,'terminal_rows',return_value={'41':dict(state='RUNNING',elapsed_seconds=30)}):
            with self.assertRaisesRegex(submit.h.Closed,'terminal'):
                submit.preview(self.proposal,self.approval)

    def test_unresolved_submission_never_silently_disappears_from_budget(self):
        self.campaign.mkdir()
        intent=self.campaign/'submission-intent-fixture.json';submit.h.put(intent,dict(nonce='fixture'))
        with self.assertRaisesRegex(submit.h.Closed,'Unresolved'):
            submit.submission_lineage(self.campaign,[])
        submit.h.put(self.campaign/'submission-42.json',dict(job_id='42',intent=submit.owner.ref(intent)))
        with self.assertRaisesRegex(submit.h.Closed,'Unregistered'):
            submit.submission_lineage(self.campaign,[])
        submit.submission_lineage(self.campaign,[dict(job_id='42')])
