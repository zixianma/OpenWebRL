"""Offline submission ownership: no actual scheduler mutation or model call."""
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import subprocess
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from scripts import submit_webprm_table5_campaign as submitter


class SubmissionTests(unittest.TestCase):
    def setUp(self):
        self.tmp=TemporaryDirectory(dir='/tmp',prefix='webprm-held-submission-');self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.proposal_path=self.root/'proposal.json';self.approval_path=self.root/'approval.json'
        self.proposal={'campaign_root':str(self.root/'campaign')};self.approval={'status':'pending'}
        self.proposal_path.write_text(json.dumps(self.proposal));self.approval_path.write_text(json.dumps(self.approval))
        self.request={'argv':['sbatch','--hold','fixture'], 'prior_job_ids':[], 'allocation_minutes':1440}

    def test_unapproved_request_never_invokes_scheduler(self):
        with patch.object(submitter.subprocess,'check_output',side_effect=AssertionError('No scheduler call')):
            with self.assertRaisesRegex(ValueError,'Exact approved'):
                submitter.preview(self.proposal_path,self.approval_path)

    def test_held_job_is_durably_registered_before_release(self):
        order=[]
        def command(argv,**kwargs):
            order.append('hold' if argv[0]=='sbatch' else 'release')
            return '42\n' if argv[0]=='sbatch' else ''
        def register(*args):
            self.assertTrue((self.root/'submission-42.json').is_file());order.append('register')
        with patch.object(submitter,'preview',return_value=(self.proposal,self.approval,self.request)),\
                patch.object(submitter.subprocess,'check_output',side_effect=command),\
                patch.object(submitter.owner,'register_allocation',side_effect=register):
            result=submitter.submit(self.proposal_path,self.approval_path)
        self.assertEqual(order,['hold','register','release']);self.assertEqual(result['job_id'],'42')
        self.assertTrue((self.root/'released-42.json').is_file())

    def test_registration_failure_cancels_only_new_held_job_without_release(self):
        with patch.object(submitter,'preview',return_value=(self.proposal,self.approval,self.request)),\
                patch.object(submitter.subprocess,'check_output',return_value='42\n') as check,\
                patch.object(submitter.subprocess,'run',return_value=SimpleNamespace(returncode=0)) as cancel,\
                patch.object(submitter.owner,'register_allocation',side_effect=ValueError('registration failed')):
            with self.assertRaisesRegex(RuntimeError,'Held submission 42 failed'):
                submitter.submit(self.proposal_path,self.approval_path)
        self.assertEqual(check.call_count,1);self.assertEqual(cancel.call_args.args[0],['scancel','42'])
        self.assertFalse((self.root/'released-42.json').exists())
        self.assertTrue((self.root/'submission-failure-42.json').is_file())

    def test_initial_and_failure_receipt_write_errors_still_cancel_identified_held_job(self):
        with patch.object(submitter,'preview',return_value=(self.proposal,self.approval,self.request)),\
                patch.object(submitter.subprocess,'check_output',return_value='42\n') as check,\
                patch.object(submitter.subprocess,'run',return_value=SimpleNamespace(returncode=0)) as cancel,\
                patch.object(submitter.owner,'put',side_effect=OSError('receipt unavailable')) as put,\
                patch.object(submitter.owner,'register_allocation') as register:
            with self.assertRaisesRegex(RuntimeError,'Held submission 42 failed.*failure_receipt_error=OSError'):
                submitter.submit(self.proposal_path,self.approval_path)
        self.assertEqual(check.call_count,1);self.assertEqual(cancel.call_args.args[0],['scancel','42'])
        self.assertEqual(put.call_count,2);register.assert_not_called()

    def prior_request(self):
        root=Path(self.proposal['campaign_root']);old=root/'attempt-41';old.mkdir(parents=True)
        (old/'controller-closure.json').write_text(json.dumps({'remaining_owned_processes':{}}))
        (old/'storage-final.json').write_text(json.dumps({'apparent_bytes':123}))
        batch=self.root/'campaign.sbatch';batch.write_text('# offline fixture')
        self.proposal.update(resources={'total_seconds':86400},allocation_policy={'maximum_attempts':3},limits={'cleanup_seconds':180,'min_start_seconds':1800},
            protocol_sha256='a'*64,ephemeral_policy={'fixture':True},source_sha256={str(batch):submitter.owner.sha(batch)})
        self.approval.update(status='approved',proposal_sha256=submitter.owner.digest(self.proposal),
            resources=self.proposal['resources'],limits=self.proposal['limits'],protocol_sha256='a'*64,
            ephemeral_policy_sha256=submitter.owner.digest(self.proposal['ephemeral_policy']),
            maximum_allocation_attempts=3,job_id=None,attempt_job_ids=['41'],previous_closed_epoch_sha256=None)
        self.proposal_path.write_text(json.dumps(self.proposal));self.approval_path.write_text(json.dumps(self.approval))

    def test_exhausted_slots_block_submission_without_exception(self):
        self.prior_request()
        self.approval.update(attempt_job_ids=['39','40','41']);self.approval_path.write_text(json.dumps(self.approval))
        with patch.object(submitter.owner,'validate_proposal'),\
                patch.object(submitter.owner,'allocation_history',return_value=[{'job_id':j} for j in ('39','40','41')]),\
                patch.object(submitter.subprocess,'check_output',side_effect=AssertionError('No scheduler call')):
            with self.assertRaisesRegex(ValueError,'remaining allocation slot'):
                submitter.preview(self.proposal_path,self.approval_path)

    def test_running_predecessor_blocks_submission_before_any_new_job(self):
        self.prior_request()
        with patch.object(submitter.owner,'validate_proposal'),                patch.object(submitter.owner,'allocation_history',return_value=[{'job_id':'41'}]),                patch.object(submitter.subprocess,'check_output',return_value='41|RUNNING|3601|\n') as check:
            with self.assertRaisesRegex(ValueError,'terminal and accounted'):
                submitter.preview(self.proposal_path,self.approval_path)
        self.assertEqual(check.call_count,1);self.assertEqual(check.call_args.args[0][0],'sacct')

    def test_failed_predecessor_time_is_deducted_and_rounded_down(self):
        self.prior_request()
        with patch.object(submitter.owner,'validate_proposal'),                patch.object(submitter.owner,'allocation_history',return_value=[{'job_id':'41'}]),                patch.object(submitter.subprocess,'check_output',return_value='41|FAILED|3601|\n') as check:
            request=submitter.preview(self.proposal_path,self.approval_path)[2]
        self.assertEqual(request['prior_elapsed_seconds'],3601);self.assertEqual(request['allocation_minutes'],1379)
        self.assertIn('--time=1379',request['argv']);self.assertEqual(check.call_count,1)

    def test_financial_handoff_rejects_open_or_modified_epoch(self):
        campaign=Path(self.proposal['campaign_root']);campaign.mkdir()
        path=campaign/'shared-budget.json.events.jsonl';events=[];previous='0'*64
        for index,kind in enumerate(['header','epoch_started','epoch_closed']):
            event=dict(sequence=index,event=kind,previous_sha256=previous)
            event['event_sha256']=submitter.owner.digest(event);previous=event['event_sha256'];events.append(event)
        path.write_text('\n'.join(json.dumps(e) for e in events)+'\n')
        submitter.financial_handoff(self.proposal,{'previous_closed_epoch_sha256':previous})
        path.write_text('\n'.join(json.dumps(e) for e in events[:2])+'\n')
        with self.assertRaisesRegex(ValueError,'not independently closed'):
            submitter.financial_handoff(self.proposal,{'previous_closed_epoch_sha256':None})
        events[0]['event']='changed';path.write_text(json.dumps(events[0])+'\n')
        with self.assertRaisesRegex(ValueError,'chain differs'):
            submitter.financial_handoff(self.proposal,{'previous_closed_epoch_sha256':None})


    def test_multiday_campaign_caps_each_submission_and_accounts_failed_time(self):
        self.prior_request()
        self.proposal['resources']['total_seconds']=172800
        self.approval.update(resources=self.proposal['resources'],proposal_sha256=submitter.owner.digest(self.proposal))
        self.proposal_path.write_text(json.dumps(self.proposal));self.approval_path.write_text(json.dumps(self.approval))
        with patch.object(submitter.owner,'validate_proposal'),\
                patch.object(submitter.owner,'allocation_history',return_value=[{'job_id':'41'}]),\
                patch.object(submitter.subprocess,'check_output',return_value='41|FAILED|3601|\n'):
            value=submitter.preview(self.proposal_path,self.approval_path)[2]
        self.assertEqual(value['allocation_minutes'],1440)
        self.assertEqual(value['prior_elapsed_seconds'],3601)
        with patch.object(submitter.owner,'validate_proposal'),\
                patch.object(submitter.owner,'allocation_history',return_value=[{'job_id':'41'}]),\
                patch.object(submitter.subprocess,'check_output',return_value='41|FAILED|100001|\n'):
            value=submitter.preview(self.proposal_path,self.approval_path)[2]
        self.assertEqual(value['allocation_minutes'],1213)

if __name__=='__main__':unittest.main()
