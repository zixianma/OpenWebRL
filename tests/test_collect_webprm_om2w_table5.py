"""Artifact tampering and incomplete-primary accounting checks; no network or browser."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from scripts import collect_webprm_om2w_table5 as c
from openwebrl.webprm_webarena import digest


class CollectorIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir='/tmp'); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.campaign = self.root/'campaign'
        self.cell = dict(cell_id='om2w_t/a__no_search', actor='a', selector='no_search', task_id='t',
                         stage='pilot10', task_config_sha256='task', domain='om2w-pilot10')
        template = dict(protocol_contract={'fixed': True}, code_sha256={}, prices_usd_per_token={'a':[1,2]})
        self.folder = self.campaign/'attempt-1/cells'/hashlib.sha256(self.cell['cell_id'].encode()).hexdigest()
        self.result_path = self.folder/'episodes/om2w_t/a__no_search/result.json'
        self.episode = self.result_path.parent
        self.put(self.root/'template.json', template)
        cohort = dict(cells=[self.cell])
        self.put(self.root/'cohort.json', cohort)
        proposal = dict(campaign_root=str(self.campaign),stage='pilot10',
            source_sha256={str(self.root/'cohort.json'):c.sha(self.root/'cohort.json'),
                           str(self.root/'template.json'):c.sha(self.root/'template.json')},
            cohort_path=str(self.root/'cohort.json'),cohort_sha256=digest(cohort),worker_template=str(self.root/'template.json'),
            task_file=str(self.root/'tasks.jsonl'), models={'arbiter7b':dict(served_model='critic')},
            limits=dict(cleanup_seconds=300,api_usd=50),resources=dict(total_seconds=28800))
        self.put(self.root/'proposal.json',proposal)
        attempt=self.folder.parents[1]
        context=dict(job_id='1',proposal_sha256=digest(proposal),output=str(attempt),deadline=10700,
            prior_seconds=2201,scratch=str(self.root/'scratch'),scheduler_epochs=dict(job_id='1',start_epoch=1000,end_epoch=11000,limit_seconds=10000))
        self.put(attempt/'execution-context.json',context)
        self.put(attempt/'arbiter7b-command.json',dict(argv=['serve','--host','127.0.0.1','--port','20000']))
        from scripts.launch_webprm_om2w_table5 import cell_plan
        self.plan=cell_plan(template,proposal,context,self.cell,socket_path=self.root/'scratch/broker/socket',ports={'arbiter7b':20000})
        self.put(self.folder/'worker-plan.json',self.plan)
        self.put(self.campaign/'queue/manifest.json',dict(cell_count=1,cells_sha256=c.identity([self.cell])))
        self.key = c.identity(self.cell['cell_id'])
        self.claim = dict(cell_id=self.cell['cell_id'],cell_sha256=c.identity(self.cell),owner='a'*32,epoch=1200)
        self.claim_path = self.campaign/'queue/claims'/(self.key+'.json')
        self.outcome_path = self.campaign/'queue/outcomes'/(self.key+'.json')
        self.put(self.claim_path,self.claim)
        self.put(self.episode/'watchdog.json',dict(remaining_owned_processes={}))
        self.put(self.result_path,dict(identity_sha256=digest(self.plan),task_id='om2w_t',arm='a__no_search',
            actor_model='a',selector='no_search',attempted=True,valid=False,success=0,
            watchdog=dict(remaining_owned_processes={}),error_type='RuntimeError'))
        result,outcome=c.verified_result(self.result_path,self.cell,self.plan)
        self.put(self.folder/'cell-outcome.json',outcome)
        self.put(self.outcome_path,dict(self.claim,result=outcome))

    def put(self,path,value):
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))

    def test_closed_invalid_record_is_audited_attempt_not_valid(self):
        rows=c.stage_records(self.root)
        self.assertEqual(rows[0][2],'recorded')
        summary=c.summarize(rows)
        self.assertFalse(summary['rates_withheld'])
        self.assertEqual(summary['by_condition'][0]['valid'],0)
        self.assertEqual(summary['by_condition'][0]['attempted'],1)

    def test_subset_manifest_and_accounting_match_launched_cells(self):
        extra=dict(self.cell,cell_id='om2w_other/a__no_search',task_id='other')
        cohort=dict(cells=[self.cell,extra]);self.put(self.root/'cohort.json',cohort)
        proposal=c.read(self.root/'proposal.json');proposal['cohort_sha256']=digest(cohort)
        proposal['source_sha256'][str(self.root/'cohort.json')]=c.sha(self.root/'cohort.json')
        proposal['cell_ids']=[self.cell['cell_id']];self.put(self.root/'proposal.json',proposal)
        context_path=self.folder.parents[1]/'execution-context.json'
        context=c.read(context_path);context['proposal_sha256']=digest(proposal);self.put(context_path,context)
        # A pending unselected cell must not appear in the queue or planned denominator.
        rows=c.stage_records(self.root)
        self.assertEqual([cell['cell_id'] for cell,_,_ in rows],[self.cell['cell_id']])
        proposal['cell_ids']=[extra['cell_id']];self.put(self.root/'proposal.json',proposal)
        with self.assertRaises(ValueError):c.stage_records(self.root)

    def test_changed_result_bytes_rejected_even_when_identity_still_valid(self):
        value=c.read(self.result_path);value['error_type']='OtherError';self.put(self.result_path,value)
        with self.assertRaisesRegex(ValueError,'hash'):c.stage_records(self.root)

    def test_changed_claim_owner_rejected(self):
        self.put(self.claim_path,dict(self.claim,owner='another'))
        with self.assertRaisesRegex(ValueError,'owner'):c.stage_records(self.root)

    def test_empty_or_foreign_claim_cannot_disappear_from_accounting(self):
        self.put(self.claim_path,{})
        with self.assertRaisesRegex(ValueError,'identity'):c.stage_records(self.root)
        self.put(self.claim_path,self.claim)
        self.put(self.campaign/'queue/claims/foreign.json',self.claim)
        with self.assertRaisesRegex(ValueError,'outside fixed stage'):c.stage_records(self.root)

    def test_changed_protocol_rejected(self):
        self.put(self.folder/'worker-plan.json',dict(self.plan,protocol_contract={'different':True}))
        with self.assertRaisesRegex(ValueError,'protocol'):c.stage_records(self.root)

    def test_unapproved_prices_endpoints_models_task_path_or_extra_knob_rejected(self):
        for change in (dict(prices_usd_per_token={'a':[0,0]}),dict(arbiter_endpoint='https://external.invalid'),
                       dict(arbiter_model='other'),dict(task_file='/tmp/other.jsonl'),dict(temperature=100)):
            self.put(self.folder/'worker-plan.json',dict(self.plan,**change))
            with self.assertRaisesRegex(ValueError,'reconstruction'):c.stage_records(self.root)

    def test_claim_outside_owning_attempt_clock_rejected(self):
        for epoch in (999,10700):
            claim=dict(self.claim,epoch=epoch)
            self.put(self.claim_path,claim)
            with self.assertRaisesRegex(ValueError,'allocation context'):c.stage_records(self.root)

    def test_live_or_missing_worker_closure_rejected(self):
        self.put(self.episode/'watchdog.json',dict(remaining_owned_processes={'7':123}))
        with self.assertRaisesRegex(ValueError,'closure'):c.stage_records(self.root)

    def test_modified_frozen_cohort_rejected(self):
        self.put(self.root/'cohort.json',dict(cells=[]))
        with self.assertRaisesRegex(ValueError,'source'):c.stage_records(self.root)

    def test_valid_judge_is_bound_to_terminal_images_and_executed_trajectory(self):
        self.put(self.episode/'rollout.json',[])
        (self.episode/'judge-final.jpg').write_bytes(b'saved-jpeg')
        png=self.episode/'native-evaluation/terminal.png';png.parent.mkdir();png.write_bytes(b'saved-png')
        judge=dict(model=c.JUDGE,finish_reason='stop',raw='Status: success',parsed=1.0,termination='stop',
                   final_screenshot_sha256=c.sha(self.episode/'judge-final.jpg'),
                   trajectory_sha256=hashlib.sha256(c.judge_trajectory([],None,'stop').encode()).hexdigest())
        self.put(self.episode/'om2w-judge.json',judge)
        self.put(self.episode/'native-evaluation/native-verdict.json',dict(valid=True,score=1,success=True,judge=judge))
        result=c.read(self.result_path);result.update(valid=True,success=1,terminal=dict(screenshot_sha256=c.sha(png)))
        self.put(self.result_path,result)
        self.assertTrue(c.verified_result(self.result_path,self.cell,self.plan)[0]['valid'])
        (self.episode/'judge-final.jpg').write_bytes(b'different-jpeg')
        with self.assertRaisesRegex(ValueError,'evidence'):c.verified_result(self.result_path,self.cell,self.plan)

    def repair_fixture(self):
        """Exercise collector dispatch after the separate strict parent verifier.

        The launcher's tests cover approval/source/cohort parity. Here the mock
        represents that already verified pair and isolates per-attempt binding.
        """
        parent=c.read(self.root/'proposal.json');template=c.read(self.root/'template.json')
        package=self.root/'repaired';package.mkdir()
        repaired_template=dict(template,code_sha256={'operational-broker':'new-code'})
        self.put(package/'template.json',repaired_template)
        repaired=dict(parent,worker_template=str(package/'template.json'),
            repair_of={'proposal':{'path':str(self.root/'proposal.json'),'sha256':c.sha(self.root/'proposal.json')},
                       'approval':{'path':str(self.root/'approved-request.json'),'sha256':'original-approval'},
                       'reason':'provider429_paid_dispatch_pacing','paid_dispatch_interval_seconds':3.0},
            source_sha256={str(self.root/'cohort.json'):c.sha(self.root/'cohort.json'),
                           str(package/'template.json'):c.sha(package/'template.json')})
        self.put(package/'proposal.json',repaired)
        return package,repaired,repaired_template,(parent,template)

    def test_repaired_package_preserves_old_result_and_original_full_plan(self):
        package,proposal,template,parent=self.repair_fixture()
        before={p:p.read_bytes() for p in (self.result_path,self.folder/'worker-plan.json',self.claim_path,self.outcome_path)}
        with patch('scripts.launch_webprm_om2w_table5.repair_parent',return_value=parent,create=True):
            rows=c.stage_records(package)
        self.assertEqual(rows[0][2],'recorded')
        self.assertFalse(rows[0][1]['valid'])
        self.assertEqual(before,{p:p.read_bytes() for p in before})
        self.assertEqual(c.attempt_protocol(proposal,template,self.folder.parents[1],parent),parent)

    def test_new_attempt_uses_only_repaired_template_and_proposal(self):
        package,proposal,template,parent=self.repair_fixture()
        attempt=self.folder.parents[1];context=c.read(attempt/'execution-context.json')
        context['proposal_sha256']=digest(proposal);self.put(attempt/'execution-context.json',context)
        plan=c.assigned_plan(template,proposal,self.cell,attempt,self.claim)
        self.put(self.folder/'worker-plan.json',plan)
        result=c.read(self.result_path);result['identity_sha256']=digest(plan);self.put(self.result_path,result)
        _,outcome=c.verified_result(self.result_path,self.cell,plan)
        self.put(self.folder/'cell-outcome.json',outcome);self.put(self.outcome_path,dict(self.claim,result=outcome))
        with patch('scripts.launch_webprm_om2w_table5.repair_parent',return_value=parent,create=True):
            self.assertEqual(c.stage_records(package)[0][2],'recorded')
            # Replacing a new attempt's plan with its valid historical counterpart
            # cannot pass merely because the cohort and scientific knobs match.
            self.put(self.folder/'worker-plan.json',self.plan)
            with self.assertRaisesRegex(ValueError,'reconstruction'):c.stage_records(package)

    def test_parent_not_explicitly_verified_cannot_be_inferred_from_same_cells(self):
        package,proposal,template,parent=self.repair_fixture()
        with patch('scripts.launch_webprm_om2w_table5.repair_parent',return_value=None,create=True):
            with self.assertRaisesRegex(ValueError,'repair lineage'):c.stage_records(package)

    def test_unknown_or_byte_hash_context_cannot_select_parent(self):
        package,proposal,template,parent=self.repair_fixture()
        context_path=self.folder.parents[1]/'execution-context.json';context=c.read(context_path)
        for wrong in ('unregistered-proposal',c.sha(self.root/'proposal.json')):
            context['proposal_sha256']=wrong;self.put(context_path,context)
            with patch('scripts.launch_webprm_om2w_table5.repair_parent',return_value=parent,create=True):
                with self.assertRaisesRegex(ValueError,'repair lineage'):c.stage_records(package)

    def test_repair_parent_verification_failure_is_not_ignored(self):
        package,_,_,_=self.repair_fixture()
        with patch('scripts.launch_webprm_om2w_table5.repair_parent',side_effect=ValueError('Parent source changed'),create=True):
            with self.assertRaisesRegex(ValueError,'Parent source changed'):c.stage_records(package)

    def stopped_fixture(self,broker=False):
        watchdog=dict(remaining_owned_processes={},reason=None,supervisor_signal=15,supervisor_error=None,
            child_exitcode=0 if broker else -15,last_phase={'identity_sha256':digest(self.plan),'phase':'completed' if broker else 'actor_request'})
        result=c.read(self.result_path)
        result.update(watchdog=watchdog,error_type='BrokerError' if broker else 'EpisodeWorkerExited',
            error='Host broker rejected or failed request; no retry' if broker else 'Episode child exited without result (exit -15)')
        if broker:result['cleanup_status']='completed'
        else:
            result.pop('actor_model');result.pop('selector')
        outcome=dict(status='invalid',reason='worker_exception',error_type='ConnectionRefusedError' if broker else 'ValueError')
        self.put(self.episode/'watchdog.json',watchdog);self.put(self.result_path,result)
        self.put(self.outcome_path,dict(self.claim,result=outcome))
        (self.folder/'cell-outcome.json').rename(self.folder/'test-preserved-normal-outcome.json')
        return result,watchdog,outcome

    def test_closed_sigterm_fallback_and_broker_invalids_remain_raw_invalid(self):
        result,_,_=self.stopped_fixture()
        rows=c.stage_records(self.root)
        self.assertEqual(rows[0][1],result)
        self.assertEqual(rows[0][2],'recorded')
        self.assertEqual(c.summarize(rows)['by_condition'][0]['successes'],0)

    def test_closed_broker_failure_retains_complete_identity(self):
        result,_,_=self.stopped_fixture(broker=True)
        self.assertEqual(c.stage_records(self.root)[0][1],result)
        changed=dict(result,selector='other');self.put(self.result_path,changed)
        with self.assertRaises(ValueError):c.stage_records(self.root)

    def test_stopped_fallback_rejects_live_wrong_exit_signal_and_positive_verdicts(self):
        result,watchdog,outcome=self.stopped_fixture()
        for change in ({'remaining_owned_processes':{'7':123}},{'supervisor_signal':9},
                       {'supervisor_signal':True},{'child_exitcode':0},{'supervisor_error':'error'},
                       {'reason':'phase_timeout'},{'last_phase':{'identity_sha256':'different'}}):
            wrong=dict(watchdog,**change);self.put(self.episode/'watchdog.json',wrong)
            self.put(self.result_path,dict(result,watchdog=wrong))
            with self.assertRaises(ValueError):c.stage_records(self.root)
        self.put(self.episode/'watchdog.json',watchdog)
        for change in ({'valid':True},{'success':1},{'success':False},{'identity_sha256':'different'},
                       {'actor_model':self.cell['actor']},{'error':'different'}):
            self.put(self.result_path,dict(result,**change))
            with self.assertRaises(ValueError):c.stage_records(self.root)

    def test_stopped_fallback_cannot_override_saved_native_judge(self):
        self.stopped_fixture()
        self.put(self.episode/'native-evaluation/native-verdict.json',{'valid':True,'success':True})
        with self.assertRaisesRegex(ValueError,'closed invalid'):c.stage_records(self.root)


if __name__=='__main__':unittest.main()
