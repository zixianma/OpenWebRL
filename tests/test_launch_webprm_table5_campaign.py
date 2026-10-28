"""Offline full-controller contracts; no allocation, model, browser or HTTP."""
import asyncio
from copy import deepcopy
import hashlib
import io
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import time
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

from scripts import launch_webprm_table5_campaign as owner
from openwebrl.webprm_webarena_worker import protocol_definition

R = owner.RUNTIME/'next-stage-preparation'


def save(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value))


class ReadinessTests(unittest.IsolatedAsyncioTestCase):
    async def test_cold_probe_timeout_is_reaped_and_retried_before_actor_admission(self):
        probes=[Mock(),Mock()]; command=Mock(side_effect=probes); check=Mock()
        with patch.object(owner.time,'time',return_value=100), \
                patch.object(owner,'wait_process',new=AsyncMock(side_effect=[TimeoutError('cold HTTP'),None])) as wait, \
                patch.object(owner,'_stop') as stop, patch.object(owner.asyncio,'sleep',new=AsyncMock()):
            await owner.wait_site_readiness(command, ['nsenter'], 'shopping', Path('/tmp'), 200, check)
        self.assertEqual(command.call_count,2)
        self.assertEqual([x.args[1] for x in wait.await_args_list],[115,115])
        self.assertEqual([x.args[0] for x in stop.call_args_list],probes)
        self.assertEqual(command.call_args_list[1].args[1],Path('/tmp/readiness-1.log'))

    async def test_site_deadline_is_never_extended_after_probe_timeout(self):
        probe=Mock();command=Mock(return_value=probe)
        with patch.object(owner.time,'time',side_effect=[100,100,110]), \
                patch.object(owner,'wait_process',new=AsyncMock(side_effect=TimeoutError('cold HTTP'))) as wait, \
                patch.object(owner,'_stop') as stop:
            with self.assertRaisesRegex(TimeoutError,'HTTP readiness failed'):
                await owner.wait_site_readiness(command, [], 'shopping', Path('/tmp'), 105, Mock())
        self.assertEqual(wait.await_args.args[1],105)
        self.assertEqual(command.call_count,1);stop.assert_called_once_with(probe,seconds=1)

    async def test_resource_guard_failure_does_not_become_a_probe_retry(self):
        command=Mock();check=Mock(side_effect=RuntimeError('scratch cap reached'))
        with self.assertRaisesRegex(RuntimeError,'scratch cap reached'):
            await owner.wait_site_readiness(command, [], 'shopping', Path('/tmp'), 200, check)
        command.assert_not_called()


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.tmp=TemporaryDirectory(prefix='webprm-t5-controller-',dir='/tmp')
        self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        block=patch('socket.socket.connect',side_effect=AssertionError('Network forbidden'))
        block.start();self.addCleanup(block.stop)
        self.cohort=owner.fixed_cohort(owner.read(owner.TASKS))
        self.cell=next(c for c in self.cohort['cells'] if c['domain']=='reddit' and
            c['actor']=='gpt-4o-2024-05-13' and c['selector']=='no_search')
        self.template=owner.read(R/'reddit-no-search-guarded-v4-cpu-v1/pending-worker-plan.json')
        self.template.update(site_runtime=deepcopy(owner.SITE_RUNTIME_CONFIGURATION),
            native_helpers=deepcopy(owner.HELPER_CONFIGURATION),evaluator_observations=deepcopy(owner.TRANSCRIPT_CONFIGURATION))
        self.template.update({key:value for key,value in owner.CAMPAIGN_SCIENCE.items() if key!='protocol'})
        self.template['campaign_science']=deepcopy(owner.CAMPAIGN_SCIENCE)
        definition=protocol_definition(self.template['arbiter_source'],turn_limit=owner.CORRECTED_TURN_LIMIT,
            piotr_representation=owner.CORRECTED_PIOTR_REPRESENTATION,no_search_prompt=owner.GUARDED_NO_SEARCH,
            scoring_protocol=owner.GUARDED_EVALUATOR,helper_configuration=owner.HELPER_CONFIGURATION,
            site_configuration=owner.SITE_RUNTIME_CONFIGURATION,observation_configuration=owner.TRANSCRIPT_CONFIGURATION,
            campaign_science=owner.CAMPAIGN_SCIENCE)
        self.template.update(protocol=definition['identity'],protocol_contract=definition)
        self.cohort=owner.frozen_cohort(owner.digest(definition))
        source={str(owner.REPO/p):owner.sha(owner.REPO/p) for p in (*owner.REQUIRED_CODE,*owner.EXTRA_CODE)}
        self.template['code_sha256']=source.copy()
        save(self.root/'template.json',self.template);save(self.root/'cohort.json',{'cohort':self.cohort})
        prior=owner.read(owner.CRITIC_REFERENCE)
        routes,cell_routes=owner.route_contract(self.cohort['actors'])
        reddit=dict(endpoint='http://127.0.0.1',state_apparent_bytes=48089754285,
            **{k:prior[k] for k in ('download_receipt','raw_verification','startup_review')})
        for path in [owner.CRITIC_REFERENCE,*(owner.RUNTIME/p for p in ('prepare_reddit_original.py',
            'run_reddit_instance.py','start_reddit_services.sh')),*(reddit[k] for k in
            ('download_receipt','raw_verification','startup_review'))]:source[str(path)]=owner.sha(path)
        self.proposal=dict(protocol=owner.CONTROLLER_PROTOCOL,campaign_root=str(self.root/'campaign'),
            cohort_path=str(self.root/'cohort.json'),cohort_file_sha256=owner.sha(self.root/'cohort.json'),
            cohort_sha256=owner.digest(self.cohort),worker_template=str(self.root/'template.json'),
            worker_template_sha256=owner.sha(self.root/'template.json'),source_sha256=source,
            broker_routes=routes,cell_routes=cell_routes,env_file=prior['env_file'],
            protocol_sha256=owner.digest(definition),resources=dict(gpus=1,gpu_type='H200',cpus=8,memory_gib=120,total_seconds=7200),
            limits=dict(api_usd=2.,paid_requests=100,local_requests=100,total_requests=200,scratch_bytes=300*2**30,
                runtime_reserve_bytes=16*2**30,max_workers=2,per_cell_memory_bytes=8*2**30,fixed_memory_bytes=80*2**30,
                min_start_seconds=300,cleanup_seconds=180,retained_bytes=64*2**30),models={k:dict(v,gpu_index=0) for k,v in prior['models'].items()},
            domains={'reddit':reddit},allocation_policy={'maximum_attempts':3},ephemeral_policy={'protocol':owner.DISPOSAL_PROTOCOL,'synthetic_test_only':True},
            nltk_environment=dict(PYTHONPATH=str(owner.RUNTIME/'nltk-runtime/venv/lib/python3.12/site-packages')+':'+str(owner.REPO),
                NLTK_DATA=str(owner.RUNTIME/'nltk-runtime/nltk_data')))

    def test_native_authentication_failure_stops_only_fresh_admission(self):
        runner=owner.CampaignRunner.__new__(owner.CampaignRunner)
        runner.proposal=self.proposal;runner.context=dict(deadline=200);runner.output=self.root
        runner.admission_halt_reason='native_authentication_failure';runner.check=Mock()
        with patch.object(owner.time,'time',return_value=100), \
                patch.object(owner,'allocation_memory_bytes',side_effect=AssertionError('No resource probing after halt')):
            self.assertFalse(runner.can_start(self.cell,1))
        decision=owner.read(self.root/'admission-decisions/00000.json')
        self.assertEqual(decision['reason'],'native_authentication_failure')
        self.assertNotIn('halt_reason',runner.__dict__)

    def test_ordinary_invalid_actor_result_does_not_trigger_authentication_guard(self):
        failure=dict(valid=False,error_type='EvaluationNotReady',error='Native authentication failed: rejected login')
        self.assertTrue(owner.native_authentication_failure(failure))
        self.assertFalse(owner.native_authentication_failure(dict(failure,valid=True)))
        self.assertFalse(owner.native_authentication_failure(dict(failure,error='Native evaluator unavailable')))
        self.assertFalse(owner.native_authentication_failure(dict(failure,error_type='ValueError')))

    def test_cms_startup_serializes_first_episode_before_resource_admission(self):
        runner=owner.CampaignRunner.__new__(owner.CampaignRunner)
        runner.proposal=deepcopy(self.proposal)
        runner.proposal['domains']={'shopping_admin':{}}
        runner.context=dict(deadline=1000,job_id='42');runner.output=self.root;runner.check=Mock()
        with patch.object(owner.time,'time',return_value=100), \
                patch.object(owner,'allocation_memory_bytes',side_effect=RuntimeError('resource gate reached')):
            self.assertFalse(runner.can_start(self.cell,1))
            runner.cms_startup_verified=True
            with self.assertRaisesRegex(RuntimeError,'resource gate reached'):
                runner.can_start(self.cell,1)
        self.assertEqual(owner.read(self.root/'admission-decisions/00000.json')['reason'],
                         'cms_startup_canary_inflight')

    def test_actual_fixed_grid_preview_is_only_reddit_ready_without_execution(self):
        with patch.object(owner,'_command',side_effect=AssertionError('No launched process')):
            value=owner.preview(self.proposal)
        self.assertEqual(value['planned_cells'],1550)
        self.assertEqual(value['currently_implemented_domains'],['reddit'])
        self.assertEqual(set(value['blocked_domains']),{'map','shopping','shopping_admin','gitlab'})
        self.assertEqual(value['scratch_accounting'],'apparent_bytes')

    def test_changed_cohort_science_or_critic_configuration_is_rejected(self):
        for change in ('cohort','science','critic','route','seed','judge'):
            proposal=deepcopy(self.proposal)
            if change=='cohort':proposal['cohort_sha256']='0'*64
            elif change=='critic':proposal['models']['arbiter7b']['context_length']+=1
            elif change=='route':proposal['cell_routes'][self.cell['actor']]['no_search'].append(['unapproved','policy'])
            else:
                template=deepcopy(self.template)
                if change=='seed':template['seed']=43
                elif change=='judge':template['fuzzy_enabled']=False
                else:template['evaluator_protocol']='native_evaluator_v1'
                save(self.root/'changed.json',template)
                proposal.update(worker_template=str(self.root/'changed.json'),worker_template_sha256=owner.sha(self.root/'changed.json'))
            with self.subTest(change=change),self.assertRaises(ValueError):owner.validate_proposal(proposal)

    def test_protocol_bound_cohort_is_identical_to_independent_auditor_membership(self):
        from scripts.audit_webprm_table5_cell import audit_identity
        cell=self.cell
        plan=owner.cell_plan(self.template,self.proposal,dict(job_id='42',deadline=time.time()+100),cell,
            socket_path='/tmp/fixture.sock',ports=dict(zip(owner.CRITICS,(33001,33002,33003))))
        # Identity/source checks are real. Only runtime-only source path policy is
        # deferred until the prospective source snapshot exists.
        with patch('scripts.audit_webprm_table5_cell.audit_code_freeze'):
            task=audit_identity(dict(cell=cell),plan,self.cohort)
        self.assertEqual(task.released_task_id,cell['task_id'])
        self.assertIsNone(self.cohort['resource_approval_sha256'])
        prepared=owner.fixed_cohort(owner.read(owner.TASKS))
        save(self.root/'cohort.json',{'cohort':prepared})
        proposal=dict(self.proposal,cohort_file_sha256=owner.sha(self.root/'cohort.json'),cohort_sha256=owner.digest(prepared))
        with self.assertRaisesRegex(ValueError,'membership, order or source'):
            owner.validate_proposal(proposal)

    def test_unapproved_execute_does_not_query_scheduler_or_create_campaign(self):
        with patch.object(owner.subprocess,'check_output',side_effect=AssertionError('No scheduler query')):
            with self.assertRaisesRegex(ValueError,'approval required'):
                owner.execute(self.proposal,dict(status='pending'))
        self.assertFalse(Path(self.proposal['campaign_root']).exists())

    def test_both_actor_plans_bind_actual_cell_and_common_protocol(self):
        context={'job_id':'42','deadline':time.time()+3600}
        for actor in self.cohort['actors']:
            cell=dict(self.cell,actor=actor)
            plan=owner.cell_plan(self.template,self.proposal,context,cell,socket_path='/tmp/fixture.sock',
                ports=dict(zip(owner.CRITICS,(33001,33002,33003))))
            self.assertEqual(plan['actors'],[actor]);self.assertEqual(plan['task_ids'],[cell['task_id']])
            self.assertEqual(plan['selectors'],[cell['selector']])
            self.assertEqual(plan['actor_model'],actor);self.assertEqual(plan['selector'],cell['selector'])
            self.assertEqual(owner.digest(plan['protocol_contract']),self.proposal['protocol_sha256'])
            self.assertEqual(owner.broker_cell_id(cell),f"webarena_lite_{cell['task_id']:03d}/{actor}__no_search")
        with self.assertRaises(ValueError):
            owner.cell_plan(self.template,self.proposal,context,dict(self.cell,task_config_sha256='0'*64),
                socket_path='/tmp/a',ports={})

    def test_all_released_tasks_preserve_cohort_hash_and_worker_hash_with_unicode(self):
        tasks = owner.read(Path(self.template['arbiter_source'])/owner.TASK_FILE)
        indexed = {t['task_id']:t for t in tasks}
        checked, unicode_differences = set(), 0
        proposal = deepcopy(self.proposal)
        proposal['domains'] = {domain:dict(endpoint='http://127.0.0.1') for domain in self.cohort['domain_counts']}
        for cell in self.cohort['cells']:
            if cell['task_id'] in checked: continue
            checked.add(cell['task_id']); raw=indexed[cell['task_id']]
            if owner.digest(raw) != owner.cohort_task_digest(raw): unicode_differences += 1
            plan = owner.cell_plan(self.template,proposal,dict(job_id='42',deadline=time.time()+100),cell,
                                   socket_path='/tmp/offline.sock', ports=dict(zip(owner.CRITICS,(1,2,3))))
            self.assertEqual(plan['tasks_sha256'], owner.digest([raw]))
            self.assertEqual(cell['task_config_sha256'], owner.cohort_task_digest(raw))
        self.assertEqual(len(checked),155)
        self.assertGreater(unicode_differences,0)

    def test_scheduler_epochs_caps_and_prior_attempt_time_are_enforced(self):
        now=int(time.time())
        approval=dict(status='approved',proposal_sha256=owner.digest(self.proposal),job_id='42',
            resources=self.proposal['resources'],limits=self.proposal['limits'],protocol_sha256=self.proposal['protocol_sha256'],
            ephemeral_policy_sha256=owner.digest(self.proposal['ephemeral_policy']))
        scheduler=f'JobId=42 JobState=RUNNING UserId=fixture({os.getuid()}) NumNodes=1 NumCPUs=8 AllocTRES=cpu=8,mem=120G,gres/gpu:h200=1'
        value=lambda n:dict(set=True,infinite=False,number=n)
        data=dict(errors=[],jobs=[dict(job_id=42,job_state=['RUNNING'],user_id=os.getuid(),restart_cnt=0,
            start_time=value(now-10),end_time=value(now+3590),time_limit=value(60))])
        with patch.dict(os.environ,SLURM_JOB_ID='42'):
            result=owner.approved_context(self.proposal,approval,scheduler,json.dumps(data),0,now=now)
            self.assertEqual(result['deadline'],now+3590-180)
            with self.assertRaises(ValueError):owner.approved_context(self.proposal,approval,scheduler,json.dumps(data),4000,now=now)
            with self.assertRaises(ValueError):owner.approved_context(self.proposal,approval,scheduler.replace('120G','240G'),json.dumps(data),0,now=now)

    def test_result_exit_zero_cannot_substitute_for_worker_identity_and_closure(self):
        folder=self.root/'episode';folder.mkdir()
        plan={'fixture':True};watchdog={'remaining_owned_processes':{}}
        result=dict(identity_sha256=owner.digest(plan),task_id=f"webarena_lite_{self.cell['task_id']:03d}",
            actor_model=self.cell['actor'],selector=self.cell['selector'],
            arm=self.cell['actor']+'__'+self.cell['selector'],attempted=True,valid=True,success=0,watchdog=watchdog)
        save(folder/'result.json',result);save(folder/'watchdog.json',watchdog)
        self.assertFalse(owner.cell_result(folder/'result.json',self.cell,plan)['audit_verified'])
        result['watchdog']={'remaining_owned_processes':{'123':{}}};save(folder/'result.json',result)
        with self.assertRaises(ValueError):owner.cell_result(folder/'result.json',self.cell,plan)

    def test_active_lane_admission_checks_actual_job_memory_and_shared_budget(self):
        runner=object.__new__(owner.CampaignRunner)
        runner.proposal=self.proposal;runner.context={'deadline':time.time()+1000,'job_id':'42','prior_retained_apparent_bytes':0}
        runner.scratch=self.root;runner.output=self.root;runner.reserved={};runner.check=lambda:None
        runner.broker=SimpleNamespace(status=lambda:dict(admission_open=True,budget=dict(halt_reason=None,
            remaining='2',paid_dispatches=0,local_dispatches=0)))
        with patch.object(owner,'apparent',return_value=0),patch.object(owner,'allocation_memory_bytes',return_value=1),\
                patch.object(owner.shutil,'disk_usage',return_value=SimpleNamespace(free=300*2**30)):
            self.assertTrue(runner.can_start(self.cell,0))
            with patch.object(owner,'allocation_memory_bytes',return_value=120*2**30):
                self.assertFalse(runner.can_start(self.cell,0))
        runner.release(self.cell);self.assertEqual(runner.reserved,{})
        decisions=[json.loads(p.read_text()) for p in sorted((self.root/'admission-decisions').glob('*.json'))]
        self.assertEqual([v['reason'] for v in decisions],['all_gates_passed','allocation_memory_capacity'])

    def test_failed_paid_reservation_stops_new_cells_with_positive_residual(self):
        from openwebrl.webprm_budget_journal import JournalBudget
        runner=object.__new__(owner.CampaignRunner)
        runner.proposal=self.proposal;runner.context={'deadline':time.time()+1000,'job_id':'42','prior_retained_apparent_bytes':0}
        runner.scratch=self.root;runner.output=self.root;runner.reserved={};runner.check=lambda:None
        with JournalBudget(self.root/'admission-budget.json',1,policy_sha256='a'*64) as budget:
            budget.reserve('.8','actor')
            with self.assertRaises(RuntimeError):budget.reserve('.3','actor')
            runner.broker=SimpleNamespace(status=lambda:dict(admission_open=True,budget=budget.summary()))
            with patch.object(owner,'apparent',return_value=0),patch.object(owner,'allocation_memory_bytes',return_value=1),\
                    patch.object(owner.shutil,'disk_usage',return_value=SimpleNamespace(free=300*2**30)):
                self.assertFalse(runner.can_start(self.cell,0))
            self.assertEqual(runner.reserved,{})
            decision=json.loads((self.root/'admission-decisions/00000.json').read_text())
            self.assertEqual(decision['reason'],'paid_reservation_rejected')
            self.assertEqual(decision['budget_remaining_usd'],'0.2')
            self.assertEqual(decision['broker_reservation_rejection']['amount'],'0.3')

    def test_startup_rejects_base_only_storage_fit_before_any_model_call(self):
        runner=object.__new__(owner.CampaignRunner);runner.proposal=deepcopy(self.proposal)
        runner.context={'prior_retained_apparent_bytes':225*2**30};runner.scratch=self.root;runner.output=self.root
        with patch.object(owner,'apparent',return_value=0),\
                patch.object(owner.shutil,'disk_usage',return_value=SimpleNamespace(free=300*2**30)):
            # An immutable base fits in 300 GiB, but a base plus one live cell does not.
            with self.assertRaisesRegex(ValueError,'base and one cell'):runner.startup_capacity()
        proof=json.loads((self.root/'startup-capacity.json').read_text())
        self.assertFalse(proof['admission_allowed'])
        self.assertGreater(proof['minimum_projected_scratch_bytes'],proof['scratch_cap_bytes'])

    def test_storage_pressure_at_lane_handoff_is_released_without_cap_change(self):
        runner=object.__new__(owner.CampaignRunner);runner.proposal=deepcopy(self.proposal)
        runner.proposal['limits']['scratch_bytes']=512*2**30
        runner.proposal['domains']['reddit']['state_apparent_bytes']=50*2**30
        runner.context={'deadline':time.time()+1000,'job_id':'42','prior_retained_apparent_bytes':336670263878}
        runner.scratch=self.root;runner.output=self.root;runner.check=lambda:None
        runner.reserved={'live-cell':50*2**30}
        runner.broker=SimpleNamespace(status=lambda:dict(admission_open=True,budget=dict(halt_reason=None,
            remaining='2',paid_dispatches=0,local_dispatches=0)))
        with patch.object(owner,'allocation_memory_bytes',return_value=1),\
                patch.object(owner.shutil,'disk_usage',return_value=SimpleNamespace(free=300*2**30)):
            with patch.object(owner,'apparent',return_value=2*48092985316):
                self.assertFalse(runner.can_start(self.cell,1))
            runner.release({'cell_id':'live-cell'})
            with patch.object(owner,'apparent',return_value=48092985316):
                self.assertTrue(runner.can_start(self.cell,0))
        decisions=[json.loads(p.read_text()) for p in sorted((self.root/'admission-decisions').glob('*.json'))]
        self.assertEqual([v['reason'] for v in decisions],['cumulative_scratch_capacity','all_gates_passed'])
        self.assertTrue(all(v['scratch_cap_bytes']==512*2**30 for v in decisions))

    def registration_approval(self,proposal=None,prefix=None):
        proposal=self.proposal if proposal is None else proposal
        return dict(status='approved',proposal_sha256=owner.digest(proposal),resources=proposal['resources'],
            limits=proposal['limits'],protocol_sha256=proposal['protocol_sha256'],
            ephemeral_policy_sha256=owner.digest(proposal['ephemeral_policy']),maximum_allocation_attempts=3,
            attempt_job_ids=[] if prefix is None else prefix,job_id=None)

    def test_durable_allocation_registration_rejects_omissions_and_fourth_attempt(self):
        first=self.registration_approval();owner.register_allocation(self.proposal,first,'41')
        owner.register_allocation(self.proposal,first,'41')  # idempotent held-job registration
        with self.assertRaisesRegex(ValueError,'omit or reorder'):
            owner.register_allocation(self.proposal,first,'42')
        second=self.registration_approval(prefix=['41']);owner.register_allocation(self.proposal,second,'42')
        owner.validate_registered_allocation(self.proposal,dict(second,job_id='42',attempt_job_ids=['41','42']))
        with self.assertRaisesRegex(ValueError,'registered allocation'):
            owner.validate_registered_allocation(self.proposal,dict(second,job_id='42',attempt_job_ids=['42']))
        owner.register_allocation(self.proposal,self.registration_approval(prefix=['41','42']),'43')
        with self.assertRaisesRegex(ValueError,'three distinct'):
            owner.register_allocation(self.proposal,self.registration_approval(prefix=['41','42','43']),'44')

    def test_fourth_attempt_requires_exact_exception_and_keeps_contract(self):
        for job,prefix in (('41',[]),('42',['41']),('43',['41','42'])):
            owner.register_allocation(self.proposal,self.registration_approval(prefix=prefix),job)
        extended=deepcopy(self.proposal)
        extended['allocation_exception']=dict(additional_attempts=1,cumulative_seconds=extended['resources']['total_seconds'])
        approval=self.registration_approval(extended,['41','42','43'])
        with self.assertRaisesRegex(ValueError,'exact approved exception'):
            owner.register_allocation(extended,approval,'44')
        approval.update(maximum_allocation_attempts=4,allocation_exception_user_approval='I approve',
            allocation_exception_sha256=owner.digest(extended['allocation_exception']))
        with self.assertRaisesRegex(ValueError,'exact approved exception'):
            owner.register_allocation(extended,dict(approval,allocation_exception_user_approval=' '),'44')
        owner.register_allocation(extended,approval,'44')
        rows=owner.validate_registered_allocation(extended,dict(approval,job_id='44',attempt_job_ids=['41','42','43','44']))
        self.assertEqual(rows[-1]['allocation_contract_sha256'],owner.allocation_contract(self.proposal))
        with self.assertRaisesRegex(ValueError,'At most 4'):
            owner.register_allocation(extended,dict(approval,attempt_job_ids=['41','42','43','44']),'45')
        reset=deepcopy(extended);reset['allocation_exception']['cumulative_seconds']+=1
        with self.assertRaisesRegex(ValueError,'exact approved exception'):
            owner.register_allocation(reset,dict(self.registration_approval(reset,['41','42','43']),maximum_allocation_attempts=4,
                allocation_exception_user_approval='I approve',allocation_exception_sha256=owner.digest(reset['allocation_exception'])),'44')

    def test_per_node_scratch_accounting_requires_exact_exception(self):
        here=self.root/'t5-1';here.mkdir()
        finals=[dict(scratch=str(here),apparent_bytes=7),dict(scratch=str(self.root/'elsewhere'),apparent_bytes=300)]
        self.assertEqual(owner.scratch_accounting(self.proposal,{}),'cumulative')
        self.assertEqual(owner.prior_scratch_charge('cumulative',finals),307)
        extended=deepcopy(self.proposal)
        extended['scratch_accounting_exception']=dict(rule='per_node_live',scratch_bytes=extended['limits']['scratch_bytes'])
        approval=dict(scratch_accounting_exception_sha256=owner.digest(extended['scratch_accounting_exception']),
            scratch_accounting_exception_user_approval='yes I approve')
        self.assertEqual(owner.scratch_accounting(extended,approval),'per_node_live')
        self.assertEqual(owner.prior_scratch_charge('per_node_live',finals),7)
        for bad in (dict(approval,scratch_accounting_exception_user_approval=' '),
                dict(approval,scratch_accounting_exception_sha256='0'*64)):
            with self.assertRaisesRegex(ValueError,'exact approved exception'):owner.scratch_accounting(extended,bad)
        raised=deepcopy(extended);raised['scratch_accounting_exception']['scratch_bytes']+=1
        with self.assertRaisesRegex(ValueError,'exact approved exception'):
            owner.scratch_accounting(raised,dict(approval,scratch_accounting_exception_sha256=owner.digest(raised['scratch_accounting_exception'])))
        # The exception is outside the cumulative allocation contract.
        self.assertEqual(owner.allocation_contract(extended),owner.allocation_contract(self.proposal))

    def test_unregistered_old_attempt_is_not_treated_as_unused_budget(self):
        root=Path(self.proposal['campaign_root']);root.mkdir(mode=0o700);(root/'attempt-40').mkdir()
        with self.assertRaisesRegex(ValueError,'Unregistered historical'):
            owner.register_allocation(self.proposal,self.registration_approval(),'41')

    def test_operational_source_lineage_preserves_cumulative_allocation_contract(self):
        owner.register_allocation(self.proposal,self.registration_approval(),'41')
        fixed=deepcopy(self.proposal);fixed['source_sha256']['reviewed_operational_fix']='1'*64
        owner.register_allocation(fixed,self.registration_approval(fixed,['41']),'42')
        rows=owner.validate_registered_allocation(fixed,dict(self.registration_approval(fixed,['41']),
            job_id='42',attempt_job_ids=['41','42']))
        self.assertNotEqual(rows[0]['proposal_sha256'],rows[1]['proposal_sha256'])
        expanded=deepcopy(fixed);expanded['resources']['total_seconds']+=1
        with self.assertRaisesRegex(ValueError,'resource/science contract'):
            owner.register_allocation(expanded,self.registration_approval(expanded,['41','42']),'43')

    def test_retained_artifact_cap_is_independent_of_node_scratch(self):
        runner=object.__new__(owner.CampaignRunner)
        runner.proposal=self.proposal;runner.context={'deadline':time.time()+1000,'prior_retained_apparent_bytes':0}
        runner.scratch=self.root/'scratch';runner.last_storage=0.;runner.halt_reason=None
        runner.servers=[];runner.broker=None
        cap=self.proposal['limits']['retained_bytes']
        with patch.object(owner,'apparent',side_effect=[1,cap+1]):
            with self.assertRaisesRegex(RuntimeError,'Durable campaign artifact cap'):
                runner.check()
        runner.last_storage=0.
        with patch.object(owner,'apparent',side_effect=[1,cap]):runner.check()
        self.assertEqual(runner.retained,cap)

    def test_storage_timeout_is_preserved_and_never_becomes_a_partial_sample(self):
        import subprocess
        runner=object.__new__(owner.CampaignRunner)
        runner.proposal=self.cms_proposal();runner.output=self.root
        runner.context={'deadline':time.time()+3600,'prior_retained_apparent_bytes':0}
        runner.scratch=self.root/'scratch';runner.last_storage=0.;runner.halt_reason=None
        runner.servers=[];runner.broker=None;runner.used=123;runner.retained=456
        failure=subprocess.TimeoutExpired(['du','--apparent-size',str(runner.scratch)],1800,
            output='999999\tpartial\n')
        with patch.object(owner,'apparent',side_effect=failure):
            with self.assertRaises(subprocess.TimeoutExpired):runner.check()
        receipt=json.loads((self.root/'storage-measurement-timeout.json').read_text())
        self.assertEqual(receipt['role'],'node_scratch')
        self.assertEqual(receipt['timeout_seconds'],1800)
        self.assertFalse(receipt['partial_sample_used'])
        self.assertFalse(receipt['admissions_may_continue'])
        self.assertEqual((runner.used,runner.retained),(123,456))
        original=(self.root/'storage-measurement-timeout.json').read_bytes()
        runner.halt_reason='Worker closure is incomplete; fresh admissions halted'
        with self.assertRaisesRegex(RuntimeError,'Worker closure'):runner.check()
        self.assertEqual((self.root/'storage-measurement-timeout.json').read_bytes(),original)

    def test_start_registers_both_actors_and_all_fixed_critic_routes_without_live_calls(self):
        output=self.root/'output';scratch=self.root/'scratch';output.mkdir();scratch.mkdir()
        context=dict(job_id='42',deadline=time.time()+1000,prior_retained_apparent_bytes=0,
            approval={})
        ready_models=json.dumps(dict(data=[{'id':m['served_model']} for m in self.proposal['models'].values()]))
        fake_process=SimpleNamespace(poll=lambda:None,returncode=None)
        broker=SimpleNamespace(start=lambda:broker,process=SimpleNamespace(is_alive=lambda:True))
        cells=[next(c for c in self.cohort['cells'] if c['domain']=='reddit' and c['actor']==a and
            c['selector']=='no_search') for a in self.cohort['actors']]
        with patch.dict(os.environ),patch.object(owner,'environment',return_value=dict(os.environ)),\
                patch.object(owner,'apparent',return_value=0),\
                patch.object(owner.shutil,'disk_usage',return_value=SimpleNamespace(free=400*2**30)),\
                patch('scripts.runtime_ports.lease_ports',return_value=(Mock(),33001)),\
                patch('scripts.prepare_webprm_piotr_processor.validate_view',return_value={}),\
                patch.object(owner,'loopback_urlopen',side_effect=lambda *a,**k:io.StringIO(ready_models)),\
                patch('openwebrl.webprm_concurrent_broker.ConcurrentBrokerProcess',return_value=broker) as broker_class,\
                patch('openwebrl.webprm_broker.make_broker_client',return_value=object()),\
                patch('openwebrl.webprm_reddit_pilot_preflight.validate_local_critics',return_value={'offline_fixture':True}),\
                patch('openwebrl.webprm_reddit_pilot_preflight.validate_guarded_fuzzy',return_value={'offline_fixture':True}),\
                patch.object(owner.asyncio,'sleep',new_callable=AsyncMock):
            runner=owner.CampaignRunner(self.proposal,context,output,scratch)
            def command(argv,log,env=None):
                Path(log).write_text('offline subprocess double\n')
                if '--mode' in argv:
                    save(output/'original-build/receipt.json',{'offline_fixture':True})
                    return SimpleNamespace(poll=lambda:0,returncode=0)
                return fake_process
            runner.command=command
            asyncio.run(runner.start(cells))
        registered=broker_class.call_args.kwargs['cells']
        self.assertEqual(set(registered),{owner.broker_cell_id(c) for c in cells}|{'local_preflight','native_fuzzy_preflight'})
        for cell in cells:
            value=registered[owner.broker_cell_id(cell)]
            self.assertEqual(value['config']['actor_model'],cell['actor'])
            self.assertIn([cell['actor'],'policy'],value['allowed_routes'])
            self.assertEqual(value['config']['identity_sha256'],owner.digest(runner.plans[cell['cell_id']]))
        self.assertEqual(broker_class.call_args.kwargs['financial_contract']['cohort_sha256'],self.proposal['cohort_sha256'])
        self.assertIsNone(broker_class.call_args.kwargs['previous_closed_epoch_sha256'])
        self.assertTrue((output/'base-evidence.json').is_file())

    def test_original_copy_and_service_require_the_exact_fresh_cell_claim(self):
        from scripts.run_webprm_table5_reddit import stage_paths
        from openwebrl.webprm_ephemeral_state import claim_state
        campaign=self.root/'campaign';scratch=self.root/'scratch';campaign.mkdir(mode=0o700);scratch.mkdir(mode=0o700)
        (scratch/'states').mkdir();(scratch/'baseline').mkdir()
        output=campaign/'attempt-42';output.mkdir();folder=output/'cells'/owner.cell_key(self.cell);folder.mkdir(parents=True)
        state=scratch/'states'/owner.cell_key(self.cell)
        claim=claim_state(folder/'state-claim.json',state,scratch_root=scratch,campaign_root=campaign,
            cell_id=self.cell['cell_id'],base_path=scratch/'baseline',policy_sha256=owner.digest(self.proposal['ephemeral_policy']))
        owned=dict(scratch=str(scratch),output=str(output),proposal=self.proposal)
        self.assertEqual(stage_paths(owned,'copy',state,folder/'copy'),(state,folder/'copy'))
        with self.assertRaises(ValueError):stage_paths(owned,'serve',state,folder/'site')
        state.mkdir();self.assertEqual(stage_paths(owned,'serve',state,folder/'site'),(state,folder/'site'))
        with self.assertRaises(ValueError):stage_paths(owned,'copy',state,folder/'copy')
        save(folder/'state-claim.json',dict(claim,policy_sha256='0'*64))
        with self.assertRaises(ValueError):stage_paths(owned,'serve',state,folder/'site')

    def test_concrete_reddit_command_never_uses_hardcoded_task5_or_mini_actor(self):
        self._concrete_reddit_command_audit_result(False)

    def test_verified_invalid_reddit_audit_propagates_without_disposing_state(self):
        self._concrete_reddit_command_audit_result(True)

    def _concrete_reddit_command_audit_result(self,audit_verified):
        runner=object.__new__(owner.CampaignRunner)
        campaign=self.root/'campaign';campaign.mkdir(mode=0o700)
        scratch=self.root/'scratch';scratch.mkdir(mode=0o700);(scratch/'states').mkdir();(scratch/'baseline').mkdir()
        output=campaign/'attempt-42';output.mkdir()
        cell=deepcopy(self.cell);cell['task_id']=14;cell['cell_id']='webarena_lite_014__'+cell['actor']+'__no_search'
        folder=output/'cells'/owner.cell_key(cell);folder.mkdir(parents=True)
        plan={'fixture':True,'budget_path':str(campaign/'budget.json')}
        save(folder/'worker-plan.json',plan);save(output/'financial-contract.json',{'fixture':True});(output/'broker.jsonl').write_text('')
        save(output/'base-evidence.json',{'fixture':True})
        runner.proposal=dict(self.proposal,campaign_root=str(campaign));runner.context={'deadline':time.time()+1000}
        runner.output=output;runner.scratch=scratch;runner.plans={cell['cell_id']:plan};runner.folders={cell['cell_id']:folder}
        runner.check=lambda:None;runner.broker=SimpleNamespace(status=lambda:dict(active_cell_ids=[],budget=dict(
            budget_prefix_bytes=0,budget_ledger_sha256=hashlib.sha256(b'').hexdigest())))
        commands=[]
        process=SimpleNamespace(poll=lambda:0,returncode=0)
        def command(argv,log,env=None):
            commands.append(argv)
            if '--mode' in argv:
                mode=argv[argv.index('--mode')+1]
                if mode=='copy':
                    (scratch/'states'/owner.cell_key(cell)).mkdir();save(folder/'copy/receipt.json',{'fixture':True})
                if mode=='serve':
                    save(folder/'site/instance.json',{'service_pid':123})
                    save(folder/'site/closure.json',{'remaining_owned_processes':{}})
            if '--execute' in argv:
                episode=folder/'episodes/webarena_lite_014'/(cell['actor']+'__no_search')
                watchdog={'remaining_owned_processes':{}}
                save(episode/'watchdog.json',watchdog)
                save(episode/'result.json',dict(identity_sha256=owner.digest(plan),task_id='webarena_lite_014',
                    actor_model=cell['actor'],selector='no_search',arm=cell['actor']+'__no_search',
                    attempted=True,valid=False,success=0,watchdog=watchdog))
            if any('audit_webprm_table5_cell.py' in str(arg) for arg in argv):
                save(folder/'audit.json',dict(audit_verified=audit_verified,complete_for_disposal=False))
                code=0 if audit_verified else 2
                return SimpleNamespace(poll=lambda:code,returncode=code)
            return process
        runner.command=command
        with patch.object(owner,'_stop'):
            result=asyncio.run(runner.run_one(cell))
        launch=next(argv for argv in commands if '--execute' in argv)
        self.assertEqual(launch[launch.index('--task-id')+1],'14')
        self.assertEqual(launch[launch.index('--actor')+1],'gpt-4o-2024-05-13')
        self.assertTrue(result['state_preserved']);self.assertEqual(result['audit_verified'],audit_verified)
        self.assertEqual(result['audit_status'],'verified' if audit_verified else 'rejected')
        self.assertTrue((scratch/'states'/owner.cell_key(cell)).exists())
        self.assertTrue((folder/'state-claim.json').exists())
        self.assertTrue((folder/'audit-input.json').exists())


    def test_shopping_memory_accounting_excludes_only_reclaimable_page_cache(self):
        for controllers,relative,current,stat in (('','/system.slice/job_77/step_0','memory.current','anon 30\nfile 80\nshmem 10\n'),
                ('memory','/slurm/uid_1/job_77/step_0','memory.usage_in_bytes','total_rss 30\ntotal_cache 80\ntotal_shmem 10\n')):
            with self.subTest(controllers=controllers):
                root=self.root/('cg-'+(controllers or 'v2'));proc=root/'proc-cgroup';proc.parent.mkdir()
                proc.write_text(f'5:{controllers}:{relative}\n' if controllers else f'0::{relative}\n')
                job=root/'fs'/(controllers or '')/relative.lstrip('/');job=job.parent;job.mkdir(parents=True)
                (job/current).write_text('100\n');(job/'memory.stat').write_text(stat)
                self.assertEqual(owner.allocation_memory_bytes('77',proc=proc,cgroups=root/'fs'),100)
                self.assertEqual(owner.allocation_memory_bytes('77','cgroup_minus_reclaimable_file_cache',proc=proc,cgroups=root/'fs'),30)
        with self.assertRaises(ValueError):owner.allocation_memory_bytes('77','free_memory')
        self.assertEqual(owner.MEMORY_ACCOUNTING,{'shopping':'cgroup_minus_reclaimable_file_cache'})

    def cms_proposal(self):
        from openwebrl import webprm_cms_site as cms
        proposal=deepcopy(self.proposal);source=proposal['source_sha256']
        for name in owner.CMS_CODE:source[str(owner.REPO/name)]=owner.sha(owner.REPO/name)
        for ref in (cms.HOSTING_PROPOSAL,cms.HOSTING_VALIDATION):source[ref['path']]=ref['sha256']
        proposal['domains']={'shopping_admin':dict(endpoint=cms.ENDPOINT,state_apparent_bytes=15*2**30,
            hosting_proposal=cms.HOSTING_PROPOSAL,hosting_validation=cms.HOSTING_VALIDATION,
            base_content_sha256=cms.BASE_CONTENT_SHA256)}
        return proposal

    def test_cms_campaign_hosts_only_the_validated_cms_and_reddit_preview_is_unchanged(self):
        proposal=self.cms_proposal()
        with patch.object(owner,'_command',side_effect=AssertionError('No launched process')):
            value=owner.preview(proposal)
        self.assertEqual(value['currently_implemented_domains'],['shopping_admin'])
        self.assertEqual(set(value['blocked_domains']),{'map','shopping','reddit','gitlab'})
        self.assertEqual(owner.preview(self.proposal)['currently_implemented_domains'],['reddit'])
        entry=proposal['domains']['shopping_admin']
        for change in ('mixed','endpoint','validation','content','bytes','unfrozen','extra'):
            changed=deepcopy(proposal);item=changed['domains']['shopping_admin']
            if change=='mixed':changed['domains']['reddit']=deepcopy(self.proposal['domains']['reddit'])
            elif change=='endpoint':item['endpoint']='http://127.0.0.1:7780'
            elif change=='validation':item['hosting_validation']=dict(entry['hosting_validation'],sha256='0'*64)
            elif change=='content':item['base_content_sha256']='0'*64
            elif change=='bytes':item['state_apparent_bytes']=0
            elif change=='extra':item['unreviewed']=True
            else:changed['source_sha256'].pop(str(owner.REPO/'scripts/run_webprm_table5_cms.py'))
            with self.subTest(change=change),self.assertRaises(ValueError):owner.validate_proposal(changed)

    def shopping_proposal(self):
        from openwebrl import webprm_shopping_site as shop
        proposal=deepcopy(self.proposal);source=proposal['source_sha256']
        for name in owner.SHOPPING_CODE:source[str(owner.REPO/name)]=owner.sha(owner.REPO/name)
        for ref in (shop.HOSTING_PROPOSAL,shop.HOSTING_VALIDATION):source[ref['path']]=ref['sha256']
        proposal['domains']={'shopping':dict(endpoint=shop.ENDPOINT,state_apparent_bytes=65*2**30,
            hosting_proposal=shop.HOSTING_PROPOSAL,hosting_validation=shop.HOSTING_VALIDATION,
            base_content_sha256=shop.BASE_CONTENT_SHA256)}
        return proposal

    def test_shopping_campaign_hosts_only_validated_shopping_with_its_storage_policy(self):
        proposal=self.shopping_proposal()
        with patch.object(owner,'_command',side_effect=AssertionError('No launched process')):
            value=owner.preview(proposal)
        self.assertEqual(value['currently_implemented_domains'],['shopping'])
        self.assertEqual(set(value['blocked_domains']),{'map','shopping_admin','reddit','gitlab'})
        entry=proposal['domains']['shopping']
        for change in ('mixed','cms','endpoint','validation','content','unfrozen'):
            changed=deepcopy(proposal);item=changed['domains']['shopping']
            if change=='mixed':changed['domains']['reddit']=deepcopy(self.proposal['domains']['reddit'])
            elif change=='cms':changed['domains']['shopping_admin']=deepcopy(self.cms_proposal()['domains']['shopping_admin'])
            elif change=='endpoint':item['endpoint']='http://127.0.0.1:7780'
            elif change=='validation':item['hosting_validation']=dict(entry['hosting_validation'],sha256='0'*64)
            elif change=='content':item['base_content_sha256']='0'*64
            else:changed['source_sha256'].pop(str(owner.REPO/'openwebrl/webprm_shopping_site.py'))
            with self.subTest(change=change),self.assertRaises(ValueError):owner.validate_proposal(changed)
        runner=object.__new__(owner.CampaignRunner);runner.proposal=proposal;runner.output=self.root
        self.assertEqual(runner.site_domain,'shopping');self.assertEqual(runner.du_timeout,1800)
        self.assertTrue(runner.service_command('copy',state=self.root/'s',folder=self.root/'f')[2].endswith('scripts/run_webprm_table5_cms.py'))
        self.assertIn('webprm_shopping_site',owner.READINESS['shopping'])
        self.assertEqual(owner.AUDITORS['shopping'],'scripts/audit_webprm_table5_cms_cell.py')
        self.assertEqual(owner.STORAGE_SAMPLE_SECONDS['shopping_admin'],300)

    def test_cms_runner_uses_cms_stage_state_bound_and_readiness(self):
        runner=object.__new__(owner.CampaignRunner);runner.proposal=self.cms_proposal();runner.output=self.root
        self.assertEqual(runner.site_domain,'shopping_admin')
        self.assertEqual(runner.du_timeout,1800)
        argv=runner.service_command('copy',state=self.root/'s',folder=self.root/'f')
        self.assertTrue(argv[2].endswith('scripts/run_webprm_table5_cms.py'))
        reddit=object.__new__(owner.CampaignRunner);reddit.proposal=self.proposal;reddit.output=self.root
        self.assertTrue(reddit.service_command('copy',state=self.root/'s',folder=self.root/'f')[2].endswith('run_webprm_table5_reddit.py'))
        self.assertIn('webprm_cms_site',owner.READINESS['shopping_admin'])
        self.assertGreater(owner.SITE_STOP_SECONDS['shopping_admin'],owner.SITE_STOP_SECONDS['reddit'])
        runner.context={'prior_retained_apparent_bytes':0};runner.scratch=self.root
        with patch.object(owner,'apparent',return_value=0),\
                patch.object(owner.shutil,'disk_usage',return_value=SimpleNamespace(free=300*2**30)):
            runner.startup_capacity()
        proof=json.loads((self.root/'startup-capacity.json').read_text())
        self.assertEqual(proof['base_bytes'],15*2**30);self.assertTrue(proof['admission_allowed'])


if __name__=='__main__':unittest.main()
