"""Offline controller guards; no scheduler, browser, container or model startup."""
import json
from copy import deepcopy
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from scripts import launch_webprm_reddit_pilot as pilot


class PilotControllerTests(unittest.TestCase):
    def setUp(self):
        self.root=Path(tempfile.mkdtemp(prefix='wpilot-',dir='/tmp'))
        self.scope=patch.object(pilot,'RUNTIME',self.root);self.scope.start()
        self.addCleanup(self.scope.stop)
        self.proposal=dict(campaign_root=str(self.root/'campaign'),resources=dict(pilot.RESOURCES),
            limits=dict(pilot.LIMITS),rows=list(pilot.ROWS),task_ids=[5],actor=pilot.ACTOR,
            restore='verified_local_oci',models={name:dict(path=str(self.root),config_sha256='',weight_sizes={})
                for name in ('shepherd8b','arbiter7b','piotr_selectionarm')},source_sha256={},
            worker_plan=str(self.root/'plan.json'),worker_plan_sha256='',execution_blockers=['pending judge'])
        self.proposal['approval_only_blockers']=['pending judge']
        self.proposal['activation_changes']=dict(worker_scientific_fields=dict(fuzzy_enabled=True,
            fuzzy_model='gpt-4o-2024-05-13',fuzzy_protocol='webarena_fuzzy_gpt4o_20240513_v1',
            fuzzy_model_change_approved=True))
        self.save(self.root/'plan.json',{})
        self.save(self.root/'config.json',{})
        self.proposal['worker_plan_sha256']=pilot.sha(self.root/'plan.json')
        for model in self.proposal['models'].values():model['config_sha256']=pilot.sha(self.root/'config.json')

    def save(self,path,value):
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value))

    def corrected(self):
        from openwebrl.webprm_webarena_worker import protocol_definition
        from openwebrl.webprm_reddit_pilot_preflight import NATIVE_FUZZY_PREFLIGHT
        source='/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-baselines-20261009/sources/arbiter'
        self.proposal.update(resource_profile=pilot.CORRECTED_PROFILE,
            native_fuzzy_preflight=dict(NATIVE_FUZZY_PREFLIGHT),
            piotr_representation=pilot.CORRECTED_PIOTR_REPRESENTATION,
            resources=dict(pilot.CORRECTED_RESOURCES),limits=dict(pilot.CORRECTED_LIMITS),
            turn_limit_protocol=pilot.CORRECTED_TURN_LIMIT,
            previous_cohort=dict(campaign_root=str(self.root/'reddit-pilot-campaign-v1'),
                relationship='separate_preserved_diagnostic',budget_transfer=False,rows_reused=False))
        self.save(self.root/'plan.json',dict(arbiter_source=source,
            turn_limit_protocol=pilot.CORRECTED_TURN_LIMIT,
            piotr_representation=pilot.CORRECTED_PIOTR_REPRESENTATION,
            protocol='webarena_table5_viewer_reconstruction_v3',
            protocol_contract=protocol_definition(source,turn_limit=pilot.CORRECTED_TURN_LIMIT,
                piotr_representation=pilot.CORRECTED_PIOTR_REPRESENTATION)))
        self.proposal['worker_plan_sha256']=pilot.sha(self.root/'plan.json')
        return self.proposal

    def prior(self,job='10',paid=1,local=2,launched=('no_search',)):
        folder=self.root/'campaign'/('attempt-'+job);folder.mkdir(parents=True)
        self.save(folder/'execution-context.json',dict(job_id=job,proposal_sha256=pilot.digest(self.proposal),scratch='/tmp/retained'))
        self.save(folder/'controller-closure.json',dict(remaining_owned_processes={}))
        self.save(folder/'storage-final.json',dict(scratch_path='/tmp/retained',apparent_bytes=1234))
        rows=[dict(event='dispatch_started',request_id=str(i),local=i>=paid) for i in range(paid+local)]
        (folder/'broker.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
        for row in launched:self.save(folder/('worker-launch-'+row+'.json'),dict(selector=row))
        self.save(folder.parent/'shared-budget.json',dict(cap=15,charged=1,calls=[dict(state='reserved')]))
        return folder

    def cpu_diagnostic(self):
        from openwebrl.webprm_webarena_worker import protocol_definition
        from openwebrl.webprm_reddit_pilot_preflight import GUARDED_FUZZY_PREFLIGHT
        self.corrected()
        self.proposal.update(resource_profile=pilot.CPU_VALIDATION_PROFILE,
            resources=pilot.CPU_VALIDATION_RESOURCES,limits=pilot.CPU_VALIDATION_LIMITS,
            diagnostic_protocol=pilot.CPU_VALIDATION_PROTOCOL,maximum_allocation_attempts=1,
            rows=['no_search'],models={},broker_routes=[dict(local=False)],cell_routes=dict(no_search=[]),
            no_search_prompt_version=pilot.GUARDED_NO_SEARCH,evaluator_protocol=pilot.GUARDED_EVALUATOR,
            native_fuzzy_preflight=GUARDED_FUZZY_PREFLIGHT,
            previous_corrected_cohort=dict(campaign_root=str(self.root/'previous-v3'),
                relationship='separate_preserved_diagnostic',budget_transfer=False,rows_reused=False))
        plan=pilot.read(self.root/'plan.json')
        plan.update(diagnostic_protocol=pilot.CPU_VALIDATION_PROTOCOL,selectors=['no_search'],
            no_search_prompt_version=pilot.GUARDED_NO_SEARCH,evaluator_protocol=pilot.GUARDED_EVALUATOR)
        plan['protocol_contract']=protocol_definition(plan['arbiter_source'],turn_limit=pilot.CORRECTED_TURN_LIMIT,
            piotr_representation=pilot.CORRECTED_PIOTR_REPRESENTATION,
            no_search_prompt=pilot.GUARDED_NO_SEARCH,scoring_protocol=pilot.GUARDED_EVALUATOR)
        plan['protocol']=plan['protocol_contract']['identity']
        self.save(self.root/'plan.json',plan)
        self.proposal['worker_plan_sha256']=pilot.sha(self.root/'plan.json')
        return self.proposal

    def test_cpu_validation_is_fixed_no_model_startup_or_approval_reuse(self):
        proposal=self.cpu_diagnostic();pilot.validate(proposal)
        self.assertEqual(pilot.resource_envelope(proposal),(pilot.CPU_VALIDATION_RESOURCES,pilot.CPU_VALIDATION_LIMITS))
        for changes in (dict(models={'arbiter7b':{}}),dict(rows=list(pilot.ROWS)),
                dict(resource_profile=pilot.CORRECTED_PROFILE),dict(maximum_allocation_attempts=2),
                dict(broker_routes=[dict(local=True)]),dict(evaluator_protocol=None),
                dict(no_search_prompt_version=None),dict(previous_corrected_cohort={})):
            with self.subTest(changes=changes),self.assertRaises(ValueError):
                pilot.validate(dict(proposal,**changes))
        with patch.object(pilot.subprocess,'check_output',side_effect=AssertionError('scheduler forbidden')),\
                patch.object(pilot.subprocess,'Popen',side_effect=AssertionError('spawn forbidden')):
            with self.assertRaisesRegex(ValueError,'not approved'):
                pilot.execute(proposal,dict(status='pending'),self.root/'unstarted')

    def test_cpu_epoch_clock_does_not_interpret_scheduler_local_time(self):
        proposal=self.cpu_diagnostic();now=int(time.time())
        finite=lambda n:dict(set=True,infinite=False,number=n)
        snapshot=dict(errors=[],jobs=[dict(job_id=42,job_state=['RUNNING'],user_id=os.getuid(),restart_cnt=0,
            start_time=finite(now-1),end_time=finite(now+2702),time_limit=finite(45))])
        proof=pilot.cpu_scheduler_epochs(json.dumps(snapshot),'42')
        approval=dict(status='approved',proposal_sha256=pilot.digest(proposal),resource_profile=pilot.CPU_VALIDATION_PROFILE,
            resources=pilot.CPU_VALIDATION_RESOURCES,limits=pilot.CPU_VALIDATION_LIMITS,job_id='42',
            piotr_representation=pilot.CORRECTED_PIOTR_REPRESENTATION,piotr_representation_approved=True,
            turn_limit_protocol=pilot.CORRECTED_TURN_LIMIT,turn_limit_protocol_approved=True,
            no_search_prompt_version=pilot.GUARDED_NO_SEARCH,no_search_prompt_version_approved=True,
            evaluator_protocol=pilot.GUARDED_EVALUATOR,evaluator_protocol_approved=True,
            terminal_judge_replacement_approved=True,terminal_judge_model='gpt-4o-2024-05-13',scheduler_epoch_proof=proof)
        scheduler=f'JobId=42 JobState=RUNNING UserId=test({os.getuid()}) NumNodes=1 NumCPUs=8 AllocTRES=cpu=8,mem=16G,node=1 TimeLimit=00:45:00 EndTime=not-parsed-local-time'
        with patch.dict(os.environ,SLURM_JOB_ID='42'),patch.object(Path,'read_text',return_value='0::/job_42/step_batch'):
            result=pilot.approved_context(proposal,approval,scheduler,0)
            self.assertEqual(result['deadline'],now-1+2700-180)
            for changes in (dict(scheduler_epoch_proof={}),dict(evaluator_protocol_approved=False),
                    dict(no_search_prompt_version_approved=False)):
                with self.assertRaises(ValueError):pilot.approved_context(proposal,dict(approval,**changes),scheduler,0)
            with self.assertRaises(ValueError):
                pilot.approved_context(proposal,approval,scheduler.replace('node=1','node=1,gres/gpu:h200=1'),0)
        snapshot['jobs'][0]['restart_cnt']=1
        with self.assertRaises(ValueError):pilot.cpu_scheduler_epochs(json.dumps(snapshot),'42')

    def test_single_diagnostic_completion_not_five_row_completion(self):
        result=pilot.campaign_summary([dict(selector='no_search',attempted=True,valid=True,success=0)],[],['no_search'])
        self.assertEqual(result['planned_cells'],1)
        self.assertTrue(result['all_mechanics_valid'])
        self.assertEqual(result['missing_rows'],[])

    def test_pending_approval_cannot_query_scheduler_or_spawn(self):
        with patch.object(pilot.subprocess,'check_output',side_effect=AssertionError('scheduler forbidden')),\
                patch.object(pilot.subprocess,'Popen',side_effect=AssertionError('spawn forbidden')):
            with self.assertRaisesRegex(ValueError,'not approved'):
                pilot.execute(self.proposal,dict(status='pending'),self.root/'unused')
        self.assertFalse((self.root/'campaign').exists())

    def test_resource_request_scope_is_exact(self):
        for key,value in [('resources',dict(pilot.RESOURCES,total_seconds=10801)),
                          ('limits',dict(pilot.LIMITS,api_usd=16)),('rows',['first5']),
                          ('task_ids',[6]),('restore','sif')]:
            with self.subTest(key=key),self.assertRaises(ValueError):pilot.validate(dict(self.proposal,**{key:value}))

    def test_corrected_profile_is_explicit_and_cannot_widen_legacy_approval(self):
        legacy=deepcopy(self.proposal)
        pilot.validate(legacy)
        corrected=self.corrected();pilot.validate(corrected)
        self.assertEqual(pilot.resource_envelope(corrected),(pilot.CORRECTED_RESOURCES,pilot.CORRECTED_LIMITS))
        for changes in (dict(resource_profile='unknown'),dict(resource_profile=None),
                        dict(resources=dict(pilot.RESOURCES)),dict(limits=dict(pilot.LIMITS)),
                        dict(limits=dict(pilot.CORRECTED_LIMITS,total_requests=976))):
            with self.subTest(changes=changes),self.assertRaises(ValueError):
                pilot.validate(dict(corrected,**changes))
        legacy['worker_plan_sha256']=corrected['worker_plan_sha256']
        with self.assertRaisesRegex(ValueError,'Legacy approval'):
            pilot.validate(legacy)
        self.assertEqual(pilot.RESOURCES['total_seconds'],10800)
        self.assertEqual(pilot.LIMITS['local_requests'],320)

    def test_corrected_contract_and_separate_cohort_are_required(self):
        proposal=self.corrected()
        for changes in (dict(turn_limit_protocol=None),dict(previous_cohort={}),
                        dict(native_fuzzy_preflight=None),
                        dict(native_fuzzy_preflight=dict(proposal['native_fuzzy_preflight'],max_calls=2)),
                        dict(piotr_representation=None),
                        dict(previous_cohort=dict(proposal['previous_cohort'],budget_transfer=True)),
                        dict(campaign_root=str(self.root/'reddit-pilot-campaign-v1')),
                        dict(campaign_root=str(self.root/'reddit-pilot-campaign-v1'/'nested'))):
            with self.subTest(changes=changes),self.assertRaises(ValueError):
                pilot.validate(dict(proposal,**changes))
        plan=pilot.read(proposal['worker_plan']);plan['turn_limit_protocol']='strict_actual_answer_v1'
        self.save(Path(proposal['worker_plan']),plan)
        proposal['worker_plan_sha256']=pilot.sha(proposal['worker_plan'])
        with self.assertRaisesRegex(ValueError,'v3 N/A fallback'):
            pilot.validate(proposal)

    def test_corrected_pending_approval_is_zero_scheduler_and_zero_spawn(self):
        proposal=self.corrected()
        with patch.object(pilot.subprocess,'check_output',side_effect=AssertionError('scheduler forbidden')),\
                patch.object(pilot.subprocess,'Popen',side_effect=AssertionError('spawn forbidden')):
            with self.assertRaisesRegex(ValueError,'not approved'):
                pilot.execute(proposal,dict(status='pending'),self.root/'unstarted')
        self.assertFalse((self.root/'campaign').exists())

    def test_corrected_clock_charges_every_prior_second_and_explicit_fallback_approval(self):
        proposal=self.corrected()
        approval=dict(status='approved',proposal_sha256=pilot.digest(proposal),
            resource_profile=pilot.CORRECTED_PROFILE,resources=pilot.CORRECTED_RESOURCES,
            limits=pilot.CORRECTED_LIMITS,job_id='42',terminal_judge_replacement_approved=True,
            terminal_judge_model='gpt-4o-2024-05-13',turn_limit_protocol=pilot.CORRECTED_TURN_LIMIT,
            turn_limit_protocol_approved=True,piotr_representation=pilot.CORRECTED_PIOTR_REPRESENTATION,
            piotr_representation_approved=True)
        end=pilot.datetime.fromtimestamp(time.time()+3600).strftime('%Y-%m-%dT%H:%M:%S')
        scheduler=f'JobId=42 JobState=RUNNING UserId=test({os.getuid()}) NumNodes=1 NumCPUs=8 AllocTRES=cpu=8,mem=120G,gres/gpu:h200=1 TimeLimit=01:00:00 EndTime={end}'
        with patch.dict(os.environ,SLURM_JOB_ID='42'),patch.object(Path,'read_text',return_value='0::/job_42/step_batch'):
            context=pilot.approved_context(proposal,approval,scheduler,1800)
            self.assertEqual(context['prior_consumed_seconds'],1800)
            with self.assertRaisesRegex(ValueError,'shared90min'):
                pilot.approved_context(proposal,approval,scheduler,1801)
            for changes in (dict(resource_profile=None),dict(turn_limit_protocol_approved=False),
                            dict(piotr_representation=None),dict(piotr_representation_approved=False),
                            dict(turn_limit_protocol='strict_actual_answer_v1'),dict(limits=pilot.LIMITS)):
                with self.subTest(changes=changes),self.assertRaises(ValueError):
                    pilot.approved_context(proposal,dict(approval,**changes),scheduler,0)

    def test_corrected_recovery_keeps_unknown_calls_storage_and_primary_claims(self):
        self.corrected()
        self.prior(paid=20,local=350)
        history=pilot.campaign_history(self.proposal,['10'])
        self.assertEqual(history['remaining_paid_requests'],555)
        self.assertEqual(history['remaining_local_requests'],50)
        self.assertEqual(history['prior_retained_scratch_bytes'],1234)
        self.assertNotIn('no_search',history['remaining_rows'])
        self.assertEqual(history['prior_missing_rows'],['no_search'])
        self.assertEqual(pilot.read(self.root/'campaign/shared-budget.json')['calls'][0]['state'],'reserved')
        folder=self.root/'campaign/attempt-10'
        with (folder/'broker.jsonl').open('a') as stream:
            for number in range(51):
                stream.write(json.dumps(dict(event='dispatch_started',request_id='extra'+str(number),local=True))+'\n')
        with self.assertRaisesRegex(ValueError,'caps already exceeded'):
            pilot.campaign_history(self.proposal,['10'])

    def test_corrected_cannot_absorb_previous_profile_attempt_or_budget(self):
        self.prior()
        self.corrected()
        with self.assertRaisesRegex(ValueError,'Campaign identity'):
            pilot.campaign_history(self.proposal,['10'])
        with self.assertRaisesRegex(ValueError,'omits'):
            pilot.campaign_history(self.proposal,[])

    def test_resource_approval_alone_cannot_activate_judge(self):
        approval=dict(status='approved',proposal_sha256=pilot.digest(self.proposal),resources=pilot.RESOURCES,
            limits=pilot.LIMITS,job_id='42',terminal_judge_model='gpt-4o-2024-05-13',
            terminal_judge_replacement_approved=False)
        end=pilot.datetime.fromtimestamp(time.time()+10800).strftime('%Y-%m-%dT%H:%M:%S')
        scheduler=f'JobId=42 JobState=RUNNING UserId=test({os.getuid()}) NumNodes=1 NumCPUs=8 AllocTRES=cpu=8,mem=120G,gres/gpu:h200=1 TimeLimit=03:00:00 EndTime={end}'
        with patch.dict(os.environ,SLURM_JOB_ID='42'),patch.object(Path,'read_text',return_value='0::/job_42/step_batch'):
            with self.assertRaisesRegex(ValueError,'judge'):pilot.approved_context(self.proposal,approval,scheduler,0)
            approval['terminal_judge_replacement_approved']=True
            result=pilot.approved_context(self.proposal,approval,scheduler,0)
            self.assertAlmostEqual(result['deadline'],pilot.datetime.strptime(end,'%Y-%m-%dT%H:%M:%S').timestamp()-180)
            with self.assertRaisesRegex(ValueError,'shared3h'):pilot.approved_context(self.proposal,approval,scheduler,1)
            for original,replacement in [('gres/gpu:h200=1','gres/gpu:h200=2'),('NumCPUs=8','NumCPUs=16'),
                    ('mem=120G','mem=240G'),('NumNodes=1','NumNodes=2'),('JobState=RUNNING','JobState=PENDING'),
                    ('TimeLimit=03:00:00','TimeLimit=03:00:01')]:
                with self.subTest(replacement=replacement),self.assertRaises(ValueError):
                    pilot.approved_context(self.proposal,approval,scheduler.replace(original,replacement),0)

    def test_prior_unknown_calls_states_and_launched_cell_stay_charged(self):
        self.prior()
        result=pilot.campaign_history(self.proposal,['10'])
        self.assertEqual((result['remaining_paid_requests'],result['remaining_local_requests']),(574,318))
        self.assertEqual(result['prior_retained_scratch_bytes'],1234)
        self.assertNotIn('no_search',result['remaining_rows'])
        self.assertEqual(result['prior_missing_rows'],['no_search'])
        ledger=pilot.read(self.root/'campaign/shared-budget.json')
        self.assertEqual(ledger['calls'][0]['state'],'reserved')

    def test_internal_warmups_from_old_attempts_are_conservatively_charged(self):
        folder=self.prior()
        self.save(folder/'piotr_selectionarm-command.json',dict(argv=['python','-m','sglang.launch_server']))
        self.save(folder/'arbiter7b-command.json',dict(argv=['python','-m','sglang.launch_server','--skip-server-warmup']))
        history=pilot.campaign_history(self.proposal,['10'])
        self.assertEqual(history['remaining_local_requests'],317)

    def test_piotr_rope_translation_preserves_every_original_field(self):
        config=dict(text_config=dict(rope_parameters=dict(mrope_interleaved=True,mrope_section=[24,20,20],
            rope_type='default',rope_theta=5000000),hidden_size=2560),vision_config=dict(patch_size=16))
        path=self.root/'config.json';self.save(path,config)
        changed=pilot.piotr_compatible_config(dict(path=str(self.root)))
        text=changed['text_config']
        self.assertEqual(text.pop('rope_scaling'),dict(mrope_interleaved=True,mrope_section=[24,20,20],rope_type='default'))
        self.assertEqual(text.pop('rope_theta'),5000000)
        self.assertEqual(changed,config);self.assertEqual(pilot.read(path),config)
        config['text_config']['rope_parameters']['rope_theta']=10000;self.save(path,config)
        with self.assertRaisesRegex(ValueError,'RoPE'):pilot.piotr_compatible_config(dict(path=str(self.root)))

    def test_piotr_explicit_template_bypasses_checkpoint_path_name_guessing(self):
        model=dict(path='/tmp/checkpoints/piotr',processor_path='/tmp/processor',
            served_model='piotr',max_total_tokens=32768,context_length=32768)
        with patch('scripts.prepare_webprm_piotr_processor.validate_view'),patch.object(pilot,'piotr_compatible_config',return_value={}):
            command=pilot.model_command('piotr_selectionarm',model,12345,compatible_config='/tmp/compat.json')
        self.assertEqual(command[command.index('--chat-template')+1],'/tmp/processor/chat_template.jinja')
        self.assertEqual(command[command.index('--decrypted-config-file')+1],'/tmp/compat.json')
        self.assertIn('--skip-server-warmup',command)

    def test_readiness_keeps_login_cookie_and_rejects_database_failure(self):
        from http.server import BaseHTTPRequestHandler, HTTPServer
        from threading import Thread
        from urllib.error import HTTPError
        seen=[]
        forum_status=[200]
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args):pass
            def do_GET(self):
                seen.append(self.path)
                if self.path=='/login':
                    self.send_response(302)
                    self.send_header('Set-Cookie','probe=ok; Path=/')
                    self.send_header('Location','/login?_cookie_check=1')
                elif self.path.startswith('/login?'):
                    self.send_response(200 if 'probe=ok' in self.headers.get('Cookie','') else 403)
                else:self.send_response(forum_status[0])
                self.end_headers();self.wfile.write(b'ready')
        server=HTTPServer(('127.0.0.1',0),Handler)
        thread=Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            url=f'http://127.0.0.1:{server.server_port}'
            pilot.website_ready(url)
            self.assertEqual(seen,['/login','/login?_cookie_check=1','/f/books'])
            forum_status[0]=503
            with self.assertRaises(HTTPError):pilot.website_ready(url)
        finally:
            server.shutdown();server.server_close();thread.join(2)

    def test_operational_revision_preserves_prior_attempt_accounting(self):
        source=str(pilot.REPO/'scripts/prepare_webprm_piotr_processor.py')
        self.proposal['source_sha256']={source:'old'}
        self.prior()
        original=deepcopy(self.proposal)
        parent=self.root/'original-proposal.json';self.save(parent,original)
        repaired=deepcopy(original)
        repaired['source_sha256'][source]='new'
        repaired['operational_parent']=dict(proposal_path=str(parent),
            proposal_sha256=pilot.digest(original),reason='Cross-node stat portability repair')
        self.assertEqual(pilot.proposal_lineage(repaired),[pilot.digest(repaired),pilot.digest(original)])
        history=pilot.campaign_history(repaired,['10'])
        self.assertEqual(history['remaining_paid_requests'],574)
        self.assertEqual(history['remaining_local_requests'],318)
        self.assertEqual(history['prior_retained_scratch_bytes'],1234)
        self.assertNotIn('no_search',history['remaining_rows'])
        for key,value in [('limits',dict(pilot.LIMITS,api_usd=16)),('task_ids',[6]),
                          ('activation_changes',{})]:
            changed=deepcopy(repaired);changed[key]=value
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'scientific settings'):
                pilot.proposal_lineage(changed)
        changed=deepcopy(repaired)
        changed['source_sha256'][str(pilot.REPO/'openwebrl/webprm_broker.py')]='new'
        with self.assertRaisesRegex(ValueError,'unsupported source'):pilot.proposal_lineage(changed)
        changed=deepcopy(repaired);changed['operational_parent']['proposal_sha256']='wrong'
        with self.assertRaisesRegex(ValueError,'lineage'):pilot.proposal_lineage(changed)

    def test_worker_initialization_repair_preserves_every_scientific_field(self):
        worker=str(pilot.REPO/'openwebrl/webprm_webarena_worker.py')
        original=deepcopy(self.proposal)
        original['source_sha256']={worker:'old'}
        old_plan=dict(max_turns=30,seed=42,code_sha256={worker:'old'})
        self.save(self.root/'plan.json',old_plan)
        original['worker_plan_sha256']=pilot.sha(self.root/'plan.json')
        parent=self.root/'original-proposal.json';self.save(parent,original)
        repaired=deepcopy(original)
        repaired['source_sha256'][worker]='new'
        repaired['worker_plan']=str(self.root/'new-plan.json')
        new_plan=dict(old_plan,code_sha256={worker:'new'})
        self.save(self.root/'new-plan.json',new_plan)
        repaired['worker_plan_sha256']=pilot.sha(self.root/'new-plan.json')
        repaired['operational_parent']=dict(proposal_path=str(parent),
            proposal_sha256=pilot.digest(original),reason='Pre-reset browser initialization repair')
        self.assertEqual(len(pilot.proposal_lineage(repaired)),2)
        self.save(self.root/'new-plan.json',dict(new_plan,max_turns=31))
        with self.assertRaisesRegex(ValueError,'digest'):pilot.proposal_lineage(repaired)
        repaired['worker_plan_sha256']=pilot.sha(self.root/'new-plan.json')
        with self.assertRaisesRegex(ValueError,'scientific'):pilot.proposal_lineage(repaired)
        self.save(self.root/'new-plan.json',new_plan)
        repaired['worker_plan_sha256']=pilot.sha(self.root/'new-plan.json')
        repaired['source_sha256'][worker]='different'
        with self.assertRaisesRegex(ValueError,'worker code'):pilot.proposal_lineage(repaired)

    def test_missing_attempt_chain_or_orphan_fails_closed(self):
        folder=self.prior()
        with self.assertRaisesRegex(ValueError,'omits'):pilot.campaign_history(self.proposal,[])
        self.save(folder/'controller-closure.json',dict(remaining_owned_processes={'123':{}}))
        with self.assertRaisesRegex(ValueError,'cleanup'):pilot.campaign_history(self.proposal,['10'])

    def test_host_cleared_scratch_requires_original_node_and_terminal_job(self):
        storage=dict(scratch_path=str(self.root/'absent-scratch'),apparent_bytes=900)
        context=dict(job_id='10')
        with patch.object(pilot,'SCRATCH',self.root),patch.object(pilot.socket,'gethostname',return_value='g008'):
            for record,charged in [('10|FAILED|g008\n',0),('10|FAILED|g009\n',900),
                                   ('10|RUNNING|g008\n',900),('11|FAILED|g008\n',900)]:
                with self.subTest(record=record),patch.object(pilot.subprocess,'check_output',return_value=record):
                    self.assertEqual(pilot.retained_storage(storage,context)[0],charged)
            Path(storage['scratch_path']).mkdir()
            with patch.object(pilot.subprocess,'check_output',side_effect=AssertionError('existing scratch stays charged')):
                self.assertEqual(pilot.retained_storage(storage,context)[0],900)

    def test_preexecution_rejection_stays_in_attempt_chain_without_consuming_calls(self):
        folder=self.root/'campaign/attempt-10';folder.mkdir(parents=True)
        log=self.root/'failed.log';log.write_text('failure before child startup')
        receipt=dict(job_id='10',proposal_sha256=pilot.digest(self.proposal),no_children_spawned=True,
            retained_scratch_bytes=0,batch_log=str(log),batch_log_sha256=pilot.sha(log))
        self.save(folder/'preexecution-rejection.json',receipt)
        history=pilot.campaign_history(self.proposal,['10'])
        self.assertEqual(history['remaining_paid_requests'],575)
        self.assertEqual(history['remaining_rows'],list(pilot.ROWS))
        self.assertEqual(history['prior_attempt_chain'][0]['job_id'],'10')
        log.write_text('changed')
        with self.assertRaisesRegex(ValueError,'rejection evidence'):pilot.campaign_history(self.proposal,['10'])

    def test_lost_shared_paid_ledger_fails_closed(self):
        self.prior()
        with patch.object(pilot,'read',side_effect=lambda path: (_ for _ in ()).throw(ValueError('lost'))
                if str(path).endswith('shared-budget.json') else json.loads(Path(path).read_text())):
            with self.assertRaisesRegex(ValueError,'lost'):pilot.campaign_history(self.proposal,['10'])

    def test_nonfinite_budget_negative_storage_and_duplicate_dispatch_rejected(self):
        folder=self.prior()
        self.save(folder.parent/'shared-budget.json',dict(cap=15,charged=float('nan')))
        with self.assertRaisesRegex(ValueError,'budget'):pilot.campaign_history(self.proposal,['10'])
        self.save(folder.parent/'shared-budget.json',dict(cap=15,charged=1))
        self.save(folder/'storage-final.json',dict(scratch_path='/tmp/retained',apparent_bytes=-1))
        with self.assertRaisesRegex(ValueError,'storage'):pilot.campaign_history(self.proposal,['10'])

    def test_timeout_result_preserved_and_orphan_result_rejected(self):
        path=self.root/'cell/result.json';watchdog=dict(remaining_owned_processes={},child_exitcode=-15)
        result=dict(arm=pilot.ACTOR+'__no_search',task_id='webarena_lite_005',identity_sha256='x',
                    attempted=True,valid=False,success=0,watchdog=watchdog)
        self.save(path,result);self.save(path.parent/'watchdog.json',watchdog)
        outcome=pilot.cell_outcome(path,'no_search','x')
        self.assertTrue(outcome['attempted']);self.assertFalse(outcome['valid'])
        watchdog['remaining_owned_processes']={'123':{}}
        self.save(path.parent/'watchdog.json',watchdog)
        with self.assertRaisesRegex(RuntimeError,'closure'):pilot.cell_outcome(path,'no_search','x')

    def test_campaign_never_conflates_partial_or_invalid_with_valid_completion(self):
        outcomes=[dict(selector=row,attempted=True,valid=row!='no_search',success=0) for row in pilot.ROWS]
        full=pilot.campaign_summary(outcomes,[])
        self.assertEqual(full['status'],'fixed_attempts_completed');self.assertEqual(full['valid'],4)
        self.assertFalse(full['all_mechanics_valid'])
        partial=pilot.campaign_summary(outcomes[1:],['no_search'])
        self.assertEqual(partial['status'],'partial_preserved_attempts');self.assertFalse(partial['all_mechanics_valid'])
        self.assertFalse(pilot.campaign_summary([],[])['all_attempted'])

    def test_cache_paths_override_inherited_home_aliases(self):
        scratch=self.root/'scratch';scratch.mkdir()
        with patch.object(pilot,'SCRATCH',self.root),patch.dict(os.environ,dict(HF_HUB_CACHE='/gpfs/home/zixianma/cache',
                SGLANG_CACHE_DIR='/gpfs/home/zixianma/cache',OPENAI_API_KEY='secret')):
            env=pilot.environment(scratch)
        for key in ('HF_HUB_CACHE','SGLANG_CACHE_DIR','SGLANG_DG_CACHE_DIR','WEBPRM_BROWSER_TMP_ROOT','TRANSFORMERS_CACHE'):
            self.assertTrue(env[key].startswith(str(scratch)+'/'))
        self.assertNotIn('OPENAI_API_KEY',env);self.assertTrue(Path(env['CUDA_HOME']+'/bin/nvcc').is_file())

    def test_cache_symlink_rejected_before_write(self):
        scratch=self.root/'scratch';scratch.mkdir();(scratch/'tmp').symlink_to('/gpfs/home/zixianma')
        with patch.object(pilot,'SCRATCH',self.root),self.assertRaises(ValueError):pilot.environment(scratch)

    def test_stop_reaps_only_owned_session(self):
        own=subprocess.Popen([sys.executable,'-B','-c','import time; time.sleep(30)'],start_new_session=True)
        other=subprocess.Popen([sys.executable,'-B','-c','import time; time.sleep(30)'],start_new_session=True)
        try:
            pilot._stop(own,seconds=.2)
            self.assertIsNotNone(own.poll());self.assertIsNone(other.poll())
        finally:pilot._stop(other,seconds=.2)

    def test_real_selector_preflight_uses_eleven_local_requests(self):
        from tests.test_webprm_webarena_worker import Scripted
        from openwebrl.webprm_reddit_pilot_preflight import validate_local_critics
        root=Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-webarena-20261011')
        config=pilot.read(root/'pilot-integration-v2/pending-reddit-adapter-plan.json')
        config.update(shepherd_model='shepherd',arbiter_model='arbiter',
            shepherd_endpoint='http://127.0.0.1:1/v1',arbiter_endpoint='http://127.0.0.1:2/v1',
            piotr_endpoint='http://127.0.0.1:3/v1')
        scripted=Scripted()
        with patch('socket.socket.connect',side_effect=AssertionError('network forbidden')):
            proof=validate_local_critics(config,scripted,self.root)
        self.assertEqual(proof['expected_local_dispatches'],11)
        self.assertEqual(len(scripted.requests),11)
        self.assertTrue(all(row['local'] for row in scripted.requests))
        self.assertEqual(proof['screenshot_sha256'],'4bfe4fc5b867b1cbe44af5b63663e23da19b83f464e15bf1315e89e6cc50af7c')


if __name__=='__main__':unittest.main()
