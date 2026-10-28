"""Offline notification tests; no scheduler, queue, browser or allocation calls."""
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch
from scripts import supervise_webprm_table5 as monitor


def current(state='PENDING',job='42',failed=False,done=False,stalled=False):
    return dict(jobs=[dict(job_id=job,state=state,failure_artifact=failed,completion_artifact=done,stalled=stalled)])


def response(stdout='',stderr='',returncode=0):
    return SimpleNamespace(stdout=stdout,stderr=stderr,returncode=returncode)


class SchedulerTests(unittest.TestCase):
    expired='slurm_load_jobs error: Invalid job id specified\n'

    def test_expired_terminal_id_uses_exact_accounting_without_claiming_closure(self):
        run=Mock(side_effect=[response('42|FAILED|47|1:0\n'),response(stderr=self.expired,returncode=1)])
        result=monitor.scheduler(['42'],run=run)
        self.assertEqual(result,{'42':dict(state='FAILED',elapsed_seconds=47,exit_code='1:0')})
        self.assertEqual([call.args[0][0] for call in run.call_args_list],['sacct','squeue'])
        self.assertTrue(monitor.decision({},0,current('FAILED'),1000)[2])

    def test_mixed_active_and_expired_ids_requery_only_unresolved_ids(self):
        run=Mock(side_effect=[response('42|FAILED|47|1:0\n43|RUNNING|120|0:0\n'),
            response('43|PENDING\n',self.expired,1),response('43|RUNNING\n')])
        result=monitor.scheduler(['42','43'],run=run)
        self.assertEqual(result['42']['state'],'FAILED');self.assertEqual(result['43']['state'],'RUNNING')
        self.assertEqual(run.call_args_list[-1].args[0],['squeue','-h','-j','43','-o','%i|%T'])
        self.assertEqual(len(run.call_args_list),3)

    def test_queue_state_overrides_stale_terminal_accounting_when_available(self):
        run=Mock(side_effect=[response('42|COMPLETED|47|0:0\n'),response('42|RUNNING\n')])
        self.assertEqual(monitor.scheduler(['42'],run=run)['42']['state'],'RUNNING')

    def test_missing_or_nonterminal_accounting_does_not_explain_invalid_queue_id(self):
        for accounting in ['', '42.batch|FAILED|47|1:0\n', '42|RUNNING|47|0:0\n']:
            with self.subTest(accounting=accounting):
                run=Mock(side_effect=[response(accounting),response(stderr=self.expired,returncode=1)])
                with self.assertRaisesRegex(RuntimeError,'lacks exact terminal accounting'):monitor.scheduler(['42'],run=run)

    def test_real_queue_error_is_not_hidden_by_terminal_accounting(self):
        for error,code in [('slurm_load_jobs error: Socket timed out',1),(self.expired,2),
                (self.expired+'additional unexpected failure',1)]:
            with self.subTest(error=error,code=code):
                run=Mock(side_effect=[response('42|FAILED|47|1:0\n'),response(stderr=error,returncode=code)])
                with self.assertRaisesRegex(RuntimeError,'squeue failed'):monitor.scheduler(['42'],run=run)

    def test_unresolved_mixed_id_must_be_present_in_successful_requery(self):
        run=Mock(side_effect=[response('42|FAILED|47|1:0\n'),response(stderr=self.expired,returncode=1),response()])
        with self.assertRaisesRegex(RuntimeError,'Missing queue ID'):monitor.scheduler(['42','43'],run=run)

    def test_failed_requery_alerts_without_recursive_retries(self):
        run=Mock(side_effect=[response('42|FAILED|47|1:0\n'),response(stderr=self.expired,returncode=1),
            response(stderr=self.expired,returncode=1)])
        with self.assertRaisesRegex(RuntimeError,'squeue unresolved IDs failed'):monitor.scheduler(['42','43'],run=run)
        self.assertEqual(run.call_count,3)

    def test_malformed_or_duplicate_accounting_cannot_certify_terminal_state(self):
        for accounting in ['42|FAILED|unknown|1:0\n','42|FAILED|47|unknown\n',
                '42|FAILED|47|1:0\n42|COMPLETED|47|0:0\n']:
            with self.subTest(accounting=accounting):
                run=Mock(return_value=response(accounting))
                with self.assertRaisesRegex(ValueError,'Exact unique Slurm accounting'):monitor.scheduler(['42'],run=run)

    def test_accounting_failure_and_queue_timeout_remain_errors(self):
        run=Mock(return_value=response(stderr='Accounting storage unavailable',returncode=1))
        with self.assertRaisesRegex(RuntimeError,'sacct failed'):monitor.scheduler(['42'],run=run)
        run=Mock(side_effect=[response('42|FAILED|47|1:0\n'),monitor.subprocess.TimeoutExpired('squeue',20)])
        with self.assertRaises(monitor.subprocess.TimeoutExpired):monitor.scheduler(['42'],run=run)

    def test_successful_absent_queries_report_unknown_instead_of_terminal(self):
        run=Mock(side_effect=[response(),response()])
        self.assertEqual(monitor.scheduler(['42'],run=run)['42']['state'],'UNKNOWN')
        self.assertTrue(monitor.decision({},0,current('UNKNOWN'),1000)[2])


class SupervisorTests(unittest.TestCase):
    def setUp(self):
        self.tmp=TemporaryDirectory(dir='/tmp',prefix='webprm-supervisor-');self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.pointer=self.root/'pointer.json'

    def test_pre_actor_queue_failure_alerts_only_for_its_own_attempt(self):
        import hashlib
        campaign=self.root/'campaign';cell_id='fixed_cell';key=hashlib.sha256(cell_id.encode()).hexdigest()
        cell=campaign/'attempt-42/cells'/key;cell.mkdir(parents=True)
        (cell/'state-claim.json').write_text('{}')
        outcomes=campaign/'queue/outcomes';outcomes.mkdir(parents=True)
        (outcomes/'one.json').write_text(json.dumps(dict(cell_id=cell_id,
            result=dict(reason='worker_exception',error_type='TimeoutError',status='invalid'))))
        row=dict(job_id='42',package=str(self.root),campaign_root=str(campaign))
        result=monitor.artifacts(row)
        self.assertTrue(result['failure_artifact']);self.assertEqual(result['pre_actor_failures'],1)
        replacement=monitor.artifacts(dict(row,job_id='43'))
        self.assertFalse(replacement['failure_artifact']);self.assertEqual(replacement['pre_actor_failures'],0)
        (cell/'worker-launch.json').write_text('{}')
        self.assertEqual(monitor.artifacts(row)['pre_actor_failures'],0)

    def test_first_native_login_failure_alerts_before_the_campaign_exits(self):
        campaign=self.root/'campaign';cell=campaign/'attempt-42/cells/one';episode=cell/'episodes/task/actor'
        episode.mkdir(parents=True);result=episode/'result.json'
        result.write_text(json.dumps(dict(valid=False,error_type='EvaluationNotReady',
                                         error='Native authentication failed: navigation timeout')))
        (cell/'cell-outcome.json').write_text(json.dumps(dict(valid=False,result_path=str(result))))
        files=monitor.artifacts(dict(job_id='42',package=str(self.root),campaign_root=str(campaign)))
        self.assertTrue(files['failure_artifact']);self.assertEqual(files['native_authentication_failures'],1)
        old=dict(queued_epoch=1000,signature=monitor.decision({},0,current('RUNNING'),1000)[1])
        self.assertTrue(monitor.decision(old,0,current('RUNNING',failed=files['failure_artifact']),1010)[2])
        result.write_text(json.dumps(dict(valid=False,error_type='ValueError',error='Malformed actor output')))
        files=monitor.artifacts(dict(job_id='42',package=str(self.root),campaign_root=str(campaign)))
        self.assertFalse(files['failure_artifact'])

    def test_initial_pending_queue_is_deduplicated_until_ack_and_routine_cadence(self):
        send,signature,urgent,pending=monitor.decision({},0,current(),1000)
        self.assertTrue(send);self.assertFalse(urgent)
        old=dict(signature=signature,queued_epoch=1000)
        self.assertFalse(monitor.decision(old,0,current(),4599)[0])
        self.assertTrue(monitor.decision(old,0,current(),4600)[0])
        self.assertFalse(monitor.decision(old,1100,current(),4699)[0])
        self.assertTrue(monitor.decision(old,1100,current(),4700)[0])

    def test_hourly_routine_cadence_keeps_urgent_recovery_immediate(self):
        signature=monitor.decision({},0,current('RUNNING'),1000)[1]
        old=dict(signature=signature,queued_epoch=1000)
        self.assertFalse(monitor.decision(old,1100,current('RUNNING'),2000)[0])
        self.assertFalse(monitor.decision(old,1100,current('RUNNING'),4699)[0])
        self.assertTrue(monitor.decision(old,1100,current('RUNNING'),4700)[0])
        for value in [current('FAILED'),current('COMPLETED',done=True),current('RUNNING',stalled=True)]:
            self.assertTrue(monitor.decision(old,1100,value,1101)[0])

    def test_each_additional_startup_failure_bypasses_pending_callback_once(self):
        one=current('RUNNING',failed=True);one['jobs'][0]['native_authentication_failures']=1
        signature=monitor.decision({},0,one,1000)[1]
        old=dict(queued_epoch=1000,signature=signature)
        two=current('RUNNING',failed=True);two['jobs'][0]['native_authentication_failures']=2
        send,signature,urgent,pending=monitor.decision(old,0,two,1010)
        self.assertTrue(send and urgent and pending)
        self.assertFalse(monitor.decision(dict(queued_epoch=1010,signature=signature),0,two,1020)[0])

    def test_new_start_failure_completion_and_stall_bypass_pending_only_once(self):
        signature=monitor.decision({},0,current(),1000)[1];old=dict(signature=signature,queued_epoch=1000)
        for next_value in [current('RUNNING'),current('FAILED'),current('COMPLETED',done=True),current('RUNNING',stalled=True)]:
            send,signature,urgent,pending=monitor.decision(old,0,next_value,1010)
            self.assertTrue(send and urgent and pending)
            advanced=dict(queued_epoch=1010,signature=signature)
            self.assertFalse(monitor.decision(advanced,0,next_value,2000)[0])

    def test_idle_keep_alive_reloads_registration_then_exits_at_verified_completion(self):
        self.pointer.write_text(json.dumps(dict(verified_complete=False)))
        def complete(_):
            self.pointer.write_text(json.dumps(dict(verified_complete=True)))
        with patch.object(monitor,'poll',side_effect=[False,False]) as poll,\
                patch.object(monitor.time,'sleep',side_effect=complete) as sleep:
            monitor.loop(self.root,self.pointer,False,keep_alive=True)
        self.assertEqual(poll.call_count,2);sleep.assert_called_once_with(60)

    def test_verified_completion_suppresses_scheduler_queue_and_duplicate_reports(self):
        self.pointer.write_text(json.dumps(dict(verified_complete=True,next_stage_jobs=[{'job_id':'42'}])))
        with patch.object(monitor,'scheduler',side_effect=AssertionError('No scheduler query')),\
                patch.object(monitor,'queue_review',side_effect=AssertionError('No callback')):
            self.assertFalse(monitor.poll(self.root,self.pointer,1000))
        self.assertTrue(monitor.read(self.root/'finished.json')['verified_complete'])

    def test_replacement_pointer_and_stalled_progress_follow_current_job_only(self):
        self.pointer.write_text(json.dumps(dict(next_stage_jobs=[dict(job_id='41',kind='reddit',verified_complete=True),
            dict(job_id='42',kind='map',package=str(self.root),supervision_required=True)])))
        with patch.object(monitor,'scheduler',return_value={'42':dict(state='RUNNING',elapsed_seconds=1300,exit_code='')}) as scheduler,\
                patch.object(monitor,'artifacts',return_value=dict(fingerprint='a',files=1,latest_mtime=700,
                    failure_artifact=False,completion_artifact=False)):
            value=monitor.snapshot(self.pointer,{},2000)
        scheduler.assert_called_once_with(['42']);self.assertTrue(value['jobs'][0]['stalled'])
        self.assertEqual(set(value['progress']),{'42'})

    def test_queue_targets_exact_thread_with_preserved_codex_home_and_safe_paths(self):
        inherited=os.environ.get('CODEX_HOME');self.assertEqual(inherited,monitor.CODEX_HOME)
        run=Mock(return_value=SimpleNamespace(returncode=0,stdout='queued message fixture',stderr=''))
        state=monitor.queue_review(self.root,self.pointer,current(),{},0,1000,run=run)
        argv=run.call_args.args[0];env=run.call_args.kwargs['env']
        self.assertEqual(argv[:4],[monitor.CODEX,'queue','--thread',monitor.THREAD])
        self.assertEqual(env['CODEX_HOME'],inherited);self.assertEqual(os.environ['CODEX_HOME'],inherited)
        self.assertEqual(env['TMPDIR'],'/tmp')
        for key in ('XDG_CACHE_HOME','XDG_CONFIG_HOME','XDG_DATA_HOME','XDG_STATE_HOME','XDG_RUNTIME_DIR'):
            self.assertIn(self.root,Path(env[key]).parents)
        self.assertLess(len(monitor.prompt(monitor.ROOT,monitor.POINTER).encode()),1000)
        self.assertEqual(state['queued_epoch'],1000);self.assertTrue((self.root/'notifications.jsonl').is_file())

    def test_snapshot_stats_live_partial_file_and_filters_other_job_completion(self):
        attempt=self.root/'attempt-42';attempt.mkdir();partial=attempt/'archive.partial';partial.write_bytes(b'abc')
        (attempt/'failure.json').write_text('{}');(self.root/'completed.json').write_text(json.dumps({'job_id':'41'}))
        result=monitor.artifacts(dict(job_id='42',package=str(self.root)))
        self.assertEqual(result['files'],2);self.assertTrue(result['failure_artifact']);self.assertFalse(result['completion_artifact'])

    def test_static_benchmark_progress_and_failed_final_alert_before_scheduler_exit(self):
        attempt=self.root/'attempt-42';transport=attempt/'transport';transport.mkdir(parents=True)
        response=transport/'batch-0001.json';response.write_text('{"responses":[]}')
        row=dict(job_id='42',kind='webprmbench',package=str(self.root))
        initial=monitor.artifacts(row)
        self.assertFalse(initial['completion_artifact'])
        response.write_text('{"responses":["A"]}')
        self.assertNotEqual(initial['fingerprint'],monitor.artifacts(row)['fingerprint'])
        final=attempt/'final.json';final.write_text(json.dumps(dict(verified_complete=False,error='server exited')))
        result=monitor.artifacts(row)
        self.assertTrue(result['completion_artifact']);self.assertTrue(result['failure_artifact'])
        final.write_text(json.dumps(dict(verified_complete=True)))
        result=monitor.artifacts(row)
        self.assertTrue(result['completion_artifact']);self.assertFalse(result['failure_artifact'])
        final.write_text('{}')
        with self.assertRaisesRegex(ValueError,'explicit completion status'):monitor.artifacts(row)

    def test_static_benchmark_registration_keeps_completion_independent(self):
        self.pointer.write_text(json.dumps(dict(next_stage_jobs=[dict(job_id='42',kind='webprmbench',
            package=str(self.root),supervision_required=True)])))
        with patch.object(monitor,'scheduler',return_value={'42':dict(state='RUNNING',elapsed_seconds=2,exit_code='0:0')}):
            value=monitor.snapshot(self.pointer,{},1000)
        self.assertEqual(value['jobs'][0]['kind'],'webprmbench')
        self.assertFalse(value['verified_complete'])

    def test_om2w_registry_is_supervised_without_allowing_other_runtime_roots(self):
        self.assertEqual(monitor.checked(monitor.OM2W/'pilot-v1'),monitor.OM2W/'pilot-v1')
        with self.assertRaises(ValueError):monitor.checked(monitor.OM2W.parent/'unrelated'/'pilot')
        self.pointer.write_text(json.dumps(dict(next_stage_jobs=[dict(job_id='43',kind='om2w',
            package=str(self.root),supervision_required=True)])))
        with patch.object(monitor,'scheduler',return_value={'43':dict(state='RUNNING',elapsed_seconds=2,exit_code='0:0')}),\
                patch.object(monitor,'artifacts',return_value=dict(fingerprint='a',files=1,latest_mtime=1000,
                    failure_artifact=False,completion_artifact=False)):
            self.assertEqual(monitor.snapshot(self.pointer,{},1000)['jobs'][0]['kind'],'om2w')


if __name__=='__main__':unittest.main()
