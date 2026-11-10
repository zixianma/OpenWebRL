import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import supervise_uniform8_rl_control as supervisor


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))


class UniformEightSupervisorTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.base=Path(self.temp.name)
        self.root=self.base/'control'
        self.root.mkdir()
        self.pointer=self.base/'pointer.json'
        self.runroot=self.base/'run'
        self.evalroot=self.base/'evaluation'
        self.approval=dict(approved=True,resources=supervisor.RESOURCES,**supervisor.CAPS,
            target_iteration=10,full300_evaluations=[10],
            includes_all_startup_retries_and_evaluations=True,unused_prior_budgets_added=False)
        self.attempts=dict(approved_total_seconds=57600,
            attempts=[dict(job_id='101',maximum_seconds=57600,charged_seconds=0)])
        write(self.root/'approval-request.json',self.approval)
        write(self.root/'attempts.json',self.attempts)
        write(self.root/'launcher-preparation.json',dict(task_count=2102,source=str(self.base/'source'),
            source_compilation_passed=True,native_launcher_dry_runs_passed=True))
        write(self.pointer,dict(root=str(self.root),job_id='101',owning_thread_id='root-thread',
            run_directory=str(self.runroot),evaluation_root=str(self.evalroot),verified_complete=False))
        write(self.root/'status.json',dict(job_id='101',stage='collection',complete=False))
        os.utime(self.root/'status.json',(1000,1000))
        self.calls=[]

    def tearDown(self):
        self.temp.cleanup()

    def runner(self,state='RUNNING',elapsed=100,omit=(),badqueue=False,rows=None):
        def run(command,**kwargs):
            self.calls.append(command)
            if command[0]=='/usr/bin/sacct':
                identities=command[command.index('-j')+1].split(',')
                result=[]
                for job in identities:
                    if job in omit:
                        continue
                    current=(rows or {}).get(job,(state,elapsed))
                    result.append(f'{job}|{current[0]}|{current[1]}|0:0|node|cpu=64,mem=960G,gres/gpu=8,gres/gpu:h200=8|\n')
                return SimpleNamespace(returncode=0,stderr='',stdout=''.join(result))
            self.assertEqual(command[1:4],['queue','--thread','root-thread'])
            return SimpleNamespace(returncode=0,stderr='',
                stdout='ok' if badqueue else 'Queued message fixture-123 for thread root-thread.')
        return run

    def poll(self,now=1000,**kwargs):
        return supervisor.poll(self.root,self.pointer,'root-thread',now=now,**kwargs)

    def snapshot(self):
        return supervisor.read(self.root/'supervisor-latest.json')

    def queues(self):
        return [c for c in self.calls if c[0]!='/usr/bin/sacct']

    def test_first_poll_proves_queue_receipt_and_exact_remaining_budget(self):
        self.assertTrue(self.poll(run=self.runner()))
        self.assertEqual(len(self.queues()),1)
        self.assertEqual(self.snapshot()['remaining_scheduler_seconds'],57500)
        self.assertFalse(self.snapshot()['errors'])
        self.assertTrue(supervisor.read(self.root/'active-continuation.json')['active_agent_queue_verified'])

    def test_pending_routine_does_not_repeat(self):
        self.poll(run=self.runner())
        self.poll(now=2000,run=self.runner())
        self.assertEqual(len(self.queues()),1)

    def test_live_evaluation_log_prevents_false_stale_wrapper_alert(self):
        write(self.root/'status.json',dict(job_id='101',stage='evaluation',complete=False))
        os.utime(self.root/'status.json',(1000,1000))
        self.evalroot.mkdir()
        log=self.evalroot/'evaluation.log'
        log.write_text('worker is collecting trajectories')
        os.utime(log,(1950,1950))
        self.poll(now=2000,run=self.runner(elapsed=2000))
        self.assertFalse(self.snapshot()['stale_heartbeat'])

    def test_fresh_evaluation_record_counts_when_worker_log_is_buffered(self):
        write(self.root/'status.json',dict(job_id='101',stage='evaluation',complete=False))
        os.utime(self.root/'status.json',(1000,1000))
        record=self.evalroot/'rollouts/task.json'
        write(record,dict(task_id='task'))
        os.utime(record,(1950,1950))
        self.poll(now=2000,run=self.runner(elapsed=2000))
        self.assertFalse(self.snapshot()['stale_heartbeat'])

    def test_evaluation_without_recent_activity_still_alerts(self):
        write(self.root/'status.json',dict(job_id='101',stage='evaluation',complete=False))
        os.utime(self.root/'status.json',(1000,1000))
        self.evalroot.mkdir()
        log=self.evalroot/'evaluation.log'
        log.write_text('old activity')
        os.utime(log,(1000,1000))
        self.poll(now=2000,run=self.runner(elapsed=2000))
        self.assertTrue(self.snapshot()['stale_heartbeat'])

    def test_unrelated_evaluation_activity_does_not_hide_training_stall(self):
        self.evalroot.mkdir()
        log=self.evalroot/'evaluation.log'
        log.write_text('old evaluation continues elsewhere')
        os.utime(log,(1950,1950))
        self.poll(now=2000,run=self.runner(elapsed=2000))
        self.assertTrue(self.snapshot()['stale_heartbeat'])

    def test_failure_bypasses_pending_exactly_once(self):
        self.poll(run=self.runner())
        self.poll(now=1060,run=self.runner(state='FAILED'))
        self.poll(now=1120,run=self.runner(state='FAILED'))
        self.assertEqual(len(self.queues()),2)
        self.assertTrue(supervisor.read(self.root/'supervisor-state.json')['pending_callback_bypassed'])

    def test_controller_failure_before_scheduler_is_urgent(self):
        self.poll(run=self.runner())
        write(self.root/'status.json',dict(job_id='101',failed=True,stage='stopped'))
        self.poll(now=1060,run=self.runner())
        self.assertEqual(len(self.queues()),2)

    def test_running_ack_allows_fifteen_minute_callback(self):
        self.poll(run=self.runner())
        write(self.root/'supervisor-ack.json',dict(epoch=1020))
        self.poll(now=1060,run=self.runner())
        self.poll(now=1919,run=self.runner())
        self.assertEqual(len(self.queues()),1)
        self.poll(now=1920,run=self.runner())
        self.assertEqual(len(self.queues()),2)

    def test_unchanged_scheduler_queue_only_calls_agent_hourly(self):
        self.poll(run=self.runner(state='PENDING'))
        write(self.root/'supervisor-ack.json',dict(epoch=1020))
        for now in (1060,1920,2820,3720,4619):
            self.poll(now=now,run=self.runner(state='PENDING'))
        self.assertEqual(len(self.queues()),1)
        self.poll(now=4620,run=self.runner(state='PENDING'))
        self.assertEqual(len(self.queues()),2)

    def test_allocation_start_bypasses_pending_queue_once(self):
        self.poll(run=self.runner(state='PENDING'))
        self.poll(now=1060,run=self.runner())
        self.poll(now=1120,run=self.runner())
        self.assertEqual(len(self.queues()),2)
        self.assertTrue(supervisor.read(self.root/'supervisor-state.json')['pending_callback_bypassed'])

    def test_allocation_start_is_prompt_after_queue_ack(self):
        self.poll(run=self.runner(state='PENDING'))
        write(self.root/'supervisor-ack.json',dict(epoch=1020))
        self.poll(now=1060,run=self.runner())
        self.assertEqual(len(self.queues()),2)
        self.assertEqual(supervisor.read(self.root/'supervisor-state.json')['notification_kind'],'urgent')

    def test_verified_complete_stale_callback_does_not_query_or_queue(self):
        pointer=supervisor.read(self.pointer)
        pointer['verified_complete']=True
        write(self.pointer,pointer)
        self.assertFalse(self.poll(run=lambda *a,**k:self.fail('Unexpected external action')))
        self.assertTrue((self.root/'supervisor-finished.json').exists())
        self.assertTrue((self.root/'supervisor-ack.json').exists())

    def test_exact_approval_is_checked_before_external_calls(self):
        for change in [dict(approved=False),dict(judge_cap_usd=50),dict(judge_cap_usd=101),
                       dict(judge_call_cap=6000),dict(judge_call_cap=8001),dict(target_iteration=20),
                       dict(resources=dict(supervisor.RESOURCES,gpus=4)),dict(unused_prior_budgets_added=True)]:
            write(self.root/'approval-request.json',dict(self.approval,**change))
            with self.assertRaisesRegex(ValueError,'approval'):
                self.poll(run=lambda *a,**k:self.fail('Unexpected external action'))

    def test_receipt_is_required_even_if_codex_exits_zero(self):
        with self.assertRaisesRegex(RuntimeError,'verified receipt'):
            self.poll(run=self.runner(badqueue=True))
        self.assertFalse((self.root/'active-continuation.json').exists())

    def test_all_attempts_charged_latest_replacement_followed(self):
        self.attempts['attempts'].append(dict(job_id='201',maximum_seconds=55600,charged_seconds=0))
        self.attempts['attempts'][0]['charged_seconds']=2000
        write(self.root/'attempts.json',self.attempts)
        self.poll(run=self.runner(rows={'101':('FAILED',1900),'201':('RUNNING',100)}))
        snap=self.snapshot()
        self.assertEqual(snap['job_id'],'201')
        self.assertEqual(snap['all_attempt_scheduler_seconds'],2100)
        self.assertEqual(snap['remaining_scheduler_seconds'],55500)
        self.assertFalse(snap['errors'])

    def test_missing_prior_accounting_never_invents_remaining_budget(self):
        self.attempts['attempts'].insert(0,dict(job_id='100',maximum_seconds=100,charged_seconds=80))
        self.attempts['attempts'][1]['maximum_seconds']=57520
        write(self.root/'attempts.json',self.attempts)
        self.poll(run=self.runner(omit=('100',)))
        self.assertIsNone(self.snapshot()['remaining_scheduler_seconds'])
        self.assertIn('scheduler_accounting_missing:100',self.snapshot()['errors'])

    def test_active_predecessor_and_oversized_retry_are_alerted(self):
        self.attempts['attempts'].append(dict(job_id='201',maximum_seconds=57600))
        write(self.root/'attempts.json',self.attempts)
        self.poll(run=self.runner())
        errors=self.snapshot()['errors']
        self.assertIn('prior_scheduler_attempt_still_active_or_queued',errors)
        self.assertIn('current_allocation_exceeds_remaining_original_budget',errors)

    def test_resource_mismatch_is_detected(self):
        base=self.runner()
        def changed(command,**kwargs):
            value=base(command,**kwargs)
            if command[0]=='/usr/bin/sacct':
                value.stdout=value.stdout.replace('mem=960G','mem=1200G')
            return value
        self.poll(run=changed)
        self.assertIn('unexpected_or_unverified_allocation:101',self.snapshot()['errors'])

    def test_replacement_during_poll_suppresses_old_failure(self):
        base=self.runner(state='FAILED')
        def replace(command,**kwargs):
            result=base(command,**kwargs)
            self.attempts['attempts'].append(dict(job_id='201',maximum_seconds=57000))
            write(self.root/'attempts.json',self.attempts)
            return result
        self.poll(run=replace)
        self.assertFalse(self.queues())
        self.assertFalse((self.root/'supervisor-latest.json').exists())

    def test_budget_halt_prompts_urgent_review(self):
        self.poll(run=self.runner())
        write(self.root/'judge-budget/halt.json',dict(reason='cap'))
        write(self.root/'judge-budget/ledger.json',dict(schema_version=1,caps=supervisor.CAPS,
            calls=8001,browser_dispatches=12001,charged_or_reserved_usd=100.1))
        self.poll(now=1060,run=self.runner())
        snap=self.snapshot()
        self.assertTrue(snap['halt_sha256'])
        self.assertEqual(len(snap['errors']),3)
        self.assertEqual(len(self.queues()),2)

    def test_extended_lifetime_judge_caps_allow_existing_usage_and_exact_ceilings(self):
        for calls, dollars in ((6001,50.1),(8000,100)):
            with self.subTest(calls=calls,dollars=dollars):
                write(self.root/'judge-budget/ledger.json',dict(schema_version=1,caps=supervisor.CAPS,
                    calls=calls,browser_dispatches=12000,charged_or_reserved_usd=dollars))
                self.poll(run=self.runner())
                self.assertFalse(self.snapshot()['errors'])

    def test_stale_judge_ledger_caps_are_rejected(self):
        write(self.root/'judge-budget/ledger.json',dict(schema_version=1,
            caps=dict(supervisor.CAPS,judge_cap_usd=50,judge_call_cap=6000),
            calls=6000,browser_dispatches=12000,charged_or_reserved_usd=50))
        self.poll(run=self.runner())
        self.assertIn('judge_browser_ledger_protocol_or_caps_changed',self.snapshot()['errors'])

    def test_existing_receipts_without_ledger_are_alerted(self):
        write(self.root/'judge-budget/browser-000001.json',dict(dispatch_id=1))
        self.poll(run=self.runner())
        self.assertIn('missing_budget_ledger_with_existing_physical_receipts',self.snapshot()['errors'])

    def test_progress_stall_and_log_stall_are_distinct(self):
        self.poll(run=self.runner())
        self.poll(now=3800,run=self.runner(elapsed=3000))
        snap=self.snapshot()
        self.assertTrue(snap['stale_heartbeat'])
        self.assertTrue(snap['stalled_progress'])

    def test_eval_stage_long_status_age_uses_fresh_worker_log(self):
        write(self.root/'status.json',dict(job_id='101',stage='evaluation'))
        os.utime(self.root/'status.json',(1000,1000))
        log=self.root/'evaluation-101-iter10.log'
        log.write_text('useful progress')
        os.utime(log,(3700,3700))
        self.poll(now=3800,run=self.runner(elapsed=3000))
        self.assertFalse(self.snapshot()['stale_heartbeat'])

    def test_checkpoint_and_eval_evidence_never_auto_verify_completion(self):
        self.runroot.mkdir()
        (self.runroot/'latest_checkpointed_iteration.txt').write_text('9')
        (self.runroot/'iter_0000009').mkdir()
        for index in range(300):
            write(self.evalroot/f'rollouts/{index}.json',dict(task_id=str(index),judge_model='gpt-4.1',
                judge_prompt_variant='action_history',metrics=dict(successes=1,valid_trajectories=1),
                rollout_file=f'{index}.pt'))
        write(self.evalroot/'status.json',dict(complete=True,returncode=0))
        write(self.root/'status.json',dict(job_id='101',stage='awaiting_agent_artifact_review',complete=True))
        self.poll(run=self.runner(state='COMPLETED'))
        snap=self.snapshot()
        self.assertEqual(snap['artifacts']['durable_iteration'],10)
        self.assertEqual(snap['artifacts']['evaluation']['canonical_records'],300)
        self.assertFalse(snap['errors'])
        self.assertFalse(supervisor.read(self.pointer)['verified_complete'])

    def test_completion_without_artifacts_alerts(self):
        write(self.root/'status.json',dict(job_id='101',complete=True))
        self.poll(run=self.runner(state='COMPLETED'))
        self.assertIn('controller_completion_before_checkpoint10_and_full300',self.snapshot()['errors'])

    def test_prompt_cap_and_only_observation_plus_authorized_queue(self):
        root=supervisor.RUNTIME/'training-controls/uniform8-original2102-20261006'
        prompt=supervisor.continuation_prompt(root,supervisor.DEFAULT_POINTER)
        self.assertLessEqual(len(prompt.encode()),1000)
        self.assertIn('Adaptive run stays paused',prompt)
        self.assertIn('No added budget',prompt)
        self.assertIn('8000 judge calls/$100 lifetime',prompt)
        self.assertIn('8H200/64CPU/960GiB/57600s TOTAL,12000 browsers/64concurrent',prompt)
        self.poll(run=self.runner())
        self.assertEqual({c[0] for c in self.calls},{'/usr/bin/sacct','codex'})


if __name__=='__main__':
    unittest.main()
