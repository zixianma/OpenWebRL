import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import supervise_sft_pass8_extension as supervisor


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class SupervisorTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.root = self.base/'campaign'
        self.root.mkdir()
        self.pointer = self.base/'pointer.json'
        self.source = self.base/'source'
        write(self.source/'reference_manifest.json', {'frozen': True})
        self.plan = dict(root=str(self.root), source=str(self.source),
            source_manifest_sha256=hashlib.sha256((self.source/'reference_manifest.json').read_bytes()).hexdigest(),
            historical_task_order=[f'task{i}' for i in range(2000)],
            extension_task_order=[f'task{i}' for i in range(707)],
            shards=[[f'task{i}' for i in range(j,707,4)] for j in range(4)], new_primary_episodes=2121,
            resources=supervisor.RESOURCES, limits=supervisor.LIMITS)
        self.approval = dict(approved=True, plan_sha256=supervisor.digest(self.plan),
            resources=supervisor.RESOURCES, limits=supervisor.LIMITS,
            attempts=[dict(job_id=str(101+i), shard=i, time_limit_seconds=36000) for i in range(4)])
        write(self.root/'plan.json', self.plan)
        write(self.root/'approval.json', self.approval)
        write(self.pointer, dict(root=str(self.root), plan_sha256=supervisor.digest(self.plan),
            owning_thread_id='root-thread', verified_complete=False,
            shards=[dict(shard=i, root=str(self.root/f'shard-{i}'), job_id=str(101+i)) for i in range(4)]))
        for i in range(4):
            path = self.root/f'shard-{i}'/'status.json'
            write(path, dict(job_id=str(101+i), stage='running', epoch=1000))
            os.utime(path, (1000, 1000))
        self.calls = []

    def tearDown(self):
        self.tmp.cleanup()

    def runner(self, state='RUNNING', elapsed=60, queue_code=0, queue_output=None, omit=()):
        def run(command, **kwargs):
            self.calls.append(command)
            if command[0] == '/usr/bin/sacct':
                jobs = command[command.index('-j')+1].split(',')
                return SimpleNamespace(returncode=0, stderr='', stdout=''.join(
                    f'{job}|{state}|{elapsed}|0:0|node|cpu=8,mem=120G,gres/gpu=1,gres/gpu:h200=1|\n'
                    for job in jobs if job not in omit))
            self.assertEqual(command[1:4], ['queue', '--thread', 'root-thread'])
            return SimpleNamespace(returncode=queue_code,
                stdout=queue_output if queue_output is not None else 'Queued message message-123 for thread root-thread.',
                stderr='unavailable' if queue_code else '')
        return run

    def poll(self, now=1000, **kwargs):
        return supervisor.poll(self.root, self.pointer, 'root-thread', now=now, **kwargs)

    def queue_calls(self):
        return [c for c in self.calls if c[0] != '/usr/bin/sacct']

    def test_first_poll_preserves_four_shards_and_proves_successful_queue(self):
        self.assertTrue(self.poll(run=self.runner()))
        self.assertEqual(len(self.queue_calls()), 1)
        snapshot = supervisor.read(self.root/'supervisor-latest.json')
        self.assertEqual(len(snapshot['shards']), 4)
        self.assertEqual([s['results']['expected_primary_records'] for s in snapshot['shards']], [531,531,531,528])
        self.assertEqual(snapshot['shards'][0]['remaining_scheduler_seconds'], 35940)
        receipt = supervisor.read(self.root/'active-continuation.json')
        self.assertTrue(receipt['active_agent_queue_verified'])
        self.assertEqual(receipt['owning_thread_id'], 'root-thread')

    def test_pending_routine_callback_does_not_repeat(self):
        self.poll(run=self.runner())
        self.poll(now=2000, run=self.runner())
        self.assertEqual(len(self.queue_calls()), 1)

    def test_failure_bypasses_pending_routine_exactly_once(self):
        self.poll(run=self.runner())
        self.poll(now=1060, run=self.runner(state='FAILED'))
        self.poll(now=1120, run=self.runner(state='FAILED'))
        self.assertEqual(len(self.queue_calls()), 2)
        state = supervisor.read(self.root/'supervisor-state.json')
        self.assertTrue(state['pending_callback_bypassed'])
        self.assertEqual(state['notification_kind'], 'urgent')

    def test_controller_failure_is_urgent_before_scheduler_catches_up(self):
        self.poll(run=self.runner())
        write(self.root/'shard-0/status.json', dict(job_id='101', stage='failed', epoch=1060))
        self.poll(now=1060, run=self.runner())
        self.assertEqual(len(self.queue_calls()), 2)
        self.assertTrue(supervisor.read(self.root/'supervisor-state.json')['pending_callback_bypassed'])

    def test_acknowledged_review_repeats_at_fifteen_minutes(self):
        self.poll(run=self.runner())
        write(self.root/'supervisor-ack.json', dict(epoch=1020))
        self.poll(now=1919, run=self.runner())
        self.assertEqual(len(self.queue_calls()), 1)
        self.poll(now=1920, run=self.runner())
        self.assertEqual(len(self.queue_calls()), 2)

    def test_ordinary_job_start_after_recent_ack_waits_fifteen_minutes(self):
        self.poll(run=self.runner(state='PENDING'))
        write(self.root/'supervisor-ack.json',dict(epoch=1020))
        self.poll(now=1060,run=self.runner(state='RUNNING'))
        self.poll(now=1919,run=self.runner(state='RUNNING'))
        self.assertEqual(len(self.queue_calls()),1)
        self.poll(now=1920,run=self.runner(state='RUNNING'))
        self.assertEqual(len(self.queue_calls()),2)
        self.assertEqual(supervisor.read(self.root/'supervisor-state.json')['notification_kind'],'routine')

    def test_initial_dispatch_is_allowed_despite_a_recent_preexisting_ack(self):
        write(self.root/'supervisor-ack.json',dict(epoch=995))
        self.poll(now=1000,run=self.runner())
        self.assertEqual(len(self.queue_calls()),1)

    def test_unchanged_completed_shard_does_not_make_peer_startup_urgent(self):
        completed=dict(shard=0,job_id='101',state='COMPLETED',stage='collection_complete',
            stale_heartbeat=False,stalled_results=False,halt_sha256=None,collection_complete=True,errors=[])
        pending=dict(shard=1,job_id='102',state='PENDING',stage='awaiting_current_controller',
            stale_heartbeat=False,stalled_results=False,halt_sha256=None,collection_complete=False,errors=[])
        snapshot=dict(shards=[completed,pending])
        _,signature,_,_=supervisor.notification_decision({},0,snapshot,1000)
        old=dict(queued_epoch=1000,signature=signature)
        pending=dict(pending,state='RUNNING',stage='running')
        dispatch,_,urgent,_=supervisor.notification_decision(old,1020,dict(shards=[completed,pending]),1060)
        self.assertFalse(dispatch)
        self.assertFalse(urgent)

    def test_verified_complete_silently_acks_without_reads_or_scheduler(self):
        value = supervisor.read(self.pointer)
        value['verified_complete'] = True
        write(self.pointer, value)
        self.assertFalse(self.poll(run=lambda *args, **kwargs: self.fail('No external actions for stale callback')))
        self.assertTrue((self.root/'supervisor-finished.json').exists())
        self.assertTrue((self.root/'supervisor-ack.json').exists())
        self.assertFalse((self.root/'supervisor-state.json').exists())

    def test_exact_approval_source_and_pointer_mismatch_block_queries(self):
        def forbidden(*args, **kwargs):
            self.fail('Invalid authorization must not query or queue from poll')
        for changed in [dict(self.approval, approved=False), dict(self.approval, plan_sha256='bad'),
                        dict(self.approval, limits=dict(supervisor.LIMITS, judge_usd_total=51.))]:
            write(self.root/'approval.json', changed)
            with self.assertRaisesRegex(ValueError, 'approval'):
                self.poll(run=forbidden)
        write(self.root/'approval.json', self.approval)
        write(self.source/'reference_manifest.json', {'frozen': False})
        with self.assertRaisesRegex(ValueError, 'source manifest'):
            self.poll(run=forbidden)

    def test_missing_extra_or_reselected_extension_targets_are_rejected(self):
        original = json.loads(json.dumps(self.plan))
        for mutation in ('missing', 'extra', 'reorder'):
            self.plan = json.loads(json.dumps(original))
            if mutation == 'missing':
                self.plan['extension_task_order'].pop()
            elif mutation == 'extra':
                self.plan['extension_task_order'].append('task707')
            else:
                self.plan['extension_task_order'][0:2] = self.plan['extension_task_order'][1::-1]
            write(self.root/'plan.json', self.plan)
            approval = dict(self.approval, plan_sha256=supervisor.digest(self.plan))
            write(self.root/'approval.json', approval)
            pointer = supervisor.read(self.pointer)
            pointer['plan_sha256'] = supervisor.digest(self.plan)
            write(self.pointer, pointer)
            with self.assertRaisesRegex(ValueError, 'Filtered extension'):
                self.poll(run=lambda *args, **kwargs: self.fail('Invalid cohort must not query'))

    def test_exit_zero_without_matching_queue_receipt_cannot_claim_continuation(self):
        for output in ['', 'ok', 'Queued message abc for thread wrong-thread.']:
            with self.subTest(output=output), self.assertRaisesRegex(RuntimeError, 'verified receipt'):
                self.poll(run=self.runner(queue_output=output))
        with self.assertRaises(RuntimeError):
            self.poll(run=self.runner(queue_code=1))
        self.assertFalse((self.root/'active-continuation.json').exists())
        self.assertFalse((self.root/'supervisor-state.json').exists())

    def test_all_registered_attempts_charged_and_latest_replacement_followed(self):
        self.approval['attempts'].append(dict(job_id='201', shard=0, time_limit_seconds=35000))
        write(self.root/'approval.json', self.approval)
        self.poll(run=self.runner(state='PENDING', elapsed=80))
        item = supervisor.read(self.root/'supervisor-latest.json')['shards'][0]
        self.assertEqual(item['job_id'], '201')
        self.assertEqual(item['all_attempt_scheduler_seconds'], 160)
        self.assertEqual(item['remaining_scheduler_seconds'], 35840)
        self.assertEqual(len(item['scheduler_attempts']), 2)

    def test_missing_prior_accounting_never_invents_remaining_budget(self):
        self.approval['attempts'].insert(0, dict(job_id='100', shard=0, time_limit_seconds=100))
        write(self.root/'approval.json', self.approval)
        self.poll(run=self.runner(omit=('100',)))
        item = supervisor.read(self.root/'supervisor-latest.json')['shards'][0]
        self.assertFalse(item['scheduler_accounting_complete'])
        self.assertIsNone(item['remaining_scheduler_seconds'])
        self.assertIn('scheduler_accounting_missing:100', item['errors'])

    def test_larger_allocation_is_detected_from_scheduler_evidence(self):
        base = self.runner()
        def larger(command, **kwargs):
            value = base(command, **kwargs)
            if command[0] == '/usr/bin/sacct':
                value.stdout = value.stdout.replace('gres/gpu=1,gres/gpu:h200=1', 'gres/gpu=2,gres/gpu:h200=2')
            return value
        self.poll(run=larger)
        item = supervisor.read(self.root/'supervisor-latest.json')['shards'][0]
        self.assertIn('unexpected_or_unverified_allocation:101', item['errors'])

    def test_failure_snapshot_suppressed_if_replacement_registered_during_poll(self):
        base = self.runner(state='FAILED')
        def replace(command, **kwargs):
            result = base(command, **kwargs)
            self.approval['attempts'].append(dict(job_id='201', shard=0, time_limit_seconds=30000))
            write(self.root/'approval.json', self.approval)
            return result
        self.poll(run=replace)
        self.assertEqual(len(self.queue_calls()), 0)
        self.assertFalse((self.root/'supervisor-state.json').exists())

    def test_halts_and_budget_excess_prompt_urgent_review(self):
        self.poll(run=self.runner())
        write(self.root/'shard-0/judge-budget/halt.json', {'reason': 'budget'})
        write(self.root/'shard-0/judge-budget/ledger.json', {'calls': 4801, 'charged_or_reserved_usd': 12.6})
        self.poll(now=1060, run=self.runner())
        item = supervisor.read(self.root/'supervisor-latest.json')['shards'][0]
        self.assertIn('judge_budget_exceeded', item['errors'])
        self.assertTrue(item['halt_sha256'])
        self.assertEqual(len(self.queue_calls()), 2)

    def test_stale_heartbeat_and_stalled_results_are_urgent(self):
        self.poll(run=self.runner())
        self.poll(now=5000, run=self.runner(elapsed=4000))
        item = supervisor.read(self.root/'supervisor-latest.json')['shards'][0]
        self.assertTrue(item['stale_heartbeat'])
        self.assertTrue(item['stalled_results'])
        self.assertEqual(len(self.queue_calls()), 2)

    def test_latest_record_archive_and_judge_evidence_are_visible(self):
        directory = self.root/'shard-0'
        artifact = directory/'artifact'
        archive = artifact/(hashlib.sha256(b'task').hexdigest()+'.pt')
        write(archive, {'fixture': True})
        write(archive.with_suffix('.json'), dict(terminal_status='completed', reward_metadata={'judge': 1.}))
        write(directory/'records/record.json', dict(task_id='task', attempt=5, valid=True, reward=1.,
            steps=3, seed=77, policy_id='OpenWebRL-4B-SFT:iteration0',
            judge_id='gpt-4.1/action_history', artifact=str(artifact)))
        self.poll(run=self.runner())
        item = supervisor.read(self.root/'supervisor-latest.json')['shards'][0]['results']
        self.assertEqual(item['canonical_records'], 1)
        self.assertEqual(item['latest_records'][0]['terminal_status'], 'completed')
        self.assertEqual(item['latest_records'][0]['judge_verdict'], 1.)
        self.assertGreater(item['latest_records'][0]['archive']['size_bytes'], 0)

    def test_collection_completion_queues_audit_and_does_not_mark_verified(self):
        self.poll(run=self.runner())
        write(self.root/'shard-0/collection-complete.json', dict(complete=True, primary_attempts=531))
        self.poll(now=1060, run=self.runner(state='COMPLETED'))
        self.assertEqual(len(self.queue_calls()), 2)
        self.assertFalse(supervisor.read(self.pointer)['verified_complete'])

    def test_prompt_budget_and_no_automatic_job_launch_commands(self):
        root = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/evaluations/sft-training-pass8-20261006/extension3-misses-v1')
        prompt = supervisor.continuation_prompt(root, supervisor.DEFAULT_POINTER)
        self.assertLessEqual(len(prompt.encode()), 1000)
        self.assertIn('routine at most hourly', prompt)
        self.assertIn('No added budget', prompt)
        self.assertIn('2121 new records on 707 misses', prompt)
        self.poll(run=self.runner())
        self.assertEqual({c[0] for c in self.calls}, {'/usr/bin/sacct', 'codex'})

    def test_private_log_tails_redact_credentials(self):
        path = self.root/'log'
        path.write_text('failure api_key=123secret authorization=BearerSecret sk-example-secret1234')
        tail = supervisor.file_status(path, 1000, tail_bytes=200)['tail']
        self.assertNotIn('123secret', tail)
        self.assertNotIn('BearerSecret', tail)
        self.assertNotIn('sk-example', tail)


if __name__ == '__main__':
    unittest.main()
