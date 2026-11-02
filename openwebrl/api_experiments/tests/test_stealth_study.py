"""No-provider tests for the full300 paired study's financial/time boundaries."""
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import Mock, patch
import tempfile
import threading
import time
import unittest

from openwebrl.api_experiments.reproduce import stealth_study as s


class StudyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='owrl-full300-', dir='/tmp')
        self.root = Path(self.tmp.name)
        self.limits = dict(browser_target_usd='90', browser_admission_stop_usd='80',
                           collection_calendar_seconds=604800)
        self.global_control = s.GlobalControl(self.root / 'global')
        self.global_control.initialize(time.time() - 10, self.limits)
        self.controls = []
        for index in range(2):
            root = self.root / str(index)
            s.exclusive(root / 'study-control-config.json', dict(global_control=str(self.root / 'global'), shard=index))
            control = s.Control(root / 'shared-control')
            control.initialize(time.time() + 57600)
            self.controls.append(control)

    def tearDown(self):
        self.tmp.cleanup()

    def view(self, fee, **kw):
        return dict(id='owned-session', browserCost='0', proxyCost=None if fee is None else str(fee),
                    proxyUsedMb='1', status='stopped', **kw)

    def add(self, index, name, fee):
        control = self.controls[index]
        directory = self.root / name
        control.admission(directory, {'proxyCountryCode': 'us'})
        return control.usage(directory, self.view(fee))

    def test_global_fees_cover_both_shards(self):
        self.add(0, 'a', 39)
        self.add(1, 'b', 41)
        with self.global_control.locked(readonly=True) as state:
            self.assertEqual(s.GlobalControl.total(state), Decimal('80'))
        for control in self.controls:
            with control.locked() as state:
                with self.assertRaisesRegex(ValueError, 'threshold'):
                    control.check(state, new_pair=True)

    def test_watchdog_requests_both_shards_stop_at_target(self):
        self.add(0, 'a', 39)
        self.assertFalse(self.add(1, 'b', 51))
        with self.controls[0].locked() as state:
            with self.assertRaisesRegex(ValueError, 'halted'):
                self.controls[0].check(state)

    def test_downward_fee_correction_preserves_admission_reserve(self):
        self.add(0, 'a', 12)
        self.controls[0].usage(self.root / 'a', self.view(9))
        with self.global_control.locked(readonly=True) as state:
            self.assertEqual(s.GlobalControl.total(state), Decimal('12'))
        local = self.controls[0].snapshot()['sessions'][str(self.root / 'a')]
        self.assertEqual(local['latest_provider_usage']['proxyCost'], '9')

    def test_unknown_provider_fee_halts_globally(self):
        c = self.controls[0]
        c.admission(self.root / 'a', {})
        self.assertFalse(c.usage(self.root / 'a', self.view(None)))
        with self.assertRaises(ValueError):
            self.controls[1].admission(self.root / 'b', {})

    def test_ambiguous_cloud_start_halts_other_shard(self):
        self.controls[0].halt('cloud_start_failed_TimeoutError')
        with self.assertRaises(ValueError):
            self.controls[1].admission(self.root / 'b', {})

    def test_model_pool_halt_is_local(self):
        self.controls[0].halt('model_or_terra_pool_halted')
        self.controls[1].admission(self.root / 'b', {})

    def test_duplicate_create_never_repeats(self):
        self.controls[0].admission(self.root / 'a', {})
        with self.assertRaises(ValueError):
            self.controls[0].admission(self.root / 'a', {})
        with self.global_control.locked(readonly=True) as state:
            self.assertEqual(len(state['creates']), 1)

    def test_global_create_cap_checked_before_dispatch(self):
        with self.global_control.locked() as state:
            state['creates'] = {str(i): {} for i in range(608)}
        with self.assertRaisesRegex(ValueError, 'Global browser'):
            self.controls[0].admission(self.root / 'a', {})

    def test_higher_or_changed_shared_limits_refused(self):
        with self.assertRaises(ValueError):
            self.global_control.initialize(time.time(), dict(self.limits, browser_target_usd='91'))

    def test_late_arrival_of_earlier_started_shard_tightens_calendar(self):
        now = time.time()
        before = self.global_control.initialize(now - 100, self.limits)
        after = self.global_control.initialize(now, self.limits)
        self.assertEqual(before, after)
        self.assertAlmostEqual(after, now - 100 + 604800)

    def test_new_pair_not_admitted_without_actor_drain_time(self):
        c = self.controls[0]
        with c.locked() as state:
            state['deadline'] = time.time() + 1859
        with self.assertRaises(ValueError):
            c.admission(self.root / 'a', {})

    def test_pair_release_latch_is_not_rechecked_against_next_create(self):
        c = self.controls[0]
        with c.locked() as state:
            state['arrivals']['0'] = [s.ARMS[0]]
            state['released_pair_phases']['arrivals:0'] = True
            # A released fast peer may already have admitted its next browser.
            state['creates']['next'] = {}
        c.pair(s.ARMS[1], 0)

    def test_pair_requires_stopped_late_known_fees(self):
        c = self.controls[0]
        self.add(0, 'a', 1)
        with c.locked() as state:
            state['arrivals']['0'] = [s.ARMS[0]]
        with self.assertRaisesRegex(ValueError, 'late fees'):
            c.pair(s.ARMS[1], 0)

    def test_exact_approval_does_not_follow_changed_caps(self):
        plan = dict(global_limits={}, local_limits={})
        approval = dict(approved=True, proxy_overshoot_authorized=True,
                        plan_sha256=s.plan_hash(plan), global_limits={}, local_limits={})
        s.approval_check(plan, approval)
        plan['global_limits']['api_usd'] = '61'
        with self.assertRaises(ValueError):
            s.approval_check(plan, approval)

    def test_no_overshoot_approval_means_no_execution(self):
        plan = dict(global_limits={}, local_limits={})
        with self.assertRaises(ValueError):
            s.approval_check(plan, dict(approved=True, plan_sha256=s.plan_hash(plan)))

    def test_exact_once_json_publish_is_complete_and_exclusive(self):
        path = self.root / 'intent.json'
        s.exclusive(path, {'sent': False})
        self.assertEqual(s.read(path), {'sent': False})
        with self.assertRaises(FileExistsError):
            s.exclusive(path, {'sent': True})
        self.assertEqual(s.read(path), {'sent': False})

    def scheduler(self, **changes):
        fields = dict(JobId='42', JobState='RUNNING', Partition='gpu-h200', NumCPUs='2', NumNodes='1',
                      Requeue='0', Restarts='0', ReqTRES='cpu=2,mem=16G,node=1',
                      AllocTRES='cpu=2,mem=16G,node=1', StartTime='2026-11-02T00:00:00', EndTime='2026-11-02T16:00:00')
        fields.update(changes)
        return ' '.join(k + '=' + v for k, v in fields.items())

    def test_scheduler_selects_exact_record_in_multiline_output(self):
        now = datetime(2026, 11, 2, 1, tzinfo=timezone.utc).timestamp()
        raw = self.scheduler(JobId='43', JobState='PENDING') + '\n' + self.scheduler()
        start, end, fields = s.allocation(raw, '42', now)
        self.assertEqual(end - start, 57600)
        self.assertEqual(fields['JobId'], '42')

    def test_scheduler_refuses_foreign_duplicate_gpu_or_oversized_allocation(self):
        now = datetime(2026, 11, 2, 1, tzinfo=timezone.utc).timestamp()
        for raw in (self.scheduler(JobId='43'), self.scheduler() + '\n' + self.scheduler(),
                    self.scheduler(AllocTRES='cpu=2,mem=16G,node=1,gres/gpu=1'),
                    self.scheduler(EndTime='2026-11-03T00:00:00'), self.scheduler(Partition='gpu-h200-mig')):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                s.allocation(raw, '42', now)


class ShutdownTests(unittest.TestCase):
    def worker(self, pid, *, alive=True, join_error=None):
        worker = Mock(pid=pid, exitcode=None if alive else 0)
        worker.is_alive.return_value = alive
        if join_error is not None:
            worker.join.side_effect = join_error
        return worker

    def test_failed_durable_halt_still_signals_and_joins_every_child(self):
        control = Mock()
        control.halt.side_effect = OSError('quota exhausted')
        workers = [self.worker(11001), self.worker(11002)]
        errors, signalled = [], set()
        with patch.object(s.os, 'kill') as kill:
            s._drain(workers, control, signalled, errors)
        self.assertEqual(kill.call_args_list,
                         [unittest.mock.call(11001, s.signal.SIGTERM),
                          unittest.mock.call(11002, s.signal.SIGTERM)])
        for worker in workers:
            worker.join.assert_called_once_with()
        self.assertEqual(signalled, {11001, 11002})
        self.assertEqual(errors, [dict(component='halt_record', error_type='OSError')])

    def test_failed_join_does_not_skip_remaining_owned_child(self):
        workers = [self.worker(11001, alive=False, join_error=OSError('wait failed')),
                   self.worker(11002, alive=False)]
        errors = []
        s._drain(workers, Mock(), set(), errors)
        workers[1].join.assert_called_once_with()
        self.assertEqual(errors, [dict(component='child_await', error_type='OSError')])

    def test_normal_completion_joins_without_unnecessary_abort(self):
        control = Mock()
        workers = [self.worker(11001, alive=False), self.worker(11002, alive=False)]
        errors = []
        with patch.object(s.os, 'kill') as kill:
            s._drain(workers, control, set(), errors)
        control.halt.assert_not_called()
        kill.assert_not_called()
        for worker in workers:
            worker.join.assert_called_once_with()
        self.assertEqual(errors, [])

    def test_pre_set_shutdown_creates_no_worker(self):
        shutdown = threading.Event()
        shutdown.set()
        plan = dict(source_sha256={}, runtime='/unused-runtime', campaign_output='/unused-campaign',
                    shard=0, output='/unused-output', global_limits={})
        control = Mock()
        global_control = Mock()
        global_control.initialize.return_value = time.time() + 10000
        with ExitStack() as stack:
            for name in ('approval_check', 'exclusive'):
                stack.enter_context(patch.object(s, name))
            stack.enter_context(patch.object(s, 'prepare', return_value=plan))
            stack.enter_context(patch.object(s, 'allocation',
                               return_value=(time.time() - 1, time.time() + 10000, {})))
            stack.enter_context(patch.object(s.subprocess, 'run',
                               return_value=SimpleNamespace(stdout='offline-allocation')))
            stack.enter_context(patch.object(s, 'GlobalControl', return_value=global_control))
            stack.enter_context(patch.object(s, 'Control', return_value=control))
            stack.enter_context(patch.object(s.Path, 'mkdir'))
            context = stack.enter_context(patch.object(s.multiprocessing, 'get_context'))
            stack.enter_context(patch.dict(s.os.environ, SLURM_JOB_ID='42'))
            with self.assertRaisesRegex(ValueError, 'before any worker starts'):
                s.run(plan, {}, None, shutdown_event=shutdown)
        context.assert_not_called()
        control.halt.assert_not_called()


if __name__ == '__main__':
    unittest.main()
