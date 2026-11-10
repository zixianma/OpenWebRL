"""Local CPU regressions; no allocation, application, browser, model, or network."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from openwebrl.harmful_native_runtime import (
    AsyncStorageSampler, DeadlineBudget, ProcessSupervisor, RuntimeRefused,
    StorageGate, private_path, scan_storage,
)


def sample(count=10, start=100.0, finish=101.0):
    return {'status': 'complete_scan', 'bytes': {'scratch': count},
            'started_monotonic': start, 'finished_monotonic': finish}


class DeadlineTests(unittest.TestCase):
    def test_whole_remaining_chain_includes_normal_closure(self):
        budget = DeadlineBudget(200, 20, 10)
        self.assertEqual(budget.admit([40, 20, 10], now=100), 0)
        with self.assertRaises(RuntimeRefused):
            budget.admit([40, 20, 11], now=100)

    def test_work_cannot_borrow_either_reserve(self):
        budget = DeadlineBudget(200, 20, 10)
        self.assertEqual(budget.stage_deadline(100, now=150), 170)
        self.assertEqual(budget.stage_deadline(100, now=170, diagnostic=True), 180)
        with self.assertRaises(RuntimeRefused):
            budget.stage_deadline(1, now=170)
        with self.assertRaises(RuntimeRefused):
            budget.stage_deadline(1, now=180, diagnostic=True)

    def test_epoch_clock_changes_cannot_renew_budget(self):
        with patch('openwebrl.harmful_native_runtime.time.monotonic', return_value=20), \
                patch('openwebrl.harmful_native_runtime.time.time', return_value=1000):
            budget = DeadlineBudget.from_epoch(1100, 10)
        with patch('openwebrl.harmful_native_runtime.time.time', return_value=-10000):
            self.assertEqual(budget.work_deadline, 110)

    def test_invalid_caps_are_not_admitted(self):
        for value in (0, -1, float('nan'), float('inf'), True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                DeadlineBudget(100, 10).admit([value], now=0)

    def test_injected_nonfinite_clocks_never_bypass_reserves(self):
        for value in (float('nan'), float('inf'), -1, True, '100'):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    DeadlineBudget(100, 10).admit([1], now=value)
                with self.assertRaises(ValueError):
                    DeadlineBudget(100, 10).stage_deadline(1, now=value)


class StorageGateTests(unittest.TestCase):
    def gate(self):
        gate = StorageGate({'scratch': 100}, 10, {'scratch': 5})
        gate.accept(sample(), now=101)
        return gate

    def test_absent_failed_and_stale_scans_block_admission(self):
        gate = StorageGate({'scratch': 100}, 10)
        with self.assertRaises(RuntimeRefused):
            gate.reserve('world', {'scratch': 1}, now=101)
        gate.accept(sample(), now=101)
        with self.assertRaises(RuntimeRefused):
            gate.check(now=111)
        gate.invalidate('sampler blocked')
        with self.assertRaises(RuntimeRefused):
            gate.check(now=101)

    def test_parallel_growth_reservations_count_before_launch(self):
        gate = self.gate()
        gate.reserve('first', {'scratch': 50}, now=101)
        with self.assertRaises(RuntimeRefused):
            gate.reserve('second', {'scratch': 36}, now=101)
        gate.reserve('second', {'scratch': 35}, now=101)

    def test_released_growth_cannot_disappear_before_new_scan(self):
        gate = self.gate()
        gate.reserve('first', {'scratch': 80}, now=101)
        gate.release('first', now=102)
        with self.assertRaises(RuntimeRefused):
            gate.reserve('second', {'scratch': 6}, now=102)
        with self.assertRaises(RuntimeRefused):
            gate.accept(sample(count=90, start=101.5, finish=102.5), now=103)

    def test_release_observed_by_later_scan_is_accounted(self):
        gate = self.gate()
        gate.reserve('first', {'scratch': 80}, now=101)
        gate.release('first', now=102)
        gate.accept(sample(count=90, start=103, finish=104), now=104)
        self.assertEqual(gate.reservations, {})
        gate.reserve('second', {'scratch': 5}, now=104)
        with self.assertRaises(RuntimeRefused):
            gate.reserve('third', {'scratch': 1}, now=104)

    def test_partial_scan_never_rehabilitates_old_sample(self):
        gate = self.gate()
        with self.assertRaises(RuntimeRefused):
            gate.accept(dict(sample(), status='failed_scan'), now=101)
        with self.assertRaises(RuntimeRefused):
            gate.check(now=101)

    def test_missing_root_future_negative_and_replayed_samples_reject(self):
        bad = [dict(sample(), bytes={}), dict(sample(), bytes={'other': 1}),
               dict(sample(), bytes={'scratch': -1}), sample(start=102, finish=103),
               sample(start=99, finish=100)]
        for value in bad:
            with self.subTest(value=value), self.assertRaises(RuntimeRefused):
                self.gate().accept(value, now=101)

    def test_duplicate_reservation_rejects(self):
        gate = self.gate()
        gate.reserve('first', {'scratch': 1}, now=101)
        with self.assertRaises(ValueError):
            gate.reserve('first', {'scratch': 1}, now=101)

    def test_injected_storage_clock_failures_reject(self):
        gate = self.gate()
        gate.reserve('first', {'scratch': 1}, now=101)
        for value in (float('nan'), float('inf'), -1, True, '100'):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    gate.check(now=value)
                with self.assertRaises(ValueError):
                    gate.release('first', now=value)
        with self.assertRaises(RuntimeRefused):
            gate.check(now=100)
        with self.assertRaises(ValueError):
            gate.release('first', now=100)


class StorageScanTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='native-runtime-scan-', dir='/tmp')
        self.root = Path(self.tmp.name)
        self.ephemeral = self.root / 'browser'
        self.ephemeral.mkdir()
        self.file = self.ephemeral / 'temporary'
        self.file.write_bytes(b'x' * 8192)

    def tearDown(self):
        self.tmp.cleanup()

    def test_allocated_bytes_hardlinks_and_symlinks(self):
        os.link(self.file, self.root / 'hardlink')
        (self.root / 'outside').symlink_to('/proc')
        expected = sum(path.lstat().st_blocks * 512 for path in
                       (self.root, self.ephemeral, self.file, self.root / 'outside'))
        result = scan_storage({'scratch': str(self.root)})
        self.assertEqual(result['bytes']['scratch'], expected)
        self.assertFalse(result['atomic_snapshot'])

    def test_known_temp_enoent_restarts_entire_scan(self):
        original = os.stat
        calls = []

        def churn(path, *args, **kwargs):
            if path == 'temporary':
                calls.append(path)
                if len(calls) == 1:
                    raise FileNotFoundError('Concurrent Chromium cleanup')
            return original(path, *args, **kwargs)

        with patch('openwebrl.harmful_native_runtime.os.stat', side_effect=churn):
            result = scan_storage({'scratch': str(self.root)}, [str(self.ephemeral)])
        self.assertEqual(result['attempts'], 2)
        self.assertEqual(result['churn_retries'], 1)
        self.assertEqual(result['bytes'], scan_storage({'scratch': str(self.root)})['bytes'])

    def test_persistent_churn_never_returns_partial_total(self):
        original = os.stat

        def churn(path, *args, **kwargs):
            if path == 'temporary':
                raise FileNotFoundError('Persistent churn')
            return original(path, *args, **kwargs)

        with patch('openwebrl.harmful_native_runtime.os.stat', side_effect=churn):
            with self.assertRaises(RuntimeRefused):
                scan_storage({'scratch': str(self.root)}, [str(self.ephemeral)])
            with self.assertRaises(FileNotFoundError):
                scan_storage({'scratch': str(self.root)})

    def test_permission_failure_is_not_temp_churn(self):
        original = os.stat

        def denied(path, *args, **kwargs):
            if path == 'temporary':
                raise PermissionError('Unreadable')
            return original(path, *args, **kwargs)

        with patch('openwebrl.harmful_native_runtime.os.stat', side_effect=denied), \
                self.assertRaises(PermissionError):
            scan_storage({'scratch': str(self.root)}, [str(self.ephemeral)])

    def test_missing_root_is_not_tolerated(self):
        with self.assertRaises(FileNotFoundError):
            scan_storage({'scratch': str(self.root / 'missing')}, [str(self.root / 'missing')])

    def test_path_and_ephemeral_escape_guards(self):
        for path in ('relative', '/tmp/x/../y', '/gpfs/home/zixianma/anything', '/tmp'):
            with self.subTest(path=path), self.assertRaises(ValueError):
                private_path(path)
        (self.root / 'linked').symlink_to(self.ephemeral)
        with self.assertRaises(ValueError):
            private_path(self.root / 'linked' / 'new')
        with self.assertRaises(ValueError):
            scan_storage({'scratch': str(self.ephemeral)}, [str(self.root)])


class ProcessTests(unittest.TestCase):
    def owner(self, seconds=4, reserve=1):
        owner = ProcessSupervisor(DeadlineBudget(time.monotonic() + seconds, reserve),
                                  poll_seconds=0.01, term_grace_seconds=0.08)
        self.addCleanup(lambda: owner.close() if not owner.finished else None)
        return owner

    def command(self, source):
        return [sys.executable, '-B', '-c', source]

    def test_success_means_process_completion_not_safe_native_capture(self):
        owner = self.owner()
        receipt = owner.run_stage('child', self.command('pass'), 1)
        self.assertEqual(receipt['status'], 'process_completed_and_closed')
        self.assertEqual(receipt['native_capture_status'], 'not_evaluated')
        self.assertIsNone(receipt['safe_success'])
        self.assertTrue(receipt['closure']['owned_processes'])
        self.assertEqual(owner.close()['status'], 'closure_verified')

    def test_stage_deadline_closes_term_ignoring_process(self):
        owner = self.owner()
        start = time.monotonic()
        receipt = owner.run_stage('child', self.command(
            'import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(30)'), 0.2)
        self.assertEqual(receipt['status'], 'stage_failed_unknown')
        self.assertEqual(receipt['reason'], 'stage_deadline')
        self.assertEqual(receipt['closure']['status'], 'closure_unknown')
        self.assertLess(receipt['finished_monotonic'] - receipt['stage_deadline_monotonic'], .15)
        self.assertLess(time.monotonic() - start, 1.5)
        self.assertEqual(owner.close()['remaining_owned_processes'], {})

    def test_escaped_descendant_is_owned_and_closed(self):
        owner = self.owner()
        source = ('import os,time;pid=os.fork();'
                  '\nif pid==0:\n os.setsid();time.sleep(30)'
                  '\nelse:\n time.sleep(.15)')
        receipt = owner.run_stage('tree', self.command(source), 1)
        final = owner.close()
        self.assertGreaterEqual(len(final['owned_processes']), 2)
        self.assertEqual(final['status'], 'closure_verified')
        self.assertEqual(final['remaining_owned_processes'], {})
        self.assertIsNone(receipt['safe_success'])

    def test_unattributed_double_fork_cannot_be_reported_safe(self):
        owner = self.owner()
        source = ('import os,time;pid=os.fork();'
                  '\nif pid==0:\n os.setsid();p=os.fork();'
                  '\n if p: os._exit(0)'
                  '\n time.sleep(30)'
                  '\nelse:\n time.sleep(.1)')
        receipt = owner.run_stage('tree', self.command(source), 1)
        final = owner.close()
        self.assertEqual(final['status'], 'closure_verified')
        self.assertGreaterEqual(len(final['owned_processes']), 2)
        self.assertIsNone(receipt['safe_success'])

    def test_nonzero_exit_cannot_pass(self):
        owner = self.owner()
        receipt = owner.run_stage('bad', self.command('raise SystemExit(7)'), 1)
        self.assertEqual(receipt['status'], 'stage_failed_unknown')
        self.assertEqual(receipt['returncode'], 7)

    def test_proc_observation_error_is_not_empty_closure(self):
        owner = self.owner()
        owner.start('child', self.command('import time;time.sleep(30)'))
        with patch('openwebrl.harmful_native_runtime._process_table', side_effect=PermissionError('denied')):
            receipt = owner.close(label='child')
        self.assertEqual(receipt['status'], 'closure_unknown')
        self.assertTrue(receipt['observation_errors'])
        final = owner.close()
        self.assertEqual(final['remaining_owned_processes'], {})
        self.assertEqual(final['status'], 'closure_unknown')

    def test_pid_reuse_is_never_signaled(self):
        owner = self.owner()
        with patch('openwebrl.harmful_native_runtime._process_identity', return_value={'start': 2}), \
                patch('openwebrl.harmful_native_runtime.os.kill') as kill:
            owner._signal({123: {'start': 1}}, signal.SIGKILL)
            kill.assert_not_called()

    def test_baseline_pid_reuse_does_not_hide_new_adopted_child(self):
        owner = self.owner()
        owner.baseline[123] = {'start': 1}
        with patch('openwebrl.harmful_native_runtime._process_table', return_value={
                123: {'start': 2, 'ppid': os.getpid(), 'state': 'S', 'session': 123}}):
            rows = owner.refresh()
        self.assertEqual(rows[123]['label'], 'unattributed')
        # Synthetic PID was never a real owned process; final close observes it absent.

    def test_late_unattributed_child_invalidates_label_closure(self):
        owner = self.owner()
        rows = {123: {'start': 2, 'ppid': os.getpid(), 'state': 'S', 'session': 123,
                      'label': 'unattributed'}}
        with patch.object(owner, 'refresh', side_effect=[rows, {}]), \
                patch.object(owner, '_signal'):
            result = owner.close(label='finished-stage')
        self.assertEqual(result['status'], 'closure_unknown')

    def test_required_daemon_failure_aborts_worker(self):
        owner = self.owner()
        owner.start('app', self.command('raise SystemExit(3)'), critical=True)
        receipt = owner.run_stage('work', self.command('import time;time.sleep(30)'), 1)
        self.assertEqual(receipt['status'], 'stage_failed_unknown')
        self.assertIn('daemon exited', receipt['reason'])
        self.assertEqual(owner.close()['status'], 'closure_verified')

    def test_intentional_verified_daemon_close_retires_requirement(self):
        owner = self.owner()
        owner.start('app', self.command('import time;time.sleep(30)'), critical=True)
        self.assertEqual(owner.close(label='app')['status'], 'closure_verified')
        receipt = owner.run_stage('post-close-review', self.command('pass'), 1)
        self.assertEqual(receipt['status'], 'process_completed_and_closed')

    def test_close_worker_can_stop_only_declared_daemon(self):
        owner = self.owner()
        daemon = owner.start('app', self.command('import time;time.sleep(30)'), critical=True)
        receipt = owner.run_stage('stop-app', self.command(f'import os,signal;os.kill({daemon.pid},signal.SIGTERM)'),
                                  1, closing_labels=['app'])
        self.assertEqual(receipt['status'], 'process_completed_and_closed')
        self.assertEqual(owner.close(label='app')['status'], 'closure_verified')

    def test_watchdog_exception_fails_and_closes(self):
        owner = self.owner()
        def broken(): raise KeyError('malformed lease')
        with patch('subprocess.Popen') as start:
            receipt = owner.run_stage('work', self.command('pass'), 1, watchdog_tick=broken)
            start.assert_not_called()
        self.assertEqual(receipt['status'], 'stage_failed_unknown')
        self.assertIsNone(receipt['safe_success'])

    def test_delayed_predispatch_check_cannot_extend_absolute_stage_deadline(self):
        owner = self.owner()
        deadline = time.monotonic() + .02
        def delayed(): time.sleep(.04)
        with patch('subprocess.Popen') as start:
            result = owner.run_stage('late', self.command('pass'), 1,
                                    watchdog_tick=delayed, absolute_deadline=deadline)
            start.assert_not_called()
        self.assertEqual(result['status'], 'stage_failed_unknown')

    def test_slow_last_watchdog_cannot_turn_late_child_exit_into_success(self):
        owner = self.owner()
        calls = 0
        def delayed():
            nonlocal calls
            calls += 1
            if calls == 2:
                time.sleep(.08)
        receipt = owner.run_stage('late-exit', self.command('pass'), .05, watchdog_tick=delayed)
        self.assertEqual(receipt['status'], 'stage_failed_unknown')
        self.assertEqual(receipt['reason'], 'stage_deadline')
        self.assertEqual(owner.close()['status'], 'closure_verified')

    def test_deadline_and_signal_refuse_new_launch(self):
        owner = self.owner()
        owner._stop_signal()
        with self.assertRaises(RuntimeRefused):
            owner.start('new', self.command('pass'))

    def test_global_cleanup_reserve_remains_after_stage_timeout(self):
        owner = self.owner(seconds=1, reserve=.4)
        receipt = owner.run_stage('sleep', self.command('import time;time.sleep(30)'), 10)
        self.assertEqual(receipt['reason'], 'stage_deadline')
        self.assertLessEqual(receipt['stage_deadline_monotonic'], owner.budget.work_deadline)
        self.assertLess(receipt['work_finished_monotonic'], owner.budget.diagnostic_deadline + .2)
        self.assertEqual(owner.close()['status'], 'closure_verified')

    def test_context_closes_on_exception(self):
        owner = self.owner()
        with self.assertRaises(ValueError):
            with owner:
                owner.start('child', self.command('import time;time.sleep(30)'))
                raise ValueError('adapter failure')
        self.assertEqual(owner.final_closure['status'], 'closure_verified')


class SamplerTests(unittest.TestCase):
    owner = ProcessTests.owner
    command = ProcessTests.command
    def test_real_sampler_keeps_scans_outside_watchdog(self):
        with tempfile.TemporaryDirectory(prefix='native-sampler-', dir='/tmp') as root:
            owner = self.owner()
            gate = StorageGate({'scratch': 10**7}, 3)
            sampler = AsyncStorageSampler(gate, {'scratch': root}, scan_timeout=1, interval=.5)
            until = time.monotonic() + 2
            while gate.sample is None and time.monotonic() < until:
                sampler.tick(owner)
                time.sleep(.01)
            self.assertIsNotNone(gate.sample)
            receipt = owner.run_stage('worker', self.command('import time;time.sleep(.1)'), 1,
                                      storage=gate, sampler=sampler)
            self.assertEqual(receipt['status'], 'process_completed_and_closed')
            self.assertEqual(owner.close()['status'], 'closure_verified')

    def test_blocked_sampler_does_not_block_stage_deadline(self):
        owner = self.owner()
        gate = StorageGate({'scratch': 100}, 3)
        now = time.monotonic()
        gate.accept(sample(start=now, finish=now))
        sampler = AsyncStorageSampler(gate, {'scratch': '/tmp/unused'}, scan_timeout=.15, interval=.5)
        with patch.object(sampler, '_argv', return_value=self.command(
                'import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(30)')):
            start = time.monotonic()
            result = owner.run_stage('work', self.command('import time;time.sleep(30)'), 1,
                                     storage=gate, sampler=sampler)
        self.assertEqual(result['status'], 'stage_failed_unknown')
        self.assertIn('sampler deadline', result['reason'])
        self.assertLess(time.monotonic() - start, 1)
        with self.assertRaises(RuntimeRefused):
            gate.check()
        final = owner.close()
        self.assertEqual(final['status'], 'closure_verified')
        self.assertEqual(final['remaining_owned_processes'], {})

    def test_sampler_frozen_source_rejects_changed_bytes_without_execution(self):
        import hashlib
        with tempfile.TemporaryDirectory(prefix='native-sampler-custody-', dir='/tmp') as name:
            root = Path(name)
            worker = root / 'scan.py'
            worker.write_text('raise SystemExit(0)\n')
            expected = hashlib.sha256(worker.read_bytes()).hexdigest()
            marker = root / 'executed'
            worker.write_text('from pathlib import Path;Path(' + repr(str(marker)) + ').write_text("bad")')
            gate = StorageGate({'scratch': 10**7}, 3)
            sampler = AsyncStorageSampler(gate, {'scratch': str(root)}, scan_timeout=1, interval=.5,
                                           worker_sha256=expected)
            sampler.worker_path = str(worker)
            owner = self.owner()
            with self.assertRaises(RuntimeRefused):
                until = time.monotonic() + 2
                while time.monotonic() < until:
                    sampler.tick(owner); time.sleep(.01)
            self.assertFalse(marker.exists())
            self.assertIsNone(gate.sample)
            self.assertEqual(owner.close()['status'], 'closure_verified')

    def test_failed_sampler_valid_looking_partial_output_rejects(self):
        owner = self.owner()
        gate = StorageGate({'scratch': 100}, 3)
        now = time.monotonic()
        gate.accept(sample(start=now, finish=now))
        sampler = AsyncStorageSampler(gate, {'scratch': '/tmp/unused'}, scan_timeout=1, interval=.5)
        source = "import json,time;print(json.dumps({'status':'complete_scan','bytes':{'scratch':0},'started_monotonic':time.monotonic(),'finished_monotonic':time.monotonic()}));raise SystemExit(1)"
        with patch.object(sampler, '_argv', return_value=self.command(source)):
            result = owner.run_stage('work', self.command('import time;time.sleep(30)'), 1,
                                     storage=gate, sampler=sampler)
        self.assertEqual(result['status'], 'stage_failed_unknown')
        self.assertTrue(sampler.failures)
        self.assertEqual(owner.close()['status'], 'closure_verified')


if __name__ == '__main__':
    unittest.main()
