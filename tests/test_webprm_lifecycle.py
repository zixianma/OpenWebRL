"""Real subprocess watchdog tests; no browser, provider request or GPU."""
import json
import errno
import multiprocessing
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from concurrent.futures import ProcessPoolExecutor
from unittest.mock import patch

from openwebrl.jev_eval import write_json
from openwebrl.webprm_lifecycle import (
    OwnedEpisodeTree, _processes, mark_episode_phase, run_supervised_episode, save_outcome_and_close)


def episode_fixture(task, arm, config):
    folder = Path(config['output']) / task['task_id'] / arm
    case = config['fixture_case']
    if case == 'exit':
        os._exit(7)
    if case == 'raise':
        raise RuntimeError('Uncaught zero-API fixture failure')
    if case == 'reserve_then_stall':
        from openwebrl.webprm_worker import SharedBudget
        SharedBudget(config['budget_path']).reserve(.25, 'fixture-no-provider-call')
        mark_episode_phase(config, 'actor_request')
        time.sleep(600)
    if case == 'detached_descendant':
        intermediate = os.fork()
        if intermediate == 0:
            orphan = os.fork()
            if orphan != 0:
                os._exit(0)
            os.setsid()
            (folder / 'detached.pid').write_text(str(os.getpid()))
            time.sleep(600)
            os._exit(0)
        os.waitpid(intermediate, 0)
        until = time.time() + 5
        while not (folder / 'detached.pid').exists() and time.time() < until:
            time.sleep(.01)
        mark_episode_phase(config, 'browser_step')
        time.sleep(600)
    if case in {'startup', 'browser_reset', 'native_observation', 'browser_step'}:
        mark_episode_phase(config, case)
        time.sleep(600)
    if case == 'progressing_requests':
        for candidate in range(5):
            mark_episode_phase(config, 'critic_request', candidate=candidate)
            time.sleep(config['episode_watchdog']['browser_phase_timeout_seconds'] / 4)
    result = dict(task_id=task['task_id'], arm=arm, success=1, valid=True,
                  attempted=True, identity_sha256=config['identity_sha256'],
                  started_at=time.time(), turns=1)

    class Browser:
        def close(self):
            if case == 'cleanup_hang':
                time.sleep(600)
            if case == 'cleanup_error':
                raise RuntimeError('Fixture close failure')
    save_outcome_and_close(result, folder, Browser(), config)
    if case == 'shutdown_hang':
        time.sleep(600)


def supervisor_fixture(task, arm, config):
    run_supervised_episode(task, arm, config, episode_fixture)


class EpisodeLifecycle(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix='webprm-watchdog-', dir='/tmp')
        self.root = Path(self.directory.name)
        self.task = dict(task_id='fixture')
        self.arm = 'arbiter_unguided'
        self.config = dict(output=str(self.root), identity_sha256='fixed-fixture-identity',
            deadline=time.time() + 120, budget_path=str(self.root / 'budget.json'),
            # Cold Python 3.12 imports on GPFS can exceed 1.5 seconds before
            # the spawned fixture begins. Production's explicit cap is 180 s.
            episode_watchdog=dict(browser_phase_timeout_seconds=15,
                cleanup_timeout_seconds=.5, termination_grace_seconds=.1, poll_seconds=.04))
        self.folder = self.root / self.task['task_id'] / self.arm

    def tearDown(self):
        self.directory.cleanup()

    def run_case(self, case):
        self.config['fixture_case'] = case
        return run_supervised_episode(self.task, self.arm, self.config, episode_fixture)

    def assert_reaped(self, result):
        self.assertFalse(result['watchdog']['remaining_owned_processes'])
        self.assertIsNotNone(result['watchdog']['child_exitcode'])
        for pid in result['watchdog']['owned_pids']:
            self.assertFalse(Path(f'/proc/{pid}').exists(), f'Owned PID {pid} was not reaped')

    def test_startup_reset_observation_and_action_stalls_are_invalid_without_orphans(self):
        for phase in ['startup', 'browser_reset', 'native_observation', 'browser_step']:
            with self.subTest(phase=phase):
                self.task = dict(task_id=phase)
                self.config['deadline'] = time.time() + 120
                result = self.run_case(phase)
                self.assertFalse(result['valid'])
                self.assertTrue(result['attempted'])
                self.assertEqual(result['success'], 0)
                self.assertEqual(result['watchdog']['reason'], 'phase_timeout')
                self.assertEqual(result['watchdog']['last_phase']['phase'], phase)
                self.assert_reaped(result)

    def test_completed_outcome_survives_cleanup_hang(self):
        result = self.run_case('cleanup_hang')
        self.assertTrue(result['valid'])
        self.assertEqual(result['success'], 1)
        self.assertEqual(result['cleanup_status'], 'interrupted')
        self.assertEqual(result['watchdog']['reason'], 'cleanup_timeout')
        self.assert_reaped(result)
        self.assertEqual(json.loads((self.folder / 'result.json').read_text()), result)

    def test_normal_exit_and_cleanup_error_preserve_outcome(self):
        for case in ['normal', 'cleanup_error']:
            with self.subTest(case=case):
                self.task = dict(task_id=case)
                result = self.run_case(case)
                self.assertTrue(result['valid'])
                self.assertEqual(result['success'], 1)
                self.assertEqual(result['cleanup_status'], 'error' if case == 'cleanup_error' else 'completed')
                self.assertIsNone(result['watchdog']['reason'])
                self.assert_reaped(result)

    def test_child_shutdown_after_browser_close_shares_cleanup_deadline(self):
        result = self.run_case('shutdown_hang')
        self.assertTrue(result['valid'])
        self.assertEqual(result['success'], 1)
        self.assertEqual(result['cleanup_status'], 'completed')
        self.assertEqual(result['watchdog']['last_phase']['phase'], 'completed')
        self.assertEqual(result['watchdog']['reason'], 'cleanup_timeout')
        self.assert_reaped(result)

    def test_crashed_or_exceptional_child_cannot_manufacture_result(self):
        self.config.update(actor_model='fixture-actor', selector='fixture-selector')
        for case in ['exit', 'raise']:
            with self.subTest(case=case):
                self.task = dict(task_id=case)
                result = self.run_case(case)
                self.assertFalse(result['valid'])
                self.assertEqual(result['error_type'], 'EpisodeWorkerExited')
                self.assertEqual(result['actor_model'], self.config['actor_model'])
                self.assertEqual(result['selector'], self.config['selector'])
                self.assert_reaped(result)

    def test_deadline_before_start_preserves_assigned_actor_selector(self):
        self.config.update(actor_model='fixture-actor', selector='fixture-selector',
                           deadline=time.time() - 1)
        result = self.run_case('exit')
        self.assertFalse(result['attempted'])
        self.assertFalse(result['valid'])
        self.assertEqual(result['actor_model'], self.config['actor_model'])
        self.assertEqual(result['selector'], self.config['selector'])
        self.assertEqual(result['watchdog']['owned_pids'], [])
        self.assertEqual(result['watchdog']['reason'], 'allocation_deadline_before_start')

    def test_phase_timeout_preserves_assigned_actor_selector(self):
        self.config.update(actor_model='fixture-actor', selector='fixture-selector')
        self.config['episode_watchdog']['browser_phase_timeout_seconds'] = 3
        result = self.run_case('native_observation')
        self.assertTrue(result['attempted'])
        self.assertFalse(result['valid'])
        self.assertEqual(result['error_type'], 'EpisodeWatchdogTimeout')
        self.assertEqual(result['watchdog']['reason'], 'phase_timeout')
        self.assertEqual(result['actor_model'], self.config['actor_model'])
        self.assertEqual(result['selector'], self.config['selector'])
        self.assert_reaped(result)

    def test_detached_orphan_is_reaped_and_unrelated_process_survives(self):
        unrelated = subprocess.Popen([sys.executable, '-B', '-c', 'import time; time.sleep(600)'],
                                     start_new_session=True)
        try:
            result = self.run_case('detached_descendant')
            orphan_pid = int((self.folder / 'detached.pid').read_text())
            self.assertIn(orphan_pid, result['watchdog']['owned_pids'])
            self.assertIsNone(unrelated.poll())
            self.assertNotIn(unrelated.pid, result['watchdog']['owned_pids'])
            self.assert_reaped(result)
        finally:
            unrelated.terminate()
            unrelated.wait(timeout=5)

    def test_global_deadline_interrupts_a_longer_phase(self):
        self.config['episode_watchdog']['browser_phase_timeout_seconds'] = 30
        self.config['deadline'] = time.time() + 1
        result = self.run_case('browser_step')
        self.assertEqual(result['watchdog']['reason'], 'allocation_deadline')
        self.assertFalse(result['valid'])
        self.assert_reaped(result)

    def test_unknown_provider_reservation_is_preserved_on_forced_exit(self):
        result = self.run_case('reserve_then_stall')
        budget = json.loads(Path(self.config['budget_path']).read_text())
        self.assertEqual(budget['charged'], .25)
        self.assertEqual(len(budget['calls']), 1)
        self.assertEqual(budget['calls'][0]['state'], 'reserved')
        self.assert_reaped(result)

    def test_saved_cell_skips_child_and_budget_and_checks_identity(self):
        saved = dict(task_id='fixture', arm=self.arm, identity_sha256='fixed-fixture-identity',
                     valid=True, success=0, attempted=True)
        write_json(self.folder / 'result.json', saved)
        original = (self.folder / 'result.json').read_bytes()
        with patch('openwebrl.webprm_lifecycle.multiprocessing.get_context',
                   side_effect=AssertionError('Finalized cell launched a new process')):
            self.assertEqual(self.run_case('normal'), saved)
            self.config['identity_sha256'] = 'changed'
            with self.assertRaisesRegex(ValueError, 'identity changed'):
                self.run_case('normal')
        self.assertEqual((self.folder / 'result.json').read_bytes(), original)
        self.assertFalse(Path(self.config['budget_path']).exists())

    def test_expired_allocation_never_spawns_or_attempts_a_cell(self):
        self.config['deadline'] = time.time() - 1
        with patch('openwebrl.webprm_lifecycle.multiprocessing.get_context',
                   side_effect=AssertionError('Expired allocation launched a process')):
            result = self.run_case('normal')
        self.assertFalse(result['attempted'])
        self.assertFalse(result['valid'])
        self.assertEqual(result['watchdog']['reason'], 'allocation_deadline_before_start')
        self.assertFalse(Path(self.config['budget_path']).exists())

    def test_individual_requests_extend_phase_without_aggregate_turn_timeout(self):
        result = self.run_case('progressing_requests')
        self.assertTrue(result['valid'])
        self.assertIsNone(result['watchdog']['reason'])
        self.assert_reaped(result)

    def test_pool_worker_can_spawn_watchdog_child_and_continue_after_timeout(self):
        with ProcessPoolExecutor(max_workers=1,
                mp_context=multiprocessing.get_context('spawn')) as pool:
            config = dict(self.config, fixture_case='cleanup_hang')
            result = pool.submit(run_supervised_episode, self.task, self.arm,
                                 config, episode_fixture).result(timeout=45)
            self.assertTrue(result['valid'])
            self.assertEqual(result['watchdog']['reason'], 'cleanup_timeout')
            self.assert_reaped(result)
            config['fixture_case'] = 'normal'
            second = pool.submit(run_supervised_episode, dict(task_id='second'), self.arm,
                                 config, episode_fixture).result(timeout=45)
            self.assertTrue(second['valid'])
            self.assertIsNone(second['watchdog']['reason'])
            self.assert_reaped(second)

    def test_remaining_owned_processes_preserve_result_and_fail_collection_closed(self):
        original = OwnedEpisodeTree.stop
        def incomplete_stop(tree, grace, **kwargs):
            receipt = original(tree, grace)
            receipt['remaining_owned_processes'] = {'999999999': {'state': 'S'}}
            return receipt
        with patch.object(OwnedEpisodeTree, 'stop', incomplete_stop):
            with self.assertRaisesRegex(RuntimeError, 'collection must stop'):
                self.run_case('normal')
        saved = json.loads((self.folder / 'result.json').read_text())
        self.assertTrue(saved['valid'])
        self.assertEqual(saved['success'], 1)
        self.assertTrue(saved['watchdog']['remaining_owned_processes'])

    def test_supervisor_sigterm_drains_isolated_child_before_exiting(self):
        config = dict(self.config, fixture_case='browser_step')
        config['episode_watchdog'] = dict(config['episode_watchdog'], browser_phase_timeout_seconds=30)
        supervisor = multiprocessing.get_context('spawn').Process(
            target=supervisor_fixture, args=(self.task, self.arm, config))
        supervisor.start()
        try:
            until = time.time() + 30
            while time.time() < until:
                phase_path = self.folder / 'phase.json'
                if phase_path.exists() and json.loads(phase_path.read_text())['phase'] == 'browser_step':
                    break
                time.sleep(.02)
            else:
                self.fail('Watchdog child never reached its browser phase')
            os.kill(supervisor.pid, signal.SIGTERM)
            supervisor.join(timeout=10)
            self.assertFalse(supervisor.is_alive())
            self.assertEqual(supervisor.exitcode, 0)
            result = json.loads((self.folder / 'result.json').read_text())
            self.assertEqual(result['watchdog']['reason'], 'supervisor_termination')
            self.assertEqual(result['watchdog']['supervisor_signal'], signal.SIGTERM)
            self.assertFalse(result['valid'])
            self.assert_reaped(result)
        finally:
            if supervisor.is_alive():
                supervisor.terminate()
                supervisor.join(timeout=10)

    def test_transient_procfs_error_is_retried_without_false_clean_tree(self):
        original = Path.iterdir
        attempts = []
        def intermittent(path):
            attempts.append(str(path))
            if len(attempts) == 1:
                raise OSError(errno.ENOMEM, 'Transient procfs fixture error')
            return original(path)
        with patch.object(Path, 'iterdir', intermittent):
            self.assertIn(os.getpid(), _processes())
        self.assertEqual(len(attempts), 2)
        with patch.object(Path, 'iterdir', side_effect=OSError(errno.ENOMEM, 'Persistent procfs fixture error')):
            with self.assertRaises(OSError):
                _processes()

    def test_cleanup_scan_failure_still_stops_known_child_and_fails_closed(self):
        with patch.object(OwnedEpisodeTree, '_stop_tracked',
                          side_effect=OSError(errno.ENOMEM, 'Persistent cleanup scan fixture')):
            with self.assertRaisesRegex(RuntimeError, 'collection must stop'):
                self.run_case('cleanup_hang')
        result = json.loads((self.folder / 'result.json').read_text())
        self.assertTrue(result['valid'])
        self.assertIn('verification_failed', result['watchdog']['remaining_owned_processes'])
        for pid in result['watchdog']['owned_pids']:
            self.assertFalse(Path(f'/proc/{pid}').exists())

    def test_supervisor_startup_failure_saves_unattempted_result_without_api_work(self):
        with patch('openwebrl.webprm_lifecycle.resource_tracker.ensure_running',
                   side_effect=OSError(errno.ENOMEM, 'Fixture supervisor startup failure')):
            with self.assertRaisesRegex(RuntimeError, 'supervision failed; collection must stop'):
                self.run_case('normal')
        saved = json.loads((self.folder / 'result.json').read_text())
        self.assertFalse(saved['attempted'])
        self.assertFalse(saved['valid'])
        self.assertEqual(saved['error_type'], 'EpisodeSupervisorError')
        self.assertIn('Fixture supervisor startup failure', saved['watchdog']['supervisor_error'])
        self.assertEqual(saved['watchdog']['owned_pids'], [])
        self.assertFalse(Path(self.config['budget_path']).exists())


if __name__ == '__main__':
    unittest.main()
