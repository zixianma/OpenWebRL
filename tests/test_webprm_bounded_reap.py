"""Owned reaping regressions for transient and persistent uninterruptible I/O."""
import signal
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from openwebrl import webprm_lifecycle as m


class BoundedReapTests(unittest.TestCase):
    def run_stop(self, release, deadline):
        clock = [0.]
        process = {123: dict(state='D', ppid=10, start_ticks=77)}
        tree = object.__new__(m.OwnedEpisodeTree)
        tree.seen = {123}
        tree.child = SimpleNamespace(exitcode=-9)
        def refresh():
            return process if clock[0] < release else {}
        def sleep(seconds):
            self.assertGreaterEqual(seconds, 0)
            clock[0] += seconds
        with patch.object(tree, 'refresh', side_effect=refresh), \
             patch.object(tree, 'reap') as reap, \
             patch.object(tree, 'signal') as signals, \
             patch.object(m.time, 'monotonic', side_effect=lambda: clock[0]), \
             patch.object(m.time, 'sleep', side_effect=sleep):
            receipt = tree.stop(.1, reap_deadline=deadline)
        self.assertEqual([c.args[0] for c in signals.call_args_list],
                         [signal.SIGTERM, signal.SIGKILL])
        return receipt, clock[0], reap.call_count

    def test_delayed_kill_is_reaped_before_fixed_deadline_with_initial_evidence(self):
        receipt, elapsed, reaps = self.run_stop(1.3, 2.)
        self.assertEqual(receipt['remaining_owned_processes'], {})
        self.assertEqual(receipt['initial_stop_remaining_processes'][123]['state'], 'D')
        self.assertTrue(receipt['recovered_after_initial_stop'])
        self.assertLessEqual(elapsed, 2.)
        self.assertGreater(reaps, 1)

    def test_persistent_io_retains_failure_at_original_deadline(self):
        receipt, elapsed, _ = self.run_stop(float('inf'), 1.5)
        self.assertEqual(receipt['remaining_owned_processes'][123]['state'], 'D')
        self.assertFalse(receipt['recovered_after_initial_stop'])
        self.assertAlmostEqual(elapsed, 1.5)

    def test_expired_deadline_does_not_open_a_fresh_wait_window(self):
        receipt, elapsed, _ = self.run_stop(float('inf'), 0.)
        self.assertFalse(receipt['recovered_after_initial_stop'])
        self.assertLess(elapsed, 1.2)

    def test_invalid_deadline_is_rejected_before_any_signal(self):
        tree = object.__new__(m.OwnedEpisodeTree)
        for value in (float('nan'), float('inf'), -1.):
            with self.subTest(value=value), patch.object(tree, 'signal') as signals:
                with self.assertRaisesRegex(ValueError, 'deadline'):
                    tree.stop(.1, reap_deadline=value)
                signals.assert_not_called()


if __name__ == '__main__':
    unittest.main()
