from datetime import datetime
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).parents[1]/'scripts'))
from arm_browser_startup_guard import BrowserStartupGuard, instrument_controller


class StartupGuardTests(unittest.TestCase):
    def line(self, event, pid, at):
        stamp = datetime.fromtimestamp(at).strftime('%Y-%m-%d %H:%M:%S')
        return f'[worker] [{stamp}] file.py:1 - [LocalProcessCreate] {event} pid={pid} port=18000\n'

    def attempts(self, guard, count, ready, at):
        for pid in range(count):
            guard.feed(self.line('start', pid, at))
            if pid < ready:
                guard.feed(self.line('ready', pid, at+1))

    def test_partial_success_does_not_mask_sustained_startup_failure(self):
        g=BrowserStartupGuard();self.attempts(g,100,60,1000)
        self.assertFalse(g.check(1600)['stop'])
        self.assertTrue(g.check(1660)['stop'])

    def test_recent_startups_and_short_bursts_do_not_stop(self):
        g=BrowserStartupGuard();self.attempts(g,100,60,1000)
        self.assertEqual(g.check(1100)['mature_startups'],0)
        self.assertFalse(g.check(1600)['stop'])
        for pid in range(60,100):g.feed(self.line('ready',pid,1620))
        self.assertFalse(g.check(1700)['stop'])

    def test_website_failures_do_not_count_as_startup_failures(self):
        g=BrowserStartupGuard();self.attempts(g,100,100,1000)
        for _ in range(200):g.feed('Task failed: navigation timeout; reward=0')
        self.assertFalse(g.check(1660)['stop'])

    def test_old_failures_do_not_poison_next_collection(self):
        g=BrowserStartupGuard();self.attempts(g,100,0,1000)
        g.check(1600);self.assertTrue(g.check(1660)['stop'])
        self.assertFalse(g.check(2000)['stop'])

    def test_incremental_partial_line_and_duplicate_poll(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'collection.log';line=self.line('start',1,1000)
            path.write_text(line[:25]);g=BrowserStartupGuard()
            self.assertEqual(g.poll(path,now=1500)['mature_startups'],0)
            with path.open('a') as f:f.write(line[25:])
            self.assertEqual(g.poll(path,now=1500)['mature_startups'],1)
            self.assertEqual(g.poll(path,now=1500)['mature_startups'],1)

    def test_controller_integration_is_explicit_and_idempotence_fails_closed(self):
        path=Path(__file__).parents[1]/'scripts/run_arm_turn_bonus_cycles.py'
        source=path.read_text();patched=instrument_controller(source)
        self.assertIn("root/'browser-server-logs'",patched)
        self.assertIn("if browser_health['stop']: raise RuntimeError",patched)
        with self.assertRaises(ValueError):instrument_controller(patched)


if __name__=='__main__':unittest.main()
