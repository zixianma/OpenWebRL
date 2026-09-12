"""The duration change must retain the real allocation deadline and verify cap."""
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from prepare_run_duration import duration_launcher


class DurationTest(unittest.TestCase):
    block = '''seconds = 15 * 60 if args.verify_resume_only else 8 * 3600
seconds = min(seconds, int(end - time.time()) - (30 if args.verify_resume_only else 180))
if seconds < (60 if args.verify_resume_only else 600):
    raise ValueError('Insufficient time remains')
'''

    def run_duration(self, remaining, verify=False):
        scope = {'args': SimpleNamespace(verify_resume_only=verify),
                 'end': 1000 + remaining, 'time': SimpleNamespace(time=lambda: 1000)}
        exec(duration_launcher(self.block, 16), scope)
        return scope['seconds']

    def test_long_job_uses_full_budget_but_never_exceeds_its_allocation(self):
        self.assertEqual(self.run_duration(16 * 3600), 16 * 3600 - 180)
        self.assertEqual(self.run_duration(8 * 3600), 8 * 3600 - 180)
        self.assertEqual(self.run_duration(20 * 3600), 16 * 3600)
        with self.assertRaisesRegex(ValueError, 'Insufficient'):
            self.run_duration(700)

    def test_verification_stays_short(self):
        self.assertEqual(self.run_duration(16 * 3600, verify=True), 900)
        self.assertEqual(self.run_duration(120, verify=True), 90)

    def test_unknown_launcher_and_invalid_caps_rejected(self):
        for hours in [0, -1, 25, 1.5, True]:
            with self.subTest(hours=hours), self.assertRaises(ValueError):
                duration_launcher(self.block, hours)
        with self.assertRaises(ValueError):
            duration_launcher(self.block.replace('int(end - time.time())', '9999999'), 16)


if __name__ == '__main__':
    unittest.main()
