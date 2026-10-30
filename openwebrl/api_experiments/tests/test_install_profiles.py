"""Read-only installer planning and profile upgrade checks; no downloads."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / 'reproduce/install.py'
SPEC = importlib.util.spec_from_file_location('profile_installer', SCRIPT)
installer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(installer)


class InstallationProfileTests(unittest.TestCase):
    def test_browsercode_default_has_only_required_source_libraries(self):
        plan = installer.installation_plan()
        self.assertEqual(plan['profile'], 'browsercode')
        self.assertEqual(set(plan['sources']), {'cuawright_source', 'browsercode_source'})
        self.assertIsNone(plan['legacy_requirements_sha256'])
        self.assertIn('official_webjudge', installer.PINS)

    def test_profile_upgrade_adds_sources_and_never_silently_downgrades(self):
        original = installer.installation_plan()
        upgraded = installer.installation_plan(profile='all', previous=original)
        self.assertEqual(upgraded['sources'], installer.PINS)
        retained = installer.installation_plan(previous=upgraded)
        self.assertEqual(retained['profile'], 'all')
        self.assertEqual(retained['sources'], upgraded['sources'])
        # Pre-profile manifests remain usable by native and historical CLIs.
        old = copy.deepcopy(upgraded)
        old.pop('profile')
        self.assertEqual(installer.installation_plan(previous=old)['profile'], 'all')

    def test_legacy_promotes_all_and_is_retained_on_resume(self):
        legacy = installer.installation_plan(legacy=True)
        self.assertEqual(legacy['profile'], 'all')
        self.assertIsNotNone(legacy['legacy_requirements_sha256'])
        resumed = installer.installation_plan(previous=legacy)
        self.assertEqual(resumed, legacy)

    def test_changed_or_unknown_previous_source_cannot_upgrade(self):
        for change in ('commit', 'url'):
            previous = copy.deepcopy(installer.installation_plan())
            previous['sources']['cuawright_source'][change] = 'different'
            with self.assertRaisesRegex(ValueError, 'different source pins'):
                installer.installation_plan(profile='all', previous=previous)
        previous = installer.installation_plan()
        previous['sources'] = {'unknown_source': {'url': 'untrusted', 'commit': 'unknown'}}
        with self.assertRaisesRegex(ValueError, 'different source pins'):
            installer.installation_plan(previous=previous)

    def test_real_dry_plan_selects_profile_without_creating_prefix(self):
        with tempfile.TemporaryDirectory() as folder:
            for flags, expected in [([], 'browsercode'), (['--profile', 'all'], 'all'), (['--legacy'], 'all')]:
                prefix = Path(folder) / ('prefix-' + str(len(flags)))
                process = subprocess.run([sys.executable, '-B', str(SCRIPT), '--prefix', str(prefix), *flags], check=True, capture_output=True, text=True)
                plan = json.loads(process.stdout)
                self.assertEqual(plan['profile'], expected)
                self.assertEqual('official_webjudge' in plan['sources'], expected == 'all')
                self.assertFalse(plan['install'])
                self.assertFalse(prefix.exists())


if __name__ == '__main__':
    unittest.main()
