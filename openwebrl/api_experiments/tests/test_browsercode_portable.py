"""Run with OPENWEBRL_API_DEPENDENCIES pointing to install.py's manifest.

Unlike the lightweight accounting tests, these assemble real runtimes and import
actual upstream CUAWright, frozen model adapters and judge modules in fresh
processes. They never construct an SDK client or send a paid model request.
"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('portable_builder', ROOT / 'reproduce/prepare_browsercode.py')
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


@unittest.skipUnless(os.environ.get('OPENWEBRL_API_DEPENDENCIES'), 'Install public dependencies and set OPENWEBRL_API_DEPENDENCIES for real runtime checks')
class PortableRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dependencies = json.loads(Path(os.environ['OPENWEBRL_API_DEPENDENCIES']).read_text())
        cls.python = cls.dependencies.get('python', sys.executable)
        cls.temp = tempfile.TemporaryDirectory(prefix='browsercode-portable-test-')
        cls.addClassCleanup(cls.temp.cleanup)

    def command(self, *args):
        run = subprocess.run([self.python, *map(str, args)], text=True, capture_output=True, env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'}, timeout=120)
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        return run.stdout

    def test_prepare_imports_dry_runs_and_archive_integrity(self):
        index = json.loads((ROOT / 'source-provenance.json').read_text())
        before = {e['published_path']: builder.digest(ROOT / e['published_path']) for campaign in index['snapshots'].values() for e in campaign['files']}
        for key in builder.CAMPAIGNS:
            with self.subTest(campaign=key):
                if not (ROOT / 'portable_assets/campaign_specs' / key).exists():
                    self.fail('Missing promised public campaign ' + key)
                runtime = Path(self.temp.name) / key
                # Build under the same interpreter collaborators install/use.
                self.command(ROOT / 'reproduce/prepare_browsercode.py', '--campaign', key, '--output', runtime, '--dependencies', os.environ['OPENWEBRL_API_DEPENDENCIES'])
                self.assertFalse((runtime / 'budget').exists())
                imported = json.loads(self.command(ROOT / 'reproduce/verify_runtime.py', '--runtime', runtime))
                self.assertEqual(imported['paid_requests'], 0)
                plan = json.loads((runtime / 'campaign-plan.json').read_text())
                self.assertEqual(len(imported['verified']), len(plan['arms']))
                provider = 'luna' if key == 'browsercode_luna_effort' else 'claude' if key == 'browsercode_claude' else 'gpt'
                for arm in plan['arms']:
                    output = Path(self.temp.name) / ('results-' + key + arm['arm'])
                    report = self.command(ROOT / f'reproduce/eval_{provider}.py', '--runtime', runtime, '--arm', arm['arm'], '--output', output, '--model-usd', '1', '--judge-usd', '1')
                    self.assertIn('Dry run:', report)
                    self.assertFalse(output.exists())
                if key == 'browsercode_gpt_high_wv_ds':
                    for benchmark, count in [('webvoyager', 595), ('deepshop', 150)]:
                        self.command(ROOT / 'reproduce/eval_gpt.py', '--runtime', runtime, '--benchmark', benchmark, '--limit', count, '--arm', 'serial_high', '--output', Path(self.temp.name) / benchmark, '--model-usd', '1', '--judge-usd', '1')
        self.assertEqual(before, {name: builder.digest(ROOT / name) for name in before})


if __name__ == '__main__':
    unittest.main()
