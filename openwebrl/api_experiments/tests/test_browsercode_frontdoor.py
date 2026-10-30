"""Public BrowserCode CLI checks; real setup tests use installed public dependencies."""
from __future__ import annotations

import ast
import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1] / 'reproduce/browsercode.py'
spec = importlib.util.spec_from_file_location('browsercode_frontdoor', SOURCE)
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)


class FrontDoorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='browsercode-frontdoor-test-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def runtime(self, campaign, arm, model):
        runtime = self.root / campaign
        runtime.mkdir()
        self.write(runtime / 'campaign-plan.json', {'arms': [{'arm': arm, 'model': model}]})
        self.write(runtime / 'runtime-config.json', {'portable_campaign': campaign})
        return runtime

    def test_infer_frozen_adapter_from_model_and_campaign(self):
        for campaign, arm, model, expected in [
            ('browsercode_luna_effort', 'luna_n5_medium', 'gpt-6-luna', 'luna'),
            ('browsercode_claude', 'sonnet_n5', 'claude-sonnet-5-5', 'claude'),
            ('browsercode_gpt_high', 'parallel3_select_high', 'gpt-6-luna', 'gpt'),
            ('browsercode_sol61_pair20', 'serial_high', 'gpt-6.1-sol', 'gpt'),
        ]:
            runtime = self.runtime(campaign, arm, model)
            self.assertEqual(b.provider(runtime, arm), expected)
            with self.assertRaises(ValueError):
                b.provider(runtime, 'wrong-arm')
        bad = self.runtime('bad', 'unknown', 'foreign-model')
        with self.assertRaises(ValueError):
            b.provider(bad, 'unknown')

    def test_public_parser_rejects_executor_and_provider_switches(self):
        flags = ['run', '--workspace', str(self.root), '--arm', 'serial_high', '--output', str(self.root / 'out'), '--model-usd', '1', '--judge-usd', '1']
        for extra in [('--harness', 'cuawright'), ('--provider', 'claude'), ('--native',), ('--runtime', str(self.root))]:
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                b.parser().parse_args([*flags, *extra])
        for command in ([], ['setup'], ['run'], ['check'], ['report']):
            output = io.StringIO()
            with contextlib.redirect_stdout(output), self.assertRaises(SystemExit):
                b.parser().parse_args([*command, '--help'])
            self.assertNotIn('cuawright', output.getvalue().lower())
            self.assertNotIn('--harness', output.getvalue())
            self.assertNotIn('--provider', output.getvalue())
        self.assertEqual(b.campaign('luna'), 'browsercode_luna_effort')
        self.assertEqual(b.campaign('browsercode_gpt_high'), 'browsercode_gpt_high')

    def test_run_is_dry_by_default_and_forces_the_executor(self):
        runtime = self.runtime('browsercode_gpt_high', 'serial_high', 'gpt-6-sol')
        args = b.parser().parse_args(['run', '--workspace', str(self.root), '--arm', 'serial_high', '--output', str(self.root / 'out'), '--model-usd', '2', '--judge-usd', '.3'])
        forwarded = []
        with patch.object(b, 'workspace', return_value=(runtime, Path(sys.executable))), patch.object(b, 'call', side_effect=lambda *values: forwarded.append(values) or 0):
            self.assertEqual(b.run(args), 0)
        command = list(map(str, forwarded[0]))
        self.assertEqual(command[1], 'eval_gpt.py')
        self.assertEqual(command[command.index('--harness') + 1], 'browsercode')
        self.assertNotIn('--execute', command)
        self.assertFalse((self.root / 'out').exists())

    def test_only_preflight_inventory_changes_with_valid_hashes(self):
        runtime = self.root / 'prepared'
        runtime.mkdir()
        source = "def episode():\n return 'frozen actor, judge and execution'\ndef preflight(root,budget):\n for kind in ['cuawright','browsercode']:\n  print(kind)\n"
        worker = runtime / 'worker.py'
        worker.write_text(source)
        before_episode = ast.dump(ast.parse(source).body[0])
        self.write(runtime / 'portable-build.json', {'adaptations': []})
        self.write(runtime / 'source-manifest.json', {'source_hashes': {str(worker): b.digest(worker)}})
        b.browsercode_preflight_only(runtime)
        self.assertEqual(ast.dump(ast.parse(worker.read_text()).body[0]), before_episode)
        self.assertEqual(ast.literal_eval(ast.parse(worker.read_text()).body[1].body[0].iter), ['browsercode'])
        manifest = b.read(runtime / 'source-manifest.json')
        self.assertEqual(manifest, b.read(runtime / 'execution-source-manifest.json'))
        self.assertEqual(manifest['source_hashes'][str(worker)], b.digest(worker))
        self.assertEqual(len(b.read(runtime / 'portable-build.json')['adaptations']), 1)

    def test_historical_setup_pins_declared_viewport_before_actor_release(self):
        runtime = self.root / 'historical'
        runtime.mkdir()
        original = "def setup_native(local,kind,ws,item):\n  target=live_page(ws,navigate=item['start_url']);local.start(ws,target);result={'observation':{'success':True},'native_browsercode_bound':True}\n  return result\ndef preflight(root,budget):\n for kind in ['cuawright','browsercode']:\n  pass\n"
        worker = runtime / 'worker.py'
        worker.write_text(original)
        self.write(runtime / 'portable-build.json', {'adaptations': []})
        self.write(runtime / 'source-manifest.json', {'source_hashes': {str(worker): b.digest(worker)}})
        b.browsercode_preflight_only(runtime)
        setup = ast.parse(worker.read_text()).body[0]
        executions = []
        class Workspace:
            def start(self, ws, target): executions.append(('start', ws, target))
            def execute(self, action):
                executions.append(('execute', json.loads(action['command'])))
                return {'observation': {'success': True}, 'output': 'viewport-fixed'}
            def _workspace_dir(self): return runtime
        namespace = {'json': json, 'live_page': lambda ws, navigate: 'owned-target', 'write_json': lambda *args: None}
        exec(compile(ast.Module(body=[setup], type_ignores=[]), '<generated-setup>', 'exec'), namespace)
        result = namespace['setup_native'](Workspace(), 'browsercode', 'owned-websocket', {'start_url': 'data:text/html,ready'})
        self.assertEqual(executions[0], ('start', 'owned-websocket', 'owned-target'))
        request = executions[1][1]
        self.assertIn('session.Emulation.setDeviceMetricsOverride({width:1280,height:1800,deviceScaleFactor:1,mobile:false})', request['code'])
        self.assertEqual(request['timeout'], 30000)
        self.assertEqual(result['viewport'], '1280x1800')
        manifest = b.read(runtime / 'source-manifest.json')
        for path, expected in manifest['source_hashes'].items():
            self.assertEqual(b.digest(path), expected)
        self.assertEqual(manifest, b.read(runtime / 'execution-source-manifest.json'))

    def test_report_accepts_only_browsercode_dispositions(self):
        output = self.root / 'foreign'
        self.write(output / 'reproduction-plan.json', {'harness': 'cuawright'})
        args = b.parser().parse_args(['report', '--run', 'x=' + str(output), '--output', str(self.root / 'report.json')])
        with self.assertRaises(ValueError):
            b.report(args)
        self.assertFalse((self.root / 'report.json').exists())


@unittest.skipUnless(os.environ.get('OPENWEBRL_API_DEPENDENCIES'), 'Set OPENWEBRL_API_DEPENDENCIES for real public-dependency setup checks')
class ExistingDependenciesIntegration(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='browsercode-public-real-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.dependencies = Path(os.environ['OPENWEBRL_API_DEPENDENCIES'])
        self.python = b.read(self.dependencies)['python']

    def command(self, *flags):
        result = subprocess.run([self.python, str(SOURCE), *map(str, flags)], text=True, capture_output=True, env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'}, timeout=180)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result.stdout

    def test_setup_check_and_readonly_run_from_real_sources(self):
        for campaign, arm, provider in [('luna', 'luna_n5_medium', 'luna'), ('claude', 'sonnet_n5', 'claude'), ('gpt', 'serial_high', 'gpt'), ('gpt-luna-single', 'luna_single_high', 'gpt'), ('gpt-crossbench', 'parallel5_select_high', 'gpt'), ('sol61-pair20', 'parallel3_select_low', 'gpt')]:
            with self.subTest(campaign=campaign):
                workspace = self.root / campaign
                self.command('setup', '--workspace', workspace, '--campaign', campaign, '--dependencies', self.dependencies)
                self.assertFalse((workspace / 'dependencies').exists())
                checked = json.loads(self.command('check', '--workspace', workspace))
                self.assertEqual(checked['paid_requests'], 0)
                self.assertEqual(b.provider(workspace / 'runtime', arm), provider)
                output = self.root / ('output-' + campaign)
                before = {str(p): b.digest(p) for p in (workspace / 'runtime').iterdir() if p.is_file()}
                dry = self.command('run', '--workspace', workspace, '--arm', arm, '--output', output, '--model-usd', '1', '--judge-usd', '.1')
                self.assertIn('Dry run:', dry)
                self.assertIn('"harness": "browsercode"', dry)
                self.assertFalse(output.exists())
                self.assertEqual(before, {str(p): b.digest(p) for p in (workspace / 'runtime').iterdir() if p.is_file()})
                worker = ast.parse((workspace / 'runtime/worker.py').read_text())
                preflight = next(node for node in worker.body if isinstance(node, ast.FunctionDef) and node.name == 'preflight')
                checks = [ast.literal_eval(node.iter) for node in ast.walk(preflight) if isinstance(node, ast.For) and isinstance(node.iter, ast.List) and any(isinstance(item, ast.Constant) and item.value == 'browsercode' for item in node.iter.elts)]
                self.assertEqual(checks, [['browsercode']])
                if campaign == 'sol61-pair20':
                    self.assertIn('setDeviceMetricsOverride({width:1280,height:1800', (workspace / 'runtime/worker.py').read_text())


if __name__ == '__main__':
    unittest.main()
