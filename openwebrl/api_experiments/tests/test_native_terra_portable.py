"""Offline trusted-native/Terra regressions. No real browser or provider calls."""
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import struct
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zlib

REPRODUCE = Path(__file__).resolve().parents[1] / 'reproduce'
sys.path.insert(0, str(REPRODUCE))
import native
import native_runtime
import native_terra


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def png(path):
    def chunk(kind, value):
        return struct.pack('>I', len(value)) + kind + value + struct.pack('>I', zlib.crc32(kind + value))
    raw = b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', 1280, 1800, 8, 0, 0, 0, 0))
    raw += chunk(b'IDAT', zlib.compress((b'\0' + b'\xff' * 1280) * 1800)) + chunk(b'IEND', b'')
    path.write_bytes(raw)


class Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='portable-native-terra-', dir='/tmp')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.image = self.root / 'terminal.png'
        png(self.image)

    def arguments(self, output, *extra):
        return ['--arm', 'luna_high_single', '--output', str(output), '--cuawright-source', str(self.root),
                '--webjudge-source', str(self.root), '--chromium', str(self.root / 'chrome'),
                '--browser-cache', str(self.root), *extra]

    def fake_endpoint(self, runtime, directory, task):
        return {'id': task['task_id'], 'original_valid': True, 'eligible': True,
                'cohort': {'benchmark': 'synthetic', 'harness': 'native', 'arm': 'fixture',
                           'actor_model': 'fixture', 'actor_protocol_sha256': '0' * 64},
                'task': 'Synthetic test only', 'actions': ['Synthetic selected action'],
                'screenshot': str(self.image), 'screenshot_sha256': native_terra.sha(self.image),
                'endpoint_scope': 'dedicated_terminal_capture'}

    def fake_client(self, calls, fail=False):
        def create(**request):
            calls.append(request)
            if fail:
                raise RuntimeError('Synthetic provider interruption')
            return {'model': 'gpt-5.6-terra',
                    'choices': [{'finish_reason': 'stop', 'message': {'content': 'Thoughts: fixture\nStatus: success', 'refusal': None}}],
                    'usage': {'prompt_tokens': 100, 'completion_tokens': 5, 'total_tokens': 105}}
        return SimpleNamespace(max_retries=0, close=lambda: None,
                               chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    def test_default_webjudge_credits_and_plan_preserved(self):
        output = self.root / 'default-run'
        with patch.object(native, 'verify_source'), patch.object(native, 'verify_browser'), \
             patch.object(native, 'verify_terminal_sources', side_effect=AssertionError('default must not opt in')), \
             contextlib.redirect_stdout(io.StringIO()) as stream:
            self.assertEqual(native.main(self.arguments(output, '--judge-usd', '2', '--judge-calls', '3')), 0)
        plan = json.loads(stream.getvalue())['plan']
        self.assertEqual(plan['judge'], 'official-webjudge-gpt-4o-2024-08-06')
        self.assertEqual(plan['shards'][0]['external_judge_usd_all_attempts'], 2)
        self.assertEqual(plan['shards'][0]['external_judge_calls_all_attempts'], 3)
        self.assertFalse(output.exists())

    def test_terra_dry_run_has_one_explicit_pool_and_zero_webjudge_credits(self):
        output = self.root / 'terra-plan'
        with patch.object(native, 'verify_source'), patch.object(native, 'verify_browser'), \
             contextlib.redirect_stdout(io.StringIO()) as stream:
            self.assertEqual(native.main(self.arguments(output, '--judge', 'terra', '--judge-usd', '2', '--judge-calls', '3')), 0)
        plan = json.loads(stream.getvalue())['plan']
        self.assertEqual(plan['shared_terra_pool'], {'judge_usd': 2, 'judge_calls': 3})
        self.assertEqual(plan['shards'][0]['external_judge_usd_all_attempts'], 0)
        self.assertEqual(plan['shards'][0]['external_judge_calls_all_attempts'], 0)
        self.assertEqual(plan['judge']['model'], 'gpt-5.6-terra')
        self.assertFalse(output.exists())

    def test_missing_caps_fail_before_science_or_any_runtime_write(self):
        output = self.root / 'no-caps'
        with patch.object(native_terra, 'verify_science', side_effect=AssertionError('No source/API stage yet')), \
             patch.object(native, 'build_runtime', side_effect=AssertionError('No writes yet')):
            with self.assertRaisesRegex(ValueError, 'explicit --model-usd'):
                native.main(self.arguments(output, '--judge', 'terra', '--execute'))
        self.assertFalse(output.exists())

    def test_source_tamper_fails_before_runtime_or_provider_access(self):
        package = self.root / 'support/terminal_observer'
        package.mkdir(parents=True)
        (package / 'helper.py').write_text('expected source')
        expected = native_runtime.digest(package / 'helper.py')
        (package / 'source-pins.json').write_text(json.dumps({'source_hashes': {'helper.py': expected}}))
        with patch.object(native_runtime, 'SUPPORT', package.parent):
            native_runtime.verify_terminal_sources()
            (package / 'helper.py').write_text('tampered')
            with self.assertRaisesRegex(ValueError, 'observer source mismatch'):
                native_runtime.verify_terminal_sources()

    def test_observer_bundle_exact_validated_bytes_and_preload_order(self):
        bundle = native_runtime.SUPPORT / 'terminal_observer'
        pins = native_runtime.verify_terminal_sources()
        self.assertEqual(pins['source_hashes']['trusted_terminal_capture.py'],
                         '2ed79556d2aa7c7447e64d9c053a24ebbb6006255934ed4e429483475872716b')
        target = self.root / 'observer-runtime'
        target.mkdir()
        archive = native.ROOT / 'cuawright/runtime'
        original_hashes = {}
        for name in ('browser_broker.py', 'native_episode.py', 'diagnostic_browser.py'):
            shutil.copyfile(archive / name, target / name)
            original_hashes[name] = native_runtime.digest(archive / name)
        installer = module(bundle / 'materialize.py', 'portable_observer_test_installer')
        installer.materialize(target)
        installer.install_probe(target)
        source = (target / 'native_episode.py').read_text()
        self.assertLess(source.index('from trusted_terminal_capture import request_capture'), source.index('abi = restrict'))
        self.assertEqual(source.count('from trusted_terminal_capture import request_capture'), 1)
        for name, expected in original_hashes.items():
            self.assertEqual(native_runtime.digest(archive / name), expected)
        self.assertNotIn('/gpfs', (bundle / 'materialize.py').read_text())

    def test_unauthorized_capture_rejected_without_observer_or_api(self):
        helper = module(native_runtime.SUPPORT / 'terminal_observer/trusted_terminal_capture.py', 'portable_observer_auth_fixture')
        def deny(*args):
            raise PermissionError('not original native owner')
        broker = SimpleNamespace(command_ownership=SimpleNamespace(authorize=deny))
        observed = []
        with self.assertRaises(PermissionError):
            helper.capture(broker, 1, 1, 1, observer=lambda *args: observed.append(args))
        self.assertFalse(observed)

    def test_untrusted_capture_cannot_use_saved_webjudge_label(self):
        runtime = self.root / 'invalid-endpoint'
        output = runtime / 'shard0/episodes/fixture'
        output.mkdir(parents=True)
        (output / 'result.json').write_text(json.dumps({'native_execution_complete': False, 'error': None, 'exit_status': 'Submitted'}))
        (output / 'external-judge.json').write_text(json.dumps({'valid': True, 'strict_label': 1}))
        with self.assertRaisesRegex(ValueError, 'Native execution incomplete'):
            native_terra.endpoint(runtime, output, {'task_id': 'fixture', 'intent': 'Synthetic'})

    def test_two_tasks_use_exactly_one_shared_judge_ledger(self):
        runtime = self.root / 'shared'
        runtime.mkdir()
        calls = []
        session = native_terra.TerraSession(runtime, '1', 2, 'synthetic-key', client_factory=lambda: self.fake_client(calls))
        for task_id in ('first', 'second'):
            directory = runtime / task_id
            directory.mkdir()
            verdict = session.judge_once({'task_id': task_id}, directory, endpoint_function=self.fake_endpoint)
            self.assertTrue(verdict['valid'])
        self.assertEqual(len(calls), 2)
        self.assertEqual(session.budget.snapshot()['calls'], 2)
        self.assertEqual(len(list(runtime.glob('terra-budget/ledger.json'))), 1)
        self.assertFalse(session.available())
        for request in calls:
            self.assertEqual((request['model'], request['reasoning_effort'], request['seed'], request['max_completion_tokens']),
                             ('gpt-5.6-terra', 'medium', 42, 4096))

    def test_failure_has_one_physical_call_no_sdk_retry_and_retains_reserve(self):
        runtime = self.root / 'failure'
        runtime.mkdir()
        calls = []
        session = native_terra.TerraSession(runtime, '1', 2, 'synthetic-key', client_factory=lambda: self.fake_client(calls, fail=True))
        directory = runtime / 'first'
        directory.mkdir()
        result = session.judge_once({'task_id': 'first'}, directory, endpoint_function=self.fake_endpoint)
        self.assertFalse(result['valid'])
        self.assertEqual(len(calls), 1)
        state = session.budget.snapshot()
        self.assertEqual(state['calls'], 1)
        self.assertEqual(state['unknown_calls'], 1)
        self.assertGreater(float(state['charged_or_reserved_usd']), 0)
        with self.assertRaisesRegex(ValueError, 'pool halted'):
            session.judge_once({'task_id': 'second'}, runtime / 'second', endpoint_function=self.fake_endpoint)
        self.assertEqual(len(calls), 1)

    def test_unqualified_endpoint_fails_before_client_creation(self):
        runtime = self.root / 'unqualified'
        runtime.mkdir()
        def no_client():
            raise AssertionError('Client must not be created')
        session = native_terra.TerraSession(runtime, '1', 1, 'synthetic-key', client_factory=no_client)
        def fail(*args):
            raise ValueError('No trusted capture')
        with self.assertRaisesRegex(ValueError, 'No trusted capture'):
            session.judge_once({'task_id': 'fixture'}, runtime / 'fixture', endpoint_function=fail)
        self.assertEqual(session.budget.snapshot()['calls'], 0)

    def test_gpfs_home_write_rejected_before_any_source_materialization(self):
        with patch.object(native_runtime, 'verify_source', side_effect=AssertionError('No source/writes allowed')):
            with self.assertRaisesRegex(ValueError, 'project or temporary'):
                native_runtime.build_runtime('/gpfs/home/readonly-fixture/do-not-create', '.', '.', '.', '.', {})

    def test_terminal_runtime_cannot_also_fund_webjudge_gateway(self):
        output = self.root / 'double-funded'
        plan = {'shards': [{'external_judge_usd_all_attempts': 1,
                            'external_judge_calls_all_attempts': 1}]}
        with patch.object(native_runtime, 'verify_source'), patch.object(native_runtime, 'verify_browser'):
            with self.assertRaisesRegex(ValueError, 'zero external-WebJudge gateway credits'):
                native_runtime.build_runtime(output, '.', '.', '.', '.', plan, terminal_capture=True)
        self.assertFalse(output.exists())

    def test_expired_wall_cap_prevents_endpoint_and_client_work(self):
        runtime = self.root / 'expired'
        runtime.mkdir()
        session = native_terra.TerraSession(runtime, '1', 1, 'synthetic-key')
        def forbidden(*args):
            raise AssertionError('No endpoint or provider work after deadline')
        with self.assertRaisesRegex(ValueError, 'wall deadline'):
            session.judge_once({'task_id': 'fixture'}, runtime / 'fixture', endpoint_function=forbidden,
                               deadline=time.time() - 1)
        self.assertEqual(session.budget.snapshot()['calls'], 0)

    def test_wall_deadline_interrupts_mock_provider_and_retains_reserve(self):
        runtime = self.root / 'wall-bound'
        runtime.mkdir()
        calls = []
        def create(**request):
            calls.append(request)
            time.sleep(1)
            raise AssertionError('Absolute deadline should interrupt this mock')
        client = SimpleNamespace(max_retries=0, close=lambda: None,
            chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        session = native_terra.TerraSession(runtime, '1', 1, 'synthetic-key', client_factory=lambda: client)
        directory = runtime / 'fixture'
        directory.mkdir()
        started = time.monotonic()
        verdict = session.judge_once({'task_id': 'fixture'}, directory, endpoint_function=self.fake_endpoint,
                                    deadline=time.time() + .1)
        self.assertLess(time.monotonic() - started, .5)
        self.assertFalse(verdict['valid'])
        self.assertEqual(len(calls), 1)
        self.assertEqual(session.budget.snapshot()['unknown_calls'], 1)


if __name__ == '__main__':
    unittest.main()
