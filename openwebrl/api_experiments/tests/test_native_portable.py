"""Offline portability checks; never contact a provider or launch Chromium."""
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

REPRODUCE = Path(__file__).resolve().parents[1] / 'reproduce'
sys.path.insert(0, str(REPRODUCE))
import native
import native_runtime


class PortableNativeTests(unittest.TestCase):
    def test_public_benchmark_ids_and_order(self):
        tasks = native.tasks_from_file(native.ROOT.parent / 'data/eval/online-mind2web.jsonl', limit=300)
        self.assertEqual(len(tasks), 300)
        self.assertEqual(len({t['task_id'] for t in tasks}), 300)
        for task in tasks:
            self.assertEqual(task['intent'], task['metadata']['intent'])
            self.assertEqual(task['start_url'], task['metadata']['start_url'])
        chosen = [tasks[14]['task_id'], tasks[1]['task_id']]
        selected = native.tasks_from_file(native.ROOT.parent / 'data/eval/online-mind2web.jsonl', chosen, 300)
        self.assertEqual([t['task_id'] for t in selected], [tasks[1]['task_id'], tasks[14]['task_id']])

    def test_dry_run_is_read_only_for_every_arm(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            chromium = root / 'chrome'
            chromium.write_text('#!/bin/sh\nexit 0\n')
            chromium.chmod(0o755)
            for arm in native.ARMS:
                output = root / arm
                with patch.object(native, 'verify_source'), patch.object(native, 'verify_browser'), contextlib.redirect_stdout(io.StringIO()) as stream:
                    code = native.main(['--arm', arm, '--output', str(output), '--cuawright-source', folder, '--webjudge-source', folder, '--chromium', str(chromium), '--browser-cache', folder])
                self.assertEqual(code, 0)
                self.assertFalse(output.exists())
                plan = json.loads(stream.getvalue())
                self.assertEqual(plan['mode'], 'read-only-dry-run')
                self.assertEqual(plan['plan']['max_output_tokens'], 20000)

    def test_execution_requires_all_caps_before_writing(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / 'run'
            with self.assertRaisesRegex(ValueError, 'explicit --model-usd'):
                native.main(['--arm', 'luna_high_single', '--execute', '--output', str(output), '--cuawright-source', folder, '--webjudge-source', folder, '--chromium', folder, '--browser-cache', folder])
            self.assertFalse(output.exists())

    def test_source_verification_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / 'source'
            source.mkdir()
            (source / 'file.py').write_text('x=1\n')
            expected = hashlib.sha256((source / 'file.py').read_bytes()).hexdigest()
            (root / 'fixture-source.json').write_text(json.dumps({'hashes': {'file.py': expected}}))
            with patch.object(native_runtime, 'SUPPORT', root):
                native_runtime.verify_source(source, 'fixture')
                (source / 'file.py').write_text('x=2\n')
                with self.assertRaisesRegex(ValueError, 'source mismatch'):
                    native_runtime.verify_source(source, 'fixture')

    def test_browser_verification_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as folder:
            cache = Path(folder)
            chromium = cache / 'chrome'
            chromium.write_bytes(b'pinned-browser-fixture')
            chromium.chmod(0o755)
            pin = hashlib.sha256(chromium.read_bytes()).hexdigest()
            with patch.object(native_runtime, 'CHROMIUM_SHA256', pin):
                native_runtime.verify_browser(chromium, cache)
                chromium.write_bytes(b'changed-browser-fixture')
                with self.assertRaisesRegex(ValueError, 'Chromium binary differs'):
                    native_runtime.verify_browser(chromium, cache)


    def test_transport_failure_blocks_all_extra_physical_dispatch(self):
        import time
        gateway = native.load(native.ROOT / 'cuawright/budgets/budget_gateway.py', '_native_test_gateway')
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            cap = {'shard': 0, 'task_ids': ['fixture'], 'model_max_inflight': 8, 'model_usd_all_attempts': 10, 'external_judge_usd_all_attempts': 1, 'model_calls_all_attempts': 20, 'external_judge_calls_all_attempts': 10, 'browser_starts_all_attempts': 2}
            (root / 'plan.json').write_text(json.dumps({'model': 'gpt-6-luna', 'candidate_count': 1, 'shards': [cap]}))
            (root / 'pricing-snapshot.json').write_text(json.dumps({'usd_per_million_tokens': {'input': .1, 'cached_input': .01, 'cache_write_input': .125, 'output': .5}}))
            service = gateway.ShardGateway(root, 0, time.time() + 30, 'not-a-provider-key', 'fixture-secret')
            payload = {'model': 'gpt-6-luna', 'input': [{'role': 'user', 'content': [{'type': 'input_text', 'text': 'fixture'}]}], 'reasoning': {'effort': 'high'}, 'max_output_tokens': 20000, 'tools': [{'type': 'function', 'name': 'run_command', 'parameters': {'type': 'object'}}]}
            with patch.object(gateway, '_forward', side_effect=TimeoutError) as forward:
                result = service.create('fixture', payload)
                self.assertFalse(result['usage_known'])
                # Native HTTP middleware may re-contact its loopback gateway;
                # every subsequent request must fail before the paid endpoint.
                for _ in range(6):
                    with self.assertRaisesRegex(ValueError, 'halted'):
                        service.create('fixture', payload)
                self.assertEqual(forward.call_count, 1)
            usage = gateway.totals(root, 0)
            self.assertEqual(usage['model']['calls'], 1)
            self.assertEqual(usage['model']['unknown'], 1)
            self.assertGreater(usage['model']['usd'], 0)

    def test_child_inventory_rejects_changed_source_without_scheduler(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            output = root / 'shard0/episodes/fixture'
            output.mkdir(parents=True)
            (root / 'source.py').write_text('x=1\n')
            (root / 'plan.json').write_text(json.dumps({'shards': [{'task_ids': ['fixture']}]}))
            inventory = {'passed': True, 'frozen_plan_sha256': native_runtime.digest(root / 'plan.json'), 'source_hashes': {'source.py': native_runtime.digest(root / 'source.py')}}
            (root / 'runtime-inventory.json').write_text(json.dumps(inventory))
            scope = {}
            exec('import os,json,hashlib\nfrom pathlib import Path\n' + native_runtime.PORTABLE_BINDING, scope)
            binding = scope['bind_runtime_inventory'](root, output)
            self.assertEqual(binding['schema'], 'cuawright-portable-child-source-v1')
            (root / 'source.py').write_text('x=2\n')
            with self.assertRaisesRegex(AssertionError, 'source changed'):
                scope['bind_runtime_inventory'](root, output)


if __name__ == '__main__':
    unittest.main()
