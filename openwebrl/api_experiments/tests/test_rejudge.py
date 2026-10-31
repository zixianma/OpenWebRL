"""Portable frozen-judge tests. Fixtures and HTTP responses are synthetic."""
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import threading
import types
import unittest
from unittest.mock import patch
import zlib

SOURCE = Path(__file__).resolve().parents[1] / 'reproduce/rejudge.py'
spec = importlib.util.spec_from_file_location('portable_rejudge_tests', SOURCE)
j = importlib.util.module_from_spec(spec)
spec.loader.exec_module(j)
endpoint_spec = importlib.util.spec_from_file_location('portable_archived_endpoint_tests', SOURCE.parent / 'archived_endpoint.py')
archived = importlib.util.module_from_spec(endpoint_spec)
endpoint_spec.loader.exec_module(archived)


def png(path, height=1800):
    def chunk(kind, value):
        return struct.pack('>I', len(value)) + kind + value + struct.pack('>I', zlib.crc32(kind + value))
    image = b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', 1280, height, 8, 0, 0, 0, 0))
    image += chunk(b'IDAT', zlib.compress((b'\0' + b'\xff' * 1280) * height)) + chunk(b'IEND', b'')
    path.write_bytes(image)


class RejudgeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='portable-terra-rejudge-', dir='/tmp')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.image = self.root / 'terminal.png'
        png(self.image)
        self.cohort = {'benchmark': 'synthetic', 'harness': 'browsercode', 'arm': 'fixture', 'actor_model': 'fixture-model', 'actor_protocol_sha256': '0' * 64}
        self.row = {'id': 'fixture-episode', 'original_valid': True, 'eligible': True, 'cohort': self.cohort, 'task': 'Synthetic fixture task', 'actions': ['synthetic selected action'],
                    'screenshot': 'terminal.png', 'screenshot_sha256': j.sha(self.image), 'endpoint_scope': 'dedicated_terminal_capture'}
        self.manifest = self.root / 'input.json'
        j.write(self.manifest, {'schema': 'terra-endpoint-input-v1', 'episodes': [self.row]})
        self.args = types.SimpleNamespace(manifest=self.manifest, output=self.root / 'output', judge_usd='1', judge_calls=2, wall_seconds=30, concurrency=2, execute=False, env_file=None)

    def prepare_worker(self):
        record = j.plan(self.args)
        self.args.output.mkdir()
        record.update(execute=True, output_root=str(self.args.output))
        j.write(self.args.output / 'rejudge-plan.json', record, True)
        return record

    def client(self, calls, failure=False):
        def create(**request):
            calls.append(request)
            if failure:
                raise RuntimeError('synthetic provider interruption')
            return {'model': 'gpt-5.6-terra', 'choices': [{'finish_reason': 'stop', 'message': {'content': 'Thoughts: fixture\nStatus: success', 'refusal': None}}],
                    'usage': {'prompt_tokens': 100, 'completion_tokens': 5, 'total_tokens': 105}}
        return types.SimpleNamespace(max_retries=0, close=lambda: None, chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create)))

    def test_bundled_science_and_prompt_match_provenance(self):
        judge, budget, hashes = j.science()
        self.assertEqual(judge.protocol()['reasoning_effort'], 'medium')
        self.assertEqual(judge.protocol()['max_completion_tokens'], 4096)
        self.assertEqual(judge.protocol()['seed'], 42)
        self.assertEqual(len(hashes), 4)
        for relative, expected in hashes.items():
            self.assertEqual(j.sha(j.ROOT / relative), expected)
        self.assertTrue(hasattr(budget, 'ReservationBudget'))

    def test_dry_run_never_creates_output_or_reads_credentials(self):
        with patch.object(j.subprocess, 'run', side_effect=AssertionError('must not launch worker')), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(j.run(self.args), 0)
        self.assertFalse(self.args.output.exists())

    def test_original_invalids_and_missing_endpoints_remain_separate(self):
        excluded = {'id': 'original-invalid', 'original_valid': False, 'eligible': False, 'exclusion_reason': 'original actor invalid', 'cohort': self.cohort}
        missing = dict(excluded, id='missing-evidence', original_valid=True, exclusion_reason='no terminal frame')
        j.write(self.manifest, {'schema': 'terra-endpoint-input-v1', 'episodes': [self.row, excluded, missing]})
        record = self.prepare_worker()
        calls = []
        self.assertEqual(j.perform(self.args.output, record, lambda: self.client(calls)), 0)
        cell = j.read(self.args.output / 'summary.json')['cells'][0]
        self.assertEqual((cell['requested'], cell['eligible'], cell['original_invalid_preserved'], cell['original_valid_ineligible']), (3, 1, 1, 1))
        self.assertEqual(len(calls), 1)

    def test_real_frozen_judge_and_ledger_settle_one_mock_response(self):
        record = self.prepare_worker()
        calls = []
        self.assertEqual(j.perform(self.args.output, record, lambda: self.client(calls)), 0)
        self.assertEqual(len(calls), 1)
        request = calls[0]
        self.assertEqual((request['model'], request['reasoning_effort'], request['max_completion_tokens'], request['seed']), ('gpt-5.6-terra', 'medium', 4096, 42))
        summary = j.read(self.args.output / 'summary.json')
        self.assertEqual(summary['all_attempt_judge_pool']['calls'], 1)
        self.assertEqual(summary['cells'][0]['successes'], 1)
        self.assertEqual(j.sha(self.image), self.row['screenshot_sha256'])
        self.assertEqual(len(list((self.args.output / 'episodes/fixture-episode/requests').glob('*.json'))), 1)

    def test_actual_openai_sdk_with_mock_transport_preserves_request_and_one_call(self):
        import httpx
        from openai import OpenAI
        calls = []
        def handler(request):
            calls.append(json.loads(request.content))
            return httpx.Response(200, json={'id': 'fixture-response', 'object': 'chat.completion', 'created': 0, 'model': 'gpt-5.6-terra',
                'choices': [{'index': 0, 'finish_reason': 'stop', 'message': {'role': 'assistant', 'content': 'Thoughts: fixture\nStatus: success'}}],
                'usage': {'prompt_tokens': 100, 'completion_tokens': 5, 'total_tokens': 105}})
        record = self.prepare_worker()
        factory = lambda: OpenAI(api_key='synthetic-fixture-key', max_retries=0, http_client=httpx.Client(transport=httpx.MockTransport(handler)))
        self.assertEqual(j.perform(self.args.output, record, factory), 0)
        self.assertEqual(len(calls), 1)
        self.assertEqual((calls[0]['model'], calls[0]['reasoning_effort'], calls[0]['seed'], calls[0]['max_completion_tokens']), ('gpt-5.6-terra', 'medium', 42, 4096))

    def test_real_public_frontdoor_dry_run_needs_no_workspace_or_scheduler(self):
        command = [sys.executable, '-B', str(SOURCE.parent / 'browsercode.py'), 'rejudge', '--manifest', str(self.manifest), '--output', str(self.args.output),
                   '--judge-usd', '1', '--judge-calls', '2', '--wall-seconds', '30', '--concurrency', '2']
        result = subprocess.run(command, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        value = json.loads(result.stdout)
        self.assertFalse(value['execute'])
        self.assertEqual(value['eligible'], 1)
        self.assertFalse(self.args.output.exists())

    def test_full_unknown_reservation_and_claim_survive_failed_http(self):
        record = self.prepare_worker()
        calls = []
        self.assertEqual(j.perform(self.args.output, record, lambda: self.client(calls, True)), 1)
        ledger = j.read(self.args.output / 'budget/judge/ledger.json')
        request = next(iter(ledger['requests'].values()))
        self.assertEqual(request['status'], 'usage_unknown')
        self.assertEqual(request['charged_or_reserved_usd'], request['reserved_usd'])
        self.assertTrue((self.args.output / 'episodes/fixture-episode/claimed.json').exists())
        with self.assertRaises(FileExistsError):
            j.perform(self.args.output, record, lambda: self.client(calls))
        self.assertEqual(len(calls), 1)

    def test_insufficient_reservation_cap_never_dispatches_http(self):
        self.args.judge_usd = '.000001'
        record = self.prepare_worker()
        calls = []
        self.assertEqual(j.perform(self.args.output, record, lambda: self.client(calls)), 1)
        self.assertEqual(calls, [])
        self.assertEqual(j.read(self.args.output / 'budget/judge/ledger.json')['calls'] if (self.args.output / 'budget/judge/ledger.json').exists() else 0, 0)

    def test_unsafe_duplicate_and_original_invalid_inputs_rejected(self):
        for rows in ([self.row, self.row], [dict(self.row, id='../escape')], [dict(self.row, original_valid=False)]):
            j.write(self.manifest, {'schema': 'terra-endpoint-input-v1', 'episodes': rows})
            with self.assertRaises(ValueError):
                j.plan(self.args)

    def test_original_shorter_png_preserved_and_last_action_requires_binding(self):
        png(self.image, 1657)
        self.row.update(screenshot_sha256=j.sha(self.image), endpoint_scope='last_browser_action_capture')
        j.write(self.manifest, {'schema': 'terra-endpoint-input-v1', 'episodes': [self.row]})
        with self.assertRaises(ValueError):
            j.plan(self.args)
        self.row['last_action_capture_verified'] = True
        j.write(self.manifest, {'schema': 'terra-endpoint-input-v1', 'episodes': [self.row]})
        self.assertEqual(j.plan(self.args)['episodes'][0]['screenshot_dimensions'], [1280, 1657])

    def test_modified_evidence_and_nonpositive_caps_rejected(self):
        record = j.plan(self.args)
        self.image.write_bytes(b'changed')
        with self.assertRaises(ValueError):
            j.check_inputs(record)
        for name in ('judge_calls', 'wall_seconds', 'concurrency'):
            values = types.SimpleNamespace(**vars(self.args))
            setattr(values, name, 0)
            with self.assertRaises(ValueError):
                j.plan(values)

    def test_public_execution_wall_timeout_awaits_worker_and_keeps_reserves(self):
        self.args.execute = True
        with patch.object(j.subprocess, 'run', side_effect=j.subprocess.TimeoutExpired('synthetic-worker', 30)), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(j.run(self.args), 1)
        evidence = j.read(self.args.output / 'execution.json')
        self.assertTrue(evidence['wall_deadline_terminated_worker'])
        self.assertTrue(evidence['unknown_or_inflight_reservations_never_released'])
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(ValueError):
            j.run(self.args)

    def test_call_cap_keeps_unattempted_evidence_missing(self):
        rows = [dict(self.row, id='fixture-' + str(index)) for index in range(3)]
        j.write(self.manifest, {'schema': 'terra-endpoint-input-v1', 'episodes': rows})
        self.args.judge_calls = 1
        record = self.prepare_worker()
        calls = []
        self.assertEqual(j.perform(self.args.output, record, lambda: self.client(calls)), 1)
        summary = j.read(self.args.output / 'summary.json')
        self.assertEqual(len(calls), 1)
        self.assertEqual(summary['all_attempt_judge_pool']['calls'], 1)
        self.assertEqual(summary['cells'][0]['missing_judgments'], 2)

    def test_independent_requests_overlap_without_exceeding_explicit_concurrency(self):
        rows = [dict(self.row, id='fixture-' + str(index)) for index in range(4)]
        j.write(self.manifest, {'schema': 'terra-endpoint-input-v1', 'episodes': rows})
        self.args.judge_calls = 4
        record = self.prepare_worker()
        barrier = threading.Barrier(2)
        lock = threading.Lock()
        calls, occupied, peaks = [], [0], []
        def factory():
            client = self.client(calls)
            original = client.chat.completions.create
            def create(**request):
                with lock:
                    occupied[0] += 1
                    peaks.append(occupied[0])
                try:
                    barrier.wait(timeout=5)
                    return original(**request)
                finally:
                    with lock:
                        occupied[0] -= 1
            client.chat.completions.create = create
            return client
        self.assertEqual(j.perform(self.args.output, record, factory), 0)
        self.assertEqual(len(calls), 4)
        self.assertEqual(max(peaks), 2)

    def native_fixture(self, manual=False):
        task, action, answer = 'Synthetic archived task', 'Synthetic selected action', 'Synthetic saved answer'
        step = {'browser_step': 1, 'action': action, 'screenshot_path': 'terminal.png', 'screenshot_error': None}
        if manual:
            code = self.root / 'steps/browser_step_0001.py'
            code.parent.mkdir()
            code.write_text("await p.goto('synthetic-page')\nawait p.screenshot(path=str(workspace/'terminal.png'))\nprint('saved')\n")
            step.update(screenshot_path=None, code_path='steps/browser_step_0001.py', code_sha256=archived.sha(code))
        (self.root / 'browser-steps.jsonl').write_text(json.dumps(step) + '\n')
        j.write(self.root / 'trajectory.json', {'info': {'submission': answer}})
        j.write(self.root / 'result.json', {'native_execution_complete': True, 'exit_status': 'Submitted'})
        evidence = {'task': task, 'browser_steps_sha256': archived.sha(self.root / 'browser-steps.jsonl'),
                    'trajectory_sha256': archived.sha(self.root / 'trajectory.json'), 'action_history': [action, 'Final answer: ' + answer],
                    'captures': [{'browser_step': 1, 'path': str(self.image), 'sha256': archived.sha(self.image)}]}
        j.write(self.root / 'external-judge-evidence.json', evidence)
        return task

    def test_exact_published_endpoint_algorithm_provenance(self):
        self.assertEqual(archived.sha(SOURCE.parent / 'archived_endpoint.py'), '2928f64f9988ab69197974d8018f7a7b3127ac2917457b304853720297933101')

    def test_native_helper_preserves_geometry_selected_history_and_all_bindings(self):
        png(self.image, 1657)
        task = self.native_fixture()
        before = archived.sha(self.image)
        value, bindings = archived.native_endpoint(self.root, task, {'valid': True})
        self.assertEqual(value['screenshot_dimensions'], [1280, 1657])
        self.assertEqual(value['actions'], ['Synthetic selected action', 'Final answer: Synthetic saved answer'])
        self.assertEqual(value['screenshot_sha256'], before)
        self.assertEqual(archived.sha(self.image), before)
        self.assertTrue(value['archival_geometry_preserved_no_resize'])
        for path, expected in bindings.items():
            self.assertEqual(archived.sha(path), expected)
        self.assertIn(str(self.root / 'browser-steps.jsonl'), bindings)

    def test_native_helper_accepts_final_manual_capture_and_rejects_same_line_side_effect(self):
        task = self.native_fixture(manual=True)
        value, bindings = archived.native_endpoint(self.root, task, {'valid': True})
        self.assertTrue(value['manual_capture_binding']['no_later_browser_calls'])
        code = self.root / 'steps/browser_step_0001.py'
        self.assertIn(str(code), bindings)
        code.write_text("await p.screenshot(path=str(workspace/'terminal.png')); subprocess.run(['synthetic-later-mutation'])\n")
        steps = self.root / 'browser-steps.jsonl'
        step = json.loads(steps.read_text())
        step['code_sha256'] = archived.sha(code)
        steps.write_text(json.dumps(step) + '\n')
        evidence_path = self.root / 'external-judge-evidence.json'
        evidence = j.read(evidence_path)
        evidence['browser_steps_sha256'] = archived.sha(steps)
        j.write(evidence_path, evidence)
        with self.assertRaisesRegex(ValueError, 'No unambiguous terminal capture'):
            archived.native_endpoint(self.root, task, {'valid': True})

    def test_native_helper_rejects_earlier_capture_and_changed_task_or_raw_endpoint(self):
        task = self.native_fixture()
        with self.assertRaisesRegex(ValueError, 'identity differs'):
            archived.native_endpoint(self.root, 'different task', {'valid': True})
        with self.assertRaisesRegex(ValueError, 'Original canonical invalid'):
            archived.native_endpoint(self.root, task, {'valid': False})
        j.write(self.root / 'result.json', {'native_execution_complete': False, 'exit_status': 'Submitted'})
        with self.assertRaisesRegex(ValueError, 'actor endpoint incomplete'):
            archived.native_endpoint(self.root, task, {'valid': True})
        j.write(self.root / 'result.json', {'native_execution_complete': True, 'exit_status': 'Submitted'})
        steps = self.root / 'browser-steps.jsonl'
        with steps.open('a') as stream:
            stream.write(json.dumps({'browser_step': 2, 'action': 'later browser action', 'screenshot_path': None}) + '\n')
        evidence_path = self.root / 'external-judge-evidence.json'
        evidence = j.read(evidence_path)
        evidence['browser_steps_sha256'] = archived.sha(steps)
        evidence['action_history'].insert(1, 'later browser action')
        j.write(evidence_path, evidence)
        with self.assertRaisesRegex(ValueError, 'No PNG from last logged browser interaction'):
            archived.native_endpoint(self.root, task, {'valid': True})

    def test_native_helper_rejects_capture_outside_owned_episode(self):
        task = self.native_fixture()
        steps = self.root / 'browser-steps.jsonl'
        step = json.loads(steps.read_text())
        step['screenshot_path'] = '../unowned-terminal.png'
        steps.write_text(json.dumps(step) + '\n')
        evidence_path = self.root / 'external-judge-evidence.json'
        evidence = j.read(evidence_path)
        evidence['browser_steps_sha256'] = archived.sha(steps)
        j.write(evidence_path, evidence)
        with self.assertRaisesRegex(ValueError, 'escapes episode'):
            archived.native_endpoint(self.root, task, {'valid': True})


if __name__ == '__main__':
    unittest.main()
