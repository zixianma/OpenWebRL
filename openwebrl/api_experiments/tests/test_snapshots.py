"""Exercise saved-state bindings and real frozen protocol loops without network."""
import argparse
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('portable_snapshots', ROOT / 'reproduce/snapshots.py')
SNAPSHOTS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SNAPSHOTS)


def selector_response(index):
    return {'model': 'gpt-6.1-sol', 'status': 'completed', 'output': [
        {'type': 'message', 'content': [{'type': 'output_text', 'text': json.dumps({'selected_index': index})}]}],
        'usage': {'input_tokens': 100, 'output_tokens': 10, 'input_tokens_details': {'cached_tokens': 0}, 'output_tokens_details': {'reasoning_tokens': 0}}}


@unittest.skipUnless(importlib.util.find_spec('jsonschema'), 'requires documented API dependencies')
class SnapshotTests(unittest.TestCase):
    def test_calibration_and_medium_execute_frozen_requests_without_browser(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            image = directory / 'state.png'
            image.write_bytes(b'\x89PNG\r\n\x1a\nsynthetic-evidence')
            tool = {'type': 'function', 'function': {'name': 'done', 'parameters': {'type': 'object', 'properties': {'response': {'type': 'string'}}, 'required': ['response']}}}
            manifest = directory / 'input.json'
            manifest.write_text(json.dumps({'states': [{'state_id': 'state_01', 'prompt': '<tools>\n' + json.dumps(tool) + '\n</tools><|im_end|>\nuser: Return ready', 'images': [{'path': str(image), 'sha256': hashlib.sha256(image.read_bytes()).hexdigest(), 'format': 'png'}]}]}))
            runtime = directory / 'calibration'
            SNAPSHOTS.prepare_calibration(manifest, runtime)
            calls = []
            def urlopen(request, timeout):
                body = json.loads(request.data)
                calls.append(body)
                if 'text' in body:
                    result = selector_response(1)
                    result['output'][0]['content'][0]['text'] = '{"selection":1}'
                else:
                    result = selector_response(0)
                    result['output'] = [{'type': 'function_call', 'name': 'done', 'call_id': 'test-call', 'arguments': '{"response":"ready"}'}]
                return io.BytesIO(json.dumps(result).encode())
            args = argparse.Namespace(runtime=runtime, execute=True, model_usd=20., model_calls=5)
            with mock.patch.dict('os.environ', {'OPENAI_API_KEY': 'offline-no-network'}), mock.patch('urllib.request.urlopen', side_effect=urlopen):
                report = SNAPSHOTS.run_calibration(args)
                self.assertTrue(report['complete'])
                self.assertEqual(len(calls), 5)
                actor = [row for row in calls if 'tools' in row]
                self.assertEqual({row['max_output_tokens'] for row in actor}, {4096, 12288})
                self.assertTrue(all(row['input'] == actor[0]['input'] for row in actor))
                medium = directory / 'medium'
                SNAPSHOTS.prepare_medium(runtime, medium)
                SNAPSHOTS.run_medium(argparse.Namespace(runtime=medium, execute=True, model_usd=5., model_calls=1))
                self.assertEqual(calls[-1]['reasoning'], {'effort': 'medium'})
                self.assertEqual(calls[-1]['max_output_tokens'], 4096)
                self.assertEqual(calls[-1]['input'], actor[0]['input'])
                with self.assertRaises(ValueError):
                    SNAPSHOTS.run_calibration(args)

    @unittest.skipUnless(importlib.util.find_spec('openai'), 'requires documented OpenAI SDK')
    def test_selector_pair_preserves_evidence_and_changes_only_effort(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            episode = root / 'episodes/episode01'
            decision = episode / 'decisions/000.json'
            receipt = episode / 'requests/selector.json'
            response = selector_response(0)
            request = {'model': 'gpt-6.1-sol', 'reasoning': {'effort': 'medium'}, 'max_output_tokens': 2048, 'store': False, 'service_tier': 'default', 'input': [{'role': 'user', 'content': 'Choose an action'}], 'text': {'format': {'type': 'json_schema', 'name': 'action_selection', 'strict': True, 'schema': {'type': 'object', 'properties': {'selected_index': {'type': 'integer', 'enum': [0, 1, 2]}}, 'required': ['selected_index'], 'additionalProperties': False}}}}
            SNAPSHOTS.write(decision, {'status': 'accepted', 'task_id': 'public-task', 'turn': 0, 'selected_index': 0, 'candidates': [{'name': 'done', 'arguments': {'response': str(i)}} for i in range(3)], 'selector_response': response})
            SNAPSHOTS.write(receipt, {'physical_role': 'selector', 'request': request, 'response': response})
            runtime = root / 'probe'
            SNAPSHOTS.prepare_selector(root / 'episodes', runtime, 1)
            captured = []
            class Client:
                def __init__(self, **kwargs):
                    self.responses = SimpleNamespace(create=self.create)
                    self.assert_retry = kwargs['max_retries']
                async def create(self, **kwargs):
                    captured.append(kwargs)
                    result = selector_response(int(kwargs['reasoning']['effort'] == 'medium'))
                    return SimpleNamespace(model_dump=lambda: result)
                async def close(self):
                    pass
            with mock.patch.dict('os.environ', {'OPENAI_API_KEY': 'offline-no-network'}), mock.patch('openai.AsyncOpenAI', Client):
                result = SNAPSHOTS.run_selector(argparse.Namespace(runtime=runtime, execute=True, model_usd=5., model_calls=2))
            self.assertEqual(result['pairs'], 1)
            self.assertEqual(result['paired_index_agreement'], 0)
            left, right = (dict(row) for row in captured)
            left.pop('reasoning'); right.pop('reasoning')
            self.assertEqual(left, right)
            self.assertEqual(left, {k: v for k, v in request.items() if k != 'reasoning'})


if __name__ == '__main__':
    unittest.main()
