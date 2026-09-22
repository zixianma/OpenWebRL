import copy
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch
import contextlib
import io
import json
import tempfile
import urllib.error

spec = importlib.util.spec_from_file_location('jev_audit', Path(__file__).resolve().parents[1]/'scripts/audit_arm_task_quality_jev.py')
jev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(jev)


class JevAuditTests(unittest.TestCase):
    def setUp(self):
        self.request = jev.payload({'task_name': 'Find any vegetarian recipe.', 'website': 'https://example.org',
                                    'evaluator_reference': [{'facts': ['hidden condition']} ]})
        self.response = {'model': jev.MODEL, 'answers': {k: {'type': 'choice', 'choice': 'clear',
            'probabilities': {'clear': .95, 'problem': .02, 'uncertain': .03}, 'confidence': .9}
            for k in self.request['questions']}}

    def test_rubric_cannot_repair_actor_instruction(self):
        self.assertNotIn('hidden condition', str(self.request['state']))
        for key, question in self.request['questions'].items():
            self.assertEqual('hidden condition' in str(question), key in {'rubric_alignment', 'redundant_rubric'})

    def test_pilot_never_automatically_accepts_or_rejects(self):
        self.assertEqual(jev.triage(self.response)['decision'], 'review')
        self.response['answers']['single_episode']['choice'] = 'problem'
        result = jev.triage(self.response)
        self.assertEqual(result['decision'], 'review')
        self.assertIn('single_episode', result['reason_codes'])

    def test_response_validation(self):
        jev.validate_response(self.response, self.request)
        for mutation in ['missing', 'nan', 'model', 'not_argmax']:
            response = copy.deepcopy(self.response)
            if mutation == 'missing': response['answers'].pop('self_contained')
            if mutation == 'nan': response['answers']['self_contained']['confidence'] = float('nan')
            if mutation == 'model': response['model'] = 'jev-other'
            if mutation == 'not_argmax': response['answers']['self_contained']['choice'] = 'problem'
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                jev.validate_response(response, self.request)

    def test_cache_invalidation(self):
        changed = copy.deepcopy(self.request)
        changed['state']['instruction'] = 'Different task'
        self.assertNotEqual(jev.digest(self.request), jev.digest(changed))

    def test_execute_cap_and_resume_without_duplicate_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root/'tasks.jsonl'
            source.write_text('\n'.join(json.dumps({'task_id': str(i), 'task_name': 'Find a recipe',
                'website': 'https://example.org'}) for i in range(2)))
            argv = ['audit', '--execute', '--input', str(source), '--output', str(root/'out'),
                    '--limit', '2', '--max-api-calls', '1']
            with patch('sys.argv', argv), patch.dict(jev.os.environ, {'TYPESAFE_API_KEY': 'mock'}), \
                    patch.object(jev, 'api_call', return_value=self.response) as call, contextlib.redirect_stdout(io.StringIO()):
                jev.main()
                self.assertEqual(call.call_count, 1)
                self.assertEqual(json.loads((root/'out/summary.json').read_text())['processed_tasks'], 1)
                jev.main()
                self.assertEqual(call.call_count, 2)
                self.assertEqual(json.loads((root/'out/summary.json').read_text())['processed_tasks'], 2)
                jev.main()
                self.assertEqual(call.call_count, 2)

    def test_auth_failure_stops_and_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root/'tasks.jsonl'
            source.write_text(json.dumps({'task_id': '1', 'task_name': 'Find a recipe', 'website': 'https://example.org'}))
            argv = ['audit', '--execute', '--input', str(source), '--output', str(root/'out')]
            error = urllib.error.HTTPError(jev.ENDPOINT, 401, 'Unauthorized', {}, None)
            with patch('sys.argv', argv), patch.dict(jev.os.environ, {'TYPESAFE_API_KEY': 'mock'}), \
                    patch.object(jev, 'api_call', side_effect=error) as call, contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(SystemExit) as stopped:
                    jev.main()
                self.assertEqual(stopped.exception.code, 2)
                self.assertEqual(call.call_count, 1)
                audit = json.loads((root/'out/review.json').read_text())[0]['audit']
                self.assertEqual(audit['triage']['decision'], 'review')
                self.assertEqual(audit['http_status'], 401)


if __name__ == '__main__':
    unittest.main()
