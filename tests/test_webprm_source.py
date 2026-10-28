"""Pure upstream inference parity; no models, browsers or network."""
import ast
from abc import ABC, abstractmethod
import math
from pathlib import Path
import re
import tempfile
import unittest

from openwebrl.webprm_source import (
    load_arbiter, load_arbiter_benchmark_prompts, load_shepherd, source_manifest,
)

SHEPHERD = Path('/tmp/openwebrl-webshepherd-upstream')
ARBITER = Path('/tmp/openwebrl-webarbiter-upstream')


@unittest.skipUnless(SHEPHERD.exists(), 'Reviewed upstream checkout unavailable')
class ShepherdParity(unittest.TestCase):
    def test_prompt_and_score(self):
        api = load_shepherd(SHEPHERD)
        info = dict(intent='Find a book', trajectory='', current_url='https://example.org',
                    text_observation='[a1] textbox Search', checklist='Checklist 1: Search',
                    thought='Search for the book', action="fill('a1', 'book')", start_url='https://example.org')
        messages = api.get_messages(info, 'judge_progress', 'web_shepherd', text_obs=True, image_obs=False)
        prompt_ns = {'ABC': ABC, 'abstractmethod': abstractmethod}
        for filename in ['action.py', 'eval_type.py', 'input_information.py',
                         'judge_prompt.py', 'checklist_prompt.py', 'construct_messages.py']:
            source = SHEPHERD / 'src/webprm/prompts' / filename
            parsed = ast.parse(source.read_text())
            body = [node for node in parsed.body if not isinstance(node, (ast.Import, ast.ImportFrom))]
            exec(compile(ast.Module(body=body, type_ignores=[]), str(source), 'exec'), prompt_ns)
        self.assertEqual(messages, prompt_ns['get_messages'](
            info, 'judge_progress', 'web_shepherd', text_obs=True, image_obs=False))
        self.assertEqual(messages[0], {'role': 'system', 'content': ''})
        self.assertIn('THOUGHT: Search for the book\nACTION:', messages[1]['content'])
        self.assertIn('[a1] textbox Search', messages[1]['content'])
        checklist = api.get_messages(info, 'checklist_generation', 'web_shepherd')
        self.assertIn('Find a book', checklist[1]['content'])
        # Execute the complete upstream class after replacing only its dependency
        # base. This independently retains every original method, unlike extractor.
        tree = ast.parse((SHEPHERD / 'src/webprm/models.py').read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'WebPRM')
        ns = {'BaseModel': object, 'math': math}
        exec(compile(ast.Module(body=[cls], type_ignores=[]), '<upstream>', 'exec'), ns)
        reference = object.__new__(ns['WebPRM'])
        tokens = [{'token': '<answer>'}, {'token': 'Yes', 'top_logprobs': [
            {'token': 'Yes', 'logprob': math.log(.6)},
            {'token': 'No', 'logprob': math.log(.1)},
            {'token': 'In', 'logprob': math.log(.3)}]}, {'token': '</answer>'}]
        actual = api.get_judge_probs(tokens)
        self.assertEqual(actual, reference.get_judge_probs(tokens))
        self.assertAlmostEqual(api.get_score([actual], 'prob')[0], .75)
        self.assertEqual(api.get_judge_probs([]), reference.get_judge_probs([]))
        self.assertEqual(len(api.manifest['files']), 8)

    def test_revision_pin(self):
        with self.assertRaises(ValueError):
            source_manifest(SHEPHERD, [], 'wrong-revision')


@unittest.skipUnless(ARBITER.exists(), 'Reviewed upstream checkout unavailable')
class ArbiterParity(unittest.TestCase):
    def test_native_parser(self):
        api = load_arbiter(ARBITER)
        tree = ast.parse((ARBITER / 'eval/WebPRMBench/script/evaluate_webprm_vllm.py').read_text())
        # Full module without imports and main invocation has no heavyweight calls.
        nodes = [n for n in tree.body if not isinstance(n, (ast.Import, ast.ImportFrom, ast.If))]
        ns = {'re': re, 'Path': Path, 'Iterable': __import__('typing').Iterable,
              'Tuple': __import__('typing').Tuple, 'Dict': __import__('typing').Dict}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), '<upstream>', 'exec'), ns)
        for output in ['<Answer>Response 1</Answer>', 'Response 1 then Response 2', '',
                       '<Answer>Response 2</Answer>', '<answer>Response  1</answer>']:
            self.assertEqual(api.parse_answer(output), ns['parse_answer'](output))

    def test_exact_dataset_prompt(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'bench.jsonl'
            path.write_text('{"prompt": "exact\\nupstream text", "expected": 1}\n')
            result = load_arbiter_benchmark_prompts(path)
            self.assertEqual(result['records'][0]['prompt'], 'exact\nupstream text')
            self.assertEqual(len(result['sha256']), 64)
            path.write_text('{"prompt": ""}\n')
            with self.assertRaises(ValueError):
                load_arbiter_benchmark_prompts(path)


@unittest.skipUnless(Path('/tmp/openwebrl-webprm-benchmark.jsonl').exists(),
                     'Official rendered benchmark unavailable')
class CausalRendererParity(unittest.TestCase):
    def test_all_official_prompts_roundtrip_and_live_whitelist(self):
        from openwebrl.webprm_source import (load_arbiter_causal_renderer,
                                            _arbiter_split_prompt)
        path = '/tmp/openwebrl-webprm-benchmark.jsonl'
        api = load_arbiter_causal_renderer(path)
        self.assertEqual(api.manifest['records_validated'], 4600)
        self.assertEqual(api.manifest['static_template_variants'], 2)
        import json
        with open(path) as stream:
            sample = next(json.loads(line) for line in stream
                          if json.loads(line)['source_name'] == 'webarena')
        _, fields = _arbiter_split_prompt(sample['prompt'])
        self.assertEqual(api.render(**fields), sample['prompt'])
        with self.assertRaises(ValueError):
            api.render(**fields, expected=1)
        fields['thought1'] = 'New causal thought'
        rendered = api.render(**fields)
        self.assertIn('New causal thought', rendered)
        self.assertNotIn('expected', rendered)
