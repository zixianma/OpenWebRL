"""Execute the real rollout control flow with in-memory browser/model doubles.

No browser, network, model import, or tensor allocation occurs. These tests validate
orchestration, not tokenizer correctness or Megatron/SGLang compatibility.
"""
import ast
import asyncio
from enum import Enum
import json
import logging
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

from openwebrl.rl_recipe import observation_memory

ROOT = Path(__file__).resolve().parents[1]


def load_functions(path, names, namespace, class_name=None):
    tree = ast.parse((ROOT/path).read_text())
    if class_name:
        tree = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    nodes = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in names]
    assert len(nodes) == len(names), (path, names)
    future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[])), str(path), 'exec'), namespace)
    return namespace


class Sample:
    class Status(Enum):
        PENDING = 'pending'
        COMPLETED = 'completed'
        FAILED = 'failed'
        ABORTED = 'aborted'
        TRUNCATED = 'truncated'

    def __init__(self, **kwargs):
        self.__dict__.update(status=self.Status.PENDING, remove_sample=False, metadata={},
                             reward=None, response='', response_length=0, multimodal_inputs=None)
        self.__dict__.update(kwargs)

    def get_reward_value(self, args):
        return self.reward


class Log:
    def __getattr__(self, name):
        return lambda *a, **kw: self if name == 'opt' else None


class Browser:
    def __init__(self, failure=False):
        self.failure = failure
        self.steps = 0
        self.closed = False

    async def reset(self):
        return 0, {}

    async def step(self, actions):
        if self.failure:
            raise RuntimeError('browser unavailable')
        self.steps += 1
        return self.steps, 0, actions == ['done'], False, {
            'tool_responses': [{'tool_name': actions[0], 'tool_response': f'observed-{self.steps}'}]}


class Adapter:
    token_handler = NS(apply_chat_template=lambda msgs, *a, **kw: json.dumps(msgs))
    _truncate_tool_response = staticmethod(str)
    preprocess_response = staticmethod(str)
    actions_from_parsed = staticmethod(lambda calls: calls)

    @property
    def tool_parser(self):
        return self._tool_parser

    def parse_env_info(self, info):
        return [], 'policy'

    def _get_latest_user_message(self, intent, observation, step_id=0):
        return {'role': 'user', 'content': [
            {'type': 'text', 'text': f'{intent} screenshot:\n<|vision_start|><|image_pad|><|vision_end|>'},
            {'type': 'image_url', 'image_url': f'image-{observation}'}]}

    def _convert_tool_responses_to_msgs(self, feedback, observation, **kwargs):
        return [self._get_latest_user_message(str(feedback), observation)]

    def _process_multimodal_messages(self, messages):
        text, images = [], []
        for m in messages:
            content = m['content']
            if isinstance(content, list):
                images.extend(x['image_url'] for x in content if x['type'] == 'image_url')
                content = '\n'.join(x['text'] for x in content if x['type'] == 'text')
            text.append({'role': m['role'], 'content': content})
        return text, images


class RolloutTests(unittest.TestCase):
    def run_rollout(self, outputs=('click', 'done'), memory=False, failure=False, max_steps=3, context_limit=None):
        browser = Browser(failure)
        requests = []
        iterator = iter(outputs)
        async def initialize(*args):
            return browser, Adapter(), NS(tokenizer=None, processor=None), 'unused', {'intent':'task', 'task_id':'task'}
        async def inference(url, prompt, params, images, **kw):
            requests.append((prompt, list(images)))
            output = next(iterator)
            if isinstance(output, Exception):
                raise output
            return output, [42], [-.1], output if output in ('length', 'abort') else 'stop'
        async def close(env, *a): env.closed = True
        namespace = dict(Sample=Sample, logging=logging, logger=Log(), observation_memory=observation_memory,
            _initialize_resources=initialize, _run_inference_step=inference, _safe_exit_env=close,
            ToolParser=lambda tools: NS(parse=lambda text: NS(success=text != 'bad', calls=[] if text == 'bad' else [text])),
            _get_browser_response_mode=lambda args: NS(chat_template_enable_thinking=True),
            _compress_turn_history_messages=lambda msgs,*a: msgs,
            _restore_assistant_history_blocks_in_prompt=lambda prompt,*a: prompt,
            ensure_prompt_ends_with_visible_thinking_tag=lambda prompt,*a: prompt,
            _restore_prefilled_thinking_tag_for_history=lambda response,*a: response,
            _encode_with_processor=lambda proc,tok,prompt,imgs: ([1,2], None, None, {'images':imgs}),
            _ensure_correct_sample=lambda *a: None, _should_log_task_start=lambda *a: False,
            _should_sample_llm_output=lambda *a: False, _append_host_to_blacklist_if_needed=lambda *a: None,
            _include_tool_response_in_rollout=lambda *a: True, _get_debug_trace_dir=lambda *a: None,
            _get_or_create_debug_trace_info=lambda *a: (False, None))
        load_functions('openwebrl/generate_browser.py', {
            '_generate_turn_sample_impl', '_append_to_sample', '_make_turn_sample_index',
            '_should_remove_browser_sample', '_mark_remove_sample_if_needed'}, namespace)
        args = NS(max_steps=max_steps, context_num_screenshots=1, browser_observation_memory=memory,
                  max_consecutive_parse_failures=2, rollout_max_context_len=context_limit)
        parent = Sample(index=10, group_index=2, metadata={'task_id':'task'})
        turns = asyncio.run(namespace['_generate_turn_sample_impl'](args, parent, {}))
        self.assertTrue(browser.closed)
        return turns, requests

    def test_real_loop_memory_is_causal_and_prompt_is_reused_for_training(self):
        turns, requests = self.run_rollout(memory=True)
        self.assertEqual(len(turns), 2)
        for t, (prompt, images) in zip(turns, requests):
            self.assertEqual(t.prompt, prompt)
            self.assertEqual(t.multimodal_inputs['images'], images)
            self.assertEqual(t.loss_mask, [1])
            self.assertEqual(t.rollout_log_probs, [-.1])
            self.assertEqual(t.status, Sample.Status.COMPLETED)
            self.assertFalse(t.remove_sample)
        first, second = [json.loads(r[0]) for r in requests]
        self.assertNotIn('observed-1', str(first))
        self.assertIn('observed-1', second[-1]['content'])
        self.assertEqual(requests[1][1], ['image-1'])
        self.assertEqual(str(second).count('Observed browser history'), 1)
        self.assertNotIn('Observed browser history', str(turns[-1].metadata['messages']))

    def test_default_memory_off(self):
        turns, requests = self.run_rollout()
        self.assertNotIn('Observed browser history', str(requests))
        self.assertEqual(turns[0].metadata['trajectory_id'], turns[1].metadata['trajectory_id'])
        self.assertNotEqual(turns[0].index, turns[1].index)

    def test_browser_failure_is_invalid(self):
        turns, _ = self.run_rollout(failure=True)
        self.assertTrue(all(t.remove_sample for t in turns))
        self.assertEqual(turns[-1].metadata['terminate_reason'], 'env_step_error')

    def test_agent_failures_remain_training_examples(self):
        for outputs, steps, expected in [(('bad','bad'),3,'format_error_failed'),
                                         (('click',),1,'max_steps_exhausted'),
                                         (('length',),3,'generation_length_limit')]:
            with self.subTest(expected=expected):
                turns, _ = self.run_rollout(outputs, max_steps=steps)
                self.assertEqual(turns[-1].metadata['terminate_reason'], expected)
                self.assertFalse(any(t.remove_sample for t in turns))

    def test_inference_abort_or_error_is_invalid(self):
        for outputs in [('abort',), (TimeoutError('inference timeout'),)]:
            with self.subTest(outputs=outputs):
                turns, _ = self.run_rollout(outputs)
                self.assertTrue(turns)
                self.assertTrue(all(t.remove_sample for t in turns))

    def test_initial_context_overflow_returns_terminal_sample(self):
        turns, requests = self.run_rollout(context_limit=1)
        self.assertFalse(requests)
        self.assertTrue(turns, 'An empty trajectory breaks reward and dynamic filtering')
        self.assertTrue(turns[-1].metadata['is_last_turn'])
        self.assertTrue(turns[-1].remove_sample)


if __name__ == '__main__':
    unittest.main()
