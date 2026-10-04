import ast
import json
import math
from pathlib import Path
import statistics
import tempfile
import types
import unittest

from openwebrl.rl_recipe import parse_judge_verdict, trajectory_advantages, state_advantages, observation_memory
from openwebrl.recipe_data import build_records, export_rollouts

ROOT = Path(__file__).resolve().parents[1]


def sample(reward, trajectory=0, turn=0, invalid=False, group=0, reason='task_completed'):
    s = types.SimpleNamespace(reward=reward, group_index=group, index=trajectory*100+turn,
        remove_sample=invalid, prompt=f'prefix-{trajectory}-{turn}', response='FAILED ACTION',
        multimodal_inputs={'images': ['prefix-image'], 'judge_images': ['FUTURE-IMAGE']},
        metadata={'task_id': 'task', 'trajectory_id': trajectory, 'turn_index': turn,
                  'terminate_reason': reason, 'step_tool_responses': [{'tool_name': 'click', 'tool_response': 'opened tab'}]})
    s.get_reward_value = lambda args: s.reward
    return s


def provider(args, states):
    for state in states:
        assert set(state) == {'prompt', 'images', 'turn_index'}
        assert 'FUTURE-IMAGE' not in state['images']
    return [0.25]*len(states)


class Tensor(list):
    """Tiny arithmetic adapter to run orchestration tests without installing torch."""
    def mean(self): return statistics.mean(self)
    def std(self): return statistics.stdev(self)
    def numel(self): return len(self)
    def tolist(self): return list(self)
    def __sub__(self, n): return Tensor(x-n for x in self)
    def __truediv__(self, n): return Tensor(x/n for x in self)


def actual_reward_method():
    tree = ast.parse((ROOT/'slime/ray/rollout.py').read_text())
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == '_post_process_rewards')
    node.returns = None
    for arg in node.args.args: arg.annotation = None
    def group_by(rows, key):
        out = {}
        for row in rows: out.setdefault(key(row), []).append(row)
        return out
    namespace = {'Any': object, 'Sample': object, 'group_by': group_by,
        'torch': types.SimpleNamespace(tensor=lambda x, **kw: Tensor(x), float=float),
        'logger': types.SimpleNamespace(info=lambda *a: None, warning=lambda *a: None),
        'logging_utils': types.SimpleNamespace(append_progress_log=lambda *a: None)}
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<actual reward method>', 'exec'), namespace)
    return namespace['_post_process_rewards']


class RecipeTests(unittest.TestCase):
    def setUp(self):
        self.args = types.SimpleNamespace(advantage_estimator='grpo', rewards_normalization=True,
            grpo_std_normalization=True, browser_state_advantage_mix=0,
            browser_value_provider_path=__name__+'.provider')

    def test_judge_parsing(self):
        self.assertEqual(parse_judge_verdict('Earlier NOT SUCCESS was incorrect.\nSUCCESS'), 1)
        self.assertEqual(parse_judge_verdict('Reasoning SUCCESS\n**Verdict: NOT SUCCESS**'), 0)
        self.assertEqual(parse_judge_verdict('{"verdict":"SUCCESS"}'), 1)
        self.assertEqual(parse_judge_verdict('```json\n{"verdict":"NOT SUCCESS"}\n```'), 0)
        for text in ('', 'The task may be a SUCCESS', 'SUCCESS or NOT SUCCESS', '{"verdict":true}'):
            self.assertIsNone(parse_judge_verdict(text))

    def test_actual_normalization_masks_before_statistics(self):
        rows = [sample(1, i) for i in range(4)] + [sample(0, 4, invalid=True)]
        manager = types.SimpleNamespace(args=self.args, custom_reward_post_process_func=None)
        _, actual = actual_reward_method()(manager, rows)
        self.assertEqual(actual, [0]*5)

    def test_partial_invalid_trajectory_and_variable_lengths(self):
        rows = [sample(1, 0, 0), sample(1, 0, 1), sample(0, 1), sample(0, 2, 0), sample(0, 2, 1, invalid=True)]
        manager = types.SimpleNamespace(args=self.args, custom_reward_post_process_func=None)
        _, actual = actual_reward_method()(manager, rows)
        self.assertAlmostEqual(actual[0], 1/math.sqrt(2), places=5)
        self.assertEqual(actual[0], actual[1])
        self.assertAlmostEqual(actual[2], -actual[0])
        self.assertEqual(actual[3:], [0, 0])
        self.assertTrue(rows[3].remove_sample)
        expected = trajectory_advantages(rows, self.args)
        self.assertEqual(actual, [expected[id(s)] for s in rows])

    def test_all_invalid_and_singleton(self):
        manager = types.SimpleNamespace(args=self.args, custom_reward_post_process_func=None)
        for rows in ([sample(0, invalid=True)], [sample(1)]):
            self.assertEqual(actual_reward_method()(manager, rows)[1], [0])

    def test_frozen_values_are_prefix_only_and_opt_in(self):
        rows = [sample(1), sample(0, 1)]
        self.assertEqual(state_advantages(self.args, rows)[1], list(trajectory_advantages(rows,self.args).values()))
        self.args.browser_state_advantage_mix = 1
        self.assertEqual(state_advantages(self.args, rows)[1], [.75, -.25])
        self.args.browser_state_advantage_mix = 2
        with self.assertRaises(ValueError): state_advantages(self.args, rows)

    def test_memory_is_observation_only_bounded_and_causal(self):
        turns = [sample(0, turn=i) for i in range(4)]
        ledger = observation_memory(turns, max_entries=2)
        self.assertEqual([x['step'] for x in json.loads(ledger)], [2,3])
        self.assertNotIn('FAILED ACTION', ledger)
        self.assertEqual(observation_memory(turns, max_chars=2), '[]')
        records = build_records(self.args, [[turns]])
        self.assertEqual(records['memory_targets'][0]['observed_memory'], '[]')
        self.assertEqual(len(json.loads(records['memory_targets'][1]['observed_memory'])), 1)

    def test_export_includes_rejected_failures_excludes_invalid_value_labels(self):
        groups = [[[sample(0, reason='max_steps_exhausted')], [sample(0, 1, invalid=True)]]]
        records = build_records(self.args, groups)
        self.assertEqual(len(records['value_targets']), 1)
        self.assertEqual([r['route'] for r in records['recovery_candidates']],
                         ['longer_horizon_candidate', 'infrastructure_review'])
        self.assertTrue(records['recovery_candidates'][0]['requires_live_state_verification'])
        self.assertNotIn('FUTURE-IMAGE', str(records['value_targets']))
        with tempfile.TemporaryDirectory() as directory:
            self.args.browser_recipe_export_dir = directory
            export_rollouts(self.args, groups, None)
            export_rollouts(self.args, groups, None)
            self.assertEqual(len(list(Path(directory).glob('batch-*/manifest.json'))), 2)


    def test_actual_dynamic_filter_excludes_invalid_trajectories(self):
        tree = ast.parse((ROOT/'slime/rollout/filter_hub/dynamic_sampling_filters.py').read_text())
        body = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
        ns = {'Sample': types.SimpleNamespace, 'DynamicFilterOutput': types.SimpleNamespace,
              'torch': types.SimpleNamespace(tensor=lambda x, **kw: Tensor(x), float64=float)}
        exec(compile(ast.Module(body=body, type_ignores=[]), '<actual filters>', 'exec'), ns)
        fn = ns['check_reward_nonempty_nonzero_std']
        invalid = [sample(0, 3, 0, invalid=True), sample(0, 3, 1)]
        self.assertFalse(fn(self.args, [[sample(1)], [sample(1, 1)], invalid]).keep)
        out = fn(self.args, [[sample(1)], [sample(0, 1)], invalid])
        self.assertTrue(out.keep)
        self.assertEqual(len(out.samples), 2)
        self.assertFalse(fn(self.args, [invalid]).keep)

    def test_invalid_value_provider_rejected(self):
        import sys
        module = sys.modules[__name__]
        module.bad_provider = lambda args, states: [float('nan')]*len(states)
        self.args.browser_value_provider_path = __name__+'.bad_provider'
        self.args.browser_state_advantage_mix = .5
        with self.assertRaises(ValueError): state_advantages(self.args, [sample(1)])

    def test_curriculum_is_dry_run_by_default(self):
        import subprocess
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'uncreated'
            result = subprocess.run([__import__('sys').executable,
                str(ROOT/'scripts/run_browser_curriculum.py'), '--output', str(output)],
                text=True, capture_output=True, check=True)
            self.assertIn('NUM_ROLLOUT=90', result.stdout)
            self.assertIn('NUM_ROLLOUT=140', result.stdout)
            self.assertIn('BROWSER_MAX_STEPS=30', result.stdout)
            self.assertFalse(output.exists())

    def test_package_import_does_not_load_runtime(self):
        import subprocess, sys
        subprocess.run([sys.executable, '-c',
            "import openwebrl.recipe_data, sys; assert 'torch' not in sys.modules; assert 'ray' not in sys.modules"],
            cwd=ROOT, check=True)


    def test_actual_judge_retry_errors_are_not_task_labels(self):
        import asyncio
        tree = ast.parse((ROOT/'openwebrl/reward_browser.py').read_text())
        names = {'compute_judge_reward_actionhistory', '_score_single_sample'}
        body = [n for n in tree.body if isinstance(n, ast.AsyncFunctionDef) and n.name in names]
        class JudgeSample(types.SimpleNamespace):
            Status = types.SimpleNamespace(COMPLETED='completed')
        async def no_sleep(seconds): pass
        for answer in ('malformed output', RuntimeError('API unavailable'), 'SUCCESS'):
            calls = []
            async def create(**kwargs):
                calls.append(kwargs)
                if isinstance(answer, Exception): raise answer
                return types.SimpleNamespace(choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=answer))])
            ns = {'Any': object, 'Sample': JudgeSample,
                'asyncio': types.SimpleNamespace(wait_for=asyncio.wait_for, sleep=no_sleep, TimeoutError=asyncio.TimeoutError),
                '_reward_semaphore': asyncio.Semaphore(1), 'parse_judge_verdict': parse_judge_verdict,
                'compute_format_reward': lambda *a, **kw: 1.0,
                '_extract_final_answer': lambda *a: 'answer', '_get_judge_screenshots': lambda *a: [],
                '_get_browser_response_mode': lambda *a: types.SimpleNamespace(name='browser_env'),
                '_extract_action_history_browser_env': lambda *a: 'history',
                '_build_user_content_with_step_indices': lambda *a: [],
                'JUDGE_USER_PROMPT_ACTION_HISTORY': '{task}', 'JUDGE_SYSTEM_PROMPT_ACTION_HISTORY': '',
                '_get_openai_client': lambda **kw: types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create))),
                '_should_sample_judge_output': lambda *a: False, '_save_judge_trace': lambda *a, **kw: None,
                'logger': types.SimpleNamespace(**{n: lambda *a, **kw: None for n in ('info','debug','warning','error')})}
            exec(compile(ast.Module(body=body, type_ignores=[]), '<actual judge>', 'exec'), ns)
            s = JudgeSample(metadata={'intent':'task','turn_index':0}, response='done', prompt='',
                            remove_sample=False, status='completed')
            args = types.SimpleNamespace(judge_prompt_variant='action_history', judge_api_model='test')
            score = asyncio.run(ns['_score_single_sample'](args, s))
            self.assertEqual(len(calls), 1 if answer == 'SUCCESS' else 3)
            self.assertEqual(score, 1 if answer == 'SUCCESS' else 0)
            self.assertEqual(s.remove_sample, answer != 'SUCCESS')
            self.assertEqual(s.metadata['reward']['valid'], answer == 'SUCCESS')


if __name__ == '__main__':
    unittest.main()
