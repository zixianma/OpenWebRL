"""Lightweight data, scheduling, and failure-boundary checks."""
import contextlib
import dataclasses
import importlib.util
import io
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from openwebrl.rl_recipe import state_advantages, trajectory_advantages
from test_rl_recipe import sample, actual_reward_method
from test_rollout_offline import load_functions, ROOT, Log


def module_from_file(path):
    spec = importlib.util.spec_from_file_location('curriculum_under_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PreflightTests(unittest.TestCase):
    def setUp(self):
        self.args = NS(advantage_estimator='grpo', rewards_normalization=True,
                       grpo_std_normalization=True, browser_state_advantage_mix=0)

    def test_zero_mix_masks_every_turn_of_invalid_trajectory(self):
        rows = [sample(0, 0, 0), sample(0, 0, 1, invalid=True), sample(1, 1)]
        state_advantages(self.args, rows)
        self.assertTrue(rows[0].remove_sample)
        self.assertFalse(rows[2].remove_sample)

    def test_randomized_group_normalization_matches_backend(self):
        rng = random.Random(51)
        fn = actual_reward_method()
        for standardize in [True, False]:
            self.args.grpo_std_normalization = standardize
            for _ in range(30):
                rows = []
                for group in range(3):
                    for trajectory in range(rng.randint(1,5)):
                        reward = rng.choice([-1,0,1])
                        length = rng.randint(1,4)
                        invalid_turn = rng.choice([-1, length-1])
                        rows.extend(sample(reward, trajectory, t, t == invalid_turn, group) for t in range(length))
                expected = trajectory_advantages(rows, self.args)
                manager = NS(args=self.args, custom_reward_post_process_func=None)
                actual = fn(manager, rows)[1]
                for s, value in zip(rows, actual):
                    self.assertAlmostEqual(value, expected[id(s)], places=6)

    def test_export_cli_roundtrip_and_dictionary_reward_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root/'groups.json'
            source.write_text(json.dumps([[[{'reward': {'score':0}, 'prompt':'prefix', 'response':'action',
                'index':0, 'group_index':0, 'metadata': {'turn_index':0, 'task_id':'task',
                'terminate_reason':'max_steps_exhausted'}}]]]))
            command = [sys.executable, '-m', 'openwebrl.recipe_data', '--input', str(source), '--output', str(root/'out')]
            bad = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
            self.assertNotEqual(bad.returncode, 0)
            self.assertIn('--reward-key', bad.stderr)
            subprocess.run(command+['--reward-key','score'], cwd=ROOT, check=True, capture_output=True)
            batch = next((root/'out').glob('batch-*'))
            manifest = json.loads((batch/'manifest.json').read_text())
            self.assertEqual(manifest['counts']['value_targets'], 1)
            self.assertEqual(json.loads((batch/'value_targets.jsonl').read_text())['target_return'], 0)

    def test_curriculum_checkpoint_handoff_with_mocked_launcher(self):
        module = module_from_file(ROOT/'scripts/run_browser_curriculum.py')
        for marker in [None, '3', '89']:
            with self.subTest(marker=marker), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)/'run'
                launches = []
                def launch(command, cwd, env, check):
                    launches.append(env)
                    stage = Path(env['SLIME_SAVE_DIR'])
                    stage.mkdir(parents=True)
                    if len(launches) == 1 and marker is not None:
                        (stage/'latest_checkpointed_iteration.txt').write_text(marker)
                        (stage/f'iter_{int(marker):07d}').mkdir()
                with patch.object(sys, 'argv', ['curriculum','--output',str(root),'--execute']), \
                     patch.object(module.subprocess, 'run', side_effect=launch), contextlib.redirect_stdout(io.StringIO()):
                    if marker == '89':
                        module.main()
                        self.assertEqual(len(launches), 2)
                        self.assertEqual(launches[1]['NUM_ROLLOUT'], '140')
                        self.assertEqual(launches[1]['OVERRIDE_OPT_PARAM_SCHEDULER'], '1')
                        self.assertEqual(launches[1]['SLIME_LOAD_CHECKPOINT'], str(root/'stage1'))
                    else:
                        with self.assertRaises(SystemExit): module.main()
                        self.assertEqual(len(launches), 1)

    def test_curriculum_refuses_existing_stages_before_launch(self):
        module = module_from_file(ROOT/'scripts/run_browser_curriculum.py')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'stage1').mkdir()
            with patch.object(sys, 'argv', ['curriculum','--output',str(root),'--execute']), \
                 patch.object(module.subprocess, 'run') as run, contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(SystemExit): module.main()
                run.assert_not_called()

    def test_sampler_ignores_invalid_and_missing_outcomes(self):
        ns = dict(dataclass=dataclasses.dataclass, asdict=dataclasses.asdict, Sample=NS, Any=object)
        # Load the actual stats class, without importing torch via query_selector.
        import ast
        tree = ast.parse((ROOT/'slime/rollout/query_selector.py').read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'QueryStats')
        exec(compile(ast.Module(body=[cls], type_ignores=[]), 'QueryStats', 'exec'), ns)
        ns.update(logger=Log(), append_progress_log=lambda *a:None,
                  extract_query_key=lambda s:s.metadata['task_id'],
                  extract_terminal_sample=lambda ts:ts[-1],
                  extract_completed_trajectory_length=lambda ts:None)
        load_functions('slime/rollout/data_source.py', {'update_from_rollout','_reward_to_binary_outcome'},ns,class_name='RolloutDataSource')
        manager = NS(args=self.args, _use_adaptive_query_sampling=lambda:True, _adaptive_query_stats={},
                     _normalize_rollout_group=lambda group:group, _adaptive_global_completed_traj_count=0,
                     _append_adaptive_query_summary_log=lambda _:None)
        manager._reward_to_binary_outcome = lambda value:ns['_reward_to_binary_outcome'](manager,value)
        ns['update_from_rollout'](manager, [[[sample(1)], [sample(0,1)],
            [sample(0,2,invalid=True)], [sample(None,3)]]], rollout_id=0)
        stats = manager._adaptive_query_stats['task']
        self.assertEqual((stats.success_count,stats.fail_count,stats.invalid_traj_count), (1,1,2))


if __name__ == '__main__':
    unittest.main()
