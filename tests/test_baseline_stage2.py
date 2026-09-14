"""CPU regression checks against the actual frozen launcher and scheduler."""
import ast
from contextlib import redirect_stdout, redirect_stderr
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'scripts'))
import prepare_baseline_stage2 as preparation

RUNTIME = preparation.RUNTIME
SOURCE = RUNTIME / 'reference-stage1-browsers32-20260911'
STAGE1 = RUNTIME / 'baseline_stage1_completed90_before_stage2.json'


def stage1_state():
    return json.loads((STAGE1 if STAGE1.exists() else preparation.STATE).read_text())


class ActivationTest(unittest.TestCase):
    def test_activation_requires_matching_pointer_and_gpu_receipt(self):
        root = Path(tempfile.mkdtemp(prefix='stage2-activation-test-'))
        source = root / 'source'
        (source / 'scripts').mkdir(parents=True)
        (root / 'logs').mkdir()
        (source / 'scripts/run_small_baseline.py').write_text('frozen launcher')
        current, prepared, plan_path = root / 'current.json', root / 'prepared.json', root / 'plan.json'
        preparation.write_json(current, {'training_stage': 1})
        preparation.write_json(prepared, {'durable_optimizer_updates': 1016})
        plan = dict(state=str(current), expected_state_sha256=preparation.sha(current),
                    prepared_state=str(prepared), prepared_state_sha256=preparation.sha(prepared),
                    source=str(source), initial_checkpoint=str(root / 'iter_0000089'),
                    wandb_run_id='qcq7i4ug', requested_resources={'maximum_gpu_hours': 32})
        preparation.write_json(plan_path, plan)
        receipt = dict(identity=dict(source=str(source), launcher_sha256=preparation.sha(source / 'scripts/run_small_baseline.py'),
                       resume_from=str(root), checkpoint=plan['initial_checkpoint'], optimizer_updates=1016,
                       gpus=4, job_id='42'), report={'full_model_and_optimizer_load': 'passed'}, verification_run='verified')
        receipt_path = root / 'logs/resume-42-4gpu-verification.json'
        preparation.write_json(receipt_path, receipt)
        with patch.object(preparation, 'RUNTIME', root), patch.object(preparation, 'validate_source'), patch.dict(os.environ, SLURM_JOB_ID='42'):
            original = current.read_bytes()
            with self.assertRaisesRegex(ValueError, 'inside'):
                preparation.activate(plan_path, '43')
            receipt['identity']['optimizer_updates'] = 0
            preparation.write_json(receipt_path, receipt)
            with self.assertRaisesRegex(ValueError, 'verification'):
                preparation.activate(plan_path, '42')
            self.assertEqual(current.read_bytes(), original)
            receipt['identity']['optimizer_updates'] = 1016
            preparation.write_json(receipt_path, receipt)
            current.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'pointer changed'):
                preparation.activate(plan_path, '42')
            current.write_bytes(original)
            preparation.activate(plan_path, '42')
            activated = json.loads(current.read_text())
            self.assertEqual(activated['training_stage'], 2)
            self.assertEqual(activated['queued_continuation']['job_id'], '42')


@unittest.skipUnless(SOURCE.is_dir(), 'requires the preserved local reference source')
class Stage2Test(unittest.TestCase):
    def launcher(self):
        spec = importlib.util.spec_from_file_location('stage2_test_launcher', SOURCE / 'scripts/run_small_baseline.py')
        module = importlib.util.module_from_spec(spec)
        exec(compile(preparation.upgrade_launcher((SOURCE / 'scripts/run_small_baseline.py').read_text()),
                     str(spec.origin), 'exec'), module.__dict__)
        return module

    def test_resume_environment_and_manifest(self):
        module = self.launcher()
        state = stage1_state()
        root = str(Path(state['last_valid_checkpoint']).parent)
        class CaptureEnvironment(dict):
            def copy(self):
                self.child = dict(self)
                return self.child
        env = CaptureEnvironment(os.environ)
        env.update(WANDB_PROJECT='openwebrl-evals', JUDGE_MODEL='wrong-model')
        argv = ['launcher', '--profile', 'reference', '--gpus', '4', '--resume-from', root,
                '--wandb-run-id', 'qcq7i4ug', '--dry-run', '--env-file', '/nonexistent-stage2-test-env']
        output = io.StringIO()
        with patch.object(module.os, 'environ', env), patch.object(sys, 'argv', argv), redirect_stdout(output):
            module.main()
        result = json.loads(output.getvalue())
        self.assertEqual(result['continuation']['next_rollout_id'], 90)
        self.assertEqual(result['continuation']['durable_optimizer_updates_at_start'], 1016)
        self.assertEqual(result['rollouts'], 140)
        self.assertEqual(result['max_browser_steps'], 30)
        for key, expected in dict(NUM_ROLLOUT='140', BROWSER_MAX_STEPS='30', N_SAMPLES='5',
                                  ROLLOUT_BATCH_SIZE='48', GLOBAL_BATCH_SIZE='256', TP_SIZE='4',
                                  BROWSER_CONCURRENCY='32', LEARNING_RATE='1e-6',
                                  WANDB_PROJECT='openwebrl', WANDB_RUN_ID='qcq7i4ug',
                                  JUDGE_MODEL='gpt-4.1', OVERRIDE_OPT_PARAM_SCHEDULER='1').items():
            self.assertEqual(env.child[key], expected, key)

    def test_rejects_wrong_stage_or_fresh_start(self):
        for extra in [[], ['--profile', 'reference'], ['--profile', 'reference', '--resume-from', '/tmp', '--steps', '15']]:
            with self.subTest(extra=extra), patch.object(sys, 'argv', ['launcher', '--dry-run', *extra]):
                with self.assertRaises(SystemExit), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    self.launcher().main()

    def test_scheduler_horizon_change_preserves_counter_lr_and_weight_decay(self):
        # Execute the installed scheduler class without GPU-dependent Megatron imports.
        path = RUNTIME / 'src/Megatron-LM/megatron/core/optimizer_param_scheduler.py'
        tree = ast.parse(path.read_text())
        tree.body = [node for node in tree.body if not (isinstance(node, ast.ImportFrom)
                      and node.module and node.module.startswith('megatron'))]
        namespace = dict(MegatronOptimizer=object, log_single_rank=lambda *a, **k: None)
        exec(compile(tree, str(path), 'exec'), namespace)
        import torch
        state = stage1_state()
        common = torch.load(Path(state['last_valid_checkpoint']) / 'common.pt', map_location='cpu', weights_only=False)
        saved = common['opt_param_scheduler']
        def scheduler(override):
            return namespace['OptimizerParamScheduler'](
                optimizer=SimpleNamespace(param_groups=[{}]), init_lr=0, max_lr=1e-6, min_lr=0,
                lr_warmup_steps=0, lr_decay_steps=(140 * 48 * 5 // 256) * 256,
                lr_decay_style='constant', start_wd=0.1, end_wd=0.1,
                wd_incr_steps=(140 * 48 * 5 // 256) * 256, wd_incr_style='constant',
                use_checkpoint_opt_param_scheduler=False, override_opt_param_scheduler=override)
        with self.assertRaisesRegex(AssertionError, 'total number of iterations'):
            scheduler(False).load_state_dict(saved)
        resumed = scheduler(True)
        resumed.load_state_dict(saved)
        self.assertEqual(resumed.num_steps, saved['num_steps'])
        for _ in range(1000):
            resumed.step(256)
            self.assertEqual(resumed.optimizer.param_groups[0]['lr'], 1e-6)
            self.assertEqual(resumed.optimizer.param_groups[0]['weight_decay'], 0.1)

    def test_frozen_recipe_filter_is_enabled_adaptive_sampling_not_requested(self):
        text = (SOURCE / 'scripts/run_h200_browser.sh').read_text()
        self.assertIn('--dynamic-sampling-filter-path slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonempty_nonzero_std', text)
        self.assertNotIn('--enable-adaptive-query-sampling', text)


if __name__ == '__main__':
    unittest.main()
