"""Exercise the actual initialization path with a real CPU-only scheduler."""
import ast
import importlib.util
import logging
import math
from pathlib import Path
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

# Execute the installed scheduler and repository initialization function on CPU.
# Avoid importing the entire Megatron/CUDA inference stack during collection.
# Only model construction, checkpoint I/O and distributed logging are replaced.
spec = importlib.util.find_spec('megatron')
roots = list(spec.submodule_search_locations or []) if spec else []
scheduler_path = next((Path(root)/'core/optimizer_param_scheduler.py' for root in roots
                       if (Path(root)/'core/optimizer_param_scheduler.py').is_file()), None)
if scheduler_path is None:
    raise unittest.SkipTest('Megatron scheduler source is required for this integration test')
backend = ModuleType('cpu_scheduler_backend')
backend.__dict__.update(math=math, logging=logging, logger=logging.getLogger(__name__),
    log_single_rank=lambda *a, **kw: None, torch=SimpleNamespace(version=SimpleNamespace(hip=False)),
    setup_model_and_optimizer=None, load_checkpoint=None, clear_memory=None)
for path, name in [(scheduler_path, 'OptimizerParamScheduler'),
                   (Path(__file__).resolve().parents[1]/'slime/backends/megatron_utils/model.py',
                    'initialize_model_and_optimizer')]:
    node = next(n for n in ast.parse(path.read_text()).body if getattr(n, 'name', None) == name)
    tree = ast.Module(body=[ast.ImportFrom(module='__future__',
        names=[ast.alias(name='annotations')], level=0), node], type_ignores=[])
    exec(compile(ast.fix_missing_locations(tree), str(path), 'exec'), backend.__dict__)
OptimizerParamScheduler = backend.OptimizerParamScheduler


class SchedulerResumeTest(unittest.TestCase):
    def check_restore(self, rollout_iteration, restored_steps):
        optimizer = SimpleNamespace(param_groups=[{}])
        scheduler = OptimizerParamScheduler(
            optimizer, init_lr=0.0, max_lr=1e-6, min_lr=0.0,
            lr_warmup_steps=0, lr_decay_steps=100000, lr_decay_style='linear',
            start_wd=0.1, end_wd=0.1, wd_incr_steps=100000, wd_incr_style='constant',
            use_checkpoint_opt_param_scheduler=True, override_opt_param_scheduler=False,
        )
        state = scheduler.state_dict()
        state['num_steps'] = restored_steps
        loaded_lr = []

        def load_checkpoint(model, opt, schedule, **kwargs):
            schedule.load_state_dict(state)
            loaded_lr.append(opt.param_groups[0]['lr'])
            return rollout_iteration, 0

        model = [SimpleNamespace()]
        with patch.object(backend, 'setup_model_and_optimizer', return_value=(model, optimizer, scheduler)), \
             patch.object(backend, 'load_checkpoint', side_effect=load_checkpoint), \
             patch.object(backend, 'clear_memory'):
            result = backend.initialize_model_and_optimizer(SimpleNamespace(global_batch_size=256))
        self.assertEqual(result[3], rollout_iteration)
        self.assertEqual(scheduler.num_steps, restored_steps)
        self.assertEqual(optimizer.param_groups[0]['lr'], loaded_lr[0])

    def test_resume_preserves_restored_update_count_and_learning_rate(self):
        self.check_restore(1, 7680)
        self.check_restore(7, 100 * 256)

    def test_fresh_schedule_is_not_advanced_by_model_iteration(self):
        self.check_restore(0, 0)
        self.check_restore(7, 0)

    def test_extended_rollout_target_preserves_saved_schedule(self):
        def make(length, use_checkpoint):
            optimizer = SimpleNamespace(param_groups=[{}])
            return OptimizerParamScheduler(
                optimizer, init_lr=0.0, max_lr=1e-6, min_lr=0.0,
                lr_warmup_steps=0, lr_decay_steps=length, lr_decay_style='constant',
                start_wd=0.1, end_wd=0.1, wd_incr_steps=length, wd_incr_style='constant',
                use_checkpoint_opt_param_scheduler=use_checkpoint,
                override_opt_param_scheduler=False,
            )

        saved = make(21504, True)
        saved.step(260352)
        state = saved.state_dict()
        # Reproduce the failed 90 -> 100 continuation before testing the fix.
        with self.assertRaisesRegex(AssertionError, 'do not match'):
            make(23808, False).load_state_dict(state)
        restored = make(23808, True)
        restored.load_state_dict(state)
        self.assertEqual(restored.state_dict(), state)
        self.assertEqual(restored.optimizer.param_groups, saved.optimizer.param_groups)


if __name__ == '__main__':
    unittest.main()
