"""Exercise the actual initialization path with a real CPU-only scheduler."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler
from slime.backends.megatron_utils import model as backend


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


if __name__ == '__main__':
    unittest.main()
