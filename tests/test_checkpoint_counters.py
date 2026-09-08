"""Distinguish saved Adam updates from scheduler bookkeeping after resume."""
from types import SimpleNamespace
import unittest

from scripts.inspect_training_checkpoint import inspect_counters


def checkpoint(adam_steps=(46, 46), scheduler_batches=46):
    return {
        'args': SimpleNamespace(global_batch_size=256),
        'opt_param_scheduler': {'num_steps': scheduler_batches * 256},
        'optimizer': {'optimizer': {'param_groups': [{'step': s} for s in adam_steps]}},
    }


class CheckpointCountersTest(unittest.TestCase):
    def test_matching_counters(self):
        result = inspect_counters(checkpoint(), expected_updates=46)
        self.assertEqual(result['completed_optimizer_updates'], 46)
        self.assertEqual(result['scheduler_minus_optimizer_updates'], 0)

    def test_scheduler_drift_is_rejected_by_default(self):
        with self.assertRaisesRegex(ValueError, 'Scheduler/Adam offset'):
            inspect_counters(checkpoint(scheduler_batches=47), expected_updates=46)

    def test_diagnosed_offset_does_not_invent_optimizer_updates(self):
        result = inspect_counters(checkpoint(scheduler_batches=47), 46, 1)
        self.assertEqual(result['completed_optimizer_updates'], 46)
        self.assertEqual(result['completed_optimizer_updates_from_scheduler'], 47)
        with self.assertRaisesRegex(ValueError, 'Expected 47 updates, found 46'):
            inspect_counters(checkpoint(scheduler_batches=47), 47, 1)

    def test_group_counter_disagreement_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'counters disagree'):
            inspect_counters(checkpoint(adam_steps=(45, 46)))

    def test_scheduler_only_evidence_is_explicit(self):
        result = inspect_counters(checkpoint(adam_steps=()), expected_updates=46)
        self.assertEqual(result['update_count_evidence'], 'scheduler only')
        with self.assertRaisesRegex(ValueError, 'without Adam group counters'):
            inspect_counters(checkpoint(adam_steps=(), scheduler_batches=47), 46, 1)


if __name__ == '__main__':
    unittest.main()
