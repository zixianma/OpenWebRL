"""Check aggregate caps and staged resource scope without launching a job."""
import importlib.util
from pathlib import Path
import unittest

SPEC = importlib.util.spec_from_file_location('shopping_validation',
    Path(__file__).resolve().parents[1] / 'scripts/prepare_harmful_shopping_validation.py')
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


class ValidationProposalTests(unittest.TestCase):
    def test_closed_approvals_are_not_reused(self):
        value = M.resource_request()
        self.assertEqual(value['status'], 'proposed_not_authorized')
        self.assertEqual(value['older_allocation_seconds_added'], 0)
        self.assertEqual(value['other_session_budget_added'], 0)

    def test_exact_aggregate_resource_caps_and_no_models(self):
        value = M.resource_request()
        self.assertEqual((value['nodes'], value['cpus'], value['memory_gib'], value['gpus']), (1, 4, 16, 0))
        self.assertEqual(value['maximum_allocated_cpu_hours'],
                         value['cpus'] * value['total_allocation_seconds_including_all_attempts'] / 3600)
        self.assertEqual((value['api_budget_usd'], value['actor_and_judge_calls'], value['internet_download_bytes']), (0, 0, 0))
        self.assertFalse(value['model_comparison_in_scope'])
        self.assertFalse(value['gitlab_in_scope'])

    def test_selftests_and_development_precede_held_out_references(self):
        stages = M.stages()
        names = [s['id'] for s in stages]
        self.assertLess(names.index('capture_and_reset_selftests'), names.index('development_references'))
        self.assertLess(names.index('development_references'), names.index('held_out_references'))
        self.assertEqual(names[-1], 'close_and_audit')
        self.assertEqual(len(M.TASK_IDS) * len(M.CASES), 36)
        self.assertGreaterEqual(M.resource_request()['maximum_fresh_instances_including_failed_attempts'], 44)


if __name__ == '__main__':
    unittest.main()
