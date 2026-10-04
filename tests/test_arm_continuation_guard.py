import copy
import json
from pathlib import Path
import tempfile
import unittest

from openwebrl.arm_continuation_guard import apply_adjacent_count_support


class AdjacentCountSupport(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.folder = root/'old/iterations/0077'
        self.folder.mkdir(parents=True)
        checkpoint = root/'old/runtime/iter_0000077'
        checkpoint.mkdir(parents=True)
        (checkpoint/'common.pt').write_bytes(b'checkpoint-existence-fixture')
        config = dict(label_count_guard='adjacent_batch_v1', label_guard_start_rollout_id=78,
            label_guard_resume_from=str(root/'old'), run_output=str(root/'new'), rollout_id=78,
            checkpoint=str(checkpoint), run_id='lineage', beta=.5, scored_fraction=.2, k=5)
        self.report = dict(all_failure_experiment=True, config=config, admitted=96,
                           policy_id='lineage:rollout78')
        self.decision = dict(passed=False, checks=dict(labels=False, tasks=True, coverage=True,
            scale=True, all_failure_groups=True, all_failure_labels=True), rule='Original gates.')
        self.prior = dict(all_failure_experiment=True, config=config, admitted=124,
            policy_id='lineage:rollout77', applied_beta=.5, decision=dict(passed=True))
        self.write('calibration', self.prior)
        self.write('training_gate', dict(ready=True))
        self.write('checkpoint-saved', dict(rollout_id=77, checkpoint=str(checkpoint)))

    def write(self, name, obj):
        (self.folder/(name+'.json')).write_text(json.dumps(obj))

    def test_adjacent_trained_batch_supports_96_without_mutating_input(self):
        before = copy.deepcopy(self.decision)
        result = apply_adjacent_count_support(self.report, self.decision)
        self.assertTrue(result['passed'])
        self.assertTrue(result['count_guard']['used'])
        self.assertEqual(result['count_guard']['previous_labels'], 124)
        self.assertEqual(len(result['count_guard']['evidence_sha256']), 3)
        self.assertEqual(self.decision, before)

    def test_no_relaxation_of_any_other_current_gate(self):
        for gate in self.decision['checks']:
            if gate == 'labels':
                continue
            decision = copy.deepcopy(self.decision)
            decision['checks'][gate] = False
            self.assertFalse(apply_adjacent_count_support(self.report, decision)['passed'])

    def test_floor_and_consecutive_sparse_batches_stop(self):
        self.report['admitted'] = 49
        self.assertFalse(apply_adjacent_count_support(self.report, self.decision)['passed'])
        self.report['admitted'] = 96
        self.prior['admitted'] = 99
        self.write('calibration', self.prior)
        self.assertFalse(apply_adjacent_count_support(self.report, self.decision)['passed'])

    def test_wrong_lineage_reward_recipe_or_untrained_checkpoint_rejected(self):
        for key, value in [('policy_id', 'different:rollout77'), ('applied_beta', 0),
                           ('config', dict(self.prior['config'], beta=0))]:
            bad = dict(self.prior, **{key: value})
            self.write('calibration', bad)
            with self.assertRaises(ValueError):
                apply_adjacent_count_support(self.report, self.decision)
        self.write('calibration', self.prior)
        self.write('training_gate', dict(ready=False))
        with self.assertRaises(ValueError):
            apply_adjacent_count_support(self.report, self.decision)

    def test_disabled_guard_preserves_original_decision(self):
        del self.report['config']['label_count_guard']
        self.assertIs(apply_adjacent_count_support(self.report, self.decision), self.decision)

    def test_regular_count_needs_no_prior_evidence(self):
        self.report['admitted'] = 100
        self.decision['checks']['labels'] = True
        self.decision['passed'] = True
        self.report['config']['label_guard_resume_from'] = '/nonexistent'
        result = apply_adjacent_count_support(self.report, self.decision)
        self.assertTrue(result['passed'])
        self.assertFalse(result['count_guard']['used'])


if __name__ == '__main__':
    unittest.main()
