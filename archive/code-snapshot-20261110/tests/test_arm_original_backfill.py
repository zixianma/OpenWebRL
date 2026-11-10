import hashlib
import importlib.util
from pathlib import Path
import sys
import unittest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'scripts'))
import prepare_arm_original_backfill as backfill

spec=importlib.util.spec_from_file_location('native_metrics',REPO/'slime/utils/trajectory_metrics.py')
metrics=importlib.util.module_from_spec(spec);spec.loader.exec_module(metrics)


class OriginalBackfillTests(unittest.TestCase):
    def record(self,status='completed',reward=1,remove=False,judge=True):
        payload=dict(task_id='example',turns=[dict(status=status,reward=reward,remove_sample=remove,
            metadata=dict(turn_index=0,reward=dict(combined=reward,judge=reward,
                judge_prompt_variant='action_history',judge_timeout=False) if judge else {}))])
        return backfill.recovered_record(payload,Path(hashlib.sha256(b'example').hexdigest()+'.pt'),
            metrics.trajectory_metrics)

    def test_preserves_invalid_attempts_and_valid_policy_failures(self):
        self.assertEqual(self.record()['metrics']['successes'],1)
        failed=self.record(status='failed',reward=0)['metrics']
        self.assertEqual((failed['valid_trajectories'],failed['successes']),(1,0))
        for kwargs in [dict(status='aborted',reward=None,judge=False),dict(remove=True)]:
            invalid=self.record(**kwargs)['metrics']
            self.assertEqual((invalid['invalid_trajectories'],invalid['successes']),(1,0))

    def test_refuses_valid_outcome_without_judge_evidence(self):
        with self.assertRaisesRegex(ValueError,'preserved judge evidence'):
            self.record(judge=False)

    def test_recovered_result_is_explicitly_not_a_new_judgment(self):
        record=self.record()
        self.assertFalse(record['provenance']['judge_rerun'])
        self.assertTrue(record['provenance']['recovered_from_embedded_metadata'])
        self.assertEqual(record['judge_model'],'gpt-4.1')


if __name__=='__main__':unittest.main()
