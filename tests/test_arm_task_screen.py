import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from openwebrl.arm_task_screen import POLICY, JUDGE, key, summarize, verify_record


class ScreenAccountingTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def record(self, task, attempt, valid=True, reward=0.):
        artifact = self.root/key(task, attempt)
        artifact.mkdir(exist_ok=True)
        archive = artifact/(hashlib.sha256(task.encode()).hexdigest()+'.pt')
        archive.write_bytes(b'archive-placeholder-for-accounting-test')
        archive.with_suffix('.json').write_text(json.dumps(dict(task_id=task,
            judge_model='gpt-4.1', judge_prompt_variant='action_history',
            metrics={'valid_trajectories': int(valid)}, reward_metadata={'combined': reward})))
        return dict(task_id=task, attempt=attempt, valid=valid, reward=reward if valid else None,
            policy_id=POLICY, judge_id=JUDGE, phase='screen', mode='actor', selector_calls=0,
            artifact=str(artifact))

    def test_only_five_valid_failures_are_eligible(self):
        ids = ['failed', 'mixed', 'passed', 'invalid']
        rows = [self.record(t,a,valid=not(t=='invalid' and a==4),
            reward=float(t=='passed' or t=='mixed' and a==0)) for t in ids for a in range(5)]
        result = summarize(rows,ids,True)
        self.assertEqual(result['dispositions'],dict(failed='all_failure',mixed='mixed',
            passed='all_success',invalid='unresolved_invalid'))
        self.assertEqual(result['valid_attempts'],19)
        self.assertEqual(result['successes'],6)

    def test_partial_and_duplicate_cannot_finish(self):
        rows = [self.record('task',a) for a in range(4)]
        self.assertEqual(summarize(rows,['task'])['dispositions']['task'],'pending')
        with self.assertRaises(ValueError): summarize(rows,['task'],True)
        with self.assertRaises(ValueError): summarize(rows+[rows[0]],['task'])

    def test_changed_judge_or_missing_archive_rejected(self):
        record = self.record('task',0)
        with self.assertRaises(ValueError): verify_record(dict(record,judge_id='o4-mini'), 'task',0)
        with self.assertRaises(ValueError): verify_record(dict(record,artifact=str(self.root/'missing')), 'task',0)

    def test_sidecar_reward_must_match_record(self):
        record = self.record('task',0)
        with self.assertRaises(ValueError): verify_record(dict(record,reward=1.),'task',0)


if __name__ == '__main__':
    unittest.main()
