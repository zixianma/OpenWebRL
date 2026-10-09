import copy
import sys
from pathlib import Path
import unittest

from openwebrl.arm_task_screen_sharded import (
    partition, validate_partition, pending_rows, JUDGE_USD_PER_SHARD, JUDGE_CALLS_PER_SHARD)
from openwebrl.arm_terminal_budget import CappedJudge
from openwebrl.arm_rescue_yield import stable_seed
from test_arm_task_screen import ScreenAccountingTest


class ShardTest(unittest.TestCase):
    def test_exact_partition_and_seed_stability(self):
        ids = [f'task-{i}' for i in range(2000)]
        shards = partition(ids)
        self.assertEqual([len(s) for s in shards], [250]*8)
        self.assertEqual({t for s in shards for t in s}, set(ids))
        validate_partition(shards, ids)
        for shard in shards:
            for task in shard:
                for attempt in range(5):
                    self.assertEqual(stable_seed(20260930,'OpenWebRL-4B-SFT:iteration0',task,'screen','actor',attempt),
                        stable_seed(20260930,'OpenWebRL-4B-SFT:iteration0',ids[ids.index(task)],'screen','actor',attempt))
        bad = copy.deepcopy(shards); bad[1][0] = bad[0][0]
        with self.assertRaises(ValueError): validate_partition(bad, ids)
        with self.assertRaises(ValueError): partition(ids[:-1]+[ids[0]])


    def test_total_judge_caps_do_not_multiply(self):
        self.assertEqual(JUDGE_USD_PER_SHARD*8,150.)
        self.assertEqual(JUDGE_CALLS_PER_SHARD*8,30000)


class ResumeShardTest(ScreenAccountingTest):
    def test_resume_checks_artifacts_even_for_complete_tasks(self):
        root = self.root/'records'; root.mkdir()
        import json
        from openwebrl.arm_task_screen import key
        for attempt in range(5):
            (root/(key('done',attempt)+'.json')).write_text(json.dumps(self.record('done',attempt)))
        rows = [dict(metadata=dict(task_id=t)) for t in ('done','pending')]
        self.assertEqual(pending_rows(rows,root),rows[1:])
        p = root/(key('done',2)+'.json')
        r=json.loads(p.read_text());r['artifact']=str(self.root/'missing');p.write_text(json.dumps(r))
        with self.assertRaises(ValueError): pending_rows(rows,root)

    def test_judge_restart_preserves_charges(self):
        root=self.root/'judge'
        first=CappedJudge(root,client=object(),max_usd=18.75,max_calls=3750)
        first.reserve(18.5)
        restarted=CappedJudge(root,client=object(),max_usd=18.75,max_calls=3750)
        with self.assertRaises(ValueError): restarted.reserve(.3)
        self.assertTrue((root/'halt.json').is_file())


if __name__ == '__main__': unittest.main()
