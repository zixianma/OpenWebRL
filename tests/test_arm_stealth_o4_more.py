import copy
import sys
from pathlib import Path
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import run_arm_stealth90_o4_more as code


class MoreRepeatsTests(unittest.TestCase):
    def test_separate_full_cohorts_and_seed_without_recipe_change(self):
        plans=[code.plan(m,r) for r in (2,3) for m in code.METHODS]
        self.assertEqual(len({p['output'] for p in plans}),6)
        self.assertEqual(len({p['wandb_run_id'] for p in plans}),6)
        for p in plans:
            self.assertEqual(p['expected_rollout_task_ids'],plans[0]['expected_rollout_task_ids'])
            self.assertEqual(p['optimizer_updates_requested'],0)
            self.assertEqual(p['environment']['WANDB_PROJECT'],'openwebrl-evals')
            self.assertEqual(p['command'][p['command'].index('--seed')+1],str(1234+p['repeat']-1))
            self.assertNotIn('provider_recovery',p)
            self.assertEqual(p['expected_task_count'],300)

    def test_reject_subset_wrong_seed_or_reused_identity(self):
        p=code.plan('additive',2)
        for key,value in [('expected_task_count',172),('repeat',1),('wandb_run_id','old-run')]:
            q=copy.deepcopy(p);q[key]=value
            with self.assertRaises(ValueError):code.validate(q)
        p['command'][p['command'].index('--seed')+1]='1234'
        with self.assertRaises(ValueError):code.validate(p)

    def test_old_approval_cannot_fund_new_repeats(self):
        with patch.object(code.base,'read',return_value={'approved':True,'resources':code.first.RESOURCES}),patch.object(code.subprocess,'check_output') as scheduler:
            with self.assertRaisesRegex(ValueError,'Exact resource approval'):code.execute('999','additive',2)
            scheduler.assert_not_called()


if __name__=='__main__':unittest.main()
