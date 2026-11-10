import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
import recover_arm_beta_tp4 as recovery


class Recovery(unittest.TestCase):
    def test_science_and_resume_unchanged_with_smaller_tensor_shards(self):
        root=recovery.impl.ORIGINS['weight']
        old=recovery.original_plan('PREPARE','weight-recovery',root,30)
        new=recovery.plan('PREPARE','weight-recovery',root,30)
        recovery.validate_plan(new)
        for key in ('checkpoint','source','resume_origin','replay_provenance','initial_optimizer_updates','start_rollout_id'):
            self.assertEqual(new[key],old[key],key)
        for key in ('beta','failure_beta','scored_fraction','failure_scored_fraction','failure_group_cap','candidate_gate','credit_assignment'):
            self.assertEqual(new['arm_config'][key],old['arm_config'][key],key)
        self.assertEqual(new['environment']['TP_SIZE'],'4')
        self.assertEqual(new['environment']['GLOBAL_BATCH_SIZE'],'256')
        self.assertFalse(new['fresh_optimizer'])
        self.assertEqual(round(new['requested_resources']['hours']*3600),48660)

    def test_cumulative_budget_rejects_live_or_overbudget_attempt(self):
        for raw in ('333402|2000|FAILED\n','333402|1682|RUNNING\n'):
            with patch.object(recovery.subprocess,'check_output',return_value=raw):
                with self.assertRaises(ValueError):recovery.check_budget(None)
        with patch.object(recovery.subprocess,'check_output',return_value='333402|1682|FAILED\n'):
            self.assertEqual(recovery.check_budget(None)['used_seconds'],1682)


if __name__=='__main__':unittest.main()
