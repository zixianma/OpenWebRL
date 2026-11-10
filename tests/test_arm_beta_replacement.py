import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
import prepare_arm_beta_replacement as replacement
import prepare_arm_ablation_to60 as original


class BetaReplacementTests(unittest.TestCase):
    def test_continues_existing_recipe_and_replays_saved_batch(self):
        root=original.ORIGINS['weight']
        old=original.plan('PREPARE','weight-recovery',root,30)
        new=replacement.plan('PREPARE','weight-recovery',root,30)
        replacement.validate_plan(new)
        for key in ['checkpoint','source','command','arm_config','resume_origin',
                    'replay_provenance','initial_optimizer_updates','start_rollout_id']:
            self.assertEqual(old[key],new[key],key)
        self.assertEqual(new['requested_resources']['hours'],14)
        self.assertEqual(new['environment']['TP_SIZE'],'2')
        self.assertEqual(new['environment']['OPENWEBRL_CUDA_CACHE_LIMIT_GIB'],'24')
        self.assertFalse(new['fresh_optimizer'])
        self.assertEqual(replacement.impl.PROFILES['weight-recovery']['end'],60)

    def test_missing_exact_resource_approval_blocks_execution(self):
        with tempfile.TemporaryDirectory() as folder,patch.object(replacement,'CONTROL',Path(folder)):
            p=Path(folder)/'weight-recovery-approval.json'
            p.write_text(json.dumps({'approved':False,'resources':None}))
            with patch.object(replacement.subprocess,'check_output') as query:
                with self.assertRaisesRegex(ValueError,'Exact user resource approval'):
                    replacement.check_budget(None)
                query.assert_not_called()

    def test_used_budget_or_live_predecessor_cannot_be_replaced(self):
        with tempfile.TemporaryDirectory() as folder,patch.object(replacement,'CONTROL',Path(folder)):
            (Path(folder)/'weight-recovery-approval.json').write_text(json.dumps(
                {'approved':True,'resources':replacement.impl.resources('weight-recovery')}))
            for raw in ['332004|1|CANCELLED\n','332004|0|PENDING\n']:
                with patch.object(replacement.subprocess,'check_output',return_value=raw):
                    with self.assertRaises(ValueError):replacement.check_budget(None)
            with patch.object(replacement.subprocess,'check_output',return_value='332004|0|CANCELLED\n'):
                self.assertEqual(replacement.check_budget(None)['requested_seconds'],50400)


if __name__=='__main__':unittest.main()
