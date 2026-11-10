import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import prepare_arm_ablation_to60 as original
import recover_arm_beta_memory as recovery


class BetaMemoryRecoveryTests(unittest.TestCase):
    def test_counts_both_failed_allocations(self):
        r=recovery.check_recovery_budget('330278|1729|FAILED\n331995|1210|FAILED\n')
        self.assertEqual(r['used_seconds'],2939)
        self.assertEqual(r['used_seconds']+r['requested_seconds'],50339)
        for raw in ['330278|1729|FAILED\n','330278|1729|FAILED\n331995|1210|RUNNING\n',
                    '330278|1729|FAILED\n331995|1300|FAILED\n']:
            with self.assertRaises(ValueError):recovery.check_recovery_budget(raw)

    def test_memory_change_preserves_saved_batch_topology_and_objective(self):
        root=original.ORIGINS['weight']
        before=original.plan('PREPARE','weight-recovery',root,30)
        after=recovery.plan('PREPARE','weight-recovery',root,30)
        recovery.validate_plan(after)
        for key in ['command','source','arm_config','checkpoint','resume_origin','replay_provenance',
                    'initial_optimizer_updates','start_rollout_id']:
            self.assertEqual(after[key],before[key],key)
        self.assertEqual(after['environment']['OPENWEBRL_CUDA_CACHE_LIMIT_GIB'],'24')
        self.assertEqual(after['environment']['TP_SIZE'],'2')
        self.assertEqual(after['requested_resources']['hours'],13+10/60)
        self.assertEqual(original.PROFILES['weight-recovery']['hours'],13.5)
        self.assertNotEqual(recovery.CONTROL,original.CONTROL)

    def test_follow_on_keeps_tighter_cache_with_original_fourteen_hour_budget(self):
        p=recovery.plan('PREPARE','weight-recovery',original.ORIGINS['weight'],30)
        root=original.RUNTIME/'evaluations/arm-failure-additive-PREPARE-tp2-iter40'
        origin=dict(manifest=p,checkpoint_report=dict(iteration=39,completed_optimizer_updates=560,
            checkpoint=str(root/'runtime/iter_0000039')))
        with patch.object(recovery.impl,'cached_origin',return_value=origin):
            later=recovery.plan('PREPARE','weight-to60',root,50)
        self.assertEqual(later['requested_resources']['hours'],14)
        self.assertEqual(later['environment']['OPENWEBRL_CUDA_CACHE_LIMIT_GIB'],'24')
        self.assertNotIn('OPENWEBRL_REPLAY_FIRST_BATCH',later['environment'])


if __name__=='__main__':unittest.main()
