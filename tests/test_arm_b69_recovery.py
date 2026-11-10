import ast
import copy
import os
from pathlib import Path
from unittest.mock import Mock,patch
import unittest
import recover_arm_b69 as b

class BRecoveryTests(unittest.TestCase):
    def test_budget_counts_every_failed_attempt(self):
        report=b.check_remaining_budget('330304|16909|FAILED\n331767|5|FAILED\n')
        self.assertEqual(report['used_seconds'],16914)
        self.assertEqual(report['used_seconds']+b.SECONDS,86394)
        for accounting in ('330304|16909|FAILED\n',
                           '330304|16909|FAILED\n331767|5|RUNNING\n',
                           '330304|16909|FAILED\n331767|12|FAILED\n'):
            with self.assertRaises(ValueError):b.check_remaining_budget(accounting)

    def test_resume_replays_only_unsaved_batch_with_unchanged_objective(self):
        p=b.plan('PREPARE');b.validate_plan(p)
        self.assertEqual((p['start_rollout_id'],p['requested_iterations'],p['initial_optimizer_updates']),(69,1,902))
        self.assertEqual(p['environment']['OPENWEBRL_REPLAY_ROLLOUT_ID'],'69')
        self.assertEqual(p['environment']['OPENWEBRL_CUDA_CACHE_LIMIT_GIB'],'48')
        self.assertEqual(p['environment']['TP_SIZE'],'2')
        self.assertEqual(p['environment']['WANDB_RUN_ID'],b.RUN_ID)
        self.assertFalse(p['fresh_optimizer'])
        self.assertLessEqual(4*3600+41*60+49+b.SECONDS,24*3600)
        before=b.read(b.ORIGIN/'launch_manifest.json')['arm_config']
        for key in ('beta','scored_fraction','candidate_gate','credit_assignment','failure_group_cap','seed'):
            self.assertEqual(p['arm_config'][key],before[key])

    def test_no_replay_at_later_stage_and_no_iteration100(self):
        prior=b.plan('PREPARE')
        root=b.RUNTIME/'evaluations/arm-failure-additive-PREPARE-tp2-iter70'
        origin=dict(manifest=prior,checkpoint_report=dict(iteration=69,completed_optimizer_updates=914,checkpoint=str(root/'runtime/iter_0000069')),root=str(root))
        with patch.object(b,'cached_origin',return_value=origin):
            p=b.plan('PREPARE',root,80)
            self.assertEqual(p['requested_iterations'],10)
            self.assertNotIn('OPENWEBRL_REPLAY_FIRST_BATCH',p['environment'])
            self.assertNotIn('gate_replay_origin',p['arm_config'])
            with self.assertRaises(ValueError):b.plan('PREPARE',root,100)

    def test_lower_cache_limit_releases_cached_not_live_memory(self):
        source=(b.SOURCE/'slime/utils/memory_utils.py').read_text()
        node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='release_unused_cuda_cache_under_pressure')
        cuda=Mock();cuda.memory_reserved.side_effect=[90*1024**3,22*1024**3];cuda.memory_allocated.return_value=21*1024**3
        ns=dict(os=os,torch=Mock(cuda=cuda),logger=Mock())
        exec(compile(ast.Module(body=[node],type_ignores=[]),'<cache-guard>','exec'),ns)
        with patch.dict(os.environ,OPENWEBRL_CUDA_CACHE_LIMIT_GIB='48'):
            ns[node.name]()
        cuda.empty_cache.assert_called_once();cuda.synchronize.assert_called_once()

if __name__=='__main__':unittest.main()
