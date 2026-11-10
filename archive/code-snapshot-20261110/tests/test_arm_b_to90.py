"""B continuation must not replay B49 or bypass unfinished B60/evaluations."""
import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import prepare_arm_b_to90 as b


def synthetic_origin(start=60):
    # The real B60 checkpoint does not exist yet. Only synthesize counters for
    # planning tests; never write this as a production checkpoint receipt.
    prior=b.read(b.RUNTIME/'arm-turn-bonus-preparation/gate-b-to60-20260924/plan.json')
    root=b.ORIGIN if start==60 else b.RUNTIME/f'evaluations/arm-failure-additive-TEST-tp2-iter{start}'
    return dict(root=str(root),manifest=prior,identity_sha256={},checkpoint_report=dict(
        iteration=start-1,completed_optimizer_updates=1000+start,
        checkpoint=str(root/f'runtime/iter_{start-1:07d}')))


class B90Tests(unittest.TestCase):
    def test_same_lineage_and_no_stale_batch_replay(self):
        origin=synthetic_origin()
        with patch.object(b,'cached_origin',return_value=origin):
            p=b.plan('TEST');b.validate_plan(p)
        self.assertEqual((p['start_rollout_id'],p['initial_optimizer_updates']),(60,1060))
        self.assertFalse(p['fresh_optimizer'])
        self.assertEqual(p['wandb_run_id'],b.RUN_ID)
        self.assertEqual(p['environment']['TP_SIZE'],'2')
        self.assertEqual(p['environment']['WANDB_RESUME'],'must')
        self.assertIn('--use-checkpoint-opt-param-scheduler',p['command'])
        self.assertNotIn('saved_batch_continuation',p)
        for key in ('gate_replay_origin','replay_origin'):self.assertNotIn(key,p['arm_config'])
        for key in ('OPENWEBRL_REPLAY_FIRST_BATCH','OPENWEBRL_REPLAY_ROLLOUT_ID','OPENWEBRL_ARM_REPLAY_CURSOR'):
            self.assertNotIn(key,p['environment'])
        for key in ('beta','scored_fraction','candidate_gate','credit_assignment','gate_scale_guard',
                    'failure_loss_coefficient','failure_group_cap','k','seed'):
            self.assertEqual(p['arm_config'][key],origin['manifest']['arm_config'][key])
        self.assertIn('gate_replay_origin',origin['manifest']['arm_config'])

    def test_all_three_stages_and_budget(self):
        for target in (70,80,90):
            origin=synthetic_origin(target-10)
            with self.subTest(target=target),patch.object(b,'cached_origin',return_value=origin):
                p=b.plan('TEST',Path(origin['root']),target)
                self.assertEqual(p['environment']['NUM_ROLLOUT'],str(target))
                self.assertEqual(p['requested_resources']['gpu_hours'],192)
                self.assertEqual(p['evaluation_reserve_seconds'],(100-target)//10*3600)
                self.assertEqual(p['requested_iterations'],10)

    def test_no_skipped_or_extra_stage(self):
        with patch.object(b,'cached_origin',return_value=synthetic_origin()):
            for target in (60,80,90,100):
                with self.subTest(target=target),self.assertRaises(ValueError):b.plan('TEST',target=target)

    def test_wrong_parent_and_modified_objective_rejected(self):
        with patch.object(b,'cached_origin',return_value=synthetic_origin()):
            with self.assertRaises(ValueError):b.plan('TEST',root=b.ORIGIN.parent/'another-run')
            p=b.plan('TEST');p['arm_config']['credit_assignment']='action_class'
            with self.assertRaises(ValueError):b.validate_plan(p)

    def test_predecessor_exit_is_insufficient(self):
        with patch.object(Path,'is_file',return_value=True),patch.object(b,'read',return_value=dict(stage='budget-stop',complete=False)):
            self.assertFalse(b.prerequisite_ready())

    def test_missing_preceding_eval_blocks_release(self):
        def read(path):
            if Path(path).name=='status.json':return dict(complete=True,evaluated_iterations=[50,60])
            return dict(tasks=300,saved_rollouts=299,saved_verdicts=300)
        with patch.object(Path,'is_file',return_value=True),patch.object(b,'read',side_effect=read),patch.object(b,'cached_origin',return_value=synthetic_origin()),patch('run_arm_gate_checkpoint_eval.checkpoint_ready',return_value=Path('/synthetic')):
            self.assertFalse(b.prerequisite_ready())


if __name__=='__main__':unittest.main()
