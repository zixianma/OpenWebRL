"""Bounded continuation must preserve beta's experiment and checkpoint state."""
import copy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import prepare_arm_beta_to40 as beta


class Beta40Tests(unittest.TestCase):
    def test_first_stage_is_same_lineage_at20(self):
        p=beta.plan('TEST')
        self.assertEqual((p['start_rollout_id'],p['initial_optimizer_updates']),(20,298))
        self.assertFalse(p['fresh_optimizer'])
        self.assertEqual(p['wandb_run_id'],beta.RUN_ID)
        self.assertEqual(p['environment']['WANDB_RESUME'],'must')
        self.assertEqual(p['environment']['TP_SIZE'],'2')
        self.assertEqual(p['environment']['NUM_ROLLOUT'],'30')
        self.assertIn('--use-checkpoint-opt-param-scheduler',p['command'])
        self.assertEqual(p['source'],str(beta.SOURCE))
        self.assertEqual(p['resume_origin']['checkpoint_report']['dataset_cursor'],str(beta.ORIGIN/'runtime/rollout/global_dataset_state_dict_19.pt'))

    def test_reward_and_sampling_knobs_unchanged(self):
        p=beta.plan('TEST');original=beta.read(beta.ORIGIN/'launch_manifest.json')
        for key in ['beta','scored_fraction','failure_beta','failure_scored_fraction','failure_group_cap','failure_loss_coefficient','k','seed']:
            self.assertEqual(p['arm_config'][key],original['arm_config'][key])
        beta.validate_plan(p)
        altered=copy.deepcopy(p);altered['arm_config']['failure_beta']=.5
        with self.assertRaises(ValueError):beta.validate_plan(altered)

    def test_no_target_above40(self):
        for target in [20,50,60,100]:
            with self.subTest(target=target),self.assertRaises(ValueError):beta.plan('TEST',target=target)

    def test_cannot_skip_iteration30_stage(self):
        with self.assertRaises(ValueError):beta.plan('TEST',target=40)

    def test_final_stage_stops_exactly_at40(self):
        origin=beta.cached_origin(beta.ORIGIN)
        origin['checkpoint_report']['iteration']=29
        origin['checkpoint_report']['completed_optimizer_updates']=450
        origin['checkpoint_report']['checkpoint']=str(beta.ORIGIN/'runtime/iter_0000029')
        with patch.object(beta,'cached_origin',return_value=origin):p=beta.plan('TEST',target=40)
        self.assertEqual((p['start_rollout_id'],p['requested_iterations'],p['target_completed_iterations']),(30,10,40))
        self.assertEqual(p['environment']['NUM_ROLLOUT'],'40')
        self.assertEqual(p['evaluation_reserve_seconds'],3600)
        self.assertEqual(p['requested_resources']['gpu_hours'],112)


if __name__=='__main__':unittest.main()
