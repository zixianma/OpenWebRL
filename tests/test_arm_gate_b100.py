import sys
from pathlib import Path
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import resume_arm_gate_b100 as runner


class ContinuationTests(unittest.TestCase):
    def test_finish_90_before_100(self):
        self.assertEqual(runner.next_stage(87,set()),('train',90))
        self.assertEqual(runner.next_stage(90,set()),('eval',90))
        self.assertEqual(runner.next_stage(90,{90}),('train',100))
        self.assertEqual(runner.next_stage(96,{90}),('train',100))
        self.assertEqual(runner.next_stage(100,{90}),('eval',100))
        self.assertEqual(runner.next_stage(100,{90,100}),('complete',100))

    def test_100_not_complete_without_90_evaluation(self):
        self.assertEqual(runner.next_stage(100,{100}),('eval',90))

    def test_short_retry_keeps_training_guard_and_required_evaluation(self):
        self.assertEqual(runner.evaluation_reserve(9720),0)
        self.assertEqual(runner.minimum_stage_seconds('train',9720),6000)
        self.assertEqual(runner.minimum_stage_seconds('eval',9720),2100)
        self.assertEqual(runner.next_stage(99,{90}),('train',100))
        self.assertEqual(runner.next_stage(100,{90}),('eval',100))
        self.assertEqual(runner.evaluation_reserve(57600),3600)
        self.assertEqual(runner.minimum_stage_seconds('train',57600),9600)

    def test_failed_attempts_cannot_reset_16h(self):
        approval=dict(approved=True,resources=runner.RESOURCES)
        b=dict(job_id='1',approved_total_seconds=57600,prior_used_seconds=1200,attempt_seconds=57600)
        with patch.object(runner,'read',side_effect=[approval,b]):
            with self.assertRaisesRegex(ValueError,'exceeds'):
                runner.budget('1')
        b['attempt_seconds']=56400
        with patch.object(runner,'read',side_effect=[approval,b]):
            self.assertEqual(runner.budget('1')['attempt_seconds'],56400)

    def test_keep_auxiliary_recipe_and_preserved_state(self):
        root='/gpfs/scrubbed/zixianma/openwebrl-runtime/evaluations/arm-failure-additive-fixture'
        prior=dict(source=str(runner.SOURCE),output=root,command=['launch'],environment=dict(TP_SIZE='4',
            GLOBAL_BATCH_SIZE='256',BROWSER_CONCURRENCY='64',OPENWEBRL_REPLAY_FIRST_BATCH='old83'),
            arm_config=dict(candidate_gate='min2',credit_assignment='response_index',beta=.5,scored_fraction=.2,
                failure_group_cap=8,failure_loss_coefficient=1/6,additive_failure_groups=True,
                deadline_epoch_seconds=1,replay_origin={'old':True}),gate_ablation='B',wandb_run_id=runner.previous.RUN_ID)
        report=dict(iteration=89,completed_optimizer_updates=1150,checkpoint=root+'/runtime/iter_0000089')
        with patch.object(runner,'origin',return_value=dict(manifest=prior,checkpoint_report=report)):
            p=runner.plan('PREPARE',Path(root),100)
            retry=runner.plan('PREPARE',Path(root),100,9720)
        self.assertEqual(p['initial_optimizer_updates'],1150)
        self.assertEqual(p['environment']['NUM_ROLLOUT'],'100')
        self.assertEqual(p['start_rollout_id'],90)
        self.assertEqual(p['requested_iterations'],10)
        self.assertFalse(p['fresh_optimizer'])
        self.assertEqual(p['arm_config']['failure_group_cap'],8)
        self.assertEqual(p['arm_config']['failure_loss_coefficient'],1/6)
        self.assertNotIn('replay_origin',p['arm_config'])
        self.assertNotIn('OPENWEBRL_REPLAY_FIRST_BATCH',p['environment'])
        self.assertTrue(Path(p['gpu_restore_output']).is_relative_to(root))
        self.assertEqual(retry['evaluation_reserve_seconds'],0)
        self.assertEqual(retry['arm_config']['minimum_cycle_seconds'],6000)
        self.assertEqual(retry['arm_config']['seconds_per_optimizer_update'],330)
        self.assertEqual(retry['checkpoint'],p['checkpoint'])
        self.assertEqual(retry['initial_optimizer_updates'],p['initial_optimizer_updates'])
        self.assertEqual(retry['arm_config'],p['arm_config'])

    def test_completion_allocation_counts_prior_attempts_and_exact_extension(self):
        approval = dict(approved=True, resources=runner.RESOURCES)
        extra = dict(approved=True, resources=dict(gpus=8, gpu_type='H200', cpus=64,
            memory_gib=960, allocation_seconds=10800, gpu_hours=24),
            additional_seconds=6495, consumed_seconds_at_approval=53295,
            remaining_seconds_at_approval=4305)
        b = dict(job_id='2', approved_total_seconds=64095,
            prior_used_seconds=53295, attempt_seconds=10800)
        with patch.object(runner,'read',side_effect=[approval,b,extra]):
            self.assertEqual(runner.budget('2')['prior_used_seconds'],53295)
        for changed, grant in [
            (dict(b, prior_used_seconds=0), extra),
            (dict(b, attempt_seconds=10801), extra),
            (dict(b, prior_used_seconds=53296), extra),
            (b, dict(extra, approved=False)),
            (b, dict(extra, additional_seconds=10800)),
        ]:
            with self.subTest(budget=changed, approval=grant):
                with patch.object(runner,'read',side_effect=[approval,changed,grant]):
                    with self.assertRaises(ValueError): runner.budget('2')


if __name__=='__main__':unittest.main()
