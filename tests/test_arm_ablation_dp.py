import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import prepare_arm_ablation_dp as stage


class AblationDP(unittest.TestCase):
    def test_native_checkpoint_ahead_of_controller_cannot_select_stale_resume(self):
        from requeue_arm_ablation_dp import boundary_report
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'runtime').mkdir();(root/'iterations/0000').mkdir(parents=True)
            (root/'runtime/latest_checkpointed_iteration.txt').write_text('1')
            report=dict(iteration=0,completed_optimizer_updates=16,checkpoint=str(root/'runtime/iter_0000000'))
            (root/'iterations/0000/checkpoint-validation.json').write_text(json.dumps(report))
            state=dict(stage='selector-startup',last_valid_checkpoint=report['checkpoint'],completed_optimizer_updates=16)
            self.assertIsNone(boundary_report(root,state,0))
            (root/'iterations/0001').mkdir()
            report.update(iteration=1,completed_optimizer_updates=30,checkpoint=str(root/'runtime/iter_0000001'))
            (root/'iterations/0001/checkpoint-validation.json').write_text(json.dumps(report))
            state.update(last_valid_checkpoint=report['checkpoint'],completed_optimizer_updates=30)
            self.assertEqual(boundary_report(root,state,0)['iteration'],1)

    def test_fresh_plans_preserve_initialization_and_both_scientific_recipes(self):
        from prepare_arm_failure_ablations import validate_initialization
        for variant,job in [('weight','318949'),('coverage','318950')]:
            p=stage.plan(variant,job,23.9)
            validate_initialization(p)
            self.assertEqual(p['environment']['TP_SIZE'],'2')
            self.assertEqual(p['requested_resources']['gpus'],8)
            self.assertEqual(p['environment']['GLOBAL_BATCH_SIZE'],'256')
            self.assertEqual(p['arm_config']['failure_beta'],1 if variant=='weight' else .5)
            self.assertEqual(p['arm_config']['failure_scored_fraction'],.2 if variant=='weight' else .4)
            self.assertEqual(p['start_rollout_id'],0)
            self.assertEqual(p['environment']['WANDB_RESUME'],'never')
            self.assertEqual(p['target_completed_iterations'],20)

    def test_continuation_keeps_optimizer_and_experiment_identity(self):
        from prepare_arm_failure_ablations import plan as fresh_plan
        prior=fresh_plan('weight','318949')
        initial=dict(root=str(stage.ORIGIN),manifest=prior,checkpoint_report=dict(iteration=1,
            completed_optimizer_updates=30,checkpoint=str(stage.ORIGIN/'runtime/iter_0000001')))
        with patch('resume_arm_turn_bonus.origin',return_value=initial),patch.dict(os.environ,{},clear=False):
            p=stage.plan('weight','318949',21,stage.ORIGIN)
        self.assertFalse(p['fresh_optimizer'])
        self.assertEqual(p['initial_optimizer_updates'],30)
        self.assertEqual(p['start_rollout_id'],2)
        self.assertEqual(p['environment']['WANDB_RESUME'],'must')
        self.assertEqual(p['wandb_run_id'],prior['wandb_run_id'])
        self.assertEqual(p['arm_config']['failure_beta'],1)
        self.assertIn('--use-checkpoint-opt-param-scheduler',p['command'])
        self.assertEqual(p['source'],str(stage.SOURCE))
        self.assertEqual(p['environment']['TP_SIZE'],'2')

    def test_dp_source_preserves_recipe_checks_and_auxiliary_capacity(self):
        stage.prepare_source()
        text=(stage.SOURCE/'openwebrl/arm_failure_aux.py').read_text()
        self.assertIn('beta = failure_beta(config)',text)
        self.assertIn("manifest['beta']!=failure_beta(config)",text)
        self.assertIn("[count*dp for count in counts],gbs,manifest['coefficient'],manifest.get('max_auxiliary_per_window',32)",text)
        self.assertIn('additions=shard_windows(additions,dp,rank)',text)

    def test_validation_uses_full_own_saved_batch_and_separate_tracking(self):
        p=stage.validation_plan('318949',stage.CONTROL/'test-preview')
        self.assertEqual(p['optimizer_updates'],14)
        self.assertEqual(p['expected_final_adam_updates'],30)
        self.assertEqual(p['environment']['WANDB_PROJECT'],'openwebrl-evals')
        self.assertIn(str(stage.ORIGIN/'runtime/rollout_recovery/1.pt'),p['command'])
        self.assertEqual(p['command'][p['command'].index('--load')+1],str(stage.CONTROL/'checkpoint-view'))
        self.assertNotIn('--colocate',p['command'])
        self.assertIn('--debug-train-only',p['command'])

    def test_controller_owns_validation_training_and_correct_endpoint_eval(self):
        for ready in (True,False):
            with self.subTest(ready=ready),tempfile.TemporaryDirectory() as tmp:
                r=Path(tmp);(r/'evaluations').mkdir();calls=[]
                p=dict(output=str(r/'training'))
                req=dict(variant='weight',stage='released',remaining_seconds=72000,resume_from='origin')
                def run(command,**kw):calls.append(command[command.index('--worker')+1])
                with patch.object(stage,'RUNTIME',r),patch.object(stage,'request',return_value=req),\
                     patch.object(stage,'require_ready'),patch.object(stage,'allocation'),\
                     patch.object(stage,'plan',return_value=p),patch.object(stage,'validate_plan'),\
                     patch.dict(os.environ,SLURM_JOB_ID='318949',SLURM_RESTART_COUNT='1'),\
                     patch.object(stage.subprocess,'check_output',return_value='job'),\
                     patch.object(stage.subprocess,'run',side_effect=run),\
                     patch('run_arm_iteration80_eval.checkpoint_ready',return_value=Path('checkpoint') if ready else None),\
                     patch('prepare_arm_failure_ablations.evaluation_plan',return_value={'output':tmp,'source':tmp}),\
                     patch('run_stage1_to100.audit_rollouts',return_value={'tasks':300}):
                    stage.execute('weight','318949')
                self.assertEqual(calls,['validate','train','eval'] if ready else ['validate','train'])
                state=json.loads((r/'evaluations/arm-failure-ablation-weight-318949-tp2/status.json').read_text())
                self.assertEqual(state['complete'],ready)


if __name__=='__main__':unittest.main()
