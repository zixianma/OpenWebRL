import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import prepare_arm_failure_ablations as stage


class FailureAblations(unittest.TestCase):
    def test_prepared_recipes_start_from_zero_and_isolate_knobs(self):
        for variant in stage.VARIANTS:
            p=json.loads((stage.CONTROL/f'{variant}-plan.json').read_text())
            self.assertEqual(p['start_rollout_id'],0)
            self.assertEqual(p['initial_optimizer_updates'],0)
            self.assertTrue(p['fresh_optimizer'])
            self.assertEqual(p['target_completed_iterations'],20)
            self.assertNotIn('resume_from',p)
            self.assertNotIn('resume_origin',p)
            self.assertEqual(p['checkpoint'],str(stage.training.INITIAL))
            self.assertEqual(p['environment']['SLIME_LOAD_CHECKPOINT'],str(stage.training.INITIAL))
            self.assertNotIn('--use-checkpoint-opt-param-scheduler',p['command'])
            self.assertEqual(p['environment']['WANDB_PROJECT'],'openwebrl')
            self.assertEqual(p['environment']['WANDB_RESUME'],'never')
            self.assertNotEqual(p['wandb_run_id'],p['comparison_wandb_run'])
            c=p['arm_config']
            self.assertEqual((c['beta'],c['scored_fraction'],c['failure_group_cap']),(.5,.2,8))
            self.assertEqual(c['failure_beta'],1. if variant=='weight' else .5)
            self.assertEqual(c['failure_turn_budget'],0)
            self.assertEqual(c['failure_scored_fraction'],.4 if variant=='coverage' else .2)
            self.assertEqual(p['requested_resources']['gpus'],8)
            self.assertEqual(p['browser_config']['browser_rollout_concurrency'],64)

    def test_late_checkpoint_or_optimizer_resume_is_rejected(self):
        from copy import deepcopy
        p=json.loads((stage.CONTROL/'weight-plan.json').read_text())
        changes=[{'resume_from':'old-run'},{'initial_optimizer_updates':1262},
                 {'fresh_optimizer':False},{'start_rollout_id':100}]
        for change in changes:
            with self.subTest(change=change):
                altered=deepcopy(p);altered.update(change)
                with self.assertRaisesRegex(ValueError,'iteration0'):stage.validate_initialization(altered)
        altered=deepcopy(p);altered['command'].append('--use-checkpoint-opt-param-scheduler')
        with self.assertRaisesRegex(ValueError,'iteration0'):stage.validate_initialization(altered)

    def test_nonempty_coverage_pool_requires_completed_actual_labels(self):
        file=stage.SOURCE/'openwebrl/arm_turn_bonus_runtime.py'
        spec=importlib.util.spec_from_file_location('prepared_failure_runtime',file)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        for variant,rid,labels,stop in [('coverage',0,0,True),('coverage',0,1,False),
                                       ('coverage',1,0,True),('weight',0,0,False),('control',0,0,False)]:
            with self.subTest(variant=variant,rid=rid,labels=labels), tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);config=dict(shadow_only=False,output=tmp,beta=.5,
                    failure_ablation=variant,failure_ablation_start_rollout_id=0,
                    deadline_epoch_seconds=time.time()+10000)
                path=root/'arm-config.json';path.write_text(json.dumps(config))
                report=dict(batch_rows=1024,applied_beta=.5,failure_coverage=dict(complete=True,failure_coverage=dict(groups=1,labels=labels)))
                (root/'calibration.json').write_text(json.dumps(report))
                with patch.dict(os.environ,OPENWEBRL_ARM_TURN_BONUS_CONFIG=str(path)),patch('openwebrl.arm_turn_bonus.verify_shadow_complete'),patch.object(module,'calibration_decision',return_value=dict(passed=True,checks={'native':True})):
                    self.assertEqual(module.before_optimizer(SimpleNamespace(global_batch_size=256,ppo_epochs=2),rid),stop)
                gate=json.loads((root/'training_gate.json').read_text())
                self.assertEqual(gate['ready'],not stop)

    def controller(self,checkpoint):
        with tempfile.TemporaryDirectory() as tmp:
            runtime=Path(tmp);(runtime/'evaluations').mkdir();calls=[]
            def worker(cmd,**kwargs):calls.append(cmd[cmd.index('--worker')+1])
            with patch.object(stage,'RUNTIME',runtime),patch.dict(os.environ,SLURM_JOB_ID='123'),patch.object(stage,'allocation'),patch.object(stage.subprocess,'check_output',return_value='job'),patch.object(stage,'plan',return_value={}),patch.object(stage,'validate_plan'),patch.object(stage.subprocess,'run',side_effect=worker),patch('run_arm_iteration80_eval.checkpoint_ready',return_value=checkpoint),patch.object(stage,'evaluation_plan',return_value={'output':tmp,'source':tmp}),patch('run_stage1_to100.audit_rollouts',return_value={'tasks':300}):
                if checkpoint is None:
                    with self.assertRaisesRegex(ValueError,'before20'):stage.execute('weight','123')
                else:stage.execute('weight','123')
            status=json.loads((runtime/'evaluations/arm-failure-ablation-weight-123/status.json').read_text())
            return calls,status

    def test_controller_owns_train_then_eval(self):
        calls,status=self.controller(Path('iter_0000019'))
        self.assertEqual(calls,['train','eval']);self.assertTrue(status['complete'])

    def test_short_training_does_not_evaluate_wrong_checkpoint(self):
        calls,status=self.controller(None)
        self.assertEqual(calls,['train']);self.assertFalse(status['complete'])


if __name__=='__main__':unittest.main()
