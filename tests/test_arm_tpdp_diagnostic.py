"""Check sparse auxiliary DP scheduling and the native mean-loss scaling."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from openwebrl.arm_tpdp_diagnostic import shard_windows
from openwebrl.arm_failure_additive import window_schedule


class AuxiliaryDP(unittest.TestCase):
    def test_rows_unique_collectives_equal_and_padding_zero(self):
        for n in (0,1,3,4,7,32,65):
            windows=window_schedule([{}]*n,max(n,100),[256]*3,256,1/6)
            for dp in (1,2,4):
                shards=[shard_windows(windows,dp,r) for r in range(dp)]
                seen=[]
                for w in range(3):
                    self.assertEqual(len({len(s[w]) for s in shards}),1)
                    for s in shards:
                        for i,scale,pad in s[w]:
                            if pad:self.assertEqual(scale,0)
                            else:seen.append(i)
                self.assertEqual(sorted(seen),list(range(n)))

    def test_global_aux_gradient_and_outcome_mean_are_unchanged(self):
        # Analytic scalar gradients with heterogeneous row values and signs.
        # Megatron divides by local microbatch count and averages DP gradients;
        # slime contributes local_count/global_batch_size * DP in its loss.
        for n in (1,3,7,32):
            additions=window_schedule([{}]*n,365,[256],256,1/6)
            reference=None
            for dp in (1,2,4):
                gradients=[]
                for rank in range(dp):
                    local=shard_windows(additions,dp,rank)[0]
                    outcome=sum((i+1)/256 for i in range(rank,256,dp))
                    auxiliary=sum(scale*((i+1)*(-1)**i) for i,scale,_ in local)
                    microbatches=256//dp+len(local)
                    gradients.append((outcome+auxiliary)*microbatches/256*dp/microbatches)
                gradient=sum(gradients)/dp
                if reference is None:reference=gradient
                else:self.assertAlmostEqual(gradient,reference,places=12)

    def test_reject_unvalidated_layout(self):
        with self.assertRaises(ValueError):shard_windows([],3,0)
        with self.assertRaises(ValueError):shard_windows([],4,4)


class QueuedController(unittest.TestCase):
    def test_restore_disables_actual_tracking_gate_and_preserves_training_identity(self):
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
        import prepare_arm_tpdp_test as stage
        import ast
        from types import SimpleNamespace
        command=['python','train.py','--use-wandb','--load','input','--save','output',
                 '--num-rollout','24','--wandb-mode','online']
        environment={'WANDB_RUN_ID':'diagnostic','WANDB_RESUME':'never'}
        argv,env=stage.restore_command({'output':'case','command':command},environment,Path('reload'))
        self.assertNotIn('--use-wandb',argv)
        self.assertNotIn('WANDB_RUN_ID',env);self.assertNotIn('WANDB_RESUME',env)
        self.assertEqual(env['OPENWEBRL_VERIFY_RESUME_ONLY'],'1')
        self.assertIn('--use-wandb',command);self.assertEqual(environment['WANDB_RESUME'],'never')
        # Execute the preserved adapter's actual entry; it must return before
        # creating settings or contacting W&B, even without a WANDB_MODE guard.
        tree=ast.parse((stage.PARENT/'slime/utils/wandb_utils.py').read_text())
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='init_wandb_primary')
        ns={};exec(compile(ast.Module(body=[fn],type_ignores=[]),'tracking_entry','exec'),ns)
        args=SimpleNamespace(use_wandb='--use-wandb' in argv)
        ns['init_wandb_primary'](args)
        self.assertIsNone(args.wandb_run_id)

    def test_failure_returns_to_production_without_new_allocation(self):
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
        import prepare_arm_tpdp_test as stage
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'armed.json').write_text('{}')
            (root/'readiness.json').write_text(json.dumps(dict(cpu_passed=True,fingerprint={})))
            with patch.object(stage,'CONTROL',root),patch.object(stage,'fingerprint',return_value={}),\
                    patch.dict(os.environ,SLURM_JOB_ID=stage.JOB),\
                    patch.object(stage,'allocation',return_value={'maximum_seconds':80000}),\
                    patch.object(stage.subprocess,'check_output',return_value='allocation'),\
                    patch.object(stage.subprocess,'run') as run:
                run.return_value.returncode=1
                stage.run_if_armed(stage.JOB,'C',root)
            self.assertEqual(run.call_count,1)
            argv=run.call_args.args[0]
            self.assertEqual(argv[0],'srun');self.assertNotIn('--overlap',argv)
            self.assertIn('--gres=gpu:h200:8',argv)
            self.assertIn('--time=00:21:00',argv)
            result=json.loads((root/'tpdp-diagnostic.json').read_text())
            self.assertTrue(result['continue_production']);self.assertFalse(result['production_promotion'])

    def test_unrelated_jobs_and_running_b_are_untouched(self):
        import prepare_arm_tpdp_test as stage
        with patch.object(stage.subprocess,'run') as run:
            stage.run_if_armed('318934','B',Path('/unused'))
            stage.run_if_armed('OTHER','C',Path('/unused'))
            run.assert_not_called()

    def test_followup_preserves_evaluation_reserve(self):
        import prepare_arm_tpdp_test as stage
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);control=root/'control';control.mkdir()
            (control/'finish-at-c40.json').write_text(json.dumps({'initial_attempt_seconds':250}))
            (control/'readiness.json').write_text(json.dumps({'fingerprint':{}}))
            report=root/f'evaluations/arm-gate-c-to60-{stage.JOB}';report.mkdir(parents=True)
            with patch.object(stage,'CONTROL',control),patch.object(stage,'RUNTIME',root),\
                    patch.object(stage,'fingerprint',return_value={}),patch.dict(os.environ,SLURM_JOB_ID=stage.JOB),\
                    patch.object(stage,'allocation',return_value={'maximum_seconds':3600}),\
                    patch.object(stage.subprocess,'check_output',return_value='allocation'),\
                    patch.object(stage.subprocess,'run') as run:
                stage.finish_before_evaluation(stage.JOB,'C',40)
                run.assert_not_called()
            self.assertTrue(json.loads((report/'tpdp-followup.json').read_text())['continue_evaluation'])

    def test_followup_only_at_authorized_c40_boundary(self):
        import prepare_arm_tpdp_test as stage
        with patch.object(stage.subprocess,'run') as run:
            stage.finish_before_evaluation('318934','B',40)
            stage.finish_before_evaluation(stage.JOB,'C',60)
            run.assert_not_called()

    def test_three_way_followup_fits_original_budget(self):
        import prepare_arm_tpdp_test as stage
        steps=stage.followup_steps()
        self.assertEqual([s[1] for s in steps],[2,4,8])
        self.assertLessEqual(300+sum(s[3] for s in steps),stage.CAP_SECONDS)
        self.assertTrue(all(s[2]+20<=s[3] for s in steps))

    def test_fastest_must_pass_correctness_and_restore(self):
        from copy import deepcopy
        import prepare_arm_tpdp_test as stage
        def result(tp,seconds):
            return dict(tp=tp,complete=True,batch={'sha256':'same'},
                restore={'full_model_and_optimizer_load':'passed'},
                updates=[{'grad_norm':1.,'seconds':seconds}]*2)
        cases=[result(2,31),result(4,48),result(8,80)]
        self.assertEqual(stage.comparison_result(cases)['fastest_validated_replay_tp'],2)
        self.assertFalse(stage.comparison_result(cases)['production_promotion'])
        mismatch=deepcopy(cases);mismatch[0]['updates'][1]['grad_norm']=2
        self.assertEqual(stage.comparison_result(mismatch)['fastest_validated_replay_tp'],4)
        mismatch=deepcopy(cases);mismatch[0]['batch']['sha256']='changed'
        with self.assertRaisesRegex(ValueError,'matched'):stage.comparison_result(mismatch)
        mismatch=deepcopy(cases);mismatch[0]['restore']={}
        with self.assertRaisesRegex(ValueError,'restore'):stage.comparison_result(mismatch)


if __name__=='__main__':unittest.main()
