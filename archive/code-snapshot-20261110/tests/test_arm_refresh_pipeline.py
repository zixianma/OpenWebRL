"""Prevent duplicate paid submissions and leakage in the approved refresh."""
import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import watch_arm_refresh as watcher
from run_arm_refresh_training import verify_time_budget,stage_names,completed_training_receipt
from openwebrl.arm_refresh_generation import validate_inputs,digest

class RefreshPipelineTests(unittest.TestCase):
    def test_evaluation_recovery_never_retrains_or_runs_backward(self):
        self.assertEqual(stage_names('evaluation'),('environment','grammar-smoke','evaluation'))
        with self.assertRaisesRegex(ValueError,'Unknown'):stage_names('resume-whatever')

    def test_evaluation_rejects_incomplete_training_endpoint(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            (root/'trainer_state.json').write_text(json.dumps(dict(global_step=85,max_steps=90,epoch=.95)))
            with self.assertRaisesRegex(ValueError,'completed predefined'):completed_training_receipt(root)

    def test_completed_endpoint_preserves_checkpoint_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);cp=root/'checkpoint-90';cp.mkdir()
            state=json.dumps(dict(global_step=90,max_steps=90,epoch=1.0))
            for path in (root/'trainer_state.json',cp/'trainer_state.json'):path.write_text(state)
            for name in ('adapter_model.safetensors','adapter_config.json','train_results.json','eval_results.json'):(root/name).write_text('saved')
            for name in ('optimizer.pt','scheduler.pt','rng_state.pth','adapter_model.safetensors'):(cp/name).write_text('saved')
            self.assertIn('adapter_model.safetensors',completed_training_receipt(root))
            (cp/'adapter_model.safetensors').write_text('different')
            with self.assertRaisesRegex(ValueError,'differs from final'):completed_training_receipt(root)

    def test_recovery_cannot_reset_original_budget(self):
        receipt=dict(job_id='456',time_limit_seconds=14340,
            prior_attempts=[dict(job_id='123',elapsed_seconds=29)])
        verify_time_budget({'TimeLimit':'03:59:00'},'456',receipt)
        with self.assertRaisesRegex(ValueError,'remaining approved'):
            verify_time_budget({'TimeLimit':'04:00:00'},'456',dict(receipt,time_limit_seconds=14400))
        with self.assertRaisesRegex(ValueError,'current submission'):
            verify_time_budget({'TimeLimit':'03:59:00'},'999',receipt)

    def test_completed_job_absent_from_queue_is_not_a_command_failure(self):
        from subprocess import CompletedProcess
        result=CompletedProcess([],1,'','slurm_load_jobs error: Invalid job id specified')
        with patch.object(watcher.subprocess,'run',return_value=result):
            self.assertEqual(watcher.queue_state('123'),'')

    def test_queue_failure_is_not_misclassified_as_job_completion(self):
        from subprocess import CompletedProcess
        result=CompletedProcess([],1,'','Unable to contact slurm controller')
        with patch.object(watcher.subprocess,'run',return_value=result):
            with self.assertRaisesRegex(RuntimeError,'queue lookup failed'):watcher.queue_state('123')

    def setup_state(self,root,cost=226):
        control=root/'control';c=control/'baseline40';c.mkdir(parents=True)
        def save(path,value):
            path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
        save(c/'candidate-submission.json',{'job_id':'123'})
        save(root/'evaluations/arm-refresh-candidates-123/status.json',{'complete':True})
        save(c/'data/manifest.json',{'passed':True})
        save(c/'teacher/manifest.json',{'cost_ceiling_usd':cost})
        save(control/'baseline40-approval.json',{'approved':True,'teacher_api_cap_usd':225})
        return control,c,save

    def test_cost_cap_stops_before_any_api_command(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);control,c,_=self.setup_state(root)
            with patch.multiple(watcher,R=root,CONTROL=control,C=c),patch.object(watcher,'command') as command:
                with self.assertRaisesRegex(RuntimeError,'exceeds approved cap'):watcher.tick()
                command.assert_not_called()

    def test_unresolved_sbatch_intent_prevents_duplicate_submission(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);control,c,save=self.setup_state(root,cost=100)
            save(c/'teacher/approval.json',{})
            save(c/'teacher/pilot-passed.json',{'passed':True})
            save(c/'teacher/label-status.json',{'complete':True})
            save(c/'training-readiness.json',{'passed':True})
            save(c/'training-submission.intent.json',{'created_epoch':1})
            with patch.multiple(watcher,R=root,CONTROL=control,C=c),patch.object(watcher,'command') as command,patch.object(watcher.subprocess,'check_output') as submit:
                with self.assertRaisesRegex(RuntimeError,'Unreconciled sbatch'):watcher.tick()
                command.assert_not_called();submit.assert_not_called()

    def test_worker_failure_survives_job_leaving_live_queue(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);control,c,save=self.setup_state(root,cost=100)
            for name,value in {
                'teacher/approval.json':{},'teacher/pilot-passed.json':{'passed':True},
                'teacher/label-status.json':{'complete':True},'training-readiness.json':{'passed':True},
                'training-submission.json':{'job_id':'456'},
                'training-status.json':{'job':'456','stage':'failed','complete':False},
            }.items():save(c/name,value)
            with patch.multiple(watcher,R=root,CONTROL=control,C=c),patch.object(watcher,'queue_state') as queue:
                with self.assertRaisesRegex(RuntimeError,'worker failed'):watcher.tick()
                queue.assert_not_called()

    def test_generation_rejects_same_task_in_train_and_dev(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'states.jsonl'
            path.write_text('\n'.join(json.dumps(dict(id=str(i),task_group='same',split=s)) for i,s in enumerate(('train','dev'))))
            config={'states':str(path),'states_sha256':digest(path.read_bytes()),'states_count':2}
            with self.assertRaisesRegex(ValueError,'Task split overlap'):validate_inputs(config)

if __name__=='__main__':unittest.main()
