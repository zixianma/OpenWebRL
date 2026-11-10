"""Keep the short GPU diagnostic isolated from production experiments."""
import json
import io
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

sys.path.insert(0,str(Path('scripts').resolve()))
import run_arm_sol_gpu_diagnostic as diagnostic
import arm_gpu_smoke_fixture as fixture
import run_arm_live_shadow as live_shadow
from check_arm_native_launch import check_native_actor_entry
from openwebrl import arm_turn_bonus_cycles as cycles


class GpuDiagnostic(unittest.TestCase):
    def test_live_shadow_plan_uses_real_hook_and_never_trains(self):
        p=live_shadow.plan('CPU_LIVE','sol',20,1)
        self.assertEqual(p['environment']['SLIME_BROWSER_ENV_MODE'],'local_process')
        self.assertEqual(p['environment']['SLIME_BROWSER_LOCAL_PROCESS_MAX_PROCESSES'],'8')
        self.assertEqual(p['environment']['SLIME_BROWSER_ROLLOUT_CONCURRENCY'],'8')
        self.assertTrue(p['arm_config']['shadow_only'])
        self.assertFalse(p['arm_config']['train_after_calibration'])
        self.assertFalse(p['arm_config']['synthetic_rewards_and_labels'])
        self.assertEqual(p['arm_config']['scored_fraction'],.2)
        self.assertEqual(p['command'][p['command'].index('--rollout-function-path')+1],
            'slime.rollout.sglang_rollout.generate_rollout')
        self.assertEqual(len(p['diagnostic_data']['task_ids']),4)

    def test_live_shadow_result_requires_real_labels_and_zero_updates(self):
        p=dict(diagnostic_only=True,arm_config=dict(shadow_only=True))
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'iterations/0000'; root.mkdir(parents=True)
            (root/'collection_complete.json').write_text('{"optimizer_updates":0}')
            report=dict(applied_beta=0,collection_records=10,admitted=2)
            (root/'calibration.json').write_text(json.dumps(report))
            self.assertEqual(diagnostic.training.verify_live_shadow(p,temp,True)['admitted'],2)
            with self.assertRaisesRegex(ValueError,'zero optimizer'):
                diagnostic.training.verify_live_shadow(p,temp,False)
            report['admitted']=0; (root/'calibration.json').write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError,'lacks real'):
                diagnostic.training.verify_live_shadow(p,temp,True)

    def test_clean_partial_stop_does_not_pass_the_two_cycle_diagnostic(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'status.json'
            self.assertFalse(diagnostic.training_result_passed(temp))
            for row in (dict(training_complete=False,completed_iterations=1,completed_optimizer_updates=8),
                        dict(training_complete=True,completed_iterations=2,completed_optimizer_updates=8),
                        dict(training_complete=True,completed_iterations=2,completed_optimizer_updates=16,failed=True)):
                path.write_text(json.dumps(row)); self.assertFalse(diagnostic.training_result_passed(temp))
            path.write_text(json.dumps(dict(training_complete=True,completed_iterations=2,completed_optimizer_updates=16)))
            self.assertTrue(diagnostic.training_result_passed(temp))

    def test_training_plan_uses_native_tp2_without_changing_production_plan(self):
        p=diagnostic.training_plan('1234')
        self.assertEqual(p['requested_resources']['gpus'],2)
        self.assertEqual(p['requested_resources']['memory_gib'],480)
        self.assertEqual(p['requested_iterations'],2)
        self.assertTrue(p['diagnostic_only'])
        self.assertIn('gpu-diagnostic',p['wandb_run_id'])
        self.assertNotIn('--eval-interval',p['command'])
        self.assertIn('--use-rollout-logprobs',p['command'])
        self.assertEqual(p['command'][p['command'].index('--tensor-model-parallel-size')+1],'2')
        self.assertEqual(diagnostic.training.plan('1234')['requested_resources']['gpus'],4)
        interactive=diagnostic.training_plan('1234',interactive=True,attempt=1)
        self.assertEqual((interactive['requested_resources']['cpus'],interactive['requested_resources']['memory_gib']),(8,240))
        self.assertTrue(interactive['output'].endswith('-r1'))
        self.assertEqual(interactive['browser_config']['browser_rollout_concurrency'],8)

    def test_live_pilot_covers_only_two_tasks_and_preserves_full300_report(self):
        p=diagnostic.inference_plan('1234')
        self.assertEqual(sum(len(s['indices']) for s in p['segments']),2)
        self.assertTrue(p['diagnostic_only'])
        with tempfile.TemporaryDirectory() as temp,patch.object(diagnostic.inference,'write_json') as write:
            p['output']=temp
            result=diagnostic.inference.report(p,False)
            self.assertTrue(result['diagnostic_only'])
            self.assertFalse(result['complete'])
            self.assertEqual(write.call_count,1)
            self.assertEqual(write.call_args.args[0],Path(temp)/'comparison.json')

    def test_synthetic_generator_rejects_real_run_and_marks_all_test_rows(self):
        with patch.dict(os.environ,{},clear=True):
            with self.assertRaisesRegex(ValueError,'isolated'): fixture.generate_rollout(None,0,None)
        with tempfile.TemporaryDirectory() as temp,patch.dict(os.environ,ARM_GPU_DIAGNOSTIC='1'), \
            patch.object(cycles,'config',return_value=dict(run_id='arm-sol-gpu-diagnostic-test',
                output=temp,policy_id='test:rollout0')),patch.object(fixture,'sample_templates',return_value=[
                    dict(tokens=[1,2,3,4],response='test',response_length=2,rollout_log_probs=[-.1,-.2])]*5):
            source=SimpleNamespace(get_samples=Mock())
            result=fixture.generate_rollout(None,0,source)
            self.assertEqual(len(result.samples),1200)
            self.assertEqual(sum(bool(s.metadata['arm_turn_bonus']) for s in result.samples),120)
            self.assertTrue(all(s.metadata['diagnostic_synthetic'] for s in result.samples))
            self.assertTrue(all(len(s.rollout_log_probs)==s.response_length for s in result.samples))
            source.get_samples.assert_called_once_with(24)
            self.assertTrue(json.loads(Path(temp,'synthetic-fixture.json').read_text())['diagnostic_only'])

    def test_actual_native_ppo_entry_rejects_missing_old_policy_logprobs(self):
        args=SimpleNamespace(ppo_epochs=2,use_rollout_logprobs=True,use_routing_replay=False,
            use_rollout_routing_replay=False,use_critic=False)
        self.assertTrue(check_native_actor_entry(args))
        args.use_rollout_logprobs=False
        with self.assertRaisesRegex(ValueError,'requires --use-rollout-logprobs'):
            check_native_actor_entry(args)

    def test_templates_keep_actual_token_logprob_alignment_and_reject_missing_values(self):
        args=SimpleNamespace(hf_checkpoint='test',sglang_router_ip='127.0.0.1',sglang_router_port=1234)
        result=dict(text='example',meta_info=dict(output_token_logprobs=[[-.1,8,None],[-.2,9,None]]))
        tokenizer=SimpleNamespace(apply_chat_template=lambda *a,**kw:[1,2])
        with tempfile.TemporaryDirectory() as temp,patch('transformers.AutoTokenizer.from_pretrained',return_value=tokenizer), \
            patch.object(fixture.urllib.request,'urlopen',side_effect=lambda *a,**kw:io.BytesIO(json.dumps(result).encode())) as http:
            rows=fixture.sample_templates(args,1,temp)
            self.assertEqual(http.call_count,5)
            self.assertEqual(rows[0]['tokens'],[1,2,8,9])
            self.assertEqual(rows[0]['rollout_log_probs'],[-.1,-.2])
            seeds=[json.loads(call.args[0].data)['sampling_params']['sampling_seed'] for call in http.call_args_list]
            self.assertEqual(seeds,[47,48,49,50,51])
            result['meta_info']['output_token_logprobs'][0][0]=None
            with self.assertRaisesRegex(ValueError,'Missing or invalid'): fixture.sample_templates(args,1,temp)


if __name__=='__main__': unittest.main()
