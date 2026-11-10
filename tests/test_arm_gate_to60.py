import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import prepare_arm_gate_to60 as stage
import monitor_arm_turn_bonus as monitor


class Controller(unittest.TestCase):
    def test_controller_awaits_both_training_and_evaluations_in_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            runtime=Path(tmp);(runtime/'evaluations').mkdir();calls=[]
            args=SimpleNamespace(job_id='123',variant='B',target=60,resume_from=runtime/'evaluations/origin')
            def worker(command,**kwargs):
                calls.append((command[command.index('--target')+1],command[command.index('--worker')+1]))
                self.assertIn('--gres=gpu:h200:8',command)
                self.assertIn('--cpus-per-task=64',command)
            with patch.object(stage,'RUNTIME',runtime),patch.object(stage,'prepare_sources'),patch.object(stage,'allocation'),patch.dict(os.environ,SLURM_JOB_ID='123'),patch.object(stage.subprocess,'check_output',return_value='allocation'),patch.object(stage.subprocess,'run',side_effect=worker),patch('run_arm_gate_checkpoint_eval.checkpoint_ready',return_value=Path('checkpoint')),patch.object(stage,'evaluation_plan',return_value={'output':tmp,'source':tmp}),patch('run_stage1_to100.audit_rollouts',return_value={'tasks':300}):
                stage.execute(args)
            self.assertEqual(calls,[('40','train'),('40','eval'),('60','train'),('60','eval')])
            status=json.loads((runtime/'evaluations/arm-gate-b-to60-123/status.json').read_text())
            self.assertTrue(status['complete']);self.assertEqual(status['completed_stages'],[40,60])

    def test_short_training_blocks_eval_and_next_stage(self):
        with tempfile.TemporaryDirectory() as tmp:
            runtime=Path(tmp);(runtime/'evaluations').mkdir();args=SimpleNamespace(job_id='123',variant='C',target=60,resume_from=runtime/'evaluations/origin')
            with patch.object(stage,'RUNTIME',runtime),patch.object(stage,'prepare_sources'),patch.object(stage,'allocation'),patch.dict(os.environ,SLURM_JOB_ID='123'),patch.object(stage.subprocess,'check_output',return_value='allocation'),patch.object(stage.subprocess,'run') as worker,patch('run_arm_gate_checkpoint_eval.checkpoint_ready',return_value=None):
                with self.assertRaisesRegex(ValueError,'stopped short'):stage.execute(args)
            self.assertEqual(worker.call_count,1)
            status=json.loads((runtime/'evaluations/arm-gate-c-to60-123/status.json').read_text())
            self.assertFalse(status['complete'])

    def test_monitor_follows_active_stage_not_first_stage(self):
        with tempfile.TemporaryDirectory() as tmp:
            runtime=Path(tmp);controller=runtime/'evaluations/arm-gate-c-to60-123';controller.mkdir(parents=True)
            for target in (40,60):
                root=runtime/f'evaluations/arm-failure-additive-123-iter{target}'
                (controller/'controller-plan.json').write_text(json.dumps(dict(training_root=str(root))))
                with patch.object(monitor,'RUNTIME',runtime):self.assertEqual(monitor.run_root('123'),root)


if __name__=='__main__':unittest.main()
