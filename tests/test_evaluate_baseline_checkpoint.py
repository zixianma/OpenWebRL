"""CPU checks for checkpoint identity, output isolation, and eval-only requests."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location('evaluate_baseline_checkpoint', SCRIPTS / 'evaluate_baseline_checkpoint.py')
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


class EvaluationTest(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix='openwebrl-eval-plan-test-'))
        self.ckpt = self.root / 'run/iter_0000021'
        self.ckpt.mkdir(parents=True)
        for name in ['common.pt', '.metadata', '__0_0.distcp']:
            (self.ckpt / name).write_bytes(b'fixture')
        (self.ckpt.parent / 'rollout').mkdir()
        (self.ckpt.parent / 'rollout/global_dataset_state_dict_21.pt').write_bytes(b'cursor')
        self.output = self.root / 'evaluations/new'

    def plan(self):
        with patch.object(m, 'RUNTIME', self.root), patch.object(m, 'validate_source'):
            return m.build_plan(self.root / 'source', self.ckpt, self.output, '42')

    def test_exact_checkpoint_eval_only_and_separate_wandb(self):
        p = self.plan()
        self.assertEqual(p['completed_training_iterations'], 22)
        self.assertEqual(p['environment']['NUM_ROLLOUT'], '0')
        self.assertEqual(p['environment']['SLIME_CKPT_STEP'], '21')
        self.assertNotEqual(p['wandb_run_id'], 'qcq7i4ug')
        self.assertEqual(p['environment']['RAY_ADDRESS'], 'local')
        self.assertIn('--eval-config', p['command'])
        self.assertEqual(p['command'][p['command'].index('--lr-decay-iters')+1], '1')
        self.assertIn('--use-checkpoint-opt-param-scheduler', p['command'])
        self.assertFalse(self.output.exists())

    def test_missing_checkpoint_component_rejected(self):
        self.ckpt = self.root / 'run/iter_0000022'
        self.ckpt.mkdir()
        with self.assertRaisesRegex(ValueError, 'Missing checkpoint component'):
            self.plan()

    def test_existing_output_rejected(self):
        self.output.mkdir(parents=True)
        with self.assertRaisesRegex(ValueError, 'new output'):
            self.plan()

    def test_execution_outside_allocation_rejected(self):
        with patch.dict(m.os.environ, {'SLURM_JOB_ID': 'other'}):
            with self.assertRaisesRegex(ValueError, 'authorized Slurm'):
                m.run(self.plan(), self.root / '.env')


if __name__ == '__main__':
    unittest.main()
