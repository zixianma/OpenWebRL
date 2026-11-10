"""A continued gate ablation must validate its saved optimizer, not SFT init."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import run_arm_turn_bonus_cycles as training
import resume_arm_failure_variants as continuation
import run_arm_gate_ablation as fresh


class Routing(unittest.TestCase):
    def test_trained_gate_run_uses_continuation_validation(self):
        p=dict(gate_ablation='C',gate_c_restart=True,variant_continuation=True)
        with patch.object(continuation,'validate_plan',side_effect=RuntimeError('saved optimizer validation')) as resume:
            with patch.object(fresh,'validate_plan',side_effect=RuntimeError('wrong fresh-start validation')) as start:
                with self.assertRaisesRegex(RuntimeError,'saved optimizer validation'):
                    training.execute(p)
                resume.assert_called_once_with(p)
                start.assert_not_called()

    def test_new_gate_run_keeps_fresh_start_validation(self):
        p=dict(gate_ablation='B')
        with patch.object(continuation,'validate_plan',side_effect=RuntimeError('wrong resume validation')) as resume:
            with patch.object(fresh,'validate_plan',side_effect=RuntimeError('fresh-start validation')) as start:
                with self.assertRaisesRegex(RuntimeError,'fresh-start validation'):
                    training.execute(p)
                start.assert_called_once_with(p)
                resume.assert_not_called()
