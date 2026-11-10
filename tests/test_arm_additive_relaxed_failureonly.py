"""Check real reward routing against the frozen source; CPU only."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
torch.set_num_threads(1)
from openwebrl import arm_turn_bonus as bonus
from openwebrl import arm_failure_additive as add
from openwebrl.arm_gate_recovery import calibration_guard
from test_arm_failure_bonus import failed_group
from test_arm_turn_bonus_pipeline import fixture
import prepare_arm_additive_relaxed_failureonly as launcher


class FailureOnly(unittest.TestCase):
    def test_fresh_plan_and_native_argument_resolution(self):
        p=launcher.plan();launcher.validate_recipe(p)
        self.assertEqual(p['environment']['TP_SIZE'],'2')
        self.assertEqual(p['requested_resources']['gpus'],8)
        self.assertEqual(p['evaluation_reserve_seconds'],7200)
        for key,val in [('beta',.5),('scored_fraction',.4),('credit_assignment','action_class')]:
            q=deepcopy(p);q['arm_config'][key]=val
            with self.assertRaises(ValueError):launcher.validate_recipe(q)
        q=deepcopy(p);q['resume_from']='late-checkpoint'
        with self.assertRaises(ValueError):launcher.validate_recipe(q)

    def test_guard_keeps_reliability_checks_with_zero_mixed_bonus(self):
        c=launcher.plan()['arm_config']
        report=dict(config=c,variants={'0.5':dict(bonus_to_outcome_rms=.10)})
        d=dict(passed=True,checks=dict(scale=True,labels=True,tasks=True,coverage=True),beta=.5)
        result=calibration_guard(report,d)
        self.assertTrue(result['passed']);self.assertEqual(result['beta'],0)
        self.assertTrue(result['scale_is_counterfactual_mixed_beta05'])
        d['checks']['labels']=False
        self.assertFalse(calibration_guard(report,d)['passed'])
        report['config']=dict(c,mixed_outcome_only=False)
        with self.assertRaises(ValueError):calibration_guard(report,d)

    def test_zero_mixed_bonus_preserves_failure_rewards_and_admission(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows,current,args,_,_=fixture(tmp,training=True,beta=0.)
            current['config'].update(admit_all_failure_groups=True,additive_failure_groups=True,
                failure_group_cap=8,failure_loss_coefficient=1/6,rollout_id=0,
                candidate_gate='min2',credit_assignment='response_index',
                gate_scale_guard='pretraining_coverage_audit_v1',mixed_outcome_only=True)
            # A positive and negative executed-response label remain usable even
            # when duplicate actions are present; no action-class credit here.
            group=failed_group(rows)
            for i,trajectory in enumerate(group):
                trajectory[0].metadata['arm_turn_bonus'].update(candidate_gate='min2',
                    selected_index=0 if i==0 else 1,
                    **bonus.candidate_bonus([0,0,1,2,3],0 if i==0 else 1,'response_index'))
            with patch.object(bonus,'_STATE',current):
                result=add.filter_groups(args,group)
                self.assertFalse(result.keep)  # never occupies an ordinary slot
                self.assertEqual(len(current['failure_additive_pool']),1)
                mixed=[]
                for i in range(48):
                    sample=deepcopy(rows[0]);sample.group_index=i
                    sample.metadata.pop('arm_all_failure_group',None)
                    mixed.extend([sample]*16)
                raw=[float(i%2) for i in range(len(mixed))]
                normalized=[1. if i%2 else -1. for i in range(len(mixed))]
                # Label-quality gate is independently exercised above. Stub it
                # only for this small routing fixture, not either reward path.
                with patch('openwebrl.arm_turn_bonus_runtime.calibration_decision',
                           return_value=dict(passed=True,checks={})):
                    returned_raw,returned_adv=bonus.post_process_rewards(args,mixed,raw,normalized)
                self.assertIs(returned_raw,raw);self.assertIs(returned_adv,normalized)
                manifest=json.loads((Path(tmp)/'failure_auxiliary.json').read_text())
                self.assertEqual(manifest['mixed_groups'],48)
                self.assertEqual(manifest['failure_groups'],1)
                self.assertEqual(manifest['total_failure_rows'],50)
                self.assertAlmostEqual(manifest['coefficient'],1/48)
                self.assertEqual(manifest['beta'],.5)
                values=[r['advantage'] for r in manifest['records']]
                self.assertEqual(len(values),5)
                self.assertEqual(sum(v>.0 for v in values),1)
                self.assertTrue(all(abs(v-(.4 if v>0 else -.1))<1e-9 for v in values))
                report=json.loads((Path(tmp)/'calibration.json').read_text())
                self.assertEqual(report['applied_beta'],0.)
                self.assertEqual(report['outcome_groups_preserved'],48)


if __name__=='__main__':unittest.main()
