"""Admission, native zero-variance normalization, and gradient direction."""
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
torch.set_num_threads(1)
sys.path.insert(0,str(Path(__file__).resolve().parent))
from test_arm_turn_bonus_pipeline import fixture, native
from openwebrl import arm_turn_bonus as bonus
from openwebrl import arm_turn_bonus_runtime as runtime
from openwebrl import arm_failure_bonus as failure


def failed_group(rows):
    group=[]
    for i in range(5):
        trajectory=deepcopy(rows[i*10:(i+1)*10])
        for s in trajectory:
            s.reward=0.
            s.metadata.update(terminate_reason='task_completed',
                reward=dict(combined=0.,judge=0.,judge_text='NOT SUCCESS',
                            judge_prompt_variant='action_history',judge_timeout=False))
        group.append(trajectory)
    return group


class FailureAdmission(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.rows,self.current,self.args,self.ns,self.manager=fixture(self.temp.name,training=True)
        self.current['config']['admit_all_failure_groups']=True
        self.patcher=patch.object(bonus,'_STATE',self.current);self.patcher.start();self.addCleanup(self.patcher.stop)
        env=patch.dict(os.environ,OPENWEBRL_ARM_TURN_BONUS_CONFIG=str(Path(self.temp.name)/'arm-config.json'))
        env.start();self.addCleanup(env.stop)
        self.group=failed_group(self.rows)

    def test_mixed_and_all_success_keep_native_behavior(self):
        mixed=[self.rows[i*10:(i+1)*10] for i in range(5)]
        self.assertTrue(failure.filter_groups(self.args,mixed).keep)
        for t in mixed:
            for s in t:s.reward=1.
        r=failure.filter_groups(self.args,mixed)
        self.assertFalse(r.keep);self.assertEqual(r.reason,'zero_std_1.0')

    def test_negative_only_labels_still_admit_without_winner_selection_bias(self):
        for t in self.group:
            t[0].metadata['arm_turn_bonus'].update(selected_index=1,unit_bonus=-.2)
        self.assertTrue(failure.filter_groups(self.args,self.group).keep)

    def test_missing_rewards_infra_bad_verdict_and_stale_labels(self):
        cases=[('missing',lambda g:setattr(g[0][-1],'reward',None)),
               ('timeout',lambda g:g[0][-1].metadata['reward'].update(judge_timeout=True)),
               ('api_error',lambda g:g[0][-1].metadata['reward'].update(judge_text='All judge API retry attempts exhausted.')),
               ('unknown_verdict',lambda g:g[0][-1].metadata['reward'].update(judge_text='Cannot decide')),
               ('env_error',lambda g:g[0][-1].metadata.update(terminate_reason='env_step_error')),
               ('removed',lambda g:setattr(g[0][0],'remove_sample',True)),
               ('overlap',lambda g:g[0][0].metadata.update(arm_calibration_excluded=True)),
               ('identity',lambda g:g[0][0].metadata.update(task_id='different'))]
        for name,mutate in cases:
            with self.subTest(name=name):
                g=deepcopy(self.group);mutate(g)
                self.assertFalse(failure.filter_groups(self.args,g).keep)
        self.group[0][0].metadata['arm_turn_bonus']['policy_id']='old'
        with self.assertRaisesRegex(ValueError,'provenance'):failure.filter_groups(self.args,self.group)

    def test_native_horizon_failure_is_eligible_without_a_judge_call(self):
        from slime.utils.types import Sample
        for t in self.group:
            for s in t:
                s.status=Sample.Status.FAILED;s.metadata['terminate_reason']='max_steps_exhausted'
                s.metadata['reward']['judge_text']='Judge not run for status=Status.FAILED'
        self.assertTrue(failure.filter_groups(self.args,self.group).keep)

    def test_no_labels_and_nonzero_uniform_failures_are_rejected(self):
        for t in self.group:
            for s in t:s.metadata['arm_turn_bonus']={}
        self.assertEqual(failure.filter_groups(self.args,self.group).reason,'arm_failure_no_usable_labels')
        for t in self.group:
            for s in t:s.reward=-1.
        self.assertFalse(failure.filter_groups(self.args,self.group).keep)

    def test_native_normalization_exact_bonus_and_backward(self):
        result=failure.filter_groups(self.args,self.group)
        self.assertTrue(result.keep)
        failed=[s for t in result.samples for s in t]
        # Preserve all 24 mixed groups to pass the original calibration criteria.
        for s in failed:
            s.group_index=100;s.index+=1000000;s.metadata['trajectory_id']+=100000
        samples=[*self.rows,*failed]
        raw, shaped=self.manager._post_process_rewards(samples)
        self.assertEqual(raw[-len(failed):],[0.]*len(failed))
        expected=[.5*bonus.unit_bonus(s,'p') for s in failed]
        self.assertEqual(shaped[-len(failed):],expected)
        report=json.loads((Path(self.temp.name)/'calibration.json').read_text())
        self.assertTrue(report['decision']['passed']);self.assertEqual(report['all_failure_groups'],1)
        self.assertEqual(report['all_failure_positive_turns'],1)
        self.assertEqual(report['all_failure_negative_turns'],4)
        # Exercise the actual native clipped PPO loss with these advantages.
        ns=dict(torch=torch)
        native('slime/utils/ppo_utils.py',['compute_policy_loss'],ns)
        ratios=torch.zeros(len(expected),requires_grad=True)
        losses,_=ns['compute_policy_loss'](-ratios,torch.tensor(expected),.2,.28)
        losses.sum().backward()
        self.assertLess(ratios.grad[0].item(),0.)
        self.assertGreater(ratios.grad[10].item(),0.)
        self.assertEqual(ratios.grad[1].item(),0.)

    def test_zero_outcome_denominator_never_approves_pure_proxy_batch(self):
        failure.filter_groups(self.args,self.group)
        rows=[s for t in self.group for s in t]
        report=bonus.panel(rows,[0.]*len(rows),'p')
        failure.annotate_calibration(report,rows,[0.]*len(rows),'p')
        self.assertFalse(runtime.calibration_decision(report)['passed'])
        with self.assertRaisesRegex(ValueError,'exactly zero'):
            failure.annotate_calibration(report,rows,[1.]*len(rows),'p')

    def test_label_threshold_uses_all_retained_groups_but_scale_uses_mixed(self):
        raw=[float(s.reward) for s in self.rows]
        normalized=[(.6 if r else -.4)/(.3**.5) for r in raw]
        # Full panel has 120 labels / 24 tasks; its mixed subset has only 50.
        for s in self.rows[500:]:s.metadata['arm_all_failure_group']=True
        normalized[500:]=[0.]*(len(self.rows)-500)
        report=bonus.panel(self.rows,normalized,'p')
        failure.annotate_calibration(report,self.rows,normalized,'p')
        self.assertEqual(report['mixed_outcome_panel']['admitted'],50)
        self.assertTrue(runtime.calibration_decision(report)['passed'])


if __name__=='__main__':unittest.main()
