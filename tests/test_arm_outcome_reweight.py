import math
import ast
import json
import logging
import os
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from openwebrl.arm_outcome_reweight import transform, validate_config


def row(tid, selected=None, group=1, masked=False):
    label = {} if selected is None else dict(eligible=True, policy_id='p', executed_index=0,
        selected_index=selected, unit_bonus=float(selected == 0)-.2,
        candidate_gate='min2',credit_assignment='response_index',action_class_ids=[0,0,1,1,1])
    return SimpleNamespace(group_index=group, metadata=dict(trajectory_id=tid,arm_turn_bonus=label),
                           remove_sample=False, loss_mask=[0] if masked else [1])


class Reweighting(unittest.TestCase):
    def test_original_bonus_exact_control(self):
        samples=[row('a',0),row('a',1),row('b')]
        values,_=transform(samples,[1,1,0],[1,1,-1],'p',mode='original_bonus')
        self.assertEqual(values,[1.4,.9,-1])

    def test_positive_negative_direction_and_mean(self):
        for A,R in [(1,1),(-1,0)]:
            samples=[row('a',0),row('a',1)]+[row('a') for _ in range(8)]
            values,report=transform(samples,[R]*10,[A]*10,'p')
            self.assertAlmostEqual(sum(values)/10,A)
            self.assertGreater(values[0],values[1])
            self.assertTrue(all(v*A>0 for v in values))
            self.assertLess(report['mean_advantage_error'],1e-12)
            self.assertAlmostEqual(abs(values[0]/values[1]),math.exp(.5*(1 if A>0 else -1)))

    def test_zero_lambda_and_unlabeled_turns(self):
        samples=[row('a',0),row('a',1)]
        values,_=transform(samples,[1,1],[1,1],'p',lam=0)
        self.assertEqual(values,[1,1])
        values,_=transform([row('a'),row('a')],[0,0],[-1,-1],'p')
        self.assertEqual(values,[-1,-1])

    def test_zero_advantage_single_turn_and_constant_signal(self):
        for samples,A in [([row('a',0)],1),([row('a',0),row('a',0)],1),([row('a',0),row('a',1)],0)]:
            values,_=transform(samples,[1]*len(samples),[A]*len(samples),'p')
            self.assertEqual(values,[A]*len(samples))

    def test_masked_turn_excluded_from_denominator(self):
        samples=[row('a',0),row('a',1,masked=True)]
        values,_=transform(samples,[1,1],[1,1],'p')
        self.assertEqual(values,[1,1])

    def test_group_scope_isolates_reused_trajectory_ids(self):
        samples=[row('a',0,group=1),row('a',1,group=2)]
        values,_=transform(samples,[1,0],[1,-1],'p')
        self.assertEqual(values,[1,-1])

    def test_error_group_keeps_original_objective(self):
        samples=[row('a',0),row('b',1)]
        values,report=transform(samples,[1,-1],[1,-1],'p')
        self.assertEqual(values,[1.4,-1.1])
        self.assertEqual(report['fallback_groups'],1)

    def test_rejects_trajectory_advantage_or_label_provenance_mismatch(self):
        with self.assertRaises(ValueError):transform([row('a'),row('a')],[1,1],[1,2],'p')
        with self.assertRaises(ValueError):transform([row('a',0)],[1],[1],'wrong')
        sample=row('a',0);sample.metadata['arm_turn_bonus']['candidate_gate']='distinct5'
        with self.assertRaises(ValueError):transform([sample],[1],[1],'p')

    def test_no_failure_aux_or_gate_change(self):
        cfg=dict(advantage_mode='outcome_reweight',k=5,beta=.5,scored_fraction=.2,reweight_lambda=.5,candidate_gate='min2')
        validate_config(cfg)
        for changed in [dict(max_failure_groups=8),dict(failure_group_cap=8),dict(additive_failure_groups=True),
                        dict(admit_all_failure_groups=True),dict(candidate_gate='distinct5')]:
            with self.assertRaises(ValueError):validate_config({**cfg,**changed})

    def test_response_index_credit_even_if_selected_action_is_identical(self):
        # Candidates0/1 share an action. This is gate B, not duplicate-aware C.
        values,_=transform([row('a',0),row('a',1)],[1,1],[1,1],'p',mode='original_bonus')
        self.assertEqual(values,[1.4,.9])

    def test_dp_transport_keeps_global_trajectory_weights(self):
        from openwebrl import arm_turn_bonus
        path=Path(arm_turn_bonus.__file__).parents[1]/'slime/backends/megatron_utils/actor.py'
        source=path.read_text()
        name='_build_local_rollout_data_for_epoch_from_local_shard'
        node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name==name)
        body='from __future__ import annotations\n'+ast.get_source_segment(source,node)
        # Execute the frozen production selection function, mocking only the
        # distributed count collective. No CUDA or training on the login node.
        counts=[129,128,128,128]
        def gather(destination,value,group):destination[:]=counts
        ns=dict(random=random,logger=logging.getLogger('test'),
            mpu=SimpleNamespace(get_data_parallel_group=lambda **kw:None),
            dist=SimpleNamespace(all_gather_object=gather))
        exec(compile(body,str(path),'exec'),ns)
        samples=[row(str(i//3),[0,1,None][i%3]) for i in range(513)]
        raw=[int((i//3)%2==0) for i in range(513)]
        values,_=transform(samples,raw,[1 if r else -1 for r in raw],'p')
        for epoch in (0,1):
            used=[]
            for rank in range(4):
                ids=list(range(rank,513,4))
                data=dict(tokens=[[0,1]]*len(ids),total_lengths=[2]*len(ids),
                    sample_indices=ids,rewards=[values[i] for i in ids])
                selected=ns[name](SimpleNamespace(global_batch_size=256,rollout_seed=42),data,0,epoch,rank,4)
                self.assertEqual(selected['effective_global_batch_size'],256)
                self.assertEqual(len(selected['sample_indices']),128)
                for i,value in zip(selected['sample_indices'],selected['rewards']):
                    self.assertEqual(value,values[i]);used.append(i)
            self.assertEqual(len(set(used)),512)

    def test_wrapper_preserves_raw_outcomes_and_calibration_stop(self):
        from openwebrl import arm_turn_bonus as parent
        from openwebrl.arm_outcome_reweight import post_process_rewards
        samples=[row('a',0),row('a',1)]
        raw=[1,1]; native=[1,1]
        args=SimpleNamespace(dynamic_sampling_filter_path='slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonempty_nonzero_std')
        with tempfile.TemporaryDirectory() as folder:
            config=dict(advantage_mode='outcome_reweight',k=5,beta=.5,scored_fraction=.2,
                reweight_lambda=.5,candidate_gate='min2',policy_id='p',output=folder)
            for beta in [0,.5]:
                Path(folder,'calibration.json').write_text(json.dumps(dict(applied_beta=beta)))
                old=[a+beta*u for a,u in zip(native,[.8,-.2])]
                with patch.object(parent,'state',return_value=dict(config=config)), patch.object(
                    parent,'post_process_rewards',return_value=(raw,old)), patch.dict(os.environ,{},clear=True):
                    returned,values=post_process_rewards(args,samples,raw,native)
                self.assertIs(returned,raw)
                if beta==0:self.assertIs(values,old)
                else:self.assertAlmostEqual(sum(values)/2,1.)


if __name__=='__main__':unittest.main()
