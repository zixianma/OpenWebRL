"""CPU tests of real iterator assembly and component backward scaling."""
import ast
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from openwebrl.arm_rl import LossRow, clipped_row, component_losses
from openwebrl.arm_rl_megatron import FIELDS, attach_auxiliary, check_topology, loss


def iterator_class():
    path = Path('slime/backends/megatron_utils/data.py')
    tree = ast.parse(path.read_text())
    cls = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == 'DataIterator')
    future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
    ns = {}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future, cls], type_ignores=[])), str(path), 'exec'), ns)
    return ns['DataIterator']


class Transport(unittest.TestCase):
    def setUp(self):
        self.DataIterator = iterator_class()
        self.fake_data = ModuleType('slime.backends.megatron_utils.data')
        self.fake_data.DataIterator = self.DataIterator
        self.args = SimpleNamespace(global_batch_size=2, rollout_temperature=.8,
            micro_batch_size=1, calculate_per_token_loss=False, entropy_coef=0., use_kl_loss=False,
            loss_type='policy_loss', advantage_estimator='grpo')
        self.data = dict(tokens=[torch.tensor([7, 8, 9])] * 4, response_lengths=[2]*4,
            total_lengths=[3]*4, loss_masks=[torch.ones(2)]*4, advantages=[torch.ones(2)]*4,
            rollout_log_probs=[torch.zeros(2)]*4, sample_indices=[10, 11, 12, 13])

    def test_routing_preserves_outcome_windows_and_aux_denominators(self):
        metadata = [dict(source='rescue_demo', optimizer_window=0),
                    dict(source='rescue_demo', optimizer_window=1)]
        aux = dict(tokens=[torch.tensor([1,2,3])]*2, response_lengths=[2]*2, total_lengths=[3]*2,
                   loss_masks=[torch.ones(2)]*2, arm_old_log_probs=[torch.zeros(2)]*2)
        prepared = dict(manifest={'records':metadata, 'eta':.2, 'lambda':0.}, data=aux)
        with patch.dict(sys.modules, {'slime.backends.megatron_utils.data': self.fake_data}):
            iterators, counts = attach_auxiliary(self.args, [self.DataIterator(self.data, micro_batch_size=1)], [2,2], prepared)
        self.assertEqual(counts, [3,3])
        self.assertEqual(iterators[0].rollout_data['effective_global_batch_size'], 2)
        batches = [iterators[0].get_next(['tokens','sample_weights','max_seq_lens',*FIELDS]) for _ in range(6)]
        self.assertEqual([b['arm_source'][0] for b in batches], ['outcome','outcome','rescue_demo']*2)
        self.assertEqual([b['arm_scale'][0] for b in batches], [1,1,.4]*2)
        self.assertTrue(all(b['sample_weights'] is None and b['max_seq_lens'] is None for b in batches))

    def test_local_states_follow_their_executed_turn_after_shuffle(self):
        metadata=[dict(source='arm_candidate', state_id='s', parent_sample_index=12, advantage=a) for a in [1., -1.]]
        prepared=dict(manifest={'records':metadata, 'eta':0., 'lambda':.1}, data={})
        base = self.DataIterator(self.data, micro_batch_indices=[[2],[0],[3],[1]])
        with patch.dict(sys.modules, {'slime.backends.megatron_utils.data': self.fake_data}):
            it, counts = attach_auxiliary(self.args,[base],[2,2],prepared)
        self.assertEqual(counts,[4,2])
        self.assertEqual(it[0].rollout_data['arm_source'],['outcome','outcome','arm_candidate','arm_candidate','outcome','outcome'])
        self.assertEqual(it[0].rollout_data['arm_scale'][2:4],[.2,.2])

    def test_zero_aux_returns_original_iterators(self):
        base=[self.DataIterator(self.data,micro_batch_size=1)];counts=[2,2]
        out,n=attach_auxiliary(self.args,base,counts,None)
        self.assertIs(out,base);self.assertIs(n,counts)

    def test_unvalidated_distributed_topologies_fail(self):
        check_topology(self.args)
        for key in ['dp','cp','pp']:
            with self.assertRaises(ValueError):check_topology(self.args,**{key:2})

    def test_real_loss_dispatch_has_same_gradient_as_component_reference(self):
        backend=ModuleType('slime.backends.megatron_utils.loss')
        backend.get_log_probs_and_entropy=lambda logits,**kw:(None,{'log_probs':[logits]})
        backend.policy_loss_function=lambda args,batch,logits,reducer:(clipped_row(
            LossRow(logits,batch['loss_masks'][0],batch['rollout_log_probs'][0],1.)),{})
        def run(values):
            losses=[]
            for i,source in enumerate(['outcome','outcome','rescue_demo','arm_candidate','arm_candidate']):
                b=dict(arm_source=[source],arm_scale=[1. if i<2 else .4 if i==2 else .2],
                    arm_advantage=[1. if i==3 else -1.],arm_old_log_probs=[torch.zeros(2)],
                    loss_masks=[torch.tensor([0.,1.])],rollout_log_probs=[torch.zeros(2)],
                    unconcat_tokens=[],total_lengths=[3],response_lengths=[2])
                with patch.dict(sys.modules,{'slime.backends.megatron_utils.loss':backend}):
                    term,_=loss(self.args,b,values[i],lambda x:x[1])
                # Actual dispatcher multiplies by n_micro/gbs and pipeline divides by n_micro.
                losses.append(term * 5/2/5)
            return sum(losses)
        x=[torch.tensor([-.1,-.2],dtype=torch.float64,requires_grad=True) for _ in range(5)]
        y=[t.detach().clone().requires_grad_() for t in x]
        actual=run(x)
        make=lambda t,a=1.:LossRow(t,[0,1],torch.zeros(2),a,'frozen_actor_teacher_forced')
        reference=component_losses([make(y[0]),make(y[1])],demo_rows=[make(y[2])],
            local_groups=[[make(y[3]),make(y[4],-1.)]],eta=.2,lam=.1)['total']
        actual.backward();reference.backward()
        self.assertAlmostEqual(actual.item(),reference.item())
        for a,b in zip(x,y):self.assertTrue(torch.allclose(a.grad,b.grad,atol=1e-12))


if __name__=='__main__':unittest.main()
