"""Bounded CPU tests for distribution, DPO signs/normalization, and eval coverage."""
import importlib.util
import math
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from train_arm_joint_ddp import rank_batch,dpo_loss,backward_mode
from run_arm_joint_pipeline import cohorts


class Sharding(unittest.TestCase):
    def test_two_gpu_batches_equal_one_gpu_exposure_including_tail(self):
        order=list(range(5540));seen=[]
        for begin in range(0,len(order),32):
            left,right=[rank_batch(order,begin,rank) for rank in range(2)]
            self.assertEqual(len(left),len(right))
            self.assertFalse(set(left)&set(right))
            self.assertEqual(sorted(left+right),order[begin:begin+32])
            seen+=left+right
        self.assertEqual(len(left),2)
        self.assertEqual(sorted(seen),order)

    def test_no_implicit_duplicate_padding(self):
        with self.assertRaises(ValueError):rank_batch(list(range(5)),0,0)

    def test_average_rank_gradient_equals_global_mean(self):
        for count in [4,32]:
            gradients=[(i-3)**2*.13 for i in range(count)]
            parts=[rank_batch(list(range(count)),0,r) for r in range(2)]
            averaged=sum(sum(gradients[i] for i in part)/len(part) for part in parts)/2
            self.assertAlmostEqual(averaged,sum(gradients)/count)

    def test_eval_shards_are_disjoint_and_cover_all_300(self):
        shards=cohorts()['cohorts'];a,b=shards.values()
        self.assertEqual((len(a['indices']),len(b['indices'])),(150,150))
        self.assertEqual(sorted(a['indices']+b['indices']),list(range(300)))
        self.assertFalse(set(a['task_ids'])&set(b['task_ids']))


@unittest.skipUnless(importlib.util.find_spec('torch'),'PyTorch environment required')
class Objective(unittest.TestCase):
    def test_deterministic_backward_preserves_checkpointing_training_mode(self):
        import torch
        actor=torch.nn.Sequential(torch.nn.Linear(4,4),torch.nn.Dropout(.5))
        actor.eval()
        backward_mode(actor,False)
        self.assertTrue(actor.training)
        self.assertTrue(actor[0].training)
        self.assertFalse(actor[1].training)
        value=torch.ones(2,4)
        self.assertTrue(torch.equal(actor(value),actor(value)))
        backward_mode(actor,True)
        self.assertTrue(actor[1].training)

    def test_reference_tie_and_preference_gradient_direction(self):
        import torch
        chosen=torch.tensor([-1.,-2.],requires_grad=True)
        rejected=torch.tensor([-3.,-4.],requires_grad=True)
        loss,relative=dpo_loss(chosen,rejected,-3.,-7.,.1)
        self.assertAlmostEqual(loss.item(),math.log(2),places=6)
        self.assertEqual(relative.item(),0)
        loss.backward()
        self.assertTrue(torch.all(chosen.grad<0))
        self.assertTrue(torch.all(rejected.grad>0))

    def test_sequence_sum_and_fp32(self):
        import torch
        a=torch.tensor([-.25,-.25],dtype=torch.bfloat16)
        b=torch.tensor([-1.],dtype=torch.bfloat16)
        loss,relative=dpo_loss(a,b,0.,0.,.1)
        self.assertEqual(relative.dtype,torch.float32)
        self.assertAlmostEqual(relative.item(),.5)
        self.assertAlmostEqual(loss.item(),math.log1p(math.exp(-.05)),places=6)


if __name__=='__main__':unittest.main()
