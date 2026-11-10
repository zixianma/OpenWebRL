import ast
import copy
import os
import re
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

import prepare_arm_frozen_vision as preparation
from openwebrl.vision_freeze_audit import FREEZE_PATTERN, audit


def native_freeze():
    tree = ast.parse((preparation.SOURCE/'slime/backends/megatron_utils/model_provider.py').read_text())
    fn = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='freeze_model_params')
    fn.returns = None
    for arg in fn.args.args: arg.annotation = None
    namespace = dict(re=re)
    exec(compile(ast.Module(body=[fn],type_ignores=[]),'<native freeze>','exec'),namespace)
    return namespace['freeze_model_params']


class TinyVLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_model = nn.Module()
        self.vision_model.patch_embed = nn.Linear(4,4)
        self.vision_model.pos_embed = nn.Embedding(2,4)
        self.vision_model.blocks = nn.ModuleList([nn.Linear(4,4)])
        self.vision_model.merger = nn.Linear(4,4)
        self.vision_model.deepstack_merger_list = nn.ModuleList([nn.Linear(4,4) for _ in range(3)])
        self.language_model = nn.Linear(4,1)

    def forward(self,x):
        x = self.vision_model.blocks[0](self.vision_model.patch_embed(x)+self.vision_model.pos_embed.weight[0])
        z = self.vision_model.merger(x)
        for layer in self.vision_model.deepstack_merger_list: z = z+layer(x)
        return self.language_model(z)


class FrozenVision(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.model = TinyVLM()
        self.args = SimpleNamespace(freeze_params_name_list=[FREEZE_PATTERN],only_train_params_name_list=None)

    def test_real_freeze_function_preserves_all_projector_and_language_gradients(self):
        native_freeze()(self.model,self.args)
        with patch.dict(os.environ,{'OPENWEBRL_VISION_FREEZE_AUDIT_DIR':''}):
            self.assertTrue(audit(self.model,self.args)['passed'])
        initial = {n:p.detach().clone() for n,p in self.model.named_parameters()}
        opt = torch.optim.AdamW([p for p in self.model.parameters() if p.requires_grad],lr=.01)
        self.model(torch.randn(3,4)).square().mean().backward()
        for n,p in self.model.named_parameters():
            if re.search(FREEZE_PATTERN,n): self.assertIsNone(p.grad)
            else:
                self.assertIsNotNone(p.grad)
                self.assertTrue(torch.isfinite(p.grad).all())
                self.assertGreater(p.grad.abs().sum().item(),0)
        opt.step()
        for n,p in self.model.named_parameters():
            self.assertEqual(torch.equal(initial[n],p),bool(re.search(FREEZE_PATTERN,n)),n)

    def test_audit_rejects_unfrozen_backbone_and_frozen_deepstack(self):
        with self.assertRaises(ValueError): audit(self.model,self.args)
        native_freeze()(self.model,self.args)
        self.model.vision_model.deepstack_merger_list[0].weight.requires_grad=False
        with self.assertRaises(ValueError): audit(self.model,self.args)

    def test_new_run_is_fresh_and_does_not_inherit_control_approval(self):
        p = preparation.plan('PREPARE')
        self.assertFalse(p['compute_approved'])
        self.assertEqual(p['start_rollout_id'],0)
        self.assertEqual(p['initial_optimizer_updates'],0)
        self.assertEqual(p['environment']['WANDB_RESUME'],'never')
        self.assertEqual(p['environment']['WANDB_PROJECT'],'openwebrl')
        self.assertEqual(p['requested_resources']['hours'],16)
        self.assertEqual(p['arm_config']['beta'],.5)
        self.assertEqual(p['arm_config']['scored_fraction'],.2)
        self.assertFalse(p['arm_config']['admit_all_failure_groups'])

    def test_recipe_rejects_resume_and_wider_freeze(self):
        p = preparation.plan('PREPARE')
        bad = copy.deepcopy(p); bad['resume_from']='old checkpoint'
        with self.assertRaises(ValueError): preparation.validate_recipe(bad)
        bad = copy.deepcopy(p); bad['command'][-1] = r'^vision_model\.'
        with self.assertRaises(ValueError): preparation.validate_recipe(bad)

    def test_mask_covers_real_checkpoint_backbone_but_no_projector(self):
        counts = preparation.header_audit()['counts']
        self.assertEqual(counts['backbone']['tensors'],291)
        self.assertEqual(counts['projection']['tensors'],24)
        self.assertGreater(counts['language']['parameters'],3_000_000_000)


if __name__=='__main__': unittest.main()
