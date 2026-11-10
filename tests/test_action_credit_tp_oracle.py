"""CPU regressions for the native TP oracle; no GPU, browser or service calls.

The fixture models the installed fused CE contract: its backward consumes the
saved softmax in place and returns a BF16 derivative. Each TP shard receives
deterministic full-vocabulary collective values. Actual CUDA/NCCL execution is
still covered by the separately bounded native gate.
"""
from contextlib import contextmanager
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

import torch

from openwebrl.action_credit_native_gate import tp_oracle


@contextmanager
def native_cpu_fixture(tp, rank, *, all_sampled=False, gradient_scale=1.):
    generator = torch.Generator().manual_seed(122)
    full = torch.randn(1, 8, 64, generator=generator) * 2.1
    temperature = .8
    response = (1, 14, 30, 50, 60, 9)
    sampled = (True,) * 6 if all_sampled else (True, False, True, True, False, True)
    edge = SimpleNamespace(edge_id='edge', input_ids=(1, 2, 3), response_ids=response, sampled_mask=sampled)
    batch = {'action_credit_objective': SimpleNamespace(plan=SimpleNamespace(graph=SimpleNamespace(edges=[edge]))),
             'action_credit_edge_id': 'edge'}
    args = SimpleNamespace(rollout_temperature=temperature)
    indices = [0, 1, 2, 3, 5]
    selected_full = full[0, [2 + i for i in indices]]
    targets = torch.tensor([response[i] for i in indices])
    shard = full.shape[-1] // tp
    group = object()
    calls = []

    class NativeCE(torch.autograd.Function):
        @staticmethod
        def forward(ctx, logits, target):
            # Compute independent full-vocabulary normalization, then retain
            # only the rank-local probabilities, as fused vocabulary CE does.
            scaled = selected_full / temperature
            probabilities = scaled.softmax(-1)[:, rank*shard:(rank+1)*shard].clone()
            owned = (target >= rank*shard) & (target < (rank+1)*shard)
            local_target = (target-rank*shard).clone()
            local_target[~owned] = 0
            ctx.save_for_backward(probabilities, owned, local_target)
            calls.append('forward')
            return -scaled.log_softmax(-1).gather(1, target[:, None]).flatten()

        @staticmethod
        def backward(ctx, output_gradient):
            probabilities, owned, local_target = ctx.saved_tensors
            probabilities[torch.arange(probabilities.shape[0]), local_target] -= owned.float()
            probabilities.mul_(output_gradient[:, None])
            calls.append('backward')
            return (probabilities * gradient_scale).to(torch.bfloat16), None

    def log_probs(logits, target, supplied_group):
        assert supplied_group is group
        torch.testing.assert_close(logits.detach(), selected_full[:, rank*shard:(rank+1)*shard]/temperature)
        return -NativeCE.apply(logits.clone(), target), None

    def all_gather(outputs, local, group=None):
        torch.testing.assert_close(local, selected_full[:, rank*shard:(rank+1)*shard])
        for i, output in enumerate(outputs):
            output.copy_(selected_full[:, i*shard:(i+1)*shard])

    modules = {
        'megatron.core': SimpleNamespace(mpu=SimpleNamespace(get_tensor_model_parallel_group=lambda: group)),
        'slime.utils.ppo_utils': SimpleNamespace(calculate_log_probs_and_entropy=log_probs),
    }
    native_logp = (full[0, 2:]/temperature).log_softmax(-1).gather(1, torch.tensor(response)[:, None]).flatten()
    with patch.dict(sys.modules, modules), patch('torch.distributed.get_world_size', return_value=tp), \
            patch('torch.distributed.get_rank', return_value=rank), patch('torch.distributed.all_gather', side_effect=all_gather):
        yield SimpleNamespace(args=args, batch=batch, logits=full[:, :, rank*shard:(rank+1)*shard],
            native_logp=native_logp, log_probs=log_probs, selected=selected_full[:, rank*shard:(rank+1)*shard],
            targets=targets, group=group, calls=calls)


class TPOracleNativeBackwardRegression(unittest.TestCase):
    def test_fresh_graph_and_native_rounding_all_tp2_tp8_shards(self):
        unquantized_discrepancy_observed = False
        for tp in (2, 8):
            for rank in range(tp):
                for all_sampled in (False, True):
                    with self.subTest(tp=tp, rank=rank, all_sampled=all_sampled), \
                            native_cpu_fixture(tp, rank, all_sampled=all_sampled) as f:
                        r = tp_oracle(f.args, f.batch, f.logits, f.native_logp)
                        expected_backwards = 8 if all_sampled else 4
                        self.assertEqual(f.calls.count('backward'), expected_backwards)
                        self.assertEqual(f.calls.count('forward'), 1 + expected_backwards)
                        self.assertTrue(r['fresh_backward_graph_per_probe'])
                        self.assertTrue(r['masked_gradients_exact_zero'])
                        self.assertEqual(r['native_ce_backward_dtype'], 'bfloat16')
                        self.assertEqual(len(r['ppo_gradient_max_abs_errors']), expected_backwards)
                        unquantized_discrepancy_observed |= max(r['ppo_unquantized_gradient_max_abs_errors']) > 2e-6
        self.assertTrue(unquantized_discrepancy_observed)

    def test_retained_native_graph_reproduces_inplace_failure(self):
        with native_cpu_fixture(2, 0) as f:
            selected = f.selected.clone().requires_grad_(True)
            logp, _ = f.log_probs(selected/f.args.rollout_temperature, f.targets, f.group)
            torch.autograd.grad(logp.sum(), selected, retain_graph=True)
            with self.assertRaisesRegex(RuntimeError, 'modified by an inplace operation'):
                torch.autograd.grad(logp.sum(), selected, retain_graph=True)

    def test_real_derivative_error_still_fails_original_tolerance(self):
        with native_cpu_fixture(2, 0, gradient_scale=1.02) as f:
            with self.assertRaises(AssertionError):
                tp_oracle(f.args, f.batch, f.logits, f.native_logp)

    def test_wrong_native_logprobs_still_fail(self):
        with native_cpu_fixture(8, 7) as f:
            with self.assertRaises(AssertionError):
                tp_oracle(f.args, f.batch, f.logits, f.native_logp + .01)


if __name__ == '__main__':
    unittest.main()
