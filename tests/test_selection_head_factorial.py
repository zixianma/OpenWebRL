"""Factorial head switches: interaction (joint / independent) x scorer (transformer / mlp) x loss (softce / bce)."""
import itertools
import json

import pytest
import torch

from openwebrl.selection_head import CandidateSetHead, HeadConfig, pointwise_bce_loss, selection_loss


def make(interaction, scorer):
    torch.manual_seed(0)
    config = HeadConfig(hidden_size=8, layers=(1, 2), d_model=16, n_heads=2, depth=2, dropout=0.0,
                        interaction=interaction, scorer=scorer)
    head = CandidateSetHead(config).eval()
    cand, state = torch.randn(3, 5, config.candidate_dim), torch.randn(3, config.state_dim)
    head.fit_standardizers(cand, state)
    return head, cand, state


@pytest.mark.parametrize("interaction,scorer", list(itertools.product(("joint", "independent"), ("transformer", "mlp"))))
def test_interaction_controls_cross_candidate_dependence(interaction, scorer):
    head, cand, state = make(interaction, scorer)
    with torch.no_grad():
        base = head(cand, state)[0]
        changed = cand.clone(); changed[:, 4] += 3.0          # perturb only candidate 4
        after = head(changed, state)[0]
        perm = torch.tensor([2, 0, 4, 1, 3])
        permuted = head(cand[:, perm], state)[0]
    others = (after[:, :4] - base[:, :4]).abs().max().item()
    if interaction == "independent":
        assert others < 1e-5
    else:
        assert others > 1e-4
    assert torch.allclose(permuted, base[:, perm], atol=1e-5)   # permutation-equivariant in every variant


def test_old_configs_load_as_before():
    legacy = json.dumps(dict(hidden_size=8, layers=[1, 2], pools=["thought_mean", "action_mean", "last"], d_model=16,
                             n_heads=2, depth=2, dropout=0.1, listwise=False))
    config = HeadConfig.from_json(legacy)
    assert config.resolved() == ("independent", "mlp")
    assert HeadConfig.from_json(legacy.replace("false", "true")).resolved() == ("joint", "transformer")
    assert CandidateSetHead(config).encoder is None


def test_losses_finite_and_bce_is_per_candidate():
    logits = torch.tensor([[2.0, -1.0, 0.5, 0.0, 1.0]], requires_grad=True)
    target = torch.tensor([[0.5, 0.0, 0.5, 0.0, 0.0]])
    mask = torch.ones(1, 5, dtype=torch.bool)
    for fn in (selection_loss, pointwise_bce_loss):
        loss = fn(logits, target, mask)
        loss.backward()
        assert torch.isfinite(loss)
    shifted = pointwise_bce_loss(logits + 5, target, mask)
    assert not torch.isclose(shifted, pointwise_bce_loss(logits, target, mask))   # not shift-invariant, unlike softmax


def test_pointer_readout_is_joint_equivariant_and_masks_padding():
    torch.manual_seed(0)
    config = HeadConfig(hidden_size=8, layers=(1, 2), d_model=16, n_heads=2, depth=2, dropout=0.0,
                        interaction="joint", scorer="transformer", readout="pointer")
    head = CandidateSetHead(config).eval()
    cand, state = torch.randn(3, 5, config.candidate_dim), torch.randn(3, config.state_dim)
    head.fit_standardizers(cand, state)
    mask = torch.tensor([[1, 1, 1, 0, 0]] * 3, dtype=torch.bool)
    with torch.no_grad():
        base = head(cand, state, mask)[0]
        perm = torch.tensor([2, 0, 1, 3, 4])
        permuted = head(cand[:, perm], state, mask[:, perm])[0]
        changed = cand.clone(); changed[:, 3:] += 9.0          # padded slots cannot move real scores
        after = head(changed, state, mask)[0]
    assert torch.isinf(base[:, 3:]).all() and torch.isfinite(base[:, :3]).all()
    assert torch.allclose(permuted[:, :3], base[:, perm[:3]], atol=1e-5)
    assert torch.allclose(after[:, :3], base[:, :3], atol=1e-5)
    with pytest.raises(ValueError):
        CandidateSetHead(HeadConfig(hidden_size=8, layers=(1, 2), d_model=16, n_heads=2, scorer="mlp", readout="pointer"))


def test_mlp_width_matches_transformer_parameters_and_default_layout_unchanged():
    count = lambda **k: sum(p.numel() for p in CandidateSetHead(HeadConfig(hidden_size=2560, layers=(18, 27, 36), **k)).parameters())
    transformer = count(interaction="joint", scorer="transformer")
    assert transformer == 22_562_818
    for interaction, width in (("joint", 1686), ("independent", 1856)):
        n = count(interaction=interaction, scorer="mlp", mlp_depth=2, mlp_width=width)
        assert abs(n - transformer) / transformer < 1e-3
    legacy = CandidateSetHead(HeadConfig(hidden_size=8, layers=(1, 2), d_model=16, n_heads=2, scorer="mlp", listwise=False))
    assert [k for k in legacy.state_dict() if k.startswith("pointwise")] == [
        "pointwise.0.weight", "pointwise.0.bias", "pointwise.1.weight", "pointwise.1.bias"]
