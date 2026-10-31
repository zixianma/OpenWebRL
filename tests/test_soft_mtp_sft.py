"""Gradient and masking checks for the Soft-MTP SFT loss."""

import torch
import torch.nn.functional as F

from sft.soft_mtp import soft_mtp_sft_loss


def test_soft_mtp_matches_explicit_teacher_kl_hidden_gradients() -> None:
    torch.manual_seed(7)
    labels = torch.tensor([[-100, -100, 2, 3, 1, 4, -100, 0, 1]])
    base = torch.randn(1, 9, 4)
    drafts = [torch.randn(1, 8, 4), torch.randn(1, 7, 4)]
    head = torch.nn.Linear(4, 5, bias=False)

    h0 = base.clone().requires_grad_()
    hs = [h.clone().requires_grad_() for h in drafts]
    result = soft_mtp_sft_loss(h0, hs, labels, head, mtp_weight=0.7)
    result.total.backward()
    got_ntp = h0.grad.clone()
    got_drafts = [h.grad.clone() for h in hs]
    got_output_head = head.weight.grad.clone()

    # Soft-MTP routes only hidden-state gradients; the shared LM head still
    # receives the ordinary hard-label NTP + MTP cross-entropy gradient.
    head.zero_grad(set_to_none=True)
    supervised = labels.ne(-100)
    hard_loss = F.cross_entropy(head(base[supervised]), labels[supervised])
    for k, hidden in enumerate(drafts, start=1):
        valid = supervised[:, k:].clone()
        for offset in range(k):
            valid &= supervised[:, offset : labels.shape[1] - k + offset]
        hard_loss = hard_loss + 0.7 / len(drafts) * F.cross_entropy(
            head(hidden[valid]), labels[:, k:][valid]
        )
    hard_loss.backward()
    torch.testing.assert_close(got_output_head, head.weight.grad, atol=1e-6, rtol=1e-5)
    head.zero_grad(set_to_none=True)

    h0_ref = base.clone().requires_grad_()
    hs_ref = [h.clone().requires_grad_() for h in drafts]
    valid_ntp = labels.ne(-100)
    reference = F.cross_entropy(head(h0_ref[valid_ntp]), labels[valid_ntp])
    counts = []
    for k, h in enumerate(hs_ref, start=1):
        valid = valid_ntp[:, k:].clone()
        for offset in range(k):
            valid &= valid_ntp[:, offset : labels.shape[1] - k + offset]
        counts.append(int(valid.sum()))
        teacher = F.softmax(head(h0_ref[:, k:][valid]).detach(), dim=-1)
        student = F.log_softmax(head(h[valid]), dim=-1)
        reference = reference + 0.7 / len(hs_ref) * F.kl_div(student, teacher, reduction="batchmean")
    reference.backward()

    assert result.ntp_targets == 6
    assert result.mtp_targets == tuple(counts) == (4, 2)
    torch.testing.assert_close(result.ntp, F.cross_entropy(head(base[valid_ntp]), labels[valid_ntp]))
    torch.testing.assert_close(got_ntp, h0_ref.grad, atol=1e-6, rtol=1e-5)
    for got, h in zip(got_drafts, hs_ref):
        torch.testing.assert_close(got, h.grad, atol=1e-6, rtol=1e-5)


def test_soft_mtp_skips_empty_draft_span() -> None:
    head = torch.nn.Linear(3, 6, bias=False)
    h0 = torch.randn(1, 4, 3, requires_grad=True)
    h1 = torch.randn(1, 3, 3, requires_grad=True)
    result = soft_mtp_sft_loss(h0, [h1], torch.tensor([[-100, 1, -100, 2]]), head)
    assert result.mtp_targets == (0,)
    assert result.mtp.item() == 0.0
    result.total.backward()
    assert h1.grad is None
