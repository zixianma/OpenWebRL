"""Soft-MTP loss for pre-shifted, masked supervised-language-model targets.

This is the loss/gradient-routing component, not an MTP model or trainer. The
caller supplies normalized NTP/MTP hidden states and one shared linear LM head.
See https://arxiv.org/abs/2610.38331, Appendix A and B.5.
"""

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass
class SoftMTPLoss:
    total: Tensor
    ntp: Tensor
    mtp: Tensor
    ntp_targets: int
    mtp_targets: tuple[int, ...]


def soft_mtp_sft_loss(
    ntp_hidden: Tensor,
    mtp_hidden: Sequence[Tensor],
    shifted_labels: Tensor,
    lm_head: nn.Module,
    *,
    mtp_weight: float = 1.0,
    ignore_index: int = -100,
) -> SoftMTPLoss:
    """Compute NTP plus Soft-MTP loss without materializing teacher logits.

    ``shifted_labels[b, t]`` is the token predicted by ``ntp_hidden[b, t]``;
    for an HF causal LM pass ``hidden[:, :-1]`` and ``labels[:, 1:]``.
    ``mtp_hidden[k-1][b, t]`` predicts the same label as NTP position ``t+k``.
    MTP targets are kept only when every intervening NTP target is supervised,
    so a draft never trains across a masked prompt/tool boundary. Packing of
    independently supervised examples is unsupported without a segment mask.

    The NTP forward values and gradients are unchanged. The routed gradient to
    each MTP hidden state equals the gradient of a stop-gradient NTP-teacher KL
    on the same positions, while the shared output head retains hard-label CE.
    """
    if ntp_hidden.ndim != 3 or shifted_labels.shape != ntp_hidden.shape[:2]:
        raise ValueError("expected NTP hidden [B,T,D] and shifted labels [B,T]")
    if not mtp_hidden:
        raise ValueError("at least one MTP head is required")
    if mtp_weight < 0:
        raise ValueError("mtp_weight must be nonnegative")

    batch, length, width = ntp_hidden.shape
    supervised = shifted_labels.ne(ignore_index)
    ntp_count = int(supervised.sum().item())
    if not ntp_count:
        raise ValueError("no supervised NTP targets")

    routed_hidden = ntp_hidden
    mtp_loss = ntp_hidden.new_zeros(())
    counts = []
    num_heads = len(mtp_hidden)
    for k, hidden in enumerate(mtp_hidden, start=1):
        if k >= length or hidden.shape != (batch, length - k, width):
            raise ValueError(f"MTP head {k} must have shape [B,T-{k},D]")

        # The entire draft span must belong to one contiguous supervised turn.
        valid = supervised[:, k:].clone()
        for offset in range(k):
            valid &= supervised[:, offset : length - k + offset]
        count = int(valid.sum().item())
        counts.append(count)
        if not count:
            continue

        targets = shifted_labels[:, k:][valid]
        mtp_loss = mtp_loss + F.cross_entropy(lm_head(hidden[valid]), targets)

        # Mean CE gives 1/N0 at the NTP position and 1/Nk at the MTP
        # position. This coefficient makes their one-hot terms cancel exactly.
        scale = mtp_weight * ntp_count / (num_heads * count)
        delta = scale * (hidden.detach() - hidden) * valid.unsqueeze(-1)
        routed_hidden = routed_hidden + F.pad(delta, (0, 0, k, 0))

    ntp_loss = F.cross_entropy(lm_head(routed_hidden[supervised]), shifted_labels[supervised])
    mtp_loss = mtp_weight * mtp_loss / num_heads
    return SoftMTPLoss(ntp_loss + mtp_loss, ntp_loss, mtp_loss, ntp_count, tuple(counts))
