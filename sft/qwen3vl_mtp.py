"""Training-only MTP extension for Qwen3-VL OpenWebRL SFT.

The original actor forward path remains available for generation. During SFT,
three small sequential heads use teacher-forced draft tokens and share the
actor's linear LM output head. Soft-MTP changes their hidden-state gradients;
hard MTP is the matched auxiliary-loss control. Neither mode implements a
speculative-decoding server.
"""

from __future__ import annotations

from types import MethodType
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLCausalLMOutputWithPast

from sft.soft_mtp import soft_mtp_sft_loss


class SequentialMTPHead(nn.Module):
    """Causal draft head conditioned on the previous hidden state and token."""

    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.input_projection = nn.Linear(2 * hidden_size, hidden_size, bias=False)
        self.output_projection = nn.Linear(hidden_size, hidden_size, bias=False)
        self.norm = nn.RMSNorm(hidden_size, eps=eps)

    def forward(self, previous: Tensor, draft_token_embedding: Tensor) -> Tensor:
        combined = torch.cat((previous, draft_token_embedding), dim=-1)
        update = self.output_projection(F.silu(self.input_projection(combined)))
        return self.norm(previous + update)


def draft_hidden_states(model: nn.Module, ntp_hidden: Tensor, input_ids: Tensor) -> list[Tensor]:
    """Make teacher-forced draft states with no access to their target token."""
    sequence_length = ntp_hidden.shape[1]
    embed = model.get_input_embeddings()
    previous = ntp_hidden
    states: list[Tensor] = []
    for k, head in enumerate(model.soft_mtp_heads, start=1):
        if k >= sequence_length:
            raise ValueError("MTP horizon must be shorter than the NTP sequence")
        token_embedding = embed(input_ids[:, k:sequence_length])
        previous = head(previous[:, : sequence_length - k], token_embedding)
        states.append(previous)
    return states


def hard_mtp_sft_loss(
    ntp_hidden: Tensor, mtp_hidden: list[Tensor], shifted_labels: Tensor, lm_head: nn.Module, weight: float
) -> Tensor:
    supervised = shifted_labels.ne(-100)
    if not supervised.any():
        raise ValueError("no supervised SFT targets")
    ntp = F.cross_entropy(lm_head(ntp_hidden[supervised]), shifted_labels[supervised])
    auxiliary = ntp.new_zeros(())
    length = shifted_labels.shape[1]
    for k, hidden in enumerate(mtp_hidden, start=1):
        valid = supervised[:, k:].clone()
        for offset in range(k):
            valid &= supervised[:, offset : length - k + offset]
        if valid.any():
            auxiliary = auxiliary + F.cross_entropy(lm_head(hidden[valid]), shifted_labels[:, k:][valid])
    return ntp + weight * auxiliary / len(mtp_hidden)


def attach_mtp_training(
    model: nn.Module, *, mode: Literal["hard", "soft"], horizon: int = 3, weight: float = 1.0
) -> nn.Module:
    """Attach trainable heads and an SFT loss to a loaded Qwen3-VL actor.

    Call after LLaMAFactory's ``load_model`` and before creating its trainer.
    Trainer checkpoints include the heads. Loading a final checkpoint through
    plain Qwen3-VL ignores the extra head weights and recovers the NTP actor.
    """
    if getattr(model.config, "model_type", None) != "qwen3_vl":
        raise TypeError("MTP SFT currently supports Qwen3-VL only")
    if mode not in ("hard", "soft") or horizon < 1 or weight <= 0:
        raise ValueError("expected mode hard/soft, horizon >= 1, weight > 0")
    if hasattr(model, "soft_mtp_heads"):
        raise ValueError("MTP heads already attached")

    text_config = model.config.text_config
    heads = nn.ModuleList(
        SequentialMTPHead(text_config.hidden_size, text_config.rms_norm_eps) for _ in range(horizon)
    )
    embedding_weight = model.get_input_embeddings().weight
    model.soft_mtp_heads = heads.to(device=embedding_weight.device, dtype=embedding_weight.dtype)
    model.config.soft_mtp = {"mode": mode, "horizon": horizon, "weight": weight, "architecture": "sequential_mlp"}
    original_forward = model.forward

    def training_forward(self, input_ids=None, labels=None, **kwargs):
        if labels is None:
            return original_forward(input_ids=input_ids, labels=labels, **kwargs)
        if input_ids is None:
            raise ValueError("MTP SFT needs input_ids for teacher-forced draft tokens")
        model_kwargs = dict(kwargs)
        for key in ("logits_to_keep", "num_items_in_batch", "return_dict"):
            model_kwargs.pop(key, None)
        outputs = self.model(input_ids=input_ids, **model_kwargs)
        hidden = outputs.last_hidden_state[:, :-1]
        targets = labels[:, 1:]
        drafts = draft_hidden_states(self, hidden, input_ids)
        if mode == "soft":
            loss = soft_mtp_sft_loss(hidden, drafts, targets, self.lm_head, mtp_weight=weight).total
        else:
            loss = hard_mtp_sft_loss(hidden, drafts, targets, self.lm_head, weight)
        return Qwen3VLCausalLMOutputWithPast(loss=loss, past_key_values=outputs.past_key_values)

    model.forward = MethodType(training_forward, model)
    return model
