"""Model-path checks for Qwen3-VL MTP SFT without allocating a 4B model."""

from types import SimpleNamespace

import torch

from sft.qwen3vl_mtp import attach_mtp_training, draft_hidden_states


class _TinyTrunk(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(11, 8)
        self.projection = torch.nn.Linear(8, 8)

    def forward(self, input_ids, **kwargs):
        return SimpleNamespace(
            last_hidden_state=self.projection(self.embed_tokens(input_ids)),
            past_key_values=None,
        )


class _TinyActor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(
            model_type="qwen3_vl", text_config=SimpleNamespace(hidden_size=8, rms_norm_eps=1e-6)
        )
        self.model = _TinyTrunk()
        self.lm_head = torch.nn.Linear(8, 11, bias=False)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def forward(self, **kwargs):
        return "original-actor-forward"


def test_mtp_training_backward_and_original_generation_path() -> None:
    torch.manual_seed(3)
    actor = attach_mtp_training(_TinyActor(), mode="soft", horizon=2, weight=0.7)
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    labels = torch.tensor([[-100, -100, -100, 4, 5, 6, 7, 8]])
    output = actor(input_ids=input_ids, labels=labels)
    assert output.loss.isfinite()
    output.loss.backward()
    assert actor.model.projection.weight.grad is not None
    assert all(head.input_projection.weight.grad is not None for head in actor.soft_mtp_heads)
    assert actor(input_ids=input_ids) == "original-actor-forward"


def test_draft_states_do_not_read_target_token() -> None:
    torch.manual_seed(4)
    actor = attach_mtp_training(_TinyActor(), mode="hard", horizon=2)
    original = torch.tensor([[1, 2, 3, 4, 5, 6]])
    changed = original.clone()
    changed[0, 3] = 10
    # At t=0, head 2 may read x[1] and x[2], but not target x[3].
    first = draft_hidden_states(actor, actor.model(original).last_hidden_state[:, :-1], original)[1]
    second = draft_hidden_states(actor, actor.model(changed).last_hidden_state[:, :-1], changed)[1]
    torch.testing.assert_close(first[:, 0], second[:, 0])
