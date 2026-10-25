import json
import unittest
from pathlib import Path

import torch

from openwebrl.selection_head import (
    CandidateSetHead, HeadConfig, equivalence_target, pairwise_teacher_loss, parse_calls, selection_loss,
    topk_agreement, value_loss,
)
from openwebrl.selection_head_features import candidate_tokens, candidate_tokens_from_ids, extract, pool_candidate
from openwebrl.actor_head_scoring import score as score_captured_head

SFT = Path("/gpfs/scrubbed/zixianma/checkpoints/web/OpenWebRL-4B-SFT")


def response(call, thought="reason"):
    return f"{thought}\n</think>\n\n<tool_call>\n{json.dumps(call)}\n</tool_call>"


def click(x, y):
    return {"name": "click", "arguments": {"point_2d": [x, y]}}


class Head(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.config = HeadConfig(hidden_size=8, layers=(1, 2), d_model=32, n_heads=4, dropout=0.)

    def test_listwise_head_is_permutation_equivariant_and_masks_padding(self):
        head = CandidateSetHead(self.config).eval()
        cands, state = torch.randn(2, 5, self.config.candidate_dim), torch.randn(2, self.config.state_dim)
        mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]], dtype=torch.bool)
        logits, value = head(cands, state, mask)
        perm = torch.tensor([3, 0, 4, 1, 2])
        permuted, _ = head(cands[:, perm], state, mask[:, perm])
        torch.testing.assert_close(permuted[0], logits[0, perm])
        self.assertTrue(torch.isinf(logits[1, 3:]).all())
        # Padded candidates' contents cannot change real candidates' scores.
        noisy = cands.clone(); noisy[1, 3:] = 100.
        torch.testing.assert_close(head(noisy, state, mask)[0][1, :3], logits[1, :3])
        self.assertEqual(value.shape, (2, 5))

    def test_learns_a_panel_relative_signal(self):
        # The correct candidate has the largest first feature *within its panel*;
        # a random panel-wide shift makes absolute thresholds useless.
        for listwise in (True, False):
            torch.manual_seed(1)
            config = HeadConfig(**{**self.config.__dict__, "listwise": listwise})
            def batch(n=128):
                c = torch.randn(n, 5, config.candidate_dim)
                c[..., 0] += torch.randn(n, 1) * 3
                target = torch.nn.functional.one_hot(c[..., 0].argmax(-1), 5).float()
                return c, torch.randn(n, config.state_dim), target
            head = CandidateSetHead(config)
            head.fit_standardizers(*batch(1024)[:2])
            opt = torch.optim.AdamW(head.parameters(), lr=3e-3)
            for _ in range(400):
                c, s, t = batch()
                logits, _ = head(c, s)
                loss = selection_loss(logits, t, torch.ones_like(t, dtype=torch.bool))
                opt.zero_grad(); loss.backward(); opt.step()
            c, s, t = batch(512)
            with torch.no_grad():
                accuracy = topk_agreement(head.eval()(c, s)[0], t)
            self.assertGreater(accuracy, 0.85, listwise)

    def test_value_loss_ignores_slots_without_trials(self):
        logits = torch.zeros(1, 3, requires_grad=True)
        loss = value_loss(logits, torch.tensor([[2., 0., 0.]]), torch.tensor([[3., 0., 3.]]),
                          torch.ones(1, 3, dtype=torch.bool))
        loss.backward()
        self.assertEqual(logits.grad[0, 1].item(), 0.)
        self.assertAlmostEqual(loss.item(), 0.6931, places=3)

    def test_pairwise_teacher_loss_compares_only_winner_class_to_other_actions(self):
        scores = torch.zeros(2, 5, requires_grad=True)
        targets = torch.tensor([[.5, .5, 0., 0., 0.], [.2, .2, .2, .2, .2]])
        mask = torch.ones_like(targets, dtype=torch.bool)
        loss = pairwise_teacher_loss(scores, targets, mask)
        self.assertAlmostEqual(loss.item(), torch.log(torch.tensor(2.)).item(), places=6)
        loss.backward()
        self.assertTrue((scores.grad[0, :2] < 0).all())
        self.assertTrue((scores.grad[0, 2:] > 0).all())
        self.assertTrue((scores.grad[1] == 0).all())
        better = torch.tensor([[2., 2., 0., 0., 0.]])
        self.assertLess(pairwise_teacher_loss(better, targets[:1], mask[:1]), loss)

    def test_config_round_trip(self):
        self.assertEqual(HeadConfig.from_json(self.config.to_json()), self.config)

    def test_live_scoring_uses_the_output_trained_by_pairwise_objective(self):
        class OpposedOutputs(torch.nn.Module):
            def forward(self, candidates, state):
                return torch.tensor([[5., 0., 0., 0., 0.]]), torch.tensor([[0., 5., 0., 0., 0.]])

        features = [(torch.zeros(2, 8), torch.zeros(6, 8)) for _ in range(5)]
        head = OpposedOutputs()
        self.assertEqual(max(range(5), key=lambda i: score_captured_head(head, self.config, features)[i]), 0)
        self.assertEqual(max(range(5), key=lambda i: score_captured_head(head, self.config, features, "value")[i]), 1)


class Targets(unittest.TestCase):
    def test_teacher_choice_spreads_over_near_identical_actions_only(self):
        calls = [parse_calls(r) for r in [
            response(click(568, 401)), response(click(570, 400), "other words"),
            response(click(745, 47)), "no tool call",
            response(click(567, 400)) + "\n<tool_call>\n" + json.dumps(click(745, 47)) + "\n</tool_call>"]]
        self.assertIsNone(calls[3])
        self.assertEqual(equivalence_target(calls, 0), [.5, .5, 0., 0., 0.])
        self.assertEqual(equivalence_target(calls, 3), [0., 0., 0., 1., 0.])

    def test_thinking_text_cannot_define_the_action(self):
        text = ('idea <tool_call>{"name":"stop","arguments":{}}</tool_call>\n</think>\n'
                '<tool_call>' + json.dumps(click(1, 2)) + '</tool_call>')
        self.assertEqual(parse_calls(text)[0]["name"], "click")


@unittest.skipUnless((SFT / "tokenizer.json").exists(), "actor tokenizer unavailable")
class Spans(unittest.TestCase):
    def test_spans_split_reasoning_from_action_and_end_with_im_end(self):
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(SFT, local_files_only=True)
        text = response(click(5, 6), "I will click Agree.")
        ids, spans = candidate_tokens(tok, text)
        self.assertEqual(tok.decode(ids[spans["thought"][0]:spans["thought"][1]]).strip(), "I will click Agree.")
        self.assertIn('"point_2d": [5, 6]', tok.decode(ids[spans["action"][0]:spans["action"][1]]))
        self.assertEqual(tok.decode(ids[-1:]), "<|im_end|>")
        self.assertIsNone(candidate_tokens(tok, "<tool_call>{}</tool_call>")[1]["thought"])
        # Exact generated ids give the same spans as re-tokenized text.
        self.assertEqual(candidate_tokens_from_ids(tok, ids[:-1]), (ids, spans))

    def test_pooling_layout_is_layer_major(self):
        hidden = [torch.full((4, 2), float(l)) for l in range(3)]
        hidden[2][3] = 9.
        out = pool_candidate(hidden, {"thought": (0, 2), "action": None}, layers=(1, 2))
        self.assertEqual(out[:, 0].tolist(), [1., 1., 1., 2., 9., 9.])


def tiny_qwen3_vl():
    from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration
    config = Qwen3VLConfig(
        text_config=dict(vocab_size=151936, hidden_size=64, intermediate_size=128, num_hidden_layers=3,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                         rope_scaling={"rope_type": "default", "mrope_section": [4, 2, 2],
                                       "mrope_interleaved": True}),
        vision_config=dict(depth=2, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=64,
                           num_position_embeddings=64, deepstack_visual_indexes=[1]))
    torch.manual_seed(0)
    return Qwen3VLForConditionalGeneration(config).eval()


class CachedExtraction(unittest.TestCase):
    def test_prefix_cache_reuse_matches_full_forward_with_image(self):
        model = tiny_qwen3_vl()
        ids = [100, 101, 151652] + [151655] * 4 + [151653, 102, 103]
        prefix = {"input_ids": torch.tensor([ids]), "attention_mask": torch.ones(1, len(ids), dtype=torch.long),
                  "pixel_values": torch.randn(16, 3 * 2 * 16 * 16), "image_grid_thw": torch.tensor([[1, 4, 4]])}
        cands = [([200, 201, 202, 203], {"thought": (0, 2), "action": (2, 3)}),
                 ([300, 301], {"thought": None, "action": (0, 1)})]
        s1, c1 = extract(model, prefix, cands, layers=(1, 3), reuse_prefix=True)
        s2, c2 = extract(model, prefix, cands, layers=(1, 3), reuse_prefix=False)
        torch.testing.assert_close(s1, s2)
        torch.testing.assert_close(c1, c2, atol=1e-5, rtol=1e-4)
        self.assertEqual(c1.shape, (2, 6, 64))
        # Batched, right-padded candidates match one-at-a-time encoding.
        _, batched = extract(model, prefix, cands, layers=(1, 3), batch_size=2)
        torch.testing.assert_close(batched, c1, atol=1e-5, rtol=1e-4)
        # A second candidate must not see the first candidate's tokens in the cache.
        _, alone = extract(model, prefix, cands[1:], layers=(1, 3))
        torch.testing.assert_close(alone[0], c1[1], atol=1e-5, rtol=1e-4)


if __name__ == "__main__":
    unittest.main()
