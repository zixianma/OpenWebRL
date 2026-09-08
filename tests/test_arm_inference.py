import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from openwebrl.arm_inference import (
    ActionSelector, candidate_seed, load_selection_builder, parse_selection,
    scalar_messages, selection_messages, split_response,
)
from openwebrl.arm_eval import summarize


class Contracts(unittest.TestCase):
    def test_actor_coordinates_and_multicall_response_are_preserved(self):
        action = '<tool_call>{"name":"click","arguments":{"point_2d":[1000,500]}}</tool_call>\n<tool_call>{"name":"write","arguments":{"message":"x"}} </tool_call><|im_end|>'
        candidate = split_response("<think>reason</think>\n" + action)
        self.assertEqual(candidate, {"thought": "reason", "action": action})
        msg = scalar_messages("task", "url", [], candidate)
        self.assertEqual(msg[-1]["content"], f"Action: {action}\nReasoning: reason")

    def test_invalid_verdicts_do_not_become_labels(self):
        for raw in ("1", "I prefer candidate 2", '{"selection":0}', '{"selection":6}'):
            with self.assertRaises(ValueError):
                parse_selection(raw, 5)
        self.assertEqual(parse_selection('Reason.\n{"selection":3}', 5), 2)

    def test_structured_decoder_accepts_only_in_range_selection_objects(self):
        import xgrammar as xgr
        from transformers import AutoTokenizer, AutoConfig
        from openwebrl.arm_inference import selection_schema
        base = "/gpfs/scrubbed/zixianma/checkpoints/web/OpenWebRL-4B-SFT"
        if not Path(base).exists():
            self.skipTest("Requires pinned local tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
        config = AutoConfig.from_pretrained(base, local_files_only=True)
        info = xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=config.text_config.vocab_size)
        grammar = xgr.GrammarCompiler(info).compile_json_schema(selection_schema(5))
        for value in ('{"selection":1}', '{"selection":5}'):
            self.assertTrue(xgr.GrammarMatcher(grammar).accept_string(value), value)
        for value in ('3', '{"selection":0}', '{"selection":6}', '{"selection":1,"thought":"extra"}'):
            self.assertFalse(xgr.GrammarMatcher(grammar).accept_string(value), value)

    def test_denominator_preserves_unavailable_tasks(self):
        summary = summarize([{"valid": True, "reward": 1},
                             {"valid": True, "reward": 0},
                             {"valid": False, "reward": None}], 4)
        self.assertEqual(summary["success_rate_all_scheduled"], .25)
        self.assertEqual(summary["success_rate_valid"], .5)
        self.assertEqual(summary["unavailable"], 1)

    def test_seed_depends_on_task_turn_candidate_not_completion_order(self):
        seeds = [candidate_seed(42, "task", 2, i) for i in range(5)]
        self.assertEqual(len(set(seeds)), 5)
        self.assertEqual(seeds, [candidate_seed(42, "task", 2, i) for i in range(5)])
        self.assertNotEqual(seeds[0], candidate_seed(42, "other", 2, 0))

    def test_scalar_prompt_matches_reference_training_builder(self):
        import ast
        source = Path("/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-reproduction/source")
        path = source / "data_generation/openwebrl_actor/build_scalar_rm_data.py"
        if not path.exists():
            self.skipTest("Run prepare_arm_models.py for pinned reference")
        tree = ast.parse(path.read_text())
        func = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "action_prompt")
        namespace = {}
        exec(compile(ast.Module(body=[func], type_ignores=[]), str(path), "exec"), namespace)
        history = [{"action": f"action-{i}", "thought": ""} for i in range(10)]
        messages = scalar_messages("task", "url", history, {"action": "x", "thought": "y"})
        user = "".join("<image>" if item["type"] == "image" else item["text"]
                       for item in messages[1]["content"])
        expected = namespace["action_prompt"]("task", "url", [h["action"] for h in history],
                                               None, "(see candidate below)")
        self.assertEqual(user, expected)

    def test_canonical_prompt_keeps_normalized_json_coordinates(self):
        source = "/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-reproduction/source"
        if not Path(source, "inference/selection_prompt.py").exists():
            self.skipTest("Run prepare_arm_models.py for pinned reference")
        builder = load_selection_builder(source)
        action = '<tool_call>{"name":"click","arguments":{"point_2d":[500,250]}}</tool_call>'
        msgs = selection_messages(builder, "task", "https://example.com", [],
                                  [{"action": action, "thought": ""}], b"image")
        self.assertIn(action, msgs[1]["content"][2]["text"])
        self.assertNotIn("Action History", msgs[1]["content"][0]["text"])


class AsyncContracts(unittest.IsolatedAsyncioTestCase):
    async def test_selection_executes_original_response_with_original_tokens(self):
        import httpx
        original_client = httpx.AsyncClient
        captured, proposals = [], []
        async def handle(request):
            captured.append(json.loads(request.content))
            return httpx.Response(200, json={"raw": '{"selection":4}'})
        async def infer(url, text, params, images, timeout_secs=None):
            index = len(proposals)
            result = (f"<think>reason {index}</think>action {index}", [100 + index], [-.2], "stop")
            proposals.append((params, result))
            await asyncio.sleep(0)
            return result
        transport = httpx.MockTransport(handle)
        with tempfile.TemporaryDirectory() as output:
            selector = ActionSelector("selection", "http://arm", output)
            with patch("httpx.AsyncClient", side_effect=lambda **kw: original_client(transport=transport, **kw)):
                result, metadata = await selector(
                    infer=infer, url="http://actor", input_text="same state",
                    sampling_params={"temperature": .7}, images=["same image"],
                    observation={"screenshot": b"pre-action", "active_tab_url": "https://current"},
                    history=["<think>earlier</think>previous action"],
                    task="task", task_id="id", turn=2, timeout=10)
            self.assertEqual(result, proposals[3][1])
            self.assertEqual(metadata["selected_index"], 3)
            self.assertEqual(len(proposals), 5)
            self.assertEqual(captured[0]["url"], "https://current")
            self.assertEqual(captured[0]["history"][0]["action"], "previous action")
            self.assertEqual(len({p[0]["sampling_seed"] for p in proposals}), 5)


if __name__ == "__main__":
    unittest.main()
