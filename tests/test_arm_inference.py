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

    def test_pending_arm_concurrency_override_preserves_baseline(self):
        from openwebrl.arm_eval import execution_parallel
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(execution_parallel("scalar", root / "scalar", 8), (8, None))
            config = root / "execution-settings.json"
            config.write_text(json.dumps({"parallel_by_mode": {"scalar": 16, "selection": 16}}))
            self.assertEqual(execution_parallel("baseline", root / "baseline", 8)[0], 8)
            self.assertEqual(execution_parallel("scalar", root / "scalar", 8)[0], 16)
            config.write_text(json.dumps({"parallel_by_mode": {"scalar": 99}}))
            with self.assertRaisesRegex(ValueError, "1 through 16"):
                execution_parallel("scalar", root / "scalar", 8)

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

    def test_action_only_prompt_removes_candidate_thoughts_and_preserves_context(self):
        source = "/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-reproduction/source"
        if not Path(source, "inference/selection_prompt.py").exists():
            self.skipTest("Requires pinned reference builder")
        builder = load_selection_builder(source)
        candidates = [{"action": '<tool_call>{"name":"click","arguments":{"point_2d":[500,250]}}</tool_call>',
                       "thought": "candidate-only-secret"},
                      {"action": "scroll down", "thought": "alternative-only-secret"}]
        history = [{"action": "previous action", "thought": "history-to-preserve"}]
        full = selection_messages(builder, "task", "https://example.com", history, candidates, b"image")
        compact = selection_messages(builder, "task", "https://example.com", history, candidates, b"image", "actions_only")
        self.assertEqual(full[0], compact[0])
        self.assertEqual(full[1]["content"][:2], compact[1]["content"][:2])
        expected = full[1]["content"][2]["text"]
        for candidate in candidates:
            expected = expected.replace("     Thought: " + candidate["thought"] + "\n", "")
            self.assertIn(candidate["action"], compact[1]["content"][2]["text"])
        self.assertEqual(expected, compact[1]["content"][2]["text"])
        self.assertEqual(candidates[0]["thought"], "candidate-only-secret")


class AsyncContracts(unittest.IsolatedAsyncioTestCase):
    async def test_compact_selection_and_shadow_execute_only_compact_winner(self):
        requests = []
        async def select(endpoint, payload, *args):
            requests.append(payload)
            return {"candidate_representation": payload["candidate_representation"],
                    "raw": '{"selection":2}' if payload['candidate_representation']=='actions_only' else '{"selection":1}'}
        async def infer(*args, **kwargs):
            return ('<think>candidate thought</think>action', [1], [-.2], 'stop')
        with tempfile.TemporaryDirectory() as directory:
            selector = ActionSelector('selection', 'http://arm', directory,
                                      candidate_representation='actions_only', shadow_modulus=1)
            with patch('openwebrl.arm_inference.request_selection_result', side_effect=select):
                _, meta = await selector(infer=infer, url='actor', input_text='prompt', sampling_params={},
                    images=[], observation={'screenshot':b'image'}, history=[], task='task', task_id='task', turn=0, timeout=2)
            record = json.loads(next(Path(directory).glob('*.jsonl')).read_text())
            self.assertEqual(meta['selected_index'], 1)
            self.assertEqual(record['shadow_full']['selected_index'], 0)
            self.assertEqual([p['candidate_representation'] for p in requests], ['actions_only','full'])
            self.assertEqual(requests[0]['candidates'], requests[1]['candidates'])

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
