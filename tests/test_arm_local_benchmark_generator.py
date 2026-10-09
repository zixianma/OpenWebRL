"""CPU-only protocol tests; no browser, checkpoint load, or API requests."""
import asyncio
import ast
import logging
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import arm_local_benchmark_generator as generator


class LocalGeneratorTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {"SLIME_BROWSER_ENV_MODE": "local_process"})
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_configuration_is_explicit_and_does_not_mutate_inputs(self):
        original = SimpleNamespace(judge_api_model="gpt-4.1", max_steps=15)
        sampling = {"temperature": 0, "top_k": 1, "other": 42}
        for benchmark in generator.BENCHMARKS:
            args, values = generator.configure(original, sampling, benchmark)
            self.assertEqual(args.judge_api_model, "gpt-4o")
            self.assertEqual(args.judge_prompt_variant, benchmark)
            self.assertEqual(args.browser_env, "local_process")
            self.assertEqual(args.browser_env_mode, "local_process")
            self.assertEqual((args.max_steps, args.rollout_max_context_len,
                              args.context_num_screenshots, args.judge_max_attached_imgs),
                             (30, 32768, 1, 30))
            self.assertEqual(values, dict(sampling, **generator.SAMPLING))
        self.assertEqual(original.judge_api_model, "gpt-4.1")
        self.assertEqual(original.max_steps, 15)
        self.assertEqual(sampling["temperature"], 0)

    def test_rejects_missing_or_remote_environment(self):
        for mode in ("", "browser-use", "sandbox"):
            with self.subTest(mode=mode), patch.dict(os.environ, {"SLIME_BROWSER_ENV_MODE": mode}):
                with self.assertRaises(ValueError):
                    generator.configure(SimpleNamespace(), {}, "webvoyager")

    def test_rejects_conflicting_arguments_or_nonactor_policy(self):
        for args in (SimpleNamespace(browser_env="browser-use"),
                     SimpleNamespace(browser_env_mode="sandbox"),
                     SimpleNamespace(browser_action_selector=object())):
            with self.assertRaises(ValueError):
                generator.configure(args, {}, "deepshop")

    def test_rejects_unknown_benchmark(self):
        with self.assertRaises(ValueError):
            generator.configure(SimpleNamespace(), {}, "online_mind2web")

    def test_requires_actual_frozen_yaml_to_be_local(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            for mode in ("browser-use", "sandbox"):
                path.write_text(f"mode: {mode}\n")
                with self.assertRaises(ValueError):
                    generator.validate_local_config(path)
            path.write_text("mode: local_process\n")
            self.assertEqual(generator.validate_local_config(path)["mode"], "local_process")

    def run_generation(self, benchmark, turns, rewards=None, error=None, evaluation=True,
                       yaml_mode="local_process"):
        names = ["openwebrl.generate_browser", "openwebrl.eval.reward_webvoyager",
                 "openwebrl.eval.reward_deepshop", "openwebrl.eval_monitor", "slime.utils.types"]
        fake = {name: ModuleType(name) for name in names}
        collect = AsyncMock(return_value=turns, side_effect=error)
        judges = {name: AsyncMock(return_value=rewards) for name in generator.BENCHMARKS}
        persist = AsyncMock()
        fake["openwebrl.generate_browser"].generate_turn_sample = collect
        for name, judge in judges.items():
            fake[f"openwebrl.eval.reward_{name}"].reward_func = judge
        fake["openwebrl.eval_monitor"].persist_task = persist
        fake["slime.utils.types"].Sample = SimpleNamespace(Status=SimpleNamespace(ABORTED="aborted"))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "env").mkdir()
            (root / "env/config.yaml").write_text(f"mode: {yaml_mode}\n")
            fake["openwebrl.generate_browser"].__file__ = str(root / "generate_browser.py")
            with patch.dict(sys.modules, fake):
                coroutine = getattr(generator, f"generate_{benchmark}")(
                    SimpleNamespace(), SimpleNamespace(), {}, evaluation=evaluation)
                if error or not evaluation or yaml_mode != "local_process":
                    with self.assertRaises(type(error) if error else ValueError):
                        asyncio.run(coroutine)
                else:
                    self.assertIs(asyncio.run(coroutine), turns)
        return collect, judges, persist

    @staticmethod
    def turn(status="completed"):
        return SimpleNamespace(status=status, metadata={}, remove_sample=False, reward=None)

    def test_each_wrapper_selects_only_its_native_judge(self):
        for benchmark in generator.BENCHMARKS:
            turn = self.turn()
            collect, judges, persist = self.run_generation(benchmark, [turn], [1.0])
            collect.assert_awaited_once()
            judges[benchmark].assert_awaited_once()
            judges[next(n for n in generator.BENCHMARKS if n != benchmark)].assert_not_awaited()
            persist.assert_awaited_once()
            self.assertEqual(turn.reward, 1.0)
            self.assertFalse(turn.remove_sample)
            self.assertEqual(persist.await_args.args[0].judge_prompt_variant, benchmark)

    def test_missing_verdict_is_invalid_and_cannot_trigger_fallback(self):
        for benchmark in generator.BENCHMARKS:
            turn = self.turn()
            _, _, persist = self.run_generation(benchmark, [turn], [None])
            self.assertEqual(turn.reward, 0.0)
            self.assertTrue(turn.remove_sample)
            self.assertTrue(turn.metadata["judge_invalid"])
            self.assertIsNone(turn.metadata["benchmark_judge_reward"])
            persist.assert_awaited_once()

    def test_aborted_skips_judge_and_persists(self):
        _, judges, persist = self.run_generation("deepshop", [self.turn("aborted")])
        for judge in judges.values():
            judge.assert_not_awaited()
        persist.assert_awaited_once()

    def test_failure_and_cancellation_still_persist(self):
        for error in (TimeoutError("test"), asyncio.CancelledError()):
            _, judges, persist = self.run_generation("webvoyager", [], error=error)
            for judge in judges.values():
                judge.assert_not_awaited()
            persist.assert_awaited_once()
            self.assertEqual(persist.await_args.args[-1], type(error).__name__)

    def test_wrong_yaml_and_training_mode_fail_before_collection(self):
        for kwargs in ({"yaml_mode": "browser-use"}, {"evaluation": False}):
            collect, judges, persist = self.run_generation("webvoyager", [], **kwargs)
            collect.assert_not_awaited()
            for judge in judges.values():
                judge.assert_not_awaited()
            persist.assert_not_awaited()


class ReleasedJudgeTests(unittest.TestCase):
    def invoke(self, benchmark, answer="Final answer", parsed=True):
        # Execute the released _judge body, only replacing network and parser.
        from pydantic import BaseModel, Field
        from typing import Literal

        path = Path(__file__).resolve().parents[1] / f"openwebrl/eval/reward_{benchmark}.py"
        tree = ast.parse(path.read_text())
        selected = [node for node in tree.body if
                    isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and
                    t.id in ("SYSTEM_PROMPT", "USER_PROMPT") for t in node.targets) or
                    isinstance(node, ast.ClassDef) and node.name == "DeepShopVerdict" or
                    isinstance(node, ast.AsyncFunctionDef) and node.name == "_judge"]
        message = SimpleNamespace(content="NOT SUCCESS", refusal="mock" if not parsed else None,
                                  parsed=SimpleNamespace(verdict="NOT SUCCESS", thought="test") if parsed else None)
        call = AsyncMock(return_value=SimpleNamespace(choices=[SimpleNamespace(message=message)]))
        completions = SimpleNamespace(create=call, parse=call)
        client = SimpleNamespace(chat=SimpleNamespace(completions=completions),
                                 beta=SimpleNamespace(chat=SimpleNamespace(completions=completions)))
        namespace = dict(Any=object, Sample=object, ToolParser=object, asyncio=asyncio,
                         BaseModel=BaseModel, Field=Field, Literal=Literal,
                         logger=logging.getLogger("judge-test"),
                         _extract_final_answer=lambda parser, response: answer,
                         _get_openai_client=lambda **kwargs: client)
        exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
        sample = SimpleNamespace(response="done", metadata=dict(
            intent="Original task", full_image_list=[f"image{i}" for i in range(35)]))
        result = asyncio.run(namespace["_judge"](SimpleNamespace(judge_api_model="gpt-4o"), sample, None))
        return result, call

    def test_released_judges_keep_models_prompts_and_last_thirty_images(self):
        for benchmark in generator.BENCHMARKS:
            result, call = self.invoke(benchmark)
            self.assertEqual(result[0], 0.0)
            kwargs = call.await_args.kwargs
            self.assertEqual(kwargs["model"], "gpt-4o")
            self.assertEqual(kwargs["seed"], 42)
            content = kwargs["messages"][1]["content"]
            images = [v["image_url"]["url"] for v in content if v["type"] == "image_url"]
            self.assertEqual(images, [f"data:image/png;base64,image{i}" for i in range(5, 35)])
            if benchmark == "deepshop":
                self.assertEqual(kwargs["temperature"], 0)
                self.assertEqual(kwargs["response_format"].__name__, "DeepShopVerdict")
            else:
                self.assertNotIn("temperature", kwargs)

    def test_no_final_answer_is_native_failure_without_api_call(self):
        for benchmark in generator.BENCHMARKS:
            result, call = self.invoke(benchmark, answer="")
            self.assertEqual(result, (0.0, "No answer extracted.", False))
            call.assert_not_awaited()

    def test_deepshop_unparsed_verdict_remains_invalid(self):
        result, call = self.invoke("deepshop", parsed=False)
        self.assertIsNone(result[0])
        call.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
