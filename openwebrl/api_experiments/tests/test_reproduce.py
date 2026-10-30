"""Offline boundary tests; never import a real SDK or launch a browser."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1] / "reproduce/common.py"
spec = importlib.util.spec_from_file_location("reproduce_common", SOURCE)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class ReproductionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="api-repro-test-")
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.runtime = self.base / "runtime"
        self.runtime.mkdir()
        self.output = self.base / "output"
        self.tasks = [{"task_id": str(i), "intent": "test", "start_url": "https://example.test"} for i in range(3)]
        self.config = {k: str(self.runtime / "dependency") for k in ("cuawright_source", "frozen_support_source", "chromium", "python")}
        self.config["judge_protocol"] = {"model": "gpt-5.6-terra"}
        (self.runtime / "dependency").touch()
        for name in ("browser_bridge.ts", "browsercode", "node_modules", "javascript-runtime/bun", "sandbox.py"):
            p = self.runtime / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.touch()
        self.write("tasks.json", self.tasks)
        self.write("runtime-config.json", self.config)
        self.plan = {"protocol": {"max_decisions": 30}, "task_manifest_sha256": m.digest(self.runtime / "tasks.json"), "arms": []}
        for provider, arms in m.ARMS.items():
            for arm in arms:
                count = 5 if "n5" in arm or "parallel5" in arm else 3 if "n3" in arm or "parallel3" in arm else 1
                effort = arm.rsplit("_", 1)[1] if provider == "luna" else "high"
                model = ("gpt-6-sol" if arm == "serial_high" else "gpt-6-luna") if provider in ("luna", "gpt") else "claude-opus-5-5" if arm == "opus_single" else "claude-sonnet-5-5"
                self.plan["arms"].append(dict(arm=arm, proposal_count=count, model=model, reasoning_effort=effort, output_tokens=12288))
        self.seal()

    def write(self, name, value):
        (self.runtime / name).write_text(json.dumps(value))

    def seal(self):
        self.write("campaign-plan.json", self.plan)
        self.write("source-manifest.json", {"plan_sha256": m.digest(self.runtime / "campaign-plan.json"), "source_hashes": {str(self.runtime / "runtime-config.json"): m.digest(self.runtime / "runtime-config.json")}})

    def args(self, arm="luna_n5_medium", provider="luna", extra=()):
        return m.parser(provider).parse_args(["--runtime", str(self.runtime), "--output", str(self.output), "--arm", arm, "--model-usd", "5", "--judge-usd", "1", "--wandb-mode", "disabled", *extra])

    def test_all_arms_read_only_and_exact_settings(self):
        before = {str(p): p.read_bytes() for p in self.runtime.rglob("*") if p.is_file()}
        for provider, arms in m.ARMS.items():
            for arm in arms:
                plan, tasks = m.prepare(self.args(arm, provider), provider)
                self.assertEqual(plan["candidate_count"], 5 if "n5" in arm or "parallel5" in arm else 3 if "n3" in arm or "parallel3" in arm else 1)
                self.assertEqual((plan["max_output_tokens"], plan["max_decisions"], len(tasks)), (12288, 30, 1))
                self.assertFalse(plan["execute"])
        self.assertFalse(self.output.exists())
        self.assertEqual(before, {str(p): p.read_bytes() for p in self.runtime.rglob("*") if p.is_file()})

    def test_plan_and_source_tampering_rejected(self):
        for target in ("campaign-plan.json", "runtime-config.json", "tasks.json"):
            p = self.runtime / target
            original = p.read_bytes()
            p.write_bytes(original + b" ")
            with self.assertRaises(ValueError):
                m.prepare(self.args(), "luna")
            p.write_bytes(original)
        self.assertFalse(self.output.exists())

    def test_caps_paths_and_task_ids_rejected(self):
        for flags in (("--output", "/gpfs/home/zixianma/repro-test"), ("--limit", "0"), ("--model-calls", "0"), ("--browser-starts", "0"), ("--output", str(self.runtime / "new"))):
            with self.assertRaises(ValueError):
                m.prepare(self.args(extra=flags), "luna")
        self.output.mkdir()
        with self.assertRaises(ValueError):
            m.prepare(self.args(), "luna")
        for invalid in ("0", "-1", "NaN", "Infinity"):
            with self.assertRaises(Exception):
                m.positive_money(invalid)
        self.tasks[0]["task_id"] = "../escape"
        self.write("tasks.json", self.tasks)
        self.plan["task_manifest_sha256"] = m.digest(self.runtime / "tasks.json")
        self.seal()
        self.output = self.base / "another-output"
        with self.assertRaises(ValueError):
            m.prepare(self.args(), "luna")

    def test_valid_only_summary_and_missing_metrics(self):
        result = m.summary([dict(valid=True, reward=1, task_latency_seconds=2), dict(valid=False, reward=0), dict(valid=True, reward=0)], {}, True)
        self.assertEqual((result["recorded"], result["valid"], result["invalid"], result["successes"]), (3, 2, 1, 1))
        self.assertEqual(result["valid_only_success"], .5)
        self.assertEqual(result["recorded_success_rate"], 1 / 3)
        self.assertEqual(result["valid_task_latency_seconds"], {"mean": 2, "observations": 1})
        self.assertIsNone(result["valid_selected_browser_batches"]["mean"])
        self.assertIsNone(m.summary([], {}, True)["valid_only_success"])

    def fake_execute(self, provider, halted=False, tracking=False):
        arm = "luna_n5_medium" if provider == "luna" else "serial_high" if provider == "gpt" else "sonnet_n5"
        args = self.args(arm, provider, ("--limit", "3", "--execute"))
        if tracking: args.wandb_mode = "offline"
        plan, tasks = m.prepare(args, provider)
        calls, clients, gates = [], [], []
        class Client:
            def __init__(self, **kwargs):
                self.closed = False
                assert kwargs["max_retries"] == 0
                clients.append(self)
            def close(self): self.closed = True
        class Pool:
            def __init__(self, root, *caps):
                self.root = root
                root.mkdir(parents=True)
                self.calls = 0
            def snapshot(self): return {"calls": self.calls, "charged_or_reserved_usd": str(self.calls)}
        class Model:
            def __init__(self, **kwargs):
                if "http_gate" in kwargs: gates.append(kwargs["http_gate"])
        class Gate:
            def __init__(self, root, capacity):
                self.root, self.capacity = root, capacity
        worker = types.SimpleNamespace(ReservationBudget=Pool, Budget=Pool, PairedModel=Model)
        def episode(root, task, cell, actor, judge, budgets, slot):
            worker.PairedModel()
            calls.append(task["task_id"])
            budgets["actor"].calls += 1
            budgets["judge"].calls += 1
            if halted: (budgets["actor"].root / "halt.json").write_text("{}")
            return dict(valid=len(calls) != 2, reward=1 if len(calls) == 1 else 0)
        worker.episode = episode
        worker.preflight = lambda root, browser: None
        cleaned = []
        modules = {"worker": worker, "environments": types.SimpleNamespace(cleanup_registered_groups=lambda root: cleaned.append(root)), "request_budget": types.SimpleNamespace(SharedHTTPGate=Gate), "claude_budget": types.SimpleNamespace(SharedHTTPGate=Gate)}
        class Tracking:
            def log(self, row): pass
            def finish(self): pass
        def track(**kwargs):
            self.assertEqual(kwargs["resume"], "never")
            self.assertNotEqual(kwargs["id"], "inherited-campaign")
            self.assertEqual(kwargs["project"], "openwebrl-evals")
            self.assertNotIn("WANDB_RUN_ID", os.environ)
            self.assertNotIn("WANDB_RESUME", os.environ)
            return Tracking()
        fake_wandb = types.SimpleNamespace(init=track, Settings=lambda **kwargs: None)
        before = {str(p): p.read_bytes() for p in self.runtime.rglob("*") if p.is_file()}
        with patch.dict(os.environ, {"OPENAI_API_KEY": "offline-fake", "ANTHROPIC_API_KEY": "offline-fake", "WANDB_RUN_ID": "inherited-campaign", "WANDB_RESUME": "must"}), patch.dict(sys.modules, {"wandb": fake_wandb, "openai": types.SimpleNamespace(OpenAI=Client), "anthropic": types.SimpleNamespace(Anthropic=Client)}), patch.object(m.importlib, "import_module", side_effect=modules.__getitem__), patch.object(m.signal, "signal"), patch.object(sys, "path", list(sys.path)), contextlib.redirect_stdout(io.StringIO()):
            code = m.execute(args, plan, tasks)
        self.assertEqual(code, 1)  # Invalid or budget-halted coverage must signal failure.
        self.assertEqual(len(calls), 1 if halted else 3)
        self.assertTrue(all(c.closed for c in clients))
        self.assertEqual(cleaned, [self.output])
        self.assertTrue(all(g.root == self.output / "http-gate" and g.capacity == 8 for g in gates))
        self.assertEqual(before, {str(p): p.read_bytes() for p in self.runtime.rglob("*") if p.is_file()})
        result = m.read(self.output / "summary.json")
        self.assertEqual(result["all_attempt_pools"]["actor"]["calls"], len(calls))
        self.assertEqual(result["valid_only_success"], 1 if halted else .5)

    def test_luna_execution_uses_independent_state_and_counts_invalid_cost(self):
        self.fake_execute("luna")

    def test_claude_execution_uses_independent_state_and_counts_invalid_cost(self):
        self.fake_execute("claude")

    def test_older_gpt_execution_has_isolated_budgets_and_disabled_retries(self):
        self.fake_execute("gpt")

    def test_tracking_cannot_resume_campaign_identity(self):
        self.fake_execute("luna", tracking=True)

    def test_halt_stops_without_retry(self):
        self.fake_execute("luna", halted=True)


if __name__ == "__main__":
    unittest.main()
