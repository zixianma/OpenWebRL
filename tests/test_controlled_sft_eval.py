import asyncio
import base64
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from openwebrl.controlled_sft_eval import (
    ChatResponse, HTTPTransports, ScopedBudget, generation_bindings, run_episode, runtime_worker_config,
    JevContextCapacityExceeded, is_known_jev_context_limit, maximal_jev_diagnostic_batch,
)
from openwebrl.controlled_selector_protocol import SelectorContractError, SelectorInputBudgetExceeded
from test_controlled_sft_worker import (
    Budget, CanonicalSample, canonical_binding, kev_proof, make_policy, observation, selector_response,
)


class Ledger:
    def __init__(self): self.reservations = []; self.settlements = []
    def reserve(self, kind, count=1, **context):
        receipt = dict(reservation_id=str(len(self.reservations)), kind=kind, count=count, context=context)
        self.reservations.append(receipt)
        return receipt
    def settle(self, receipt, usage): self.settlements.append((receipt, usage))


class Response:
    def __init__(self, raw, status=200):
        self.raw, self.status_code, self.text = raw, status, json.dumps(raw)
    def json(self): return self.raw
    def raise_for_status(self): raise RuntimeError("Synthetic HTTP " + str(self.status_code))


class FakeHTTP:
    def __init__(self): self.requests = []
    async def post(self, url, **kwargs):
        self.requests.append((url, kwargs))
        if url.endswith("/generate"):
            return Response(dict(text="<think>Original proposal</think>done()", meta_info=dict(
                output_token_logprobs=[[-.1, 1]], prompt_tokens=100, completion_tokens=1,
                finish_reason={"type": "stop"})))
        if url.endswith("/responses"):
            return Response(dict(selector_response("luna", 1), usage=dict(input_tokens=10, output_tokens=2)))
        if url.endswith("/systemone"):
            return Response(selector_response("jev" if "typesafe" in url else "kev", 1))
        if url.endswith("/chat/completions"):
            return Response(dict(model="o4-mini-2025-04-16", choices=[dict(message=dict(content='Status: "success"'))],
                                 usage=dict(prompt_tokens=10, completion_tokens=4)))
        raise AssertionError(url)


def worker_config(tmp_path):
    prompt = tmp_path / "prompt.md"; prompt.write_text("Browser policy fixture\n")
    tools = tmp_path / "tools.json"; tools.write_text("[]")
    reward_path = Path(__file__).parents[1] / "openwebrl/eval/reward_online_mind2web.py"
    return dict(actor_checkpoint="/fixture/sft", prompt_path=str(prompt), prompt_sha256=hashlib.sha256(prompt.read_bytes()).hexdigest(),
        tool_list_path=str(tools), tool_list_sha256=hashlib.sha256(tools.read_bytes()).hexdigest(),
        canonical_reward_sha256=hashlib.sha256(reward_path.read_bytes()).hexdigest(),
        kev_proof=kev_proof(), tasks_path="/fixture/tasks.jsonl", browser_manifest={"fixture": True},
        browser_env_config=dict(mode="local_process", width=1280, height=720, dpr=1,
            use_screenshot=True, use_a11ytree=False, resize_output_coords=True, resize_scale=1000,
            local_process=dict(server_module="openwebrl.controlled_browser_server")))


@pytest.mark.parametrize("condition,proposals,selector_kind", [
    ("L01", 1, None), ("L08", 5, "luna_selector_requests"),
    ("L10", 5, "jev_selector_requests"), ("L11", 5, "local_kev_requests")])
def test_full_episode_with_fake_transports_and_real_canonical_code(tmp_path, monkeypatch, condition, proposals, selector_kind):
    async def run():
        monkeypatch.setenv("OPENAI_API_KEY", "offline-test-no-network")
        monkeypatch.setenv("JEV_API_KEY", "offline-test-no-network")
        monkeypatch.setenv("OPENWEBRL_KEV_FORWARD_LOCK", str(tmp_path / "kev.lock"))
        config, ledger, http = worker_config(tmp_path), Ledger(), FakeHTTP()
        async def forbidden(**unused): raise AssertionError("Only injected fake transport may run")
        _, reward = canonical_binding(Budget(), lambda *unused: None, forbidden)
        class Sample(CanonicalSample):
            def __init__(self, index, prompt, metadata):
                super().__init__(); self.metadata.update(metadata)
                self.index, self.prompt, self.response = index, prompt, ""
        # Canonical isinstance check is against the parent CanonicalSample.
        reward.Sample = Sample
        closed = []
        class Browser:
            async def exit(self): closed.append(True)
            async def _post(self, path, payload, **kwargs):
                assert path == "/observe"
                obs = observation(1, "done", 2)
                obs["screenshot"] = base64.b64encode(obs["screenshot"]).decode()
                return dict(observation=obs)
            def _decode_screenshot(self, data): return base64.b64decode(data)
        async def create(*args, **kwargs): return Browser(), {}
        generation = SimpleNamespace(_apply_local_process_env_overrides=lambda value: value,
            _encode_with_processor=lambda *args: ([1] * 100, None, None, None),
            _run_inference_step=forbidden, _create_env=create,
            _ensure_im_end_w_new_line=lambda text, tokens, logprobs: text,
            _load_local_resources=lambda *args, **kwargs: (dict(start_url="https://example.com", intent="Task"), [], "Browser policy fixture\n"))
        async def generate(args, sample, sampling):
            env_config = generation._apply_local_process_env_overrides({})
            generation._load_local_resources()
            browser, _ = await generation._create_env("fixture", env_config)
            generation._encode_with_processor(None)
            args.browser_action_selector.set_training_context(dict(prompt_tokens=[1] * 100))
            output, _ = await args.browser_action_selector(infer=generation._run_inference_step,
                url="http://127.0.0.1:10000/generate", input_text="Browser policy fixture\nTask",
                sampling_params=sampling, images=[base64.b64encode(b"current-image").decode()],
                observation=observation(), history=[], task="Task", task_id="fixture", turn=0, timeout=180)
            sample.response = output[0]
            sample.status = Sample.Status.COMPLETED
            sample.metadata.update(turn_index=0, is_last_turn=True,
                messages=[dict(role="assistant", content=output[0])],
                # Realistic saved multimodal metadata must not enter the shared DB.
                full_image_list=[base64.b64encode(b"image" * 100000).decode()])
            await browser.exit()
            return [sample]
        generation.generate_turn_sample = generate
        evaluation = SimpleNamespace(EvalArgs=lambda **kwargs: SimpleNamespace(**kwargs), _to_jsonable=lambda value: value)
        claim = dict(condition=condition, task_id="fixture", attempt_id="physical-001", artifact_directory=str(tmp_path / "attempt"),
                     task=dict(id="fixture", metadata=dict(intent="Task", start_url="https://example.com")))
        result = await run_episode(claim, config, ledger, http, actor_base_url="http://127.0.0.1:10000",
            kev_base_url="http://127.0.0.1:10001", dependencies=(generation, evaluation, reward))
        assert result["valid"] and result["reward"] == 1. and result["browser_closed"]
        assert len(json.dumps(result)) < 100000 and "metadata" not in result
        artifact = Path(result["result_path"])
        assert artifact.stat().st_size > 500000 and hashlib.sha256(artifact.read_bytes()).hexdigest() == result["result_sha256"]
        assert sum(row["count"] for row in ledger.reservations if row["kind"] == "local_sft_generations") == proposals
        kinds = [row["kind"] for row in ledger.reservations]
        assert kinds.count("om2w_judge_http_attempts") == 1
        assert kinds.count(selector_kind) == 1 if selector_kind else len(kinds) == 3
        assert all(row["context"]["attempt_id"] == "physical-001" for row in ledger.reservations)
        assert any(usage == {"closed": True} for _, usage in ledger.settlements)
        assert closed == [True]
        assert generation._create_env is create
    asyncio.run(run())


def test_encoding_guard_precedes_native_early_context_guard_and_restores_bindings(tmp_path):
    config = worker_config(tmp_path)
    policy, _, _ = make_policy(tmp_path)
    encode = lambda *args: ([1] * 28672, None, None, None)
    module = SimpleNamespace(_apply_local_process_env_overrides=lambda value: value,
        _encode_with_processor=encode, _run_inference_step=None, _load_local_resources=None)
    with generation_bindings(module, config=config, policy=policy, actor_call=None):
        with pytest.raises(SelectorInputBudgetExceeded): module._encode_with_processor()
        assert policy.input_invalid and policy.decisions == 0
    assert module._encode_with_processor is encode


def test_runtime_overlay_rejects_different_pinned_kev_and_allocation(tmp_path):
    config = worker_config(tmp_path)
    frozen = deepcopy(config); frozen["kev_proof"].pop("server_card")
    runtime = dict(plan_sha256="planhash", job_id="123", actor_base_urls=["http://127.0.0.1:10000"],
        kev_base_url="http://127.0.0.1:10001", kev_proof=config["kev_proof"])
    kwargs = dict(plan_sha256="planhash", actor_base_url=runtime["actor_base_urls"][0], kev_base_url=runtime["kev_base_url"], job_id="123")
    assert runtime_worker_config({"worker_config": frozen}, runtime, **kwargs)["kev_proof"] == config["kev_proof"]
    assert "server_card" not in frozen["kev_proof"]
    changed = deepcopy(runtime); changed["kev_proof"]["revision"] = "another-model"
    with pytest.raises(SelectorContractError): runtime_worker_config({"worker_config": frozen}, changed, **kwargs)
    with pytest.raises(SelectorContractError): runtime_worker_config({"worker_config": frozen}, runtime, **dict(kwargs, job_id="124"))


def test_http400_body_is_saved_before_error_and_not_retried():
    async def run():
        rows, calls, halts = [], [], []
        async def post(*args, **kwargs): calls.append(args); return Response({"error": "context_limit"}, 400)
        transport = HTTPTransports(SimpleNamespace(post=post), actor_base_url="http://127.0.0.1:10000",
            kev_base_url="http://127.0.0.1:10001", openai_key="fixture", jev_key="fixture",
            journal=lambda event, row: rows.append((event, row)), normalize_actor=None,
            on_failure=lambda reason, **details: halts.append((reason, details, len(rows))))
        with pytest.raises(RuntimeError): await transport.selector("jev", {})
        assert len(calls) == 1 and rows[0][0] == "http_error" and "context_limit" in rows[0][1]["body"]
        assert halts[0][0] == "provider_http_error" and halts[0][2] == 1
    asyncio.run(run())


def test_shared_kev_gate_limits_multiple_worker_transports_to_one_forward(tmp_path, monkeypatch):
    async def run():
        monkeypatch.setenv("OPENWEBRL_KEV_FORWARD_LOCK", str(tmp_path / "kev-forward.lock"))
        active, maximum, calls = 0, 0, 0
        async def post(*args, **kwargs):
            nonlocal active, maximum, calls
            calls += 1; active += 1; maximum = max(maximum, active)
            await asyncio.sleep(.01)
            active -= 1
            return Response(selector_response("kev", 1))
        def transport():
            return HTTPTransports(SimpleNamespace(post=post), actor_base_url="http://127.0.0.1:10000",
                kev_base_url="http://127.0.0.1:10001", openai_key=None, jev_key=None,
                journal=lambda *args: None, normalize_actor=None)
        await asyncio.gather(*(transport().selector("kev", {}) for _ in range(4)))
        assert maximum == 1 and calls == 4 and active == 0
    asyncio.run(run())


@pytest.mark.parametrize("status,body,known", [
    (400, {"detail": {"error_type": "max_tokens_exceeded"}}, True),
    (401, {"detail": {"error_type": "max_tokens_exceeded"}}, False),
    (400, {"detail": {"error_type": "invalid_api_key"}}, False),
    (400, {"detail": "max_tokens_exceeded"}, False),
    (400, {"error_type": "max_tokens_exceeded"}, False),
])
def test_only_confirmed_jev_context_error_is_isolated(status, body, known):
    assert is_known_jev_context_limit(status, body) is known


def test_known_capacity_rejection_is_input_invalid_without_global_halt(tmp_path):
    async def run():
        rows, halts, calls = [], [], []
        async def post(*args, **kwargs):
            calls.append(args)
            return Response({"detail": {"error_type": "max_tokens_exceeded"}}, 400)
        transport = HTTPTransports(SimpleNamespace(post=post), actor_base_url="http://127.0.0.1:10000",
            kev_base_url="http://127.0.0.1:10001", openai_key=None, jev_key="fixture",
            journal=lambda event, row: rows.append((event, row)), normalize_actor=None,
            on_failure=lambda *args, **kwargs: halts.append(args))
        policy, budget, _ = make_policy(tmp_path, condition="L10", selector=transport.selector)
        async def infer(*args, **kwargs): return ("<think>Full candidate</think>done()", [1], [-.1], "stop")
        with pytest.raises(JevContextCapacityExceeded):
            await policy(infer=infer, url="unused", input_text="Browser policy fixture\nTask",
                sampling_params=dict(temperature=1., top_p=.95, top_k=-1, max_new_tokens=4096, repetition_penalty=1.),
                images=[base64.b64encode(b"current-image").decode()], observation=observation(),
                history=[], task="Task", task_id="fixture", turn=0, timeout=180)
        assert policy.input_invalid and not policy.provider_invalid and not policy.contract_invalid
        assert not halts and len(calls) == 1
        assert rows[0][0] == "http_error" and rows[1][0] == "jev_known_context_capacity_rejection"
        assert any(kind == "jev_selector_requests" for kind, _, _ in budget.reservations)
    asyncio.run(run())


def test_maximum_shared_state_diagnostic_is_bounded_reproducible_and_keeps_candidates():
    from openwebrl.controlled_selector_protocol import selector_request
    first, second = maximal_jev_diagnostic_batch(), maximal_jev_diagnostic_batch()
    assert first.state_sha256 == second.state_sha256
    assert len(first.state_json.encode()) == 262144
    state = json.loads(first.state_json)
    assert len(state["page"]["text"]) == 16000
    assert len(first.original_texts) == 5 and all("</think>click(point_2d=[50,50])" in text for text in first.original_texts)
    assert selector_request(first, "jev") == selector_request(second, "jev")


@pytest.mark.parametrize("capacity_rejected", [False, True])
def test_startup_diagnostic_preserves_accepted_or_known_capacity_result_without_browser(tmp_path, monkeypatch, capacity_rejected):
    import httpx
    from openwebrl import controlled_sft_state
    from openwebrl.controlled_sft_eval import run_jev_context_preflight
    async def run():
        reservations, calls, halts, approvals = [], [], [], []
        class State:
            plan_hash = "fixture-plan-sha"
            def __init__(self, root): assert root == tmp_path
            def require_approved_allocation(self): approvals.append(True)
            def reserve(self, kind, count, **kwargs):
                reservations.append((kind, count, kwargs))
                return dict(reservation_id="fixture-only", kind=kind, count=count)
            def halt(self, *args, **kwargs): halts.append((args, kwargs))
        class HTTP:
            def __init__(self, **kwargs): pass
            async def __aenter__(self): return self
            async def __aexit__(self, *args): pass
            async def post(self, url, **kwargs):
                assert approvals == [True] and len(reservations) == 1
                calls.append(url)
                return (Response({"detail": {"error_type": "max_tokens_exceeded"}}, 400)
                        if capacity_rejected else Response(selector_response("jev", 1)))
        monkeypatch.setattr(controlled_sft_state, "SuiteState", State)
        monkeypatch.setattr(httpx, "AsyncClient", HTTP)
        monkeypatch.setenv("SLURM_JOB_ID", "fixture-no-allocation")
        monkeypatch.setenv("JEV_API_KEY", "fixture-no-network")
        result = await run_jev_context_preflight(tmp_path)
        assert result["completed"] and result["state_bytes"] == 262144
        assert result["outcome"] == ("known_capacity_rejection" if capacity_rejected else "accepted")
        assert result["model_identity"] == (None if capacity_rejected else "jev-1.13.0")
        assert len(calls) == 1 and not halts and result["browser_episodes"] == result["sft_proposals"] == 0
        receipt = tmp_path / "scheduler-attempts/fixture-no-allocation/jev-context-preflight.json"
        assert json.loads(receipt.read_text()) == result
        with pytest.raises(SelectorContractError, match="do not replay"):
            await run_jev_context_preflight(tmp_path)
        assert len(calls) == 1
    asyncio.run(run())
