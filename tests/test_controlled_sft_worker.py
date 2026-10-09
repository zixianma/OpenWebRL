import asyncio
import ast
import base64
from copy import deepcopy
import hashlib
import json
import logging
import re
from enum import Enum
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from openwebrl.controlled_selector_protocol import SelectorContractError, SelectorInputBudgetExceeded
from openwebrl.controlled_sft_worker import (
    AttemptJournal, ControlledSFTPolicy, ControlledClient, EpisodeBoundary, KEV_IDENTITY, SAMPLING,
    CanonicalOM2WReward,
    controlled_environment_factory, judge_terminal, validate_kev_startup_proof,
    validate_actor_environment_config,
)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def observation(count=0, reason=None, sequence=1):
    image = b"current-image"
    page = dict(url="https://example.com", title="Example", text="Search", text_characters=6,
        interactive_elements=[], tabs=[dict(url="https://example.com", title="Example", index=0, active=True)],
        screen_size=[1280, 720])
    actions = [dict(attempt=i + 1, action={"name": "wait", "args": {}}, status="returned",
                    success=True, tool_response="ok") for i in range(count)]
    return dict(screenshot=image, selection_page=page, active_tab_url=page["url"],
        screen_size=[1280, 720], controlled=dict(sequence=sequence,
            screenshot_sha256=hashlib.sha256(image).hexdigest(), page_sha256=digest(page),
            capture_started_monotonic=0., capture_finished_monotonic=1.,
            action_attempts=count, executed_actions=actions, terminal_reason=reason,
            infrastructure_invalid=False))


class Budget:
    def __init__(self): self.reservations = []; self.blocked = False
    def reserve(self, kind, count, **context):
        if self.blocked: raise RuntimeError("Synthetic budget exhausted")
        self.reservations.append((kind, count, context))


def kev_proof():
    return dict(KEV_IDENTITY, files_verified=True, verified_model_path="/fixture/verified-weights",
        server_card=dict(name="kev-latest", base=KEV_IDENTITY["base"], device="cuda",
            backend="torch", dtype="bfloat16", max_state_tokens=65536, truncate_states=False,
            temperature=1.319507910772894, run="/fixture/verified-weights", cuda_graphs=True))


def make_policy(tmp_path, condition="L01", selector=None, **extra):
    path = tmp_path / "prompt.md"
    path.write_text("Browser policy fixture\n")
    budget, journal = Budget(), []
    policy = ControlledSFTPolicy(condition, task_id="fixture", budget=budget,
        journal=lambda event, row: journal.append((event, row)), selector_call=selector,
        prompt_path=path, prompt_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        kev_proof=kev_proof() if condition == "L11" else None, **extra)
    policy.set_training_context({"prompt_tokens": [1] * 100})
    return policy, budget, journal


def call_kwargs(infer, **extra):
    result = dict(infer=infer, url="unused-local-endpoint", input_text="Browser policy fixture\nUser task",
        sampling_params=dict(SAMPLING), images=[base64.b64encode(b"current-image").decode()],
        observation=observation(), history=[], task="Find something", task_id="fixture", turn=0, timeout=180)
    result.update(extra)
    return result


def selector_response(provider, index=2):
    if provider == "luna":
        return dict(model="gpt-6-luna", status="completed", output=[dict(type="message", role="assistant",
            content=[dict(type="output_text", text=json.dumps({"selected_index": index}))])])
    return dict(model="jev-1.13.0" if provider == "jev" else "kev-latest", answers={"selection": dict(
        type="choice", choice=str(index), probabilities={str(i): float(i == index) for i in range(1, 6)})})


@pytest.mark.parametrize("condition,count,provider", [("L01", 1, None), ("L08", 5, "luna"),
                                                      ("L10", 5, "jev"), ("L11", 5, "kev")])
def test_four_conditions_reserve_before_dispatch_and_return_original_tuple(tmp_path, condition, count, provider):
    async def run():
        outputs, calls = [], []
        async def select(actual_provider, request):
            calls.append((actual_provider, request))
            assert budget.reservations[-1][1] == 1
            return selector_response(actual_provider)
        policy, budget, journal = make_policy(tmp_path, condition, select)
        async def infer(*args, **kwargs):
            assert budget.reservations[0][:2] == ("local_sft_generations", count)
            output = (f"reason {len(outputs)}</think><tool_call>wait</tool_call>", [len(outputs)], [-.1], "stop")
            outputs.append(output)
            return output
        output, metadata = await policy(**call_kwargs(infer))
        assert len(outputs) == count and any(output is value for value in outputs)
        assert metadata["condition"] == condition and policy.decisions == 1
        assert len(calls) == int(provider is not None)
        assert len([e for e, _ in journal if e == "proposal_received"]) == count
        if provider == "luna":
            assert "tools" not in calls[0][1] and "previous_response_id" not in calls[0][1]
    asyncio.run(run())


@pytest.mark.parametrize("tokens,accepted", [(28671, True), (28672, False), (32768, False)])
def test_context_reserves_full4096_and_one_server_token_before_inference(tmp_path, tokens, accepted):
    async def run():
        policy, budget, _ = make_policy(tmp_path)
        policy.set_training_context({"prompt_tokens": [1] * tokens})
        calls = []
        async def infer(*args, **kwargs):
            calls.append(args[2]); return ("wait", [1], [-.1], "stop")
        if accepted:
            await policy(**call_kwargs(infer)); assert calls[0]["max_new_tokens"] == 4096
        else:
            with pytest.raises(SelectorInputBudgetExceeded): await policy(**call_kwargs(infer))
            assert not calls and not budget.reservations and policy.input_invalid
    asyncio.run(run())


@pytest.mark.parametrize("changes", [dict(images=[]), dict(images=["wrong-image"]),
    dict(input_text="policy omitted"), dict(sampling_params=dict(SAMPLING, top_k=20)),
    dict(task_id="other"), dict(turn=1), dict(observation=observation(count=30))])
def test_protocol_failure_precedes_any_model_or_budget_work(tmp_path, changes):
    async def run():
        policy, budget, _ = make_policy(tmp_path)
        async def infer(*args, **kwargs): raise AssertionError("must not dispatch")
        with pytest.raises((SelectorContractError, EpisodeBoundary)):
            await policy(**call_kwargs(infer, **changes))
        assert not budget.reservations
    asyncio.run(run())


def test_failed_proposal_waits_for_siblings_without_selecting(tmp_path):
    async def run():
        finished, calls = [], []
        async def select(*args): calls.append(args); raise AssertionError("no partial batch")
        policy, _, journal = make_policy(tmp_path, "L08", select)
        started = 0
        async def infer(*args, **kwargs):
            nonlocal started
            number = started; started += 1
            if number == 0: raise RuntimeError("synthetic failed proposal")
            await asyncio.sleep(.001)
            finished.append(number)
            return ("wait", [1], [-.1], "stop")
        with pytest.raises(RuntimeError): await policy(**call_kwargs(infer))
        assert sorted(finished) == [1, 2, 3, 4] and not calls and policy.provider_invalid
        assert len([e for e, _ in journal if e == "proposal_error"]) == 1
    asyncio.run(run())


def test_sixty_decisions_and_deadline_are_independent_of_action_counter(tmp_path):
    async def run():
        now = [0.]
        policy, budget, _ = make_policy(tmp_path, clock=lambda: now[0])
        async def infer(*args, **kwargs): return ("malformed-no-action", [1], [-.1], "stop")
        for turn in range(60): await policy(**call_kwargs(infer, turn=turn))
        with pytest.raises(EpisodeBoundary): await policy(**call_kwargs(infer, turn=60))
        assert len(budget.reservations) == 60
        policy, budget, _ = make_policy(tmp_path, clock=lambda: now[0])
        now[0] = 1800.
        with pytest.raises(EpisodeBoundary): await policy(**call_kwargs(infer))
        assert not budget.reservations
    asyncio.run(run())


@pytest.mark.parametrize("change", [dict(revision="wrong"), dict(head_sha256="wrong"),
                                    dict(files_verified=False), dict(base="wrong")])
def test_kev_alias_without_verified27b_checkpoint_is_not_enough(change):
    proof = kev_proof(); proof.update(change)
    with pytest.raises(SelectorContractError): validate_kev_startup_proof(proof)
    with pytest.raises(SelectorContractError): validate_kev_startup_proof({"model": "kev-latest"})


class CanonicalSample:
    class Status(Enum):
        COMPLETED = "completed"
        FAILED = "failed"
        ABORTED = "aborted"
    def __init__(self, status=None):
        self.status = status or self.Status.COMPLETED
        self.metadata = dict(intent="Complete the task", task_id="fixture", messages=[
            dict(role="assistant", content='<think>ORIGINAL THOUGHT</think> done()')],
            full_image_list=[base64.b64encode(b"old-image").decode()])
        self.reward = None
        self.remove_sample = False


def canonical_binding(budget, journal, transport):
    # Execute the actual unchanged source functions, omitting only imports whose
    # training dependencies are not installed in the lightweight CPU test env.
    source = Path(__file__).parents[1] / "openwebrl/eval/reward_online_mind2web.py"
    tree = ast.parse(source.read_text())
    tree.body = [node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))]
    class Parser:
        def __init__(self, *args, **kwargs): pass
        def parse(self, text):
            return SimpleNamespace(success=True, calls=[SimpleNamespace(name="done", parameters={})])
    @asynccontextmanager
    async def semaphore(): yield
    namespace = dict(__file__=str(source), __name__="canonical_fixture", asyncio=asyncio,
        json=json, logging=logging, re=re, Any=Any, Sample=CanonicalSample,
        ToolParser=Parser, _TOOLS_INFO=[], resolve_parser_type=lambda checkpoint: "fixture",
        _reward_semaphore=semaphore())
    exec(compile(tree, str(source), "exec"), namespace)
    module = SimpleNamespace(**namespace)
    binding = CanonicalOM2WReward(module, source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        chat_create=transport, budget=budget, journal=journal, task_id="fixture")
    return binding, module


def judge_args():
    return SimpleNamespace(judge_api_model="o4-mini-2025-04-16", hf_checkpoint="fixture", judge_timeout_secs=1)


def test_terminal_uses_actual_canonical_thoughts_actions_parser_and_fresh_image(tmp_path):
    async def run():
        events, received, rows = [], [], []
        policy, budget, _ = make_policy(tmp_path)
        async def close(): events.append("close")
        async def capture(client): events.append("capture"); return observation(3, "done", 2)
        async def transport(**request):
            received.append(request)
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='Status: "success"'))])
        journal = lambda event, row: rows.append((event, row))
        binding, module = canonical_binding(budget, journal, transport)
        sample = CanonicalSample()
        original_messages = deepcopy(sample.metadata["messages"])
        state = {}
        client = ControlledClient(SimpleNamespace(exit=close), state, capture=capture)
        state["client"] = client
        result = await judge_terminal(args=judge_args(), samples=sample, final_state=state, policy=policy,
            canonical_reward=binding, journal=journal)
        await client.exit()
        assert events == ["capture", "close"] and result["valid"] and result["score"] == 1.
        assert received[0]["messages"][0]["content"] == module.SYSTEM_PROMPT
        prompt = received[0]["messages"][1]["content"]
        assert "Thought 1: ORIGINAL THOUGHT" in prompt[0]["text"] and "Action 1: done()" in prompt[0]["text"]
        assert base64.b64decode(prompt[1]["image_url"]["url"].split(",", 1)[1]) == b"current-image"
        assert sample.metadata["messages"] == original_messages
        assert received[0]["seed"] == 42 and received[0]["max_completion_tokens"] == 4096
        assert [row[0] for row in budget.reservations] == ["om2w_judge_http_attempts"]
        assert sample.metadata["reward"]["protocol"] == "online_mind2web"
        # Transport binding did not replace the module's reward code or globals.
        assert module.reward_func.__code__ is binding.reward_func.__code__
        assert module.reward_func.__globals__ is not binding.reward_func.__globals__
        assert module._parse_verdict('Status: unsuccessful') == 1.  # preserve official substring parser
    asyncio.run(run())


@pytest.mark.parametrize("status", [CanonicalSample.Status.FAILED, CanonicalSample.Status.ABORTED])
def test_noncompleted_terminal_gets_canonical_zero_without_api_even_with_usable_image(tmp_path, status):
    async def run():
        policy, budget, _ = make_policy(tmp_path)
        async def forbidden(**request): raise AssertionError("Noncompleted episodes must not invoke judge")
        binding, _ = canonical_binding(budget, lambda *args: None, forbidden)
        sample = CanonicalSample(status)
        result = await judge_terminal(args=judge_args(), samples=sample,
            final_state=dict(observation=observation(30, "action_limit"), browser_closed=True),
            policy=policy, canonical_reward=binding, journal=lambda *args: None)
        assert result["valid"] and result["score"] == 0. and sample.status == status
        assert result["judge_http_attempts"] == 0 and not budget.reservations
        assert "Judge not run for status=" in sample.metadata["reward"]["judge_text"]
    asyncio.run(run())


def test_missing_fresh_capture_is_invalid_and_does_not_fall_back_to_last_image(tmp_path):
    async def run():
        policy, budget, _ = make_policy(tmp_path)
        policy.last_observation = observation()
        closed = []
        async def close(): closed.append(True)
        async def capture(client): raise RuntimeError("browser disconnected")
        async def forbidden(**request): raise AssertionError("No stale terminal image fallback")
        binding, _ = canonical_binding(budget, lambda *args: None, forbidden)
        sample = CanonicalSample()
        state = {}
        state["client"] = ControlledClient(SimpleNamespace(exit=close), state, capture=capture)
        result = await judge_terminal(args=judge_args(), samples=sample, final_state=state, policy=policy,
            canonical_reward=binding, journal=lambda *args: None)
        assert not result["valid"] and result["score"] is None and not budget.reservations and closed == [True]
    asyncio.run(run())


def test_terminal_capture_transport_margin_and_cancelled_waiter_preserve_fresh_evidence(tmp_path,monkeypatch):
    from openwebrl import controlled_browser_server as server
    # A response arrives after the capture budget but before the outer transport
    # deadline; scale the timing down without changing which data is accepted.
    monkeypatch.setattr(server,'OBSERVATION_RPC_TIMEOUT_SECONDS',.15)
    monkeypatch.setattr(server,'TERMINAL_CAPTURE_TIMEOUT_SECONDS',.2)
    async def run():
        closed=[];timeouts=[]
        async def post(path,body,*,timeout_secs):
            timeouts.append(timeout_secs)
            await asyncio.sleep(.07)
            return {'observation':observation(3,'done',2)}
        async def close():closed.append(True)
        state={}
        raw=SimpleNamespace(_post=post,_decode_screenshot=lambda x:x,exit=close)
        client=ControlledClient(raw,state)
        waiter=asyncio.create_task(client.exit())
        await asyncio.sleep(.02);waiter.cancel()
        with pytest.raises(asyncio.CancelledError):await waiter
        await client.exit()
        assert state['observation']['controlled']['terminal_reason']=='done'
        assert state['browser_closed'] and not state.get('capture_error_type')
        assert closed==[True] and timeouts==[.15]
        assert state['terminal_capture_seconds']>=.07
    asyncio.run(run())


def test_terminal_capture_stays_bounded_and_closes_browser(monkeypatch):
    from openwebrl import controlled_browser_server as server
    monkeypatch.setattr(server,'TERMINAL_CAPTURE_TIMEOUT_SECONDS',.02)
    async def run():
        state={};closed=[]
        async def capture(client):await asyncio.Event().wait()
        async def close():closed.append(True)
        client=ControlledClient(SimpleNamespace(exit=close),state,capture=capture)
        await client.exit()
        assert 'observation' not in state and state['capture_error_type']=='TimeoutError'
        assert state['browser_closed'] and closed==[True]
    asyncio.run(run())


@pytest.mark.parametrize("boundary,valid", [(None, False), ("episode_deadline", True), ("decision_limit", True)])
def test_aborted_validity_requires_known_control_boundary_without_changing_canonical_zero(tmp_path, boundary, valid):
    async def run():
        policy, budget, _ = make_policy(tmp_path)
        policy.boundary_reason = boundary
        async def forbidden(**kwargs): raise AssertionError("ABORTED never invokes the canonical judge")
        binding, _ = canonical_binding(budget, lambda *args: None, forbidden)
        result = await judge_terminal(args=judge_args(), samples=CanonicalSample(CanonicalSample.Status.ABORTED),
            final_state=dict(observation=observation(3), browser_closed=True), policy=policy,
            canonical_reward=binding, journal=lambda *args: None)
        assert result["valid"] is valid and result["score"] == 0. and not budget.reservations
    asyncio.run(run())


@pytest.mark.parametrize("condition,provider", [("L10", "jev"), ("L11", "kev")])
@pytest.mark.parametrize("fresh_capture", [True, False])
def test_native_choice_guard_ends_only_episode_without_action_fallback_or_judge(tmp_path, condition, provider, fresh_capture):
    async def run():
        calls = []
        response = selector_response(provider, 1)
        response['answers']['selection']['probabilities'] = {'1': .29, '2': .30, '3': .20, '4': .09, '5': .12}
        async def select(actual, request):
            calls.append(actual)
            return response
        policy, budget, journal = make_policy(tmp_path, condition, select)
        halts = []
        budget.halt = lambda *args, **kwargs: halts.append((args, kwargs))
        async def infer(*args, **kwargs): return ('original candidate', [1], [-.1], 'stop')
        with pytest.raises(EpisodeBoundary, match='selector_choice_rejected'):
            await policy(**call_kwargs(infer))
        assert calls == [provider] and policy.decisions == 1
        assert not halts and not policy.provider_invalid and not policy.contract_invalid
        assert not policy.input_invalid and not policy.budget_invalid
        assert [kind for kind, _, _ in budget.reservations] == ['local_sft_generations',
            'jev_selector_requests' if provider == 'jev' else 'local_kev_requests']
        rejected = [row for event, row in journal if event == 'selector_choice_rejected']
        assert len(rejected) == 1 and rejected[0]['executed'] is False
        assert rejected[0]['response'] == response
        async def forbidden(**kwargs): raise AssertionError('A rejected native choice must not invoke a judge')
        binding, _ = canonical_binding(budget, lambda *args: None, forbidden)
        sample = CanonicalSample(CanonicalSample.Status.ABORTED)
        final_state = dict(browser_closed=True)
        if fresh_capture: final_state['observation'] = observation(3)
        outcome = await judge_terminal(args=judge_args(), samples=sample, final_state=final_state,
            policy=policy, canonical_reward=binding, journal=lambda *args: None)
        assert outcome['valid'] is fresh_capture and outcome['score'] == 0.
        assert outcome['judge_http_attempts'] == 0 and sample.status == CanonicalSample.Status.ABORTED
        if fresh_capture:
            assert sample.metadata['controlled_terminal']['policy_boundary_reason'] == 'selector_choice_rejected'
    asyncio.run(run())


def test_actual_generator_step_handler_stops_at_action_limit_before_another_prompt_encoding(tmp_path):
    async def run():
        source = Path(__file__).parents[1] / "openwebrl/generate_browser.py"
        tree = ast.parse(source.read_text())
        generate = next(node for node in tree.body if isinstance(node, ast.AsyncFunctionDef)
                        and node.name == "_generate_turn_sample_impl")
        # Execute the real env.step exception handler, with its real loop break,
        # without importing CUDA/training dependencies or launching a browser.
        step_try = next(node for node in ast.walk(generate) if isinstance(node, ast.Try)
                        and any(isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                                and isinstance(call.func.value, ast.Name) and call.func.value.id == "env"
                                and call.func.attr == "step" for stmt in node.body
                                if isinstance(stmt, ast.Assign) for call in ast.walk(stmt)))
        skeleton = ast.parse('async def exercise():\n for step in range(1):\n  pass\n  encode_next()\n return turn_samples\n')
        skeleton.body[0].body[0].body[0] = deepcopy(step_try)
        policy, budget, _ = make_policy(tmp_path)
        async def step(actions): return observation(30, "action_limit", 2), 0., False, True, {}
        async def close(): pass
        async def capture(client): return observation(30, "action_limit", 3)
        state = {}
        client = ControlledClient(SimpleNamespace(step=step, exit=close), state, capture=capture)
        state["client"] = client
        sample = CanonicalSample()
        def encode_next(): raise AssertionError("Would overflow context; must never be encoded after action30")
        namespace = dict(env=client, actions=[{"name": "wait", "args": {}}], serial_variant=None,
            turn_samples=[sample], Sample=CanonicalSample, task_id="fixture", encode_next=encode_next,
            logger=SimpleNamespace(warning=lambda *args, **kwargs: None))
        exec(compile(ast.fix_missing_locations(skeleton), str(source), "exec"), namespace)
        await namespace["exercise"]()
        assert sample.status == CanonicalSample.Status.ABORTED and sample.metadata["is_last_turn"]
        async def forbidden(**kwargs): raise AssertionError("No selector, actor or judge call after operation30")
        binding, _ = canonical_binding(budget, lambda *args: None, forbidden)
        result = await judge_terminal(args=judge_args(), samples=sample, final_state=state, policy=policy,
            canonical_reward=binding, journal=lambda *args: None)
        assert result["valid"] and result["score"] == 0. and not budget.reservations
        assert not policy.input_invalid and policy.decisions == 0
    asyncio.run(run())


def test_factory_wrapper_restores_original_even_on_failure():
    async def create(*args, **kwargs): return SimpleNamespace(), {"task": "fixture"}
    module = SimpleNamespace(_create_env=create)
    with pytest.raises(RuntimeError):
        with controlled_environment_factory(module, {}, budget=Budget(), condition="L01"):
            assert module._create_env is not create
            raise RuntimeError("synthetic")
    assert module._create_env is create


def test_failed_local_reset_keeps_owned_handle_and_settles_closed_browser(monkeypatch):
    from openwebrl.env import local_process_env
    async def run():
        events, settlements = [], []
        async def close(): events.append('closed')
        async def initialize(**kwargs): raise RuntimeError('Synthetic reset failed')
        async def native_create(config):
            return SimpleNamespace(initialize=initialize,exit=close)
        async def original(task_id,config):
            client = await local_process_env.create_local_process_env(config['local_process'])
            try:
                await client.initialize(task_id=task_id)
            except BaseException:
                await client.exit()
                raise
        async def unavailable_capture(client): raise RuntimeError('No reset observation exists')
        monkeypatch.setattr(local_process_env,'create_local_process_env',native_create)
        generation = SimpleNamespace(_create_env=original)
        budget = Budget()
        budget.settle = lambda reservation,usage: settlements.append(usage)
        state = {}
        config = dict(mode='local_process',width=1280,height=720,dpr=1,
            use_screenshot=True,use_a11ytree=False,resize_output_coords=True,resize_scale=1000,
            local_process={'server_module':'openwebrl.controlled_browser_server'})
        with controlled_environment_factory(generation,state,budget=budget,condition='L01',capture=unavailable_capture):
            with pytest.raises(RuntimeError,match='Synthetic reset failed'):
                await generation._create_env('fixture',config)
        assert state['browser_closed'] is True and state['capture_error_type'] == 'RuntimeError'
        assert 'observation' not in state and isinstance(state['client'],ControlledClient)
        assert settlements == [{'closed':True}] and events == ['closed']
        assert local_process_env.create_local_process_env is native_create
        assert generation._create_env is original
    asyncio.run(run())


def test_budget_halt_never_dispatches_or_becomes_a_valid_terminal_positive(tmp_path):
    async def run():
        policy, budget, journal = make_policy(tmp_path)
        budget.blocked = True
        async def infer(*args, **kwargs): raise AssertionError("No remaining authorization")
        with pytest.raises(RuntimeError): await policy(**call_kwargs(infer))
        assert policy.budget_invalid and any(event == "budget_halt" for event, _ in journal)
        final = dict(observation=observation(3), browser_closed=True)
        async def forbidden(**request): raise AssertionError("No judge after budget halt")
        binding, _ = canonical_binding(budget, lambda *args: None, forbidden)
        result = await judge_terminal(args=judge_args(), samples=CanonicalSample(), final_state=final,
            policy=policy, canonical_reward=binding, journal=lambda *args: None)
        assert not result["valid"] and result["score"] is None
    asyncio.run(run())


def test_private_journal_appends_across_reopen_and_refuses_blob_overwrite(tmp_path):
    journal = AttemptJournal(tmp_path / "receipts")
    first = journal("reserved", {"count": 5})
    journal.blob("initial.image", b"image")
    later = AttemptJournal(tmp_path / "receipts")
    second = later("failed", {"error_type": "fixture"})
    assert first != second
    assert json.loads(Path(first).read_text())["event"] == "reserved"
    with pytest.raises(FileExistsError): later.blob("initial.image", b"replacement")
    assert (tmp_path / "receipts/initial.image").read_bytes() == b"image"


def test_real_actor_observation_serializer_excludes_selector_only_dom():
    # Exercise the actual repository method, without importing GPU dependencies
    # from the adapter's training base class or starting a browser.
    source = Path(__file__).parents[1] / "openwebrl/adapters/browser_adapter.py"
    tree = ast.parse(source.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "BrowserAdapter")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef)
                  and node.name == "_build_observation_message")
    namespace = dict(base64=base64, json=json, Any=Any)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])),
                 str(source), "exec"), namespace)
    adapter = SimpleNamespace(config=SimpleNamespace(use_a11ytree=False, use_screenshot=True))
    observed = observation()
    observed["all_tab_url"] = observed["selection_page"]["tabs"]
    observed["selection_page"]["text"] = "SELECTOR_ONLY_DOM_SENTINEL"
    observed["a11ytree"] = [{"textContent": "SELECTOR_ONLY_ELEMENT_SENTINEL"}]
    observed["controlled"]["private_marker"] = "SELECTOR_ONLY_RECEIPT_SENTINEL"
    message = namespace["_build_observation_message"](adapter, observed)
    serialized = json.dumps(message)
    assert "SELECTOR_ONLY" not in serialized
    assert "screen size: 1280 x 720" in message["content"][0]["text"]
    assert "https://example.com" in message["content"][0]["text"]
    encoded = message["content"][1]["image_url"].split(",", 1)[1]
    assert base64.b64decode(encoded) == observed["screenshot"]


def test_actor_dom_setting_is_rejected_before_browser_creation_or_reservation():
    async def run():
        calls = []
        async def create(*args, **kwargs): calls.append(args); raise AssertionError("Do not launch")
        generation = SimpleNamespace(_create_env=create)
        budget = Budget()
        config = dict(mode="local_process", width=1280, height=720, dpr=1,
            use_screenshot=True, use_a11ytree=False, resize_output_coords=True, resize_scale=1000,
            local_process={"server_module": "openwebrl.controlled_browser_server"})
        validate_actor_environment_config(config)
        config["use_a11ytree"] = True
        with controlled_environment_factory(generation, {}, budget=budget, condition="L01"):
            with pytest.raises(SelectorContractError): await generation._create_env("fixture", config)
        assert not calls and not budget.reservations
    asyncio.run(run())
