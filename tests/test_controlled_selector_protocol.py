"""Offline contract tests: no browser, model, API client or GPU is imported."""
from copy import deepcopy
import hashlib
import json

import pytest

from openwebrl.controlled_selector_protocol import (
    BASE_SEED, CANDIDATES, MODELS, PAGE_TEXT_LIMIT, POLICY, STATE_BYTE_LIMIT,
    SelectorContractError, SelectorInputBudgetExceeded, baseline_output,
    build_selection_batch, candidate_seed, selector_request, selector_result,
)


def observation(text="Search for an item"):
    return dict(active_tab_url="https://example.com/", screenshot=b"never-send-image",
        private_expected_answer="never-send-reference", selection_page=dict(
            url="https://example.com/", title="Example", text=text,
            interactive_elements=[dict(tag="input", role="textbox", ariaLabel="Search",
                                       bbox=[.1, .2, .3, .4])],
            tabs=[dict(url="https://example.com/", title="Example", index=0, active=True)],
            screen_size=[1280, 720]))


def outputs():
    # Whitespace, UTF-8, XML and exact normalized coordinates must survive.
    return [(f"  reason café {i}\n</think>\n<tool_call>"
             f'{{"name":"click","arguments":{{"point_2d":[{i},999]}}}}'
             "</tool_call>\n", [10 + i], [-.1], "stop") for i in range(CANDIDATES)]


def batch(**changes):
    kwargs = dict(task_id="fixture", task="Find café", turn=2, observation=observation(),
                  executed_actions=[{"tool": "wait", "id": i} for i in range(9)],
                  actor_outputs=outputs())
    kwargs.update(changes)
    return build_selection_batch(**kwargs)


def luna_reply(index=1, *, text=None):
    return dict(model=MODELS["luna"], status="completed", output=[
        {"type": "reasoning", "opaque_fixture": "not-interpreted"},
        dict(type="message", role="assistant", content=[dict(type="output_text",
            text=json.dumps({"selected_index": index}) if text is None else text)])])


def decision_reply(provider="jev", choice="1", probs=None):
    return dict(model=MODELS[provider], truncated=False, answers={"selection": dict(
        type="choice", choice=choice,
        probabilities=probs or {str(i): float(str(i) == choice) for i in range(1, 6)})})


def test_same_full_candidates_and_observation_for_all_selectors():
    original_outputs = outputs()
    b = batch(actor_outputs=original_outputs)
    luna = selector_request(b, "luna")
    jev = selector_request(b, "jev")
    kev = selector_request(b, "kev")
    assert json.loads(luna["input"][1]["content"]) == jev["state"] == kev["state"]
    assert jev["questions"] == kev["questions"]
    assert luna["input"][0]["content"] == jev["questions"]["selection"]["instructions"] == POLICY
    state = jev["state"]
    assert state["recent_executed_actions"] == [{"tool": "wait", "id": i} for i in range(4, 9)]
    for display, original in enumerate(b.displayed_order, 1):
        assert state["candidates"][str(display)]["text"].encode() == original_outputs[original][0].encode()
    serialized = json.dumps([luna, jev, kev])
    assert "never-send-image" not in serialized and "never-send-reference" not in serialized
    assert state["page"]["interactive_elements"] == observation()["selection_page"]["interactive_elements"]


def test_one_frozen_state_survives_mutation_of_input_and_other_provider_request():
    obs = observation()
    history = [{"action": "click"}]
    b = batch(observation=obs, executed_actions=history)
    saved = b.state_json
    obs["selection_page"]["interactive_elements"].clear()
    history[0]["action"] = "mutated"
    request = selector_request(b, "jev")
    request["state"]["page"]["text"] = "mutated"
    assert b.state_json == saved
    assert selector_request(b, "kev")["state"] == json.loads(saved)


def test_stateless_luna_contract_does_not_inherit_actor_tools_images_or_history():
    b = batch()
    request = selector_request(b, "luna")
    assert set(request) == {"model", "reasoning", "max_output_tokens", "store", "service_tier", "input", "text"}
    assert request["reasoning"] == {"effort": "medium"}
    assert request["max_output_tokens"] == 4096 and request["store"] is False
    assert request["service_tier"] == "default"
    assert len(request["input"]) == 2
    assert all(isinstance(m["content"], str) for m in request["input"])
    fmt = request["text"]["format"]
    assert fmt["strict"] is True
    assert fmt["schema"] == dict(type="object", properties={"selected_index": dict(
        type="integer", enum=[1, 2, 3, 4, 5])}, required=["selected_index"], additionalProperties=False)
    selector_result(b, "luna", luna_reply())
    assert selector_request(b, "luna") == request  # No returned state carries over.


def test_deterministic_seeds_and_shared_candidate_zero_schedule():
    expected = int.from_bytes(hashlib.sha256(f"{BASE_SEED}:fixture:2:0".encode()).digest()[:4], "big") % (2**31 - 1)
    b = batch()
    assert candidate_seed("fixture", 2, 0) == expected == b.candidate_seeds[0]
    assert b.displayed_order == batch().displayed_order
    assert b.state_sha256 == batch().state_sha256
    assert sorted(b.displayed_order) == list(range(5))
    assert len(set(b.candidate_seeds)) == 5
    assert batch(turn=3).candidate_seeds != b.candidate_seeds


@pytest.mark.parametrize("provider", ["luna", "jev", "kev"])
@pytest.mark.parametrize("displayed", [1, 5])
def test_valid_choice_returns_original_inference_object_without_rewriting(provider, displayed):
    original_outputs = outputs()
    b = batch(actor_outputs=original_outputs)
    response = luna_reply(displayed) if provider == "luna" else decision_reply(provider, str(displayed))
    selected = selector_result(b, provider, response)
    expected = b.displayed_order[displayed - 1]
    assert selected.displayed_index == displayed
    assert selected.original_index == expected
    assert selected.output is original_outputs[expected]
    assert selected.state_sha256 == b.state_sha256


def test_baseline_never_generates_a_best_of_five_control():
    original = outputs()[0]
    assert baseline_output([original]) is original
    with pytest.raises(SelectorContractError):
        baseline_output(outputs())
    with pytest.raises(SelectorContractError):
        baseline_output([])


def test_unicode_page_prefix_and_pretruncated_observation_agree():
    obs = observation("🚀" * (PAGE_TEXT_LIMIT + 5))
    full = batch(observation=obs)
    state = full.state()["page"]
    assert state["text"] == "🚀" * PAGE_TEXT_LIMIT
    assert state["text_characters"] == PAGE_TEXT_LIMIT + 5 and state["text_truncated"] is True
    obs["selection_page"].update(text="🚀" * PAGE_TEXT_LIMIT, text_characters=PAGE_TEXT_LIMIT + 5)
    assert batch(observation=obs).state_json == full.state_json
    obs["selection_page"]["text"] = "🚀" * (PAGE_TEXT_LIMIT - 1)
    with pytest.raises(SelectorContractError):
        batch(observation=obs)


@pytest.mark.parametrize("mutation", [
    lambda o: o.update(active_tab_url="https://changed.example/"),
    lambda o: o["selection_page"].update(screen_size=[1120, 780]),
    lambda o: o["selection_page"].update(text_characters=-1),
    lambda o: o["selection_page"].update(interactive_elements=None),
    lambda o: o["selection_page"].update(tabs="not-a-list"),
    lambda o: o["selection_page"]["interactive_elements"][0].update(bbox=[float("nan")]),
])
def test_missing_stale_or_invalid_observation_fails_before_request(mutation):
    obs = observation()
    mutation(obs)
    with pytest.raises(SelectorContractError):
        batch(observation=obs)


def test_input_budget_counts_utf8_candidates_and_geometry_not_only_page_text():
    original = outputs()
    original[0] = ("é" * (STATE_BYTE_LIMIT // 2), [], [], "stop")
    with pytest.raises(SelectorInputBudgetExceeded):
        batch(actor_outputs=original)
    assert original[0][0] == "é" * (STATE_BYTE_LIMIT // 2)  # No clipping.
    obs = observation()
    obs["selection_page"]["interactive_elements"] = [{"textContent": "x" * STATE_BYTE_LIMIT}]
    with pytest.raises(SelectorInputBudgetExceeded):
        batch(observation=obs)


@pytest.mark.parametrize("text", [
    '{"selected_index":0}', '{"selected_index":6}', '{"selected_index":true}',
    '{"selected_index":1.0}', '{"selected_index":"1"}', '{"selected_index":null}',
    '{"selected_index":1,"selected_index":2}', '{"selection":1}',
    '{"selected_index":1,"reason":"extra"}', '```json\n{"selected_index":1}\n```',
    '{"selected_index":1} trailing', '[]',
])
def test_luna_strict_json_rejects_invalid_outputs_without_fallback(text):
    with pytest.raises(SelectorContractError):
        selector_result(batch(), "luna", luna_reply(text=text))


@pytest.mark.parametrize("mutation", [
    lambda r: r.update(model="wrong"),
    lambda r: r.update(status="incomplete"),
    lambda r: r.update(output=[]),
    lambda r: r["output"].append(dict(type="function_call", name="click")),
    lambda r: r["output"][1].update(content=[dict(type="refusal", refusal="no")]),
    lambda r: r["output"].append(deepcopy(r["output"][1])),
])
def test_native_luna_response_identity_completeness_and_shape_are_checked(mutation):
    response = luna_reply()
    mutation(response)
    with pytest.raises(SelectorContractError):
        selector_result(batch(), "luna", response)


@pytest.mark.parametrize("provider", ["jev", "kev"])
@pytest.mark.parametrize("mutation", [
    lambda r: r.update(model="wrong"),
    lambda r: r.update(truncated=True),
    lambda r: r.update(answers={}),
    lambda r: r["answers"]["selection"].update(choice="2"),
    lambda r: r["answers"]["selection"].update(choice=1),
    lambda r: r["answers"]["selection"]["probabilities"].update({"1": True}),
    lambda r: r["answers"]["selection"]["probabilities"].update({"1": float("nan")}),
    lambda r: r["answers"]["selection"]["probabilities"].update({"1": .9}),
])
def test_classifier_rejects_wrong_identity_truncation_and_inconsistent_choice(provider, mutation):
    response = decision_reply(provider)
    mutation(response)
    with pytest.raises(SelectorContractError):
        selector_result(batch(), provider, response)


def test_jev_rounding_and_valid_argmax_ties_are_explicit():
    probs = {"1": .21, "2": .2, "3": .2, "4": .2, "5": .2}
    assert selector_result(batch(), "jev", decision_reply("jev", "1", probs)).displayed_index == 1
    with pytest.raises(SelectorContractError):
        selector_result(batch(), "kev", decision_reply("kev", "1", probs))
    assert selector_result(batch(), "kev", decision_reply("kev", "4", dict.fromkeys(probs, .2))).displayed_index == 4


def test_mutated_selected_candidate_is_never_executed():
    original = [list(output) for output in outputs()]
    b = batch(actor_outputs=original)
    original[b.displayed_order[0]][0] = "different action"
    with pytest.raises(SelectorContractError):
        selector_result(b, "luna", luna_reply(1))


def test_unknown_selector_and_nonfive_batch_are_rejected():
    with pytest.raises(SelectorContractError):
        selector_request(batch(), "other")
    with pytest.raises(SelectorContractError):
        selector_result(batch(), "other", {})
    with pytest.raises(SelectorContractError):
        batch(actor_outputs=outputs()[:4])
