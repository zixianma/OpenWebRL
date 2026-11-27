"""CPU tests for the decision actor: options, projection, policy flow, System One parsing, crossbench registration."""
import asyncio
import json

import pytest

from openwebrl.decision_actor.deciders import Decision, ProjectedActorDecider, SystemOneDecider, DecisionState
from openwebrl.decision_actor.options import HarnessConfig, build_options, clean_elements
from openwebrl.decision_actor.policy import DecisionActorPolicy
from openwebrl.decision_actor.projection import parse_tool_call, project
from openwebrl.decision_actor.typing_model import ActorTypingModel, forced_prefix

ELEMENTS = [
    dict(id=0, tag="input", type="text", bbox=[0.30, 0.02, 0.70, 0.06], placeholder="Search recipes"),
    dict(id=1, tag="button", bbox=[0.71, 0.02, 0.75, 0.06], ariaLabel="Search"),
    dict(id=2, tag="a", bbox=[0.05, 0.40, 0.30, 0.60], innerText="Chicken Stir Fry"),
    dict(id=3, tag="a", bbox=[0.05, 0.40, 0.30, 0.601], innerText="Chicken Stir Fry"),   # near-duplicate
    dict(id=4, tag="div", bbox=[0.0, 0.0, 0.001, 0.5], innerText="sliver"),               # too thin
]
OBS = dict(a11ytree=ELEMENTS, active_tab_url="https://example.com/",
           all_tab_url=[dict(index=0, url="https://example.com/", title="Ex", active=True),
                        dict(index=1, url="https://other.com/", title="Other", active=False)],
           screenshot=b"png")


def call(text):
    return f'<think>x</think>\n<tool_call>\n{json.dumps(text)}\n</tool_call>'


def test_options_dedupe_order_and_kinds():
    cfg = HarnessConfig()
    assert len(clean_elements(ELEMENTS, cfg)) == 3
    opts = build_options(OBS, cfg)
    ids = [o.id for o in opts]
    assert ids[:4] == ["t0", "e0", "e1", "e2"]                      # typable field first gets a type option
    assert {"scroll_down", "go_back", "press_enter", "done", "tab1"} <= set(ids)
    t0 = next(o for o in opts if o.id == "t0")
    assert t0.needs_text == "write" and t0.action == dict(name="click", arguments=dict(point_2d=[500, 40]))
    assert 'Search recipes' in t0.label and "(0.50, 0.04)" in t0.label


def test_harness_knobs():
    cfg = HarnessConfig.from_dict(dict(type_options=False, page_actions=["done"], grid=[2, 2], tab_options=False,
                                       label_bbox=False, max_elements=1))
    ids = [o.id for o in build_options(OBS, cfg)]
    assert ids == ["e0", "g0_0", "g0_1", "g1_0", "g1_1", "done"]
    with pytest.raises(ValueError):
        HarnessConfig.from_dict(dict(no_such_knob=1))


def test_projection_cases():
    opts = build_options(OBS, HarnessConfig(grid=(0, 0)))
    p = lambda c, **k: project(parse_tool_call(call(c)), opts, **k)
    assert p(dict(name="click", arguments=dict(point_2d=[150, 500]))).option_id == "e2"
    assert p(dict(name="click", arguments=dict(point_2d=[500, 40]))).option_id == "e0"       # plain click, not type
    near = p(dict(name="click", arguments=dict(point_2d=[310, 500])))
    assert near.option_id == "e2" and near.reason == "near_element"
    assert p(dict(name="click", arguments=dict(point_2d=[900, 900]))).option_id is None
    assert p(dict(name="scroll", arguments=dict(direction="down", amount=0.3))).option_id == "scroll_down"
    assert p(dict(name="press_keys", arguments=dict(keys=["Enter"]))).option_id == "press_enter"
    assert p(dict(name="press_keys", arguments=dict(keys=["Control", "a"], is_hotkey=True))).option_id is None
    assert p(dict(name="write", arguments=dict(message="x")), pending_write=True).option_id == "__typed__"
    assert p(dict(name="goto_url", arguments=dict(url="https://x"))).reason == "unsupported:goto_url"
    assert project(None, opts).reason == "unparsable"


def test_forced_prefix():
    assert forced_prefix("...<|im_start|>assistant\n<think>\n", "write").endswith(
        '<think>\n\n</think>\n\n<tool_call>\n{"name": "write", "arguments": {"message": "')
    assert forced_prefix("...<|im_start|>assistant\n", "done").endswith('{"name": "done", "arguments": {"response": "')


class Tok:
    def encode(self, text, add_special_tokens=False):
        return list(range(len(text) // 10 + 1))


class ScriptedDecider:
    kind = "scripted"

    def __init__(self, picks):
        self.picks = list(picks)

    async def __call__(self, state, pending_write=False):
        return Decision(self.picks.pop(0), probabilities={o.id: 0.0 for o in state.options})


def run_turns(policy, n, infer):
    outs = []
    for turn in range(n):
        out, meta = asyncio.run(policy(infer=infer, url="u", input_text="prompt<think>\n", sampling_params={},
                                       images=[], observation=OBS, history=[], task="find stir fry",
                                       task_id="task-1", turn=turn, timeout=None))
        outs.append((parse_tool_call(out[0]), meta))
    return outs


def test_policy_type_then_write_then_done(tmp_path):
    async def infer(url, text, params, images, timeout_secs=None):
        if text.endswith('"message": "'):
            return ('chicken stir fry"}}\n', [], [], "stop")
        if text.endswith('"response": "'):
            return ('Found it."}}\n', [], [], "stop")
        raise AssertionError("decision actor must not call the actor except to type")
    policy = DecisionActorPolicy(dict(decider=dict(type="projected_actor"), harness={}), tmp_path,
                                 tokenizer=Tok(), decider=ScriptedDecider(["t0", "press_enter", "done"]),
                                 typing=ActorTypingModel())
    outs = run_turns(policy, 4, infer)
    assert outs[0][0] == dict(name="click", arguments=dict(point_2d=[500, 40]))
    assert outs[1][0] == dict(name="write", arguments=dict(message="chicken stir fry"))
    assert outs[1][1]["source"] == "typing_model"
    assert outs[2][0] == dict(name="press_keys", arguments=dict(keys=["Enter"]))
    assert outs[3][0] == dict(name="done", arguments=dict(response="Found it."))
    lines = [json.loads(l) for l in (tmp_path / "trace.jsonl").read_text().splitlines()]
    assert [l["turn"] for l in lines] == [0, 1, 2, 3] and all(l["kind"] == "decision_actor" for l in lines)
    assert [l["selected_index"] for l in lines] == ["type", "typing_model", "press", "done"]


def test_projected_actor_maps_resamples_and_passes_through(tmp_path):
    replies = iter([call(dict(name="goto_url", arguments=dict(url="https://x"))),         # unmapped -> resample
                    call(dict(name="click", arguments=dict(point_2d=[150, 500]))),        # maps to e2
                    call(dict(name="done", arguments=dict(response="answer")))])          # passthrough

    async def infer(url, text, params, images, timeout_secs=None):
        return (next(replies), [], [], "stop")
    policy = DecisionActorPolicy(dict(decider=dict(type="projected_actor", max_resamples=2), harness={}), tmp_path,
                                 tokenizer=Tok())
    outs = run_turns(policy, 2, infer)
    assert outs[0][0] == dict(name="click", arguments=dict(point_2d=[175, 500]))          # canonical centre of e2
    assert outs[1][0] == dict(name="done", arguments=dict(response="answer"))
    turn0 = json.loads(next((tmp_path / "turns").glob("*/turn-000.json")).read_text())
    assert [a["reason"] for a in turn0["decision_info"]["attempts"]] == ["unsupported:goto_url", "inside_element"]


def test_projected_actor_strict_wait():
    async def infer(url, text, params, images, timeout_secs=None):
        return (call(dict(name="goto_url", arguments=dict(url="https://x"))), [], [], "stop")
    state = DecisionState(task="t", url="", title="", recent_actions=[], options=build_options(OBS, HarnessConfig()),
                          infer=infer)
    decision = asyncio.run(ProjectedActorDecider(max_resamples=1, on_unmapped="wait")(state))
    assert decision.option_id == "wait" and not decision.info["mapped"]


class FakeReply:
    def __init__(self, body):
        self.body = body

    def raise_for_status(self):
        pass

    def json(self):
        return self.body


class FakeClient:
    def __init__(self):
        self.sent = None

    async def post(self, url, json=None, headers=None, timeout=None):
        self.sent = json
        ids = list(json["questions"]["action"]["criteria"])
        probs = {k: (0.9 if k == "e2" else 0.1 / (len(ids) - 1)) for k in ids}
        return FakeReply(dict(model=json["model"], answers=dict(action=dict(type="choice", choice="e2", probabilities=probs))))


def test_systemone_request_and_argmax():
    client = FakeClient()
    decider = SystemOneDecider(endpoint="http://x/v1/systemone", model="kev-latest", client=client)
    state = DecisionState(task="find stir fry", url="https://example.com/", title="", recent_actions=["click a"],
                          options=build_options(OBS, HarnessConfig()))
    decision = asyncio.run(decider(state))
    assert decision.option_id == "e2"
    assert client.sent["questions"]["action"]["type"] == "choice"
    assert client.sent["state"]["task"] == "find stir fry"


def test_crossbench_condition_registration(tmp_path, monkeypatch):
    from openwebrl import crossbench_selector_eval as core
    from openwebrl.decision_actor import crossbench
    cfg = tmp_path / "c.json"
    cfg.write_text(json.dumps(dict(name="projected-sft", decider=dict(type="projected_actor"), harness={})))
    crossbench.install()
    cond = core.parse_condition(f"decision:{cfg}")
    assert cond["kind"] == "decision_actor" and cond["slug"].startswith("decision-projected-sft-")
    assert core.parse_condition("sft1")["kind"] == "baseline"
    proto = core.protocol_for("online_mind2web", cond)
    assert proto["selector_kind"] == "decision_actor" and proto["decision_config_sha256"] == cond["decision_config_sha256"]
    assert core.run_collection.__kwdefaults__["selector_factory"] is crossbench.make_selector


def test_report_coverage_and_cost():
    import runpy
    from pathlib import Path
    report = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/decision_actor_report.py"), run_name="x")
    steps = [dict(n_options=10, n_elements=6, seconds=1.0, chosen=dict(kind="click"),
                  decision_info=dict(mapped=True, attempts=[dict(option_id="e1", reason="inside_element")])),
             dict(n_options=12, n_elements=8, seconds=2.0, source="actor_passthrough",
                  decision_info=dict(mapped=False, attempts=[dict(option_id=None, reason="unsupported:goto_url")] * 3)),
             dict(seconds=0.5, source="typing_model", typing=dict(seconds=0.4))]
    cov = report["coverage"](steps)
    assert cov["steps"] == 2 and cov["first_sample_mapped"] == 0.5 and cov["escape_rate"] == 0.5
    c = report["cost"](steps)
    assert c["options_per_step"] == 11 and c["typing_calls"] == 1 and c["browser_steps"] == 3
