"""Offline Jev adapter tests; paid endpoints are never contacted."""
import ast
import json
import gzip
import os
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from openwebrl import jev_eval as jev


def config():
    return dict(text_provider="openai", jev_model="jev-1.13.0", max_steps=30,
                max_decisions=60, browser="local", task_timeout_seconds=600,
                judge_base_url="https://api.openai.com/v1")


def test_task_loading_strips_hidden_reference_and_rejects_duplicate(tmp_path):
    source = tmp_path / "tasks.jsonl"
    source.write_text(json.dumps(dict(task_id="x", task_name="Find a book", website="https://example.org",
        definite_answer="secret answer", metadata=dict(evaluator_reference=["hidden rubric"]))) + "\n")
    rows = jev.load_tasks(source)
    assert rows == [dict(task_id="x", intent="Find a book", start_url="https://example.org")]
    source.write_text(source.read_text() * 2)
    with pytest.raises(ValueError, match="duplicate"):
        jev.load_tasks(source)


def test_judge_uses_canonical_prompt_and_never_treats_done_as_success():
    task = dict(intent="Find a book", evaluator_reference="hidden rubric")
    messages = jev.judge_messages(task, [dict(action="Type query", text="book")], "ABC", "done")
    assert messages[0]["content"] == jev.judge_protocol()["SYSTEM_PROMPT"]
    assert "hidden rubric" not in json.dumps(messages)
    assert "not a success verdict" in json.dumps(messages)
    assert messages[1]["content"][1]["image_url"]["url"] == "data:image/jpeg;base64,ABC"
    assert jev.parse_verdict("DONE") is None
    assert jev.parse_verdict('Thoughts: failed\nStatus: "failure"') == 0


def test_parser_matches_existing_agenttrek_implementation():
    tree = ast.parse(jev.JUDGE_SOURCE.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_parse_verdict")
    scope = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "canonical", "exec"), scope)
    for text in ('Status: "success"', 'Status: "failure"', 'unknown', 'Status: unsuccessful'):
        assert jev.parse_verdict(text) == scope["_parse_verdict"](text)


def test_transport_audits_retries_and_removes_provider_specific_reasoning(tmp_path, monkeypatch):
    requests = []
    def respond(request):
        requests.append(json.loads(request.content))
        return httpx.Response(503 if len(requests) == 1 else 200, json={"choices": []})
    client = httpx.Client(transport=httpx.MockTransport(respond))
    transport = jev.ModelTransport(tmp_path, config(), client)
    monkeypatch.setattr(jev.time, "sleep", lambda _: None)
    transport.post("https://api.openai.com/v1/chat/completions", "private-key",
                   dict(model="gpt-4.1-mini", reasoning={"enabled": False}))
    assert transport.counts["text"] == 2
    assert requests[0] == dict(model="gpt-4.1-mini", temperature=0.6, top_p=0.95)
    for path in tmp_path.iterdir():
        assert "private-key" not in path.read_text()
    assert len((tmp_path / "api-attempts.jsonl").read_text().splitlines()) == 2


def test_wrong_model_and_provider_credit_exhaustion_fail_closed(tmp_path):
    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(
        200, json=dict(model="unexpected-model", answers={}))))
    transport = jev.ModelTransport(tmp_path, config(), client)
    with pytest.raises(jev.ProviderError, match="model_identity_mismatch"):
        transport.post("https://api.typesafe.ai/v1/systemone", "key", dict(model="jev-1.13.0"))
    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(402)))
    transport = jev.ModelTransport(tmp_path, config(), client)
    with pytest.raises(jev.ProviderError, match="402"):
        transport.post("https://api.typesafe.ai/v1/systemone", "key", dict(model="jev-1.13.0"))
    assert transport.counts["jev"] == 1


def test_summary_keeps_missing_and_invalid_in_overall_denominator(tmp_path):
    tasks = [dict(task_id=str(i)) for i in range(3)]
    for task, valid in zip(tasks, (True, False)):
        root = tmp_path / "tasks" / jev.digest(task["task_id"])
        jev.write_json(root / "trajectory.json", {})
        (root / "final.jpg").write_bytes(b"fixture")
        jev.write_json(root / "result.json", dict(task_id=task["task_id"], valid=valid,
            score=1 if valid else None, judge_text="Status: success" if valid else "",
            provider_blocked=False, api_attempts={}, actor_seconds=1))
    summary = jev.summarize(tasks, tmp_path)
    assert summary["overall_success_rate"] == 1 / 3
    assert summary["valid_only_success_rate"] == 1
    assert summary["missing_task_ids"] == ["2"]
    assert not summary["verified_complete"]


def test_provider_blocked_is_never_reported_as_completed_cohort(tmp_path):
    root = tmp_path / "tasks" / jev.digest("x")
    jev.write_json(root / "trajectory.json", {})
    jev.write_json(root / "result.json", dict(task_id="x", valid=False, score=None,
        judge_text="", provider_blocked=True, api_attempts={}, actor_seconds=1))
    summary = jev.summarize([dict(task_id="x")], tmp_path)
    assert summary["all_task_attempts_saved"]
    assert not summary["verified_complete"]


def test_transport_does_not_replay_ambiguous_billed_request(tmp_path):
    def respond(request):
        raise httpx.ReadTimeout("May already have been billed")
    transport = jev.ModelTransport(tmp_path, config(), httpx.Client(transport=httpx.MockTransport(respond)))
    with pytest.raises(jev.ProviderError, match="transport"):
        transport.post("https://api.typesafe.ai/v1/systemone", "key", dict(model="jev-1.13.0"))
    assert transport.counts["jev"] == 1


def test_model_http_attempt_budget_persists_retry_accounting(tmp_path):
    calls = []
    client = httpx.Client(transport=httpx.MockTransport(lambda r: calls.append(r)))
    transport = jev.ModelTransport(tmp_path, config(), client)
    transport.counts["jev"] = 180
    with pytest.raises(RuntimeError, match="budget exhausted"):
        transport.post("https://api.typesafe.ai/v1/systemone", "key", dict(model="jev-1.13.0"))
    assert not calls


def test_browser_create_not_retried_and_cleanup_tracks_only_owned_session(tmp_path, monkeypatch):
    requests = []
    def respond(request):
        requests.append((request.method, str(request.url)))
        return httpx.Response(200, json={})
    session = jev.BrowserSession(tmp_path, config())
    session.client = httpx.Client(transport=httpx.MockTransport(respond))
    session.remote_id = "owned-session"
    monkeypatch.setenv("BROWSER_USE_API_KEY", "private-key")
    assert session.close() == []
    assert requests == [("PATCH", jev.BROWSER_API + "/owned-session")]
    assert json.loads((tmp_path / "browser-session.json").read_text())["stopped"]


def test_adapter_restores_globals_without_browser_harness_daemon(monkeypatch, tmp_path):
    monkeypatch.setenv("BROWSER_HARNESS_HOME", str(tmp_path / "harness"))
    import jev_ultrafast.agent as upstream
    import jev_ultrafast.browser as browser
    import jev_ultrafast.model as model
    originals = upstream.Browser, upstream.MAX_STEPS, browser.cdp, model.post_json
    with jev.adapt_agent(SimpleNamespace(), SimpleNamespace(post=lambda *_: None), config()):
        assert upstream.MAX_STEPS == 30
        assert upstream.Browser != originals[0]
    assert (upstream.Browser, upstream.MAX_STEPS, browser.cdp, model.post_json) == originals


@pytest.mark.skipif(os.environ.get("JEV_LOCAL_BROWSER_TEST") != "1", reason="Opt-in local Chromium fixture")
def test_real_browser_fill_click_done_and_durable_judge_evidence(tmp_path, monkeypatch):
    """Full upstream loop with actual DOM mutations; all model requests are mocked."""
    fixture = tmp_path / "fixture.html"
    fixture.write_text('''<!doctype html><title>Offline search fixture</title>
      <label>Search query<input id="q"></label>
      <button onclick="document.getElementById('result').textContent='Results for '+document.getElementById('q').value">Search</button>
      <p id="result">Ready</p>''')
    calls = []
    def respond(request):
        body = json.loads(request.content)
        calls.append(body)
        if body["model"] == "jev-1.13.0":
            state, questions = body["state"], body["questions"]
            if "Results for algebra" in state["page"]["text"]:
                operation = "DONE"
            elif any(e.get("value") == "algebra" for e in state["elements"]):
                operation = "CLICK"
            else:
                operation = "TYPE_TEXT"
            answers = {}
            for name, question in questions.items():
                choices = question["criteria"]
                choice = operation if name == "operation" else next(iter(choices))
                if name == "click_target":
                    choice = next(k for k, v in choices.items() if "Search" in v["element"] and v.get("role") == "button")
                answers[name] = dict(choice=choice, confidence=1,
                                     probabilities={k: float(k == choice) for k in choices})
            return httpx.Response(200, json=dict(model="jev-1.13.0", answers=answers, usage={"input_tokens": 50}))
        if body["model"] == "o4-mini":
            text = 'Thoughts: Search results visible.\nStatus: "success"'
        else:
            text = '{"text":"algebra"}'
        return httpx.Response(200, json=dict(model=body["model"], choices=[dict(message=dict(content=text))], usage={}))
    original = jev.ModelTransport
    monkeypatch.setattr(jev, "ModelTransport", lambda root, conf: original(root, conf,
        httpx.Client(transport=httpx.MockTransport(respond))))
    monkeypatch.setenv("TEXT_MODEL_API_KEY", "fixture")
    monkeypatch.setenv("TYPESAFE_API_KEY", "fixture")
    monkeypatch.setenv("JUDGE_API_KEY", "fixture")
    monkeypatch.setenv("TYPESAFE_MODEL", "jev-1.13.0")
    monkeypatch.setenv("TEXT_MODEL", "gpt-4.1-mini-2025-04-14")
    monkeypatch.setenv("TEXT_MODEL_BASE_URL", "https://api.openai.com/v1")
    root = tmp_path / "attempt"
    result = jev.run_task(dict(task_id="fixture", intent="Search for algebra", start_url=fixture.as_uri()), root, config())
    assert result["valid"], result
    assert result["score"] == 1 and result["terminal"] == "done"
    assert result["actions"] == 2 and result["decisions"] == 3
    assert result["api_attempts"] == dict(jev=3, text=1, judge=1)
    with gzip.open(root / "state-0061.json.gz", "rt") as handle:
        state = json.load(handle)
    assert "Results for algebra" in state["page"]["text"]
    assert len((root / "final.jpg").read_bytes()) > 100
    assert (root / "judge-response.json").exists()
    for body in calls:
        if body["model"] == "jev-1.13.0":
            assert "screenshot" not in json.dumps(body)
