"""Offline checks for substituting Kev without changing the browser policy."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from openwebrl import jev_eval as jev, kev_eval as kev


def card(variant="0.8b"):
    spec = kev.model_spec(variant)
    return dict(models=[dict(name="kev-latest", run=spec["server_run"], base=spec["base"],
        device="cuda", backend="torch", dtype="bfloat16", truncate_states=False,
        max_state_tokens=65536, temperature=spec["temperature"], cuda_graphs={})])


@pytest.mark.parametrize("variant", ["0.8b", "27b"])
def test_identity_checks_checkpoint_beyond_response_alias(variant):
    assert kev.validate_card(card(variant), kev.model_spec(variant))["device"] == "cuda"
    bad = card(variant)
    bad["models"][0]["run"] = "jaredpalmer/kev-4b@v1.0"
    with pytest.raises(ValueError, match="run"):
        kev.validate_card(bad, kev.model_spec(variant))


@pytest.mark.parametrize("key,value", [("temperature", 1), ("truncate_states", True),
    ("device", "cpu"), ("dtype", "float32"), ("cuda_graphs", None)])
def test_serving_changes_fail_closed(key, value):
    payload = card()
    payload["models"][0][key] = value
    with pytest.raises(ValueError):
        kev.validate_card(payload, kev.model_spec("0.8b"))


@pytest.mark.parametrize("endpoint", ["https://remote/v1/systemone", "http://127.0.0.1:1/wrong",
    "http://secret@127.0.0.1:1/v1/systemone", "http://127.0.0.1:1/v1/systemone?token=secret"])
def test_local_endpoint_rejects_unintended_destination(endpoint):
    with pytest.raises(ValueError):
        kev.local_endpoint(endpoint)


def test_transport_reroutes_only_decisions_and_never_forwards_typesafe_secret(tmp_path):
    requests = []
    def respond(request):
        requests.append(request)
        return httpx.Response(200, json=dict(model=json.loads(request.content)["model"]))
    config = dict(decision_provider="kev", decision_endpoint="http://127.0.0.1:18761/v1/systemone",
                  jev_model="kev-latest", text_provider="openai", max_decisions=60)
    transport = jev.ModelTransport(tmp_path, config, httpx.Client(transport=httpx.MockTransport(respond)))
    transport.post("https://api.typesafe.ai/v1/systemone", "private-typesafe-key", dict(model="kev-latest"))
    transport.post("https://api.openai.com/v1/chat/completions", "private-openai-key", dict(model="gpt-4.1-mini"))
    assert str(requests[0].url) == config["decision_endpoint"]
    assert requests[0].headers["authorization"] == "Bearer local-kev"
    assert str(requests[1].url) == "https://api.openai.com/v1/chat/completions"
    assert transport.counts == dict(kev=1, text=1, judge=0)
    assert "private-" not in "".join(p.read_text() for p in tmp_path.iterdir())


def test_probability_validation_rejects_truncation_or_missing_question():
    request = dict(questions={"operation": dict(criteria={"CLICK": "click", "DONE": "stop"})})
    response = dict(model="kev-latest", answers={"operation": dict(choice="DONE", probabilities={"CLICK": .2, "DONE": .8})})
    kev.validate_answers(request, response)
    with pytest.raises(ValueError, match="truncated"):
        kev.validate_answers(request, dict(response, truncated=True))
    with pytest.raises(ValueError, match="omitted"):
        kev.validate_answers(request, dict(response, answers={}))


def test_pair_environment_removes_inherited_scientific_overrides(monkeypatch):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from evaluate_kev_pair import server_environment
    monkeypatch.setenv("KEV_TEMPERATURE", "9")
    monkeypatch.setenv("KEV_DATE_FACTS", "1")
    monkeypatch.setenv("KEV_TRUNCATE_STATES", "1")
    monkeypatch.setenv("WANDB_PROJECT", "openwebrl")
    env = server_environment()
    assert "KEV_TEMPERATURE" not in env
    assert env["KEV_DATE_FACTS"] == env["KEV_TRUNCATE_STATES"] == "0"
    assert env["WANDB_PROJECT"] == "openwebrl-evals"
    assert env["HF_HUB_OFFLINE"] == "1"


def test_pair_identity_failure_stops_owned_server_before_opening_browsers(tmp_path, monkeypatch):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import evaluate_kev_pair as pair
    monkeypatch.setenv("SLURM_JOB_ID", "offline-fixture")
    (tmp_path / "0.8b").mkdir()
    children = []
    class Process:
        def __init__(self, command, **kwargs):
            self.command, self.stopped = command, False
            children.append(self)
        def poll(self):
            return 0 if self.stopped else None
        def terminate(self):
            self.stopped = True
        def wait(self, **kwargs):
            return 0
    monkeypatch.setattr(pair.subprocess, "Popen", Process)
    def wrong_server(*_):
        raise ValueError("wrong checkpoint")
    monkeypatch.setattr(pair.kev, "check_server", wrong_server)
    args = SimpleNamespace(output=tmp_path, endpoint="http://127.0.0.1:0/v1/systemone")
    with pytest.raises(ValueError, match="wrong checkpoint"):
        pair.run(args, dict(controller_budget_seconds=6900))
    assert len(children) == 1 and children[0].stopped
    assert "kev.serve" in children[0].command
    ledger = json.loads((tmp_path / "pair-time-ledger.json").read_text())
    assert ledger["attempts"][0]["finished"]
    assert (tmp_path / "failure.json").exists()
    assert not (tmp_path / "summary.json").exists()


def test_pair_restart_does_not_restore_consumed_budget(tmp_path, monkeypatch):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import evaluate_kev_pair as pair
    monkeypatch.setenv("SLURM_JOB_ID", "offline-fixture")
    jev.write_json(tmp_path / "pair-time-ledger.json", dict(attempts=[dict(finished=True, elapsed_seconds=6600)]))
    with pytest.raises(ValueError, match="exhausted"):
        pair.run(SimpleNamespace(output=tmp_path), dict(controller_budget_seconds=6900))


def test_port_probe_allows_closed_server_connections_but_rejects_live_listener():
    import socket
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from evaluate_kev_pair import check_port_available
    with socket.socket() as listener, socket.socket() as client:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 0))
        address = listener.getsockname()
        endpoint = f"http://127.0.0.1:{address[1]}/v1/systemone"
        listener.listen()
        with pytest.raises(OSError):
            check_port_available(endpoint)
        client.connect(address)
        connection, _ = listener.accept()
        connection.shutdown(socket.SHUT_WR)
        assert client.recv(1) == b""
        client.shutdown(socket.SHUT_WR)
        assert connection.recv(1) == b""
        connection.close()
    check_port_available(endpoint)
