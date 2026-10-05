"""Private provider-error evidence; no network or browser calls."""
import json

import httpx
import pytest

from openwebrl import jev_eval as jev


def transport(tmp_path, respond, provider="jev"):
    config = dict(decision_provider=provider, max_decisions=60, jev_model="kev-latest")
    return jev.ModelTransport(tmp_path, config, httpx.Client(transport=httpx.MockTransport(respond)))


@pytest.mark.parametrize("status", [400, 422])
@pytest.mark.parametrize("provider", ["jev", "kev", "text", "judge"])
def test_error_body_redacted_persisted_and_nonretryable(tmp_path, monkeypatch, status, provider):
    key = "sk-private-credential-abcdef"
    url = "https://provider.invalid/private-capability"
    body = dict(model="kev-latest", messages=[dict(role="user", content="preserve this request")])
    error = json.dumps({"detail": "schema failure", "echo": key})
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(status, text=error, headers={"X-Private-Header": "header-secret"})

    client = transport(tmp_path, respond, "kev" if provider == "kev" else "jev")
    monkeypatch.setattr(jev.time, "sleep", lambda _: pytest.fail("400/422 must not retry"))
    with pytest.raises(jev.ProviderError, match=str(status)):
        client.request(provider, url, key, body, attempts=3, timeout=25)
    assert len(requests) == 1 and client.counts[provider] == 1
    assert json.loads(requests[0].content) == body
    assert requests[0].headers["Authorization"] == "Bearer " + key
    records = [json.loads(line) for line in (tmp_path / "api-responses.jsonl").read_text().splitlines()]
    assert len(records) == 1
    record = records[0]
    assert record["http_status"] == status
    assert record["error_body"] == error.replace(key, "[REDACTED]")
    assert record["error_body_truncated"] is False
    assert "headers" not in record and "url" not in record
    for path in tmp_path.iterdir():
        saved = path.read_text()
        assert key not in saved and url not in saved and "header-secret" not in saved


@pytest.mark.parametrize("status", [400, 422])
def test_key_redacted_before_4096_character_boundary(tmp_path, status):
    key = "sk-private-credential-straddling-boundary"
    error = "x" * 4090 + key + "tail"
    client = transport(tmp_path, lambda _: httpx.Response(status, text=error))
    with pytest.raises(jev.ProviderError, match=str(status)):
        client.request("jev", "https://provider.invalid", key, dict(model="kev-latest"),
                       attempts=3, timeout=25)
    record = json.loads((tmp_path / "api-responses.jsonl").read_text())
    assert record["error_body"] == ("x" * 4090 + "[REDACTED]tail")[:4096]
    assert len(record["error_body"]) == 4096
    assert "sk-pri" not in record["error_body"]
    assert record["error_body_truncated"] is True


def test_error_limit_is_characters_and_empty_key_does_not_expand_body(tmp_path):
    client = transport(tmp_path, lambda _: httpx.Response(422, text="\u00e9" * 4097))
    with pytest.raises(jev.ProviderError, match="422"):
        client.request("jev", "https://provider.invalid", "", dict(model="kev-latest"),
                       attempts=3, timeout=25)
    record = json.loads((tmp_path / "api-responses.jsonl").read_text())
    assert record["error_body"] == "\u00e9" * 4096
    assert record["error_body_truncated"] is True


def test_retryable_failure_body_keeps_existing_retry_policy(tmp_path, monkeypatch):
    calls = []
    sleeps = []

    def respond(request):
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(503, text="temporary unavailable: private-key")
        return httpx.Response(200, json=dict(model="kev-latest", answers={}))

    client = transport(tmp_path, respond)
    monkeypatch.setattr(jev.time, "sleep", sleeps.append)
    result = client.request("jev", "https://provider.invalid", "private-key", dict(model="kev-latest"),
                            attempts=3, timeout=25)
    assert result == dict(model="kev-latest", answers={})
    assert len(calls) == 2 and sleeps == [1]
    records = [json.loads(line) for line in (tmp_path / "api-responses.jsonl").read_text().splitlines()]
    assert records[0]["error_body"] == "temporary unavailable: [REDACTED]"
    assert records[0]["error_body_truncated"] is False
    assert "error_body" not in records[1] and "error_body_truncated" not in records[1]
