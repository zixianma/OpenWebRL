"""Pinned Kev decision-provider identity checks; no torch or training imports."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
from urllib.parse import urlsplit

UPSTREAM_COMMIT = "fe64b1274ea7f80d4095866df90666abb03e9cf6"
MODELS = {
    "0.8b": dict(repo="jaredpalmer/kev-0.8b", revision="bf75a6a8848ea6960ff2ed108d9ed44c2941174f",
                 base="Qwen/Qwen3.5-0.8B-Base", base_revision="dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68",
                 temperature=2.3510958125672174, weights="lora",
                 head_sha256="f400bd12802b2b105ae45d6b03774a158a3db4fccff42413734ddca2e5c920b6"),
    "27b": dict(repo="jaredpalmer/kev-27b", revision="af0e6d551bdc2cc724f3e9d7a8bee1cd4fb8f7bf",
                base="Qwen/Qwen3.8-27B", base_revision="1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
                temperature=1.319507910772894, weights="full",
                head_sha256="7968f17b03479c1ef9d1c0f3ab8a15e31ecb441cf40691b07ee945ab554d45ad"),
}


def model_spec(variant):
    spec = dict(MODELS[variant])
    snapshot = (Path("/gpfs/scrubbed/zixianma/openwebrl-runtime/hf-cache/hub") /
                ("models--" + spec["repo"].replace("/", "--")) / "snapshots" / spec["revision"])
    return dict(spec, variant=variant, upstream_commit=UPSTREAM_COMMIT,
                run=spec["repo"] + "@" + spec["revision"], server_run=str(snapshot))


def file_hash(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def local_endpoint(endpoint):
    parsed = urlsplit(endpoint)
    if (parsed.scheme != "http" or parsed.hostname != "127.0.0.1" or
            parsed.path != "/v1/systemone" or parsed.username or parsed.password or
            parsed.query or parsed.fragment or not parsed.port):
        raise ValueError("Kev must use an explicit localhost port and /v1/systemone")
    return endpoint


def validate_card(payload, spec):
    cards = [c for c in payload.get("models", []) if c.get("name") == "kev-latest"]
    if len(cards) != 1:
        raise ValueError("Missing or ambiguous Kev model card")
    card = cards[0]
    expected = dict(run=spec["server_run"], base=spec["base"], device="cuda", backend="torch",
                    dtype="bfloat16", truncate_states=False, max_state_tokens=65536)
    for key, value in expected.items():
        if card.get(key) != value:
            raise ValueError(f"Kev server identity/configuration mismatch: {key}")
    if not math.isclose(card.get("temperature", 0), spec["temperature"], rel_tol=1e-6):
        raise ValueError("Kev checkpoint calibration changed")
    if card.get("cuda_graphs") is None:
        raise ValueError("Expected the upstream CUDA graph serving path")
    return card


def check_server(endpoint, spec):
    import httpx
    endpoint = local_endpoint(endpoint)
    response = httpx.get(endpoint.removesuffix("systemone") + "models", timeout=10, trust_env=False)
    response.raise_for_status()
    return validate_card(response.json(), spec)


def validate_answers(request, response):
    if response.get("model") != "kev-latest" or response.get("truncated"):
        raise ValueError("Kev warmup used another model alias or truncated input")
    if set(response.get("answers", {})) != set(request["questions"]):
        raise ValueError("Kev warmup omitted questions")
    for name, question in request["questions"].items():
        answer = response["answers"][name]
        options = set(question["criteria"])
        probabilities = answer.get("probabilities", {})
        if (set(probabilities) != options or answer.get("choice") not in options or
                any(not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities.values()) or
                not math.isclose(sum(probabilities.values()), 1, abs_tol=1e-3)):
            raise ValueError("Kev warmup returned invalid choice probabilities")
