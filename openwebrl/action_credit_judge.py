"""Canonical action-history judge with per-HTTP durable cost admission.

Importing or constructing this module sends nothing. The controller must bind
an actual API approval artifact and a frozen pricing/output-limit protocol.
CPU fixtures are not API authorization. Each provider retry reserves a new
ledger subrequest before dispatch; ambiguous charges retain their reservation.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import math
from pathlib import Path
from types import FunctionType, SimpleNamespace

from openwebrl.action_credit import _sha, digest
from openwebrl.action_credit_state import BudgetExceeded


class PaidDispatchBlocked(BaseException):
    """Must escape the canonical judge's retry-on-Exception loop."""


@dataclass(frozen=True)
class JudgeHTTPProtocol:
    model: str
    approval_receipt_sha256: str
    input_bound_protocol_sha256: str
    max_completion_tokens: int
    reserved_microdollars_per_http: int
    input_microdollars_per_million_tokens: int
    output_microdollars_per_million_tokens: int

    @property
    def sha256(self):
        _sha(self.approval_receipt_sha256, "approval_receipt_sha256")
        _sha(self.input_bound_protocol_sha256, "input_bound_protocol_sha256")
        if not self.model:
            raise ValueError("explicit judge model required")
        for key, value in asdict(self).items():
            if key not in ("model", "approval_receipt_sha256", "input_bound_protocol_sha256") and (type(value) is not int or value <= 0):
                raise ValueError("explicit positive pricing/output/reservation limit required")
        return digest(asdict(self))


def _cost(amount):
    return dict(api_requests=1, api_microdollars=amount, gpu_seconds=0, browser_dispatches=0)


class MeteredJudgeClient:
    """Minimal AsyncOpenAI-compatible chat.completions.create surface.

    ``provider_create`` must have SDK retries disabled (max_retries=0); hidden
    transport retries cannot be charged by this wrapper and are prohibited.
    Token-rate arithmetic is a conservative full-price accounting convention,
    not a claim to reproduce discounts on a provider invoice.
    """
    def __init__(self, *, store, parent_attempt_id, protocol, approval_receipt,
                 provider_create, provider_max_retries, input_token_upper_bound,
                 input_bound_protocol_sha256, journal):
        if provider_max_retries != 0:
            raise ValueError("provider transport must expose each HTTP attempt; disable SDK retries")
        if (digest(approval_receipt) != protocol.approval_receipt_sha256
                or approval_receipt.get("status") != "approved"
                or approval_receipt.get("campaign_id") != store.manifest.to_dict()["campaign_id"]
                or approval_receipt.get("model") != protocol.model
                or approval_receipt.get("api_microdollars_cap") != store.manifest.to_dict()["budget_caps"]["api_microdollars"]
                or approval_receipt.get("api_requests_cap") != store.manifest.to_dict()["budget_caps"]["api_requests"]):
            raise PaidDispatchBlocked("missing or mismatched exact campaign API approval artifact")
        science = store.manifest.to_dict()["science"]["judge"]
        if science.get("http_protocol_sha256") != protocol.sha256 or science["model"] != protocol.model:
            raise ValueError("HTTP model/pricing/output-limit protocol differs from campaign")
        if (input_bound_protocol_sha256 != protocol.input_bound_protocol_sha256
                or not callable(input_token_upper_bound)):
            raise ValueError("independently validated text/image input-cost bound is required")
        self.store, self.parent_attempt_id, self.protocol = store, parent_attempt_id, protocol
        self.provider_create, self.journal, self.sequence = provider_create, journal, 0
        self.input_token_upper_bound = input_token_upper_bound
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    async def create(self, **kwargs):
        if kwargs.get("model") != self.protocol.model:
            raise PaidDispatchBlocked("canonical judge requested another model")
        payload = deepcopy(kwargs)
        configured = payload.setdefault("max_completion_tokens", self.protocol.max_completion_tokens)
        if configured != self.protocol.max_completion_tokens or "max_tokens" in payload:
            raise PaidDispatchBlocked("judge output limit differs from frozen HTTP protocol")
        prompt_bound = self.input_token_upper_bound(payload)
        if type(prompt_bound) is not int or prompt_bound < 0:
            raise PaidDispatchBlocked("input-cost verifier did not provide a conservative integer token bound")
        maximum_cost = math.ceil((prompt_bound*self.protocol.input_microdollars_per_million_tokens
            + configured*self.protocol.output_microdollars_per_million_tokens)/1_000_000)
        if maximum_cost > self.protocol.reserved_microdollars_per_http:
            raise PaidDispatchBlocked("full prompt/image and completion cost bound exceeds HTTP reservation")
        request_id = digest((self.parent_attempt_id, "judge_http", self.sequence, payload))
        self.sequence += 1
        try:
            admission = self.store.reserve_subrequest(request_id, parent_attempt_id=self.parent_attempt_id,
                payload_sha256=digest(payload), reserved_cost=_cost(self.protocol.reserved_microdollars_per_http))
        except BudgetExceeded as exc:
            raise PaidDispatchBlocked(str(exc)) from exc
        if not admission["created"]:
            raise PaidDispatchBlocked("persisted judge HTTP request requires reconciliation, never redispatch")
        self.journal("judge_http_reserved", dict(request_id=request_id, payload=payload,
                                                protocol_sha256=self.protocol.sha256))
        try:
            response = await self.provider_create(**payload)
            raw = response.model_dump() if hasattr(response, "model_dump") else response
            if not isinstance(raw, dict):
                raise ValueError("provider response must preserve complete serialized payload")
            usage = raw.get("usage") or {}
            prompt, completion = usage.get("prompt_tokens"), usage.get("completion_tokens")
            if type(prompt) is not int or type(completion) is not int or min(prompt, completion) < 0:
                raise ValueError("provider omitted exact usage; charge remains reserved")
            amount = math.ceil((prompt*self.protocol.input_microdollars_per_million_tokens
                                + completion*self.protocol.output_microdollars_per_million_tokens)/1_000_000)
            self.journal("judge_http_response", dict(request_id=request_id, response=raw,
                                                    accounted_microdollars=amount))
            self.store.settle_subrequest(request_id, status="succeeded", actual_cost=_cost(amount),
                                         response_sha256=digest(raw))
            if prompt > prompt_bound or completion > configured:
                with self.store._transaction() as state:
                    state["halt"] = "Provider usage exceeded the frozen request-cost bound"
                raise PaidDispatchBlocked("actual usage invalidates request-cost proof; campaign halted")
            return response
        except BaseException as exc:
            current = self.store.snapshot()["subrequests"][request_id]
            if current["status"] == "reserved":
                self.store.settle_subrequest(request_id, status="unknown", actual_cost=_cost(0),
                                             reason=type(exc).__name__)
            self.journal("judge_http_error", dict(request_id=request_id, error_type=type(exc).__name__))
            if isinstance(exc, BudgetExceeded):
                raise PaidDispatchBlocked(str(exc)) from exc
            raise


async def canonical_action_history_judge(*, args, samples, client, journal, source_sha256, reward_module=None):
    """Run existing reward_browser functions in isolated globals, unmodified.

    Replaces only transport acquisition and private trace persistence. The
    explicit HTTP output cap is pinned by JudgeHTTPProtocol. Native Samples are
    copied: collector's provisional remove_sample flag is removed only after
    its infrastructure checks already passed. Original diagnostics stay intact.
    """
    if reward_module is None:
        from openwebrl import reward_browser as reward_module
    if hashlib.sha256(Path(reward_module.__file__).read_bytes()).hexdigest() != source_sha256:
        raise ValueError("canonical reward source differs from frozen protocol")
    if getattr(args, "judge_prompt_variant", None) != "action_history" or getattr(args, "judge_api_model", None) != client.protocol.model:
        raise ValueError("canonical judge model/variant differs from frozen protocol")
    copied = deepcopy(list(samples))
    if not copied:
        raise ValueError("terminal native trajectory required")
    for sample in copied:
        if (sample.metadata.get("action_credit_training_admission") != "pending_frozen_scores_and_sealed_panel"
                or sample.metadata.get("judge_invalid") or sample.metadata.get("judge_timeout")):
            raise ValueError("only newly admitted, previously unjudged native episodes may be judged")
        sample.remove_sample = False
    namespace = dict(reward_module.__dict__)
    namespace["_get_openai_client"] = lambda **_: client
    namespace["_save_judge_trace"] = lambda args, sample, **record: journal("canonical_judge_verdict", record)
    for name in ("compute_judge_reward_actionhistory", "_score_single_sample", "reward_func"):
        original = getattr(reward_module, name)
        namespace[name] = FunctionType(original.__code__, namespace, name, original.__defaults__, original.__closure__)
        namespace[name].__kwdefaults__ = original.__kwdefaults__
    values = await namespace["reward_func"](args, copied)
    last = next((x for x in copied if x.metadata.get("is_last_turn")), copied[-1])
    record = deepcopy(last.metadata.get("reward", {}))
    if not record or "judge_text" not in record:
        raise ValueError("canonical judge did not preserve a verdict disposition")
    valid = bool(record.get("valid")) and not last.metadata.get("judge_invalid") and not last.metadata.get("judge_timeout")
    # This study's objective is terminal binary success; preserve the native
    # format/combined diagnostic without turning format penalties into success.
    reward = float(record.get("judge") == 1.) if valid else None
    result = dict(status="valid" if valid else "invalid", reward=reward, native_reward=record,
                  native_values=values, source_sha256=source_sha256)
    journal("terminal_outcome", result)
    return result
