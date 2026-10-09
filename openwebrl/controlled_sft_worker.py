"""Testable worker components for four controlled SFT conditions.

No CLI, model clients or implicit budget exists here. The caller supplies an
approved durable reservation object, inference/selector/judge transports and an
owned browser client. Importing the module performs no external work.
"""
from __future__ import annotations

import asyncio
import base64
from copy import deepcopy
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import time
from types import FunctionType, SimpleNamespace

from openwebrl.controlled_selector_protocol import (
    SelectorContractError, SelectorInputBudgetExceeded, SelectorChoiceRejected, baseline_output,
    build_selection_batch, candidate_seed, selector_request, selector_result,
)

CONDITIONS = {"L01": None, "L08": "luna", "L10": "jev", "L11": "kev"}
SAMPLING = dict(temperature=1., top_p=.95, top_k=-1, max_new_tokens=4096,
                repetition_penalty=1.)
KEV_IDENTITY = dict(model="jaredpalmer/kev-27b",
    revision="af0e6d551bdc2cc724f3e9d7a8bee1cd4fb8f7bf",
    base="Qwen/Qwen3.8-27B", base_revision="1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
    upstream_commit="fe64b1274ea7f80d4095866df90666abb03e9cf6",
    head_sha256="7968f17b03479c1ef9d1c0f3ab8a15e31ecb441cf40691b07ee945ab554d45ad")


class EpisodeBoundary(RuntimeError):
    """The controlled action/decision/deadline boundary has been reached."""


class AttemptJournal:
    """Append-only private receipts; reservations/results never overwrite each other."""
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.sequence = max((int(path.stem) for path in self.directory.glob("[0-9]*.json")), default=0)

    def __call__(self, event, record):
        payload = json.dumps(dict(event=event, record=record), ensure_ascii=False,
                             sort_keys=True, allow_nan=False).encode() + b"\n"
        self.sequence += 1
        target = self.directory / f"{self.sequence:08d}.json"
        with target.open("xb") as handle:
            handle.write(payload); handle.flush(); os.fsync(handle.fileno())
        return str(target)

    def blob(self, name, data):
        if Path(name).name != name or not isinstance(data, bytes):
            raise ValueError("Expected a named immutable private binary artifact")
        target = self.directory / name
        with target.open("xb") as handle:
            handle.write(data); handle.flush(); os.fsync(handle.fileno())
        return dict(path=str(target), sha256=hashlib.sha256(data).hexdigest())


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def validate_kev_startup_proof(proof):
    """Require a separately verified source/weights receipt and matching card.

    The approved controller must actually verify the pinned source/weight files
    before creating this receipt. An API alias alone cannot identify Kev27B.
    """
    if (not isinstance(proof, dict) or proof.get("files_verified") is not True
            or any(proof.get(k) != v for k, v in KEV_IDENTITY.items())):
        raise SelectorContractError("Missing verified Kev27B source/weights identity")
    card = proof.get("server_card", {})
    expected = dict(name="kev-latest", base=KEV_IDENTITY["base"], device="cuda",
        backend="torch", dtype="bfloat16", max_state_tokens=65536, truncate_states=False,
        temperature=1.319507910772894, run=proof.get("verified_model_path"))
    if (not expected["run"] or any(card.get(k) != v for k, v in expected.items())
            or card.get("cuda_graphs") is None):
        raise SelectorContractError("Kev server card does not match the verified checkpoint")
    return _digest(proof)


def validate_observation(observation):
    image = observation.get("screenshot")
    page = observation.get("selection_page")
    receipt = observation.get("controlled")
    if (not isinstance(image, bytes) or not image or not isinstance(page, dict)
            or not isinstance(receipt, dict)):
        raise SelectorContractError("Shared controlled DOM and screenshot evidence is required")
    if (page.get("url") != observation.get("active_tab_url")
            or list(observation.get("screen_size", [])) != [1280, 720]
            or receipt.get("screenshot_sha256") != hashlib.sha256(image).hexdigest()
            or receipt.get("page_sha256") != _digest(page)):
        raise SelectorContractError("Shared observation identity does not match its evidence")
    count = receipt.get("action_attempts")
    actions = receipt.get("executed_actions")
    if (type(count) is not int or not 0 <= count <= 30 or not isinstance(actions, list)
            or len(actions) != count or any(entry.get("attempt") != i + 1
                                          for i, entry in enumerate(actions))):
        raise SelectorContractError("Browser operation counter differs from its execution log")
    if (type(receipt.get("sequence")) is not int or receipt["sequence"] < 1
            or receipt.get("capture_finished_monotonic", -1)
            < receipt.get("capture_started_monotonic", 0)):
        raise SelectorContractError("Controlled capture receipt is malformed")
    return receipt


def validate_actor_environment_config(config):
    """Selector DOM stays out of the actor's screenshot/history prompt."""
    expected = dict(mode="local_process", width=1280, height=720, dpr=1,
        use_screenshot=True, use_a11ytree=False, resize_output_coords=True, resize_scale=1000)
    if any(config.get(key) != value for key, value in expected.items()):
        raise SelectorContractError("Actor browser/view configuration differs from the controlled protocol")
    local = config.get("local_process")
    if not isinstance(local, dict) or local.get("server_module") != "openwebrl.controlled_browser_server":
        raise SelectorContractError("The shared child observation server is required for all four arms")


class ControlledSFTPolicy:
    """Use the existing generate_browser selector hook with injected transports."""
    browser_coordinate_space = "normalized_1000"

    def __init__(self, condition, *, task_id, budget, journal, selector_call,
                 prompt_path, prompt_sha256, kev_proof=None, clock=time.monotonic):
        if condition not in CONDITIONS or not task_id:
            raise SelectorContractError("Unknown controlled condition or missing task")
        if not callable(getattr(budget, "reserve", None)) or not callable(journal):
            raise SelectorContractError("Explicit durable budget and journal bindings are required")
        self.prompt = Path(prompt_path).read_text(encoding="utf-8")
        if not self.prompt.strip() or hashlib.sha256(self.prompt.encode()).hexdigest() != prompt_sha256:
            raise SelectorContractError("Required actor policy prompt is missing or changed")
        self.condition, self.provider, self.task_id = condition, CONDITIONS[condition], task_id
        if self.provider and not callable(selector_call):
            raise SelectorContractError("Selector transport binding is required")
        self.kev_proof_sha256 = validate_kev_startup_proof(kev_proof) if self.provider == "kev" else None
        self.budget, self.journal, self.selector_call = budget, journal, selector_call
        self.clock, self.started = clock, clock()
        self.decisions, self.prompt_tokens, self.sequence = 0, None, 0
        self.last_observation = None
        self.input_invalid = False
        self.contract_invalid = False
        self.provider_invalid = False
        self.budget_invalid = False
        self.boundary_reason = None

    def _reserve(self, kind, count, **context):
        try:
            return self.budget.reserve(kind, count, **context)
        except BaseException as exc:
            self.budget_invalid = True
            evidence = self.journal("budget_halt", dict(kind=kind, count=count, error_type=type(exc).__name__))
            if callable(getattr(self.budget, "halt", None)):
                self.budget.halt("reservation_rejected", kind=kind, evidence_path=evidence)
            raise

    def set_training_context(self, context):
        tokens = context.get("prompt_tokens")
        if not isinstance(tokens, (list, tuple)) or not tokens:
            raise SelectorContractError("Expanded processor prompt token IDs are required")
        self.prompt_tokens = len(tokens)

    def set_context(self, messages, tools, tokenizer, prompt_tokens):
        # Compatibility with preserved instrumented generation sources.
        if type(prompt_tokens) is not int or prompt_tokens <= 0:
            raise SelectorContractError("Expanded prompt token count is invalid")
        self.prompt_tokens = prompt_tokens

    async def __call__(self, **kwargs):
        try:
            return await self._decide(**kwargs)
        except SelectorInputBudgetExceeded:
            self.input_invalid = True
            raise
        except SelectorContractError as exc:
            self.contract_invalid = True
            evidence = self.journal("contract_error", dict(error_type=type(exc).__name__, error=str(exc)))
            if callable(getattr(self.budget, "halt", None)):
                self.budget.halt("actor_or_selector_contract_error", evidence_path=evidence)
            raise

    async def _decide(self, *, infer, url, input_text, sampling_params, images,
                      observation, history, task, task_id, turn, timeout):
        if task_id != self.task_id or type(turn) is not int or turn != self.decisions:
            raise SelectorContractError("Decision belongs to another episode or turn")
        receipt = validate_observation(observation)
        if receipt["sequence"] < self.sequence:
            raise SelectorContractError("Browser observation sequence moved backwards")
        self.sequence, self.last_observation = receipt["sequence"], observation
        remaining = 1800 - (self.clock() - self.started)
        boundary = (receipt.get("terminal_reason") or
                    ("action_limit" if receipt["action_attempts"] >= 30 else None) or
                    ("decision_limit" if self.decisions >= 60 else None) or
                    ("episode_deadline" if remaining <= 0 else None))
        if boundary:
            self.boundary_reason = boundary
            self.journal("episode_boundary", dict(reason=boundary, decisions=self.decisions))
            raise EpisodeBoundary("Controlled boundary: " + boundary)
        if self.prompt_tokens is None or self.prompt not in input_text:
            raise SelectorContractError("Missing expanded context count or actor policy in prompt")
        if self.prompt_tokens + 4096 >= 32768:
            self.input_invalid = True
            self.journal("input_budget_invalid", dict(turn=turn, prompt_tokens=self.prompt_tokens))
            raise SelectorInputBudgetExceeded("Actor prompt plus4096 must be below32768 tokens")
        if any(sampling_params.get(key) != value for key, value in SAMPLING.items()):
            raise SelectorContractError("Controlled actor decoding changed")
        if len(images) != 1:
            raise SelectorContractError("The actor must receive exactly its current screenshot")
        image_payload = images[0]
        if isinstance(image_payload, dict):
            image_payload = image_payload.get("url")
        if not isinstance(image_payload, str):
            raise SelectorContractError("Expected the framework's encoded current screenshot")
        if image_payload.startswith("data:"):
            image_payload = image_payload.split(",", 1)[-1]
        try:
            image_bytes = base64.b64decode(image_payload, validate=True)
        except (ValueError, TypeError) as exc:
            raise SelectorContractError("Invalid encoded actor screenshot") from exc
        if image_bytes != observation["screenshot"]:
            raise SelectorContractError("Actor image and shared DOM capture refer to different observations")
        count = 1 if self.provider is None else 5
        seeds = [candidate_seed(task_id, turn, i) for i in range(count)]
        self._reserve("local_sft_generations", count, task_id=task_id, turn=turn)
        self.decisions += 1
        self.journal("proposal_batch_reserved", dict(turn=turn, condition=self.condition,
            prompt_sha256=hashlib.sha256(input_text.encode()).hexdigest(),
            prompt_tokens=self.prompt_tokens, sampling=SAMPLING, candidate_seeds=seeds,
            screenshot_sha256=receipt["screenshot_sha256"], page_sha256=receipt["page_sha256"]))

        async def proposal(index, seed):
            try:
                output = await asyncio.wait_for(infer(url, input_text,
                    dict(sampling_params, sampling_seed=seed), images,
                    timeout_secs=min(180, remaining)), min(180, remaining))
                if not isinstance(output, (tuple, list)) or len(output) != 4:
                    raise SelectorContractError("Malformed actor inference tuple")
                self.journal("proposal_received", dict(turn=turn, candidate=index,
                    output_text=output[0], token_ids=output[1], finish_type=output[3]))
                return output
            except BaseException as exc:
                self.journal("proposal_error", dict(turn=turn, candidate=index, error_type=type(exc).__name__))
                raise

        # Await all siblings before leaving a failed batch; never spill inference
        # into a replacement episode or manufacture a missing proposal.
        outputs = await asyncio.gather(*(proposal(i, seed) for i, seed in enumerate(seeds)),
                                       return_exceptions=True)
        failures = [output for output in outputs if isinstance(output, BaseException)]
        if failures:
            if all(isinstance(error, (TimeoutError, asyncio.TimeoutError)) for error in failures) and self.clock() - self.started >= 1800:
                self.boundary_reason = "episode_deadline"
                self.journal("episode_boundary", dict(reason=self.boundary_reason))
            else:
                self.provider_invalid = True
                if callable(getattr(self.budget, "halt", None)):
                    self.budget.halt("actor_provider_failure", error_type=type(failures[0]).__name__)
            raise failures[0]
        if self.provider is None:
            return baseline_output(outputs), dict(condition=self.condition, selected_index=0,
                                                   decision_attempts=self.decisions)
        try:
            batch = build_selection_batch(task_id=task_id, task=task, turn=turn,
                observation=observation, executed_actions=receipt["executed_actions"], actor_outputs=outputs)
        except SelectorInputBudgetExceeded:
            self.input_invalid = True
            self.journal("selector_input_budget_invalid", dict(turn=turn))
            raise
        request = selector_request(batch, self.provider)
        selector_reservation = self._reserve({"luna": "luna_selector_requests", "jev": "jev_selector_requests",
                             "kev": "local_kev_requests"}[self.provider], 1,
                            task_id=task_id, turn=turn, request=request)
        self.journal("selector_reserved", dict(turn=turn, provider=self.provider, request=request,
            state_sha256=batch.state_sha256, displayed_order=batch.displayed_order,
            kev_proof_sha256=self.kev_proof_sha256))
        selector_settled = False
        try:
            remaining = 1800 - (self.clock() - self.started)
            if remaining <= 0:
                self.boundary_reason = "episode_deadline"
                self.journal("episode_boundary", dict(reason=self.boundary_reason))
                raise EpisodeBoundary("Episode deadline reached before selector dispatch")
            response = await asyncio.wait_for(self.selector_call(self.provider, request), min(180, remaining))
            raw = response if isinstance(response, dict) else response.model_dump()
            self.journal("selector_received", dict(turn=turn, response=raw))
            if self.provider == "luna" and callable(getattr(self.budget, "settle", None)):
                self.budget.settle(selector_reservation, raw.get("usage"))
                selector_settled = True
            selected = selector_result(batch, self.provider, raw)
        except SelectorChoiceRejected as exc:
            # Preserve the native semantic guard: execute neither the returned
            # choice nor a substituted argmax. This is a model decision failure,
            # not a transport/schema failure that should halt other episodes.
            self.boundary_reason = "selector_choice_rejected"
            self.journal("selector_choice_rejected", dict(turn=turn, provider=self.provider,
                error=str(exc), executed=False, response=raw))
            raise EpisodeBoundary("Controlled boundary: selector_choice_rejected") from exc
        except BaseException as exc:
            if isinstance(exc, (TimeoutError, asyncio.TimeoutError)) and self.clock() - self.started >= 1800:
                self.boundary_reason = "episode_deadline"
                self.journal("episode_boundary", dict(reason=self.boundary_reason))
            if isinstance(exc, SelectorInputBudgetExceeded):
                self.input_invalid = True
            self.provider_invalid = not isinstance(exc, (EpisodeBoundary, SelectorInputBudgetExceeded)) and self.boundary_reason != "episode_deadline"
            evidence = self.journal("selector_error", dict(turn=turn, error_type=type(exc).__name__))
            if self.provider == "luna" and not selector_settled and callable(getattr(self.budget, "finalize_unknown", None)):
                self.budget.finalize_unknown(selector_reservation, evidence)
            if self.provider_invalid and callable(getattr(self.budget, "halt", None)):
                self.budget.halt("selector_provider_failure", evidence_path=evidence)
            raise
        return selected.output, dict(condition=self.condition,
            selected_index=selected.original_index, displayed_index=selected.displayed_index,
            selector_state_sha256=selected.state_sha256, decision_attempts=self.decisions)


class ControlledClient:
    """Capture fresh terminal evidence before the framework closes its browser."""
    def __init__(self, client, final_state, *, capture=None, budget=None, reservation=None):
        if capture is None:
            from openwebrl.controlled_browser_server import fresh_observation
            capture = fresh_observation
        self.client, self.final_state, self.capture = client, final_state, capture
        self._closing = None
        self.budget, self.reservation = budget, reservation

    def __getattr__(self, name):
        return getattr(self.client, name)

    async def step(self, actions):
        result = await self.client.step(actions)
        receipt = validate_observation(result[0])
        if result[3]:
            self.final_state["boundary_reason"] = receipt.get("terminal_reason")
            if receipt.get("terminal_reason") in ("action_limit", "requires_user_input"):
                # Existing generation ignores Gym truncated and would encode one
                # more prompt. Stop before that expansion without marking done.
                raise EpisodeBoundary("Controlled boundary: " + receipt["terminal_reason"])
            raise RuntimeError("Controlled browser stopped: " + str(receipt.get("terminal_reason")))
        return result

    async def _close(self):
        from openwebrl.controlled_browser_server import TERMINAL_CAPTURE_TIMEOUT_SECONDS
        started = time.monotonic()
        try:
            observation = await asyncio.wait_for(self.capture(self.client), TERMINAL_CAPTURE_TIMEOUT_SECONDS)
            validate_observation(observation)
            self.final_state["observation"] = observation
        except BaseException as exc:
            self.final_state["capture_error_type"] = type(exc).__name__
        finally:
            self.final_state["terminal_capture_seconds"] = time.monotonic() - started
            await self.client.exit()
            self.final_state["browser_closed"] = True
            if self.budget is not None and callable(getattr(self.budget, "settle", None)):
                self.budget.settle(self.reservation, {"closed": True})

    async def exit(self):
        if self._closing is None:
            self._closing = asyncio.create_task(self._close())
        await asyncio.shield(self._closing)


@contextmanager
def controlled_environment_factory(generation, final_state, *, budget, condition, capture=None):
    """Scoped worker-only adapter; the DOM extension runs in the child server."""
    original = generation._create_env
    if condition not in CONDITIONS or not callable(getattr(budget, "reserve", None)):
        raise SelectorContractError("Explicit browser reservation binding and condition are required")

    async def create(*args, **kwargs):
        config = args[1] if len(args) > 1 else kwargs["env_config"]
        validate_actor_environment_config(config)
        task_id = args[0] if args else kwargs["task_id"]
        reservation = budget.reserve("browser_sessions", 1, task_id=task_id, condition=condition)
        # The native local factory initializes before returning. Bind ownership
        # as soon as its child exists so a failed reset still closes/settles the
        # reserved browser and preserves a result instead of losing the handle.
        from openwebrl.env import local_process_env
        transport_create = local_process_env.create_local_process_env

        async def owned_create(local_config):
            env = await transport_create(local_config)
            wrapped = ControlledClient(env, final_state, capture=capture, budget=budget, reservation=reservation)
            final_state["client"] = wrapped
            return wrapped

        local_process_env.create_local_process_env = owned_create
        try:
            env, task_data = await original(*args, **kwargs)
        finally:
            local_process_env.create_local_process_env = transport_create
        if not isinstance(env, ControlledClient):
            # Retain injectable factory support used by offline contract tests.
            env = ControlledClient(env, final_state, capture=capture, budget=budget, reservation=reservation)
            final_state["client"] = env
        return env, task_data

    generation._create_env = create
    try:
        yield
    finally:
        generation._create_env = original


class JudgeDispatchHalt(BaseException):
    """An explicit budget stop must escape the canonical retry-on-Exception loop."""


class CanonicalOM2WReward:
    """Bind the unchanged canonical code to one explicitly metered transport.

    Only the transport dependency is rebound in a private globals dictionary;
    the imported module, rubric, parser, dispatch and retry code are unchanged.
    The common4096 output cap is an explicit suite transport setting, whereas
    the native canonical source leaves completion tokens unspecified.
    """
    def __init__(self, module, *, source_sha256, chat_create, budget, journal, task_id):
        source = Path(module.__file__).resolve()
        if (source.name != "reward_online_mind2web.py"
                or hashlib.sha256(source.read_bytes()).hexdigest() != source_sha256):
            raise SelectorContractError("Canonical reward source differs from its pinned manifest")
        if (not callable(chat_create) or not callable(getattr(budget, "reserve", None))
                or not callable(journal) or not task_id):
            raise SelectorContractError("Explicit judge transport, budget, journal and task are required")
        self.sample_class = module.Sample
        self.source_sha256 = source_sha256
        self.task_id, self.journal = task_id, journal
        self.http_attempts = 0
        self.budget_halted = False

        async def create(**request):
            if request.get("model") != "o4-mini-2025-04-16" or request.get("seed") != 42:
                raise SelectorContractError("Canonical judge identity/seed changed")
            request = dict(request, max_completion_tokens=4096)
            try:
                reservation = budget.reserve("om2w_judge_http_attempts", 1, task_id=task_id, request=request)
            except Exception as exc:
                self.budget_halted = True
                journal("judge_budget_halt", dict(error_type=type(exc).__name__))
                raise JudgeDispatchHalt("Judge HTTP reservation rejected") from exc
            self.http_attempts += 1
            journal("judge_reserved", dict(attempt=self.http_attempts, request=request,
                                           source_sha256=source_sha256))
            settled = False
            try:
                response = await chat_create(**request)
                if callable(getattr(budget, "settle", None)):
                    usage = getattr(response, "usage", None)
                    budget.settle(reservation, usage.model_dump() if hasattr(usage, "model_dump") else usage)
                    settled = True
                if getattr(response, "model", "o4-mini-2025-04-16") != "o4-mini-2025-04-16":
                    if callable(getattr(budget, "halt", None)):
                        budget.halt("judge_model_identity_changed")
                    raise SelectorContractError("Judge response model differs from the pinned model")
                journal("judge_received", dict(attempt=self.http_attempts,
                    response=response.model_dump() if hasattr(response, "model_dump") else
                    dict(text=response.choices[0].message.content,
                         model=getattr(response, "model", None))))
                return response
            except BaseException as exc:
                evidence = journal("judge_transport_error", dict(attempt=self.http_attempts,
                                                      error_type=type(exc).__name__))
                if not settled and callable(getattr(budget, "finalize_unknown", None)):
                    budget.finalize_unknown(reservation, evidence)
                raise

        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        namespace = dict(vars(module))
        namespace["_get_openai_client"] = lambda **kwargs: client
        for name in ("_judge", "_score_single", "reward_func"):
            original = getattr(module, name)
            cloned = FunctionType(original.__code__, namespace, original.__name__,
                                  original.__defaults__, original.__closure__)
            cloned.__kwdefaults__ = original.__kwdefaults__
            namespace[name] = cloned
        self.reward_func = namespace["reward_func"]

    async def __call__(self, args, samples):
        if getattr(args, "judge_api_model", None) != "o4-mini-2025-04-16":
            raise SelectorContractError("Expected the shared pinned o4-mini judge")
        return await self.reward_func(args, samples)


async def judge_terminal(*, args, samples, final_state, policy, canonical_reward, journal):
    """Preserve original messages/status and delegate canonical OM2W scoring.

    COMPLETED-only API dispatch remains inside the canonical source. Usable
    action-limit states are not promoted to COMPLETED. Infrastructure validity
    is recorded independently of the canonical score; no secondary judge exists.
    """
    if not isinstance(canonical_reward, CanonicalOM2WReward):
        raise SelectorContractError("An explicitly metered canonical reward binding is required")
    sample_class = canonical_reward.sample_class
    if isinstance(samples, sample_class):
        terminal = samples
        all_samples = [samples]
    else:
        all_samples = list(samples)
        marked = [sample for sample in all_samples if sample.metadata.get("is_last_turn")]
        if len(marked) != 1:
            raise SelectorContractError("Exactly one original terminal turn sample is required")
        terminal = marked[0]
    if terminal.metadata.get("task_id") != policy.task_id:
        raise SelectorContractError("Judge episode identity differs from the actor episode")
    client = final_state.get("client")
    if client is not None:
        await client.exit()
    observation = final_state.get("observation")
    reason = None
    receipt = None
    if observation is None or not final_state.get("browser_closed"):
        reason = "missing_fresh_terminal_evidence"
    else:
        receipt = validate_observation(observation)
        terminal.metadata.setdefault("full_image_list", []).append(
            base64.b64encode(observation["screenshot"]).decode("ascii"))
        terminal.metadata["controlled_terminal"] = dict(
            screenshot_sha256=receipt["screenshot_sha256"],
            terminal_reason=receipt.get("terminal_reason"),
            policy_boundary_reason=policy.boundary_reason,
            original_status=str(terminal.status), reward_source_sha256=canonical_reward.source_sha256)
    if ((receipt and receipt.get("infrastructure_invalid")) or policy.input_invalid
            or policy.provider_invalid or policy.budget_invalid or policy.contract_invalid):
        reason = "infrastructure_or_input_budget_invalid"
    if (terminal.status == sample_class.Status.ABORTED and not reason
            and policy.boundary_reason not in ("action_limit", "decision_limit", "episode_deadline", "requires_user_input", "selector_choice_rejected")
            and not (receipt and (receipt.get("terminal_reason") == "requires_user_input"
                     or (receipt.get("terminal_reason") == "action_limit"
                         and receipt.get("action_attempts") == 30)))):
        reason = "generation_aborted"
    # Do not use an old screenshot for a completed trajectory when fresh capture
    # failed, and do not spend judge budget on an invalid completed attempt.
    if reason and terminal.status == sample_class.Status.COMPLETED:
        for sample in all_samples:
            sample.remove_sample = True
        return dict(valid=False, reason=reason, score=None, judge_http_attempts=0)
    try:
        reward = await canonical_reward(args, samples)
    except JudgeDispatchHalt:
        return dict(valid=False, reason="judge_budget_halt", score=None,
                    judge_http_attempts=canonical_reward.http_attempts)
    score = terminal.reward
    journal("canonical_reward_applied", dict(score=score, original_status=str(terminal.status),
        reward=terminal.metadata.get("reward"), source_sha256=canonical_reward.source_sha256,
        judge_http_attempts=canonical_reward.http_attempts, invalid_reason=reason))
    return dict(valid=reason is None and score is not None, reason=reason,
        score=score, sample_rewards=reward, judge_http_attempts=canonical_reward.http_attempts)
