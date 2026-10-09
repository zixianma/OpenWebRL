"""Four-arm local-browser evaluation worker; no work occurs on import.

The CLI requires a frozen prepared root and an explicitly approved SuiteState.
One process owns one episode at a time; the allocation controller owns servers,
the shared task queue, resource accounting and all worker lifetimes.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
from contextlib import contextmanager, asynccontextmanager
from copy import deepcopy
import hashlib
import fcntl
import json
import os
import signal
import socket
from pathlib import Path
import time
from types import SimpleNamespace
from urllib.parse import urlparse

from openwebrl.controlled_selector_protocol import (
    SelectorContractError, SelectorInputBudgetExceeded, STATE_BYTE_LIMIT,
    build_selection_batch, selector_request, selector_result,
)
from openwebrl.controlled_sft_worker import (
    AttemptJournal, CanonicalOM2WReward, ControlledSFTPolicy, SAMPLING,
    controlled_environment_factory, judge_terminal, validate_actor_environment_config,
    validate_kev_startup_proof,
)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _local_base(url):
    parsed = urlparse(url)
    if (parsed.scheme != "http" or parsed.hostname not in ("127.0.0.1", "localhost")
            or not parsed.port or parsed.path not in ("", "/") or parsed.query
            or parsed.fragment or parsed.username or parsed.password):
        raise SelectorContractError("Expected an explicit loopback model server base URL")
    return url.rstrip("/")


class JevContextCapacityExceeded(SelectorInputBudgetExceeded):
    """Only the confirmed Jev HTTP400/detail.error_type capacity rejection."""


def is_known_jev_context_limit(status, body):
    return (status == 400 and isinstance(body, dict) and isinstance(body.get("detail"), dict)
            and body["detail"].get("error_type") == "max_tokens_exceeded")


class ScopedBudget:
    def __init__(self, state, claim):
        self.state = state
        self.context = {key: claim[key] for key in ("attempt_id", "condition", "task_id")}

    def reserve(self, kind, count=1, **context):
        for key, expected in self.context.items():
            if key in context and context[key] != expected:
                raise SelectorContractError("Reservation crosses episode ownership")
        return self.state.reserve(kind, count, **dict(context, **self.context))

    def settle(self, reservation, usage):
        return self.state.settle(reservation, usage)

    def halt(self, reason, **details):
        return self.state.halt(reason, **dict(details, **self.context))

    def finalize_unknown(self, reservation, evidence_path):
        return self.state.finalize_unknown(reservation, evidence_path)


@asynccontextmanager
async def kev_forward_slot():
    """One shared local Kev forward; caller's timeout also bounds queue wait."""
    location = os.getenv("OPENWEBRL_KEV_FORWARD_LOCK")
    if not location or not Path(location).is_absolute():
        raise SelectorContractError("An explicit shared Kev GPU forward lock is required")
    handle = Path(location).open("a")
    acquired = False
    try:
        while not acquired:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
            except BlockingIOError:
                await asyncio.sleep(.05)
        yield
    finally:
        if acquired:
            fcntl.flock(handle, fcntl.LOCK_UN)
        handle.close()


class ChatResponse:
    """The minimal SDK-compatible shape consumed by canonical reward code."""
    def __init__(self, raw):
        self.raw = raw
        self.model = raw.get("model")
        self.usage = raw.get("usage")
        self.choices = [SimpleNamespace(message=SimpleNamespace(content=row["message"]["content"]))
                        for row in raw["choices"]]

    def model_dump(self):
        return self.raw


class HTTPTransports:
    """No SDK retries; canonical judge alone owns its four-attempt loop."""
    def __init__(self, client, *, actor_base_url, kev_base_url, openai_key, jev_key,
                 journal, normalize_actor, on_failure=None):
        self.client = client
        self.actor = _local_base(actor_base_url)
        self.kev = _local_base(kev_base_url)
        self.openai_key, self.jev_key = openai_key, jev_key
        self.journal, self.normalize_actor = journal, normalize_actor
        self.on_failure = on_failure

    async def _post(self, label, url, request, *, key=None, timeout=180):
        headers = {"Authorization": "Bearer " + key} if key else {}
        if label == "sft":
            # Local Uvicorn and HTTPX both default to a five-second idle
            # keepalive. Never reuse a socket at that close boundary. This
            # affects connection lifetime only, and does not retry generation.
            headers["Connection"] = "close"
        started = time.monotonic()
        try:
            response = await self.client.post(url, json=request, headers=headers, timeout=timeout)
        except Exception as exc:
            import httpx
            if isinstance(exc, httpx.TransportError):
                chain, seen, cause = [], set(), exc
                while cause is not None and id(cause) not in seen and len(chain) < 8:
                    seen.add(id(cause))
                    chain.append(dict(type=type(cause).__name__, module=type(cause).__module__,
                        message=str(cause)[:2048], errno=getattr(cause, "errno", None)))
                    cause = cause.__cause__ or cause.__context__
                self.journal("http_transport_error", dict(provider=label,
                    elapsed_seconds=time.monotonic()-started, exception_chain=chain,
                    request_sha256=hashlib.sha256(json.dumps(request, sort_keys=True,
                        ensure_ascii=False, separators=(",", ":")).encode()).hexdigest(),
                    connection_close=label == "sft", response_received=False,
                    status=None, body=None, retried=False))
            raise
        if response.status_code >= 400:
            # Preserve provider diagnosis before raise_for_status loses it.
            evidence = self.journal("http_error", dict(provider=label, status=response.status_code,
                body=response.text[:131072], body_truncated=len(response.text) > 131072))
            if label == "jev":
                try:
                    body = response.json()
                except (ValueError, TypeError):
                    body = None
                if is_known_jev_context_limit(response.status_code, body):
                    self.journal("jev_known_context_capacity_rejection", dict(status=400,
                        error_type="max_tokens_exceeded", evidence_path=evidence))
                    raise JevContextCapacityExceeded("Jev rejected the unmodified state: max_tokens_exceeded")
            if self.on_failure and (label != "om2w_judge" or response.status_code in (400, 401, 403, 404, 413)):
                self.on_failure("provider_http_error", provider=label, status=response.status_code, evidence_path=evidence)
            response.raise_for_status()
        result = response.json()
        self.journal("http_received", dict(provider=label, elapsed_seconds=time.monotonic() - started,
                                           response=result))
        return result

    async def actor_call(self, url, text, sampling_params, image_data, timeout_secs=None):
        if url != self.actor + "/generate":
            raise SelectorContractError("Generation attempted to use a different actor server")
        request = dict(text=text, sampling_params=sampling_params, return_logprob=True, image_data=image_data)
        raw = await self._post("sft", url, request, timeout=timeout_secs or 180)
        meta = raw["meta_info"]
        pairs = meta["output_token_logprobs"]
        if "prompt_tokens" not in meta:
            raise SelectorContractError("Actor token accounting is missing")
        tokens, logprobs = [row[1] for row in pairs], [row[0] for row in pairs]
        return (self.normalize_actor(raw["text"], tokens, logprobs), tokens, logprobs,
                meta.get("finish_reason", {}).get("type", "stop"))

    async def selector(self, provider, request):
        if provider == "luna":
            if not self.openai_key:
                raise SelectorContractError("Missing explicitly configured OpenAI credential")
            return await self._post(provider, "https://api.openai.com/v1/responses", request,
                                    key=self.openai_key)
        if provider == "jev":
            if not self.jev_key:
                raise SelectorContractError("Missing explicitly configured Jev credential")
            return await self._post(provider, "https://api.typesafe.ai/v1/systemone", request,
                                    key=self.jev_key)
        if provider == "kev":
            async with kev_forward_slot():
                return await self._post(provider, self.kev + "/v1/systemone", request)
        raise SelectorContractError("Unknown controlled selector")

    async def judge(self, **request):
        if not self.openai_key:
            raise SelectorContractError("Missing explicitly configured OpenAI judge credential")
        return ChatResponse(await self._post("om2w_judge", "https://api.openai.com/v1/chat/completions",
                                            dict(request, store=False), key=self.openai_key, timeout=120))


@contextmanager
def generation_bindings(generation, *, config, policy, actor_call):
    """Worker-local dependency bindings; actual browser DOM capture is child-side."""
    original_config = generation._apply_local_process_env_overrides
    original_encode = generation._encode_with_processor
    original_infer = generation._run_inference_step
    original_resources = generation._load_local_resources
    frozen = deepcopy(config["browser_env_config"])
    validate_actor_environment_config(frozen)

    def encode(*args, **kwargs):
        encoded = original_encode(*args, **kwargs)
        tokens = encoded[0]
        if len(tokens) + 4096 >= 32768:
            policy.input_invalid = True
            policy.journal("input_budget_invalid", dict(prompt_tokens=len(tokens), before_generator_guard=True))
            raise SelectorInputBudgetExceeded("Actor prompt plus4096 must be below32768 tokens")
        return encoded

    def resources(*args, **kwargs):
        task, tools, prompt = original_resources(*args, **kwargs)
        expected_task = config["episode_task"]
        if (task.get("start_url") != expected_task["start_url"]
                or task.get("intent") != expected_task["intent"] or prompt != policy.prompt
                or tools != json.loads(Path(config["tool_list_path"]).read_text())):
            policy.contract_invalid = True
            raise SelectorContractError("Loaded task, tools or restored policy differs from the frozen episode")
        return task, tools, prompt

    generation._apply_local_process_env_overrides = lambda ignored: deepcopy(frozen)
    generation._encode_with_processor = encode
    generation._run_inference_step = actor_call
    generation._load_local_resources = resources
    try:
        yield
    finally:
        generation._apply_local_process_env_overrides = original_config
        generation._encode_with_processor = original_encode
        generation._run_inference_step = original_infer
        generation._load_local_resources = original_resources


def normalized_task(row, task_id):
    metadata = row.get("metadata", {})
    actual_id = row.get("task_id") or metadata.get("task_id") or row.get("id")
    intent = metadata.get("intent") or metadata.get("task") or row.get("ques") or row.get("prompt")
    start_url = row.get("start_url") or metadata.get("start_url") or row.get("web")
    if actual_id != task_id or not isinstance(intent, str) or not intent.strip() or not start_url:
        raise SelectorContractError("Claimed task is incomplete or has a different identity")
    return dict(task_id=task_id, intent=intent, start_url=start_url, index=row.get("index", 0))


def verify_worker_config(config):
    for key in ("prompt", "tool_list"):
        if _sha(config[key + "_path"]) != config[key + "_sha256"]:
            raise SelectorContractError("Frozen " + key + " file changed")
    validate_actor_environment_config(config["browser_env_config"])
    if not Path(config["actor_checkpoint"]).is_absolute():
        raise SelectorContractError("Actor checkpoint must be an absolute verified path")
    validate_kev_startup_proof(config["kev_proof"])


async def run_episode(claim, config, state, client, *, actor_base_url, kev_base_url, dependencies=None):
    """One attempt, with original canonical samples and no paid fallback path."""
    if dependencies is None:
        from openwebrl import generate_browser as generation, run_evaluate as evaluation
        from openwebrl.eval import reward_online_mind2web as reward
    else:
        generation, evaluation, reward = dependencies
    verify_worker_config(config)
    task = normalized_task(claim["task"], claim["task_id"])
    config = deepcopy(config)
    config["episode_task"] = task
    directory = Path(claim["artifact_directory"])
    journal = AttemptJournal(directory / "events")
    budget = ScopedBudget(state, claim)
    transports = HTTPTransports(client, actor_base_url=actor_base_url, kev_base_url=kev_base_url,
        openai_key=os.getenv("JUDGE_API_KEY") or os.getenv("OPENAI_API_KEY"),
        jev_key=os.getenv("TYPESAFE_API_KEY") or os.getenv("JEV_API_KEY"), journal=journal,
        normalize_actor=generation._ensure_im_end_w_new_line, on_failure=budget.halt)
    policy = ControlledSFTPolicy(claim["condition"], task_id=claim["task_id"], budget=budget,
        journal=journal, selector_call=transports.selector, prompt_path=config["prompt_path"],
        prompt_sha256=config["prompt_sha256"], kev_proof=config["kev_proof"])
    judge = CanonicalOM2WReward(reward, source_sha256=config["canonical_reward_sha256"],
        chat_create=transports.judge, budget=budget, journal=journal, task_id=claim["task_id"])
    address = urlparse(_local_base(actor_base_url))
    args = evaluation.EvalArgs(sglang_router_ip=address.hostname, sglang_router_port=address.port,
        hf_checkpoint=config["actor_checkpoint"], max_steps=60, max_consecutive_parse_failures=3, context_num_screenshots=1,
        judge_api_model="o4-mini-2025-04-16", judge_api_mode="served", judge_timeout_secs=120,
        browser_response_format_mode="browser_env", turn_history_reasoning_mode="full",
        browser_include_tool_response=1, inference_step_timeout_secs=180, task_timeout_secs=1800,
        rollout_temperature=1., rollout_top_p=.95, rollout_top_k=-1, rollout_max_response_len=4096,
        rollout_max_context_len=32768, path_to_save_generated_samples=str(directory / "rollouts"))
    args.browser_action_selector = policy
    sample = reward.Sample(index=task["index"], prompt=task["intent"], metadata=dict(task,
        _browser_task_file=config["tasks_path"]))
    manifest = dict(config["browser_manifest"], artifact_directory=str(directory / "browser"))
    manifest_path = directory / "browser-manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("x") as handle:
        json.dump(manifest, handle, sort_keys=True)
    final_state = {}
    env_updates = dict(OPENWEBRL_CONTROLLED_BROWSER_MANIFEST=str(manifest_path),
        OPENWEBRL_CONTROLLED_BROWSER_MANIFEST_SHA256=_sha(manifest_path),
        SLIME_BROWSER_ENV_MODE="local_process", SLIME_BROWSER_ROLLOUT_CONCURRENCY="1",
        SLIME_BROWSER_CHAT_TEMPLATE_ENABLE_THINKING="0", SLIME_BROWSER_THINKING_OPEN_TAG="<think>",
        SLIME_BROWSER_THINKING_CLOSE_TAG="</think>", SLIME_BROWSER_APPEND_THINKING_PREFILL="1")
    old_env = {key: os.environ.get(key) for key in env_updates}
    os.environ.update(env_updates)
    host_identity = dict(hostname=socket.gethostname(), slurm_job_id=os.getenv("SLURM_JOB_ID"),
        slurm_node_name=os.getenv("SLURMD_NODENAME"), egress_ip=None,
        egress_ip_status="unknown; no extra external lookup performed")
    journal("episode_started", dict(attempt_id=claim["attempt_id"], condition=claim["condition"],
        task=task, sampling=SAMPLING, max_decisions=60, max_operations=30,
        reward_source_sha256=judge.source_sha256, host_identity=host_identity))
    started = time.monotonic()
    try:
        with generation_bindings(generation, config=config, policy=policy, actor_call=transports.actor_call):
            with controlled_environment_factory(generation, final_state, budget=budget, condition=claim["condition"]):
                samples = await generation.generate_turn_sample(args, sample, dict(SAMPLING))
        # Native no-generated-turn/timeout samples omit turn_index. Preserve
        # their status and route that single Sample through the same canonical
        # entry point; do not manufacture a turn or mark it completed.
        reward_samples = samples if any("turn_index" in s.metadata for s in samples) else samples[-1]
        native_timeout = str(samples[-1].metadata.get("terminate_reason", ""))
        if (native_timeout == "generation_error: rollout_task_timeout after 1800.0s"
                and policy.clock() - policy.started >= 1800 and not any((
                    policy.input_invalid, policy.provider_invalid, policy.budget_invalid))):
            policy.boundary_reason = "episode_deadline"
            journal("episode_boundary", dict(reason="episode_deadline", source="native_generation_timeout"))
        outcome = await judge_terminal(args=args, samples=reward_samples, final_state=final_state,
            policy=policy, canonical_reward=judge, journal=journal)
        terminal = next((s for s in samples if s.metadata.get("is_last_turn")), samples[-1])
        receipt = (final_state.get("observation") or {}).get("controlled", {})
        halt_required = bool(policy.provider_invalid or policy.contract_invalid or policy.budget_invalid
                             or outcome.get("reason") in ("judge_budget_halt", "generation_aborted"))
        result = dict(task_id=claim["task_id"], condition=claim["condition"], attempt_id=claim["attempt_id"],
            status=str(terminal.status), reward=outcome["score"], valid=outcome["valid"], host_identity=host_identity,
            invalid_reason=outcome.get("reason"), halt_required=halt_required, judge_http_attempts=outcome["judge_http_attempts"],
            metadata=evaluation._to_jsonable(terminal.metadata), response=evaluation._to_jsonable(terminal.response),
            actor_decisions=policy.decisions, browser_operations=receipt.get("action_attempts", 0),
            elapsed_seconds=time.monotonic() - started, browser_closed=final_state.get("browser_closed", False))
        if final_state.get("observation"):
            result["final_screenshot"] = journal.blob("final.png", final_state["observation"]["screenshot"])
        journal("episode_result", result)
        raw_result = json.dumps(result, sort_keys=True, ensure_ascii=False, allow_nan=False).encode() + b"\n"
        result_file = directory / "result.json"
        with result_file.open("xb") as handle:
            handle.write(raw_result); handle.flush(); os.fsync(handle.fileno())
        descriptor = {key: value for key, value in result.items() if key not in ("metadata", "response")}
        descriptor.update(result_path=str(result_file), result_sha256=hashlib.sha256(raw_result).hexdigest())
        return descriptor
    finally:
        if final_state.get("client") is not None:
            await final_state["client"].exit()
        for key, old in old_env.items():
            if old is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old


def runtime_worker_config(plan, runtime, *, plan_sha256, actor_base_url, kev_base_url, job_id):
    if (runtime.get("plan_sha256") != plan_sha256 or runtime.get("job_id") != job_id
            or not job_id or actor_base_url not in runtime.get("actor_base_urls", [])
            or kev_base_url != runtime.get("kev_base_url")):
        raise SelectorContractError("Runtime server proof differs from this frozen allocation")
    config = deepcopy(plan["worker_config"])
    live = runtime.get("kev_proof", {})
    for key, value in config["kev_proof"].items():
        if key != "server_card" and live.get(key) != value:
            raise SelectorContractError("Live Kev proof differs from the frozen source/weights receipt")
    validate_kev_startup_proof(live)
    config["kev_proof"] = deepcopy(live)
    return config


async def run_worker(root, worker_id, actor_base_url, kev_base_url, runtime_config):
    import httpx
    from openwebrl.controlled_sft_state import SuiteState, digest
    root = Path(root)
    plan = json.loads((root / "plan.json").read_text())
    runtime = json.loads(Path(runtime_config).read_text())
    config = runtime_worker_config(plan, runtime, plan_sha256=digest(plan),
        actor_base_url=actor_base_url, kev_base_url=kev_base_url, job_id=os.getenv("SLURM_JOB_ID"))
    verify_worker_config(config)
    if os.environ.get("WANDB_PROJECT") != "openwebrl-evals":
        raise SelectorContractError("Separate evaluations require WANDB_PROJECT=openwebrl-evals")
    if not (os.getenv("JUDGE_API_KEY") or os.getenv("OPENAI_API_KEY")) or not (os.getenv("TYPESAFE_API_KEY") or os.getenv("JEV_API_KEY")):
        raise SelectorContractError("Required suite credentials are missing before any browser claim")
    state = SuiteState(root)
    # SuiteState.claim must verify exact current approval and active allocation;
    # neither importing this module nor an old historical approval permits work.
    async with httpx.AsyncClient(trust_env=False, timeout=180, follow_redirects=False) as client:
        while claim := state.claim(worker_id):
            result = await run_episode(claim, config, state, client,
                actor_base_url=actor_base_url, kev_base_url=kev_base_url)
            state.finish(claim["attempt_id"], result)
            if result.get("halt_required"):
                state.halt("systemic_provider_or_contract_failure", attempt_id=claim["attempt_id"],
                           result_path=result["result_path"], invalid_reason=result.get("invalid_reason"))
                raise RuntimeError("Suite dispatch halted; inspect preserved provider/contract evidence")


def maximal_jev_diagnostic_batch():
    """Synthetic, reproducible maximum-size shared view; no task/browser/model I/O.

    Preserve five complete proposed strings and the full allowed page prefix;
    fill the remaining state allowance with a visible element description. This
    probes the byte ceiling, not a guarantee about a provider's token capacity.
    """
    page = dict(url="about:blank", title="Controlled capacity diagnostic", text="p" * 16000,
        text_characters=16000, screen_size=[1280, 720],
        tabs=[dict(url="about:blank", title="Controlled capacity diagnostic", index=0, active=True)],
        interactive_elements=[dict(id=0, tag="button", bbox=[0., 0., .1, .1], textContent="")])
    observation = dict(selection_page=page, active_tab_url="about:blank", screen_size=[1280, 720])
    outputs = [("<think>" + (f"Candidate {i}: inspect field and task. " * 500)[:16000]
                + "</think>click(point_2d=[50,50])", [], [], "stop") for i in range(5)]
    actions = [dict(attempt=i + 1, action=dict(name="wait", args={}), success=True,
                    status="returned", tool_response="Synthetic diagnostic") for i in range(5)]
    def build():
        return build_selection_batch(task_id="diagnostic/jev-max-shared-state", task="Choose the supplied action that advances the synthetic task.",
            turn=5, observation=observation, executed_actions=actions, actor_outputs=outputs)
    initial = build()
    padding = STATE_BYTE_LIMIT - len(initial.state_json.encode("utf-8"))
    # ASCII punctuation and distinct identifiers create a token-rich valid field.
    page["interactive_elements"][0]["textContent"] = (" ! @ # $ % ^ & * ( ) _ + field_0123456789 " * (padding // 40 + 2))[:padding]
    batch = build()
    if len(batch.state_json.encode("utf-8")) != STATE_BYTE_LIMIT:
        raise SelectorContractError("Maximum-state diagnostic construction changed")
    return batch


async def run_jev_context_preflight(root):
    """One separately reserved startup diagnostic; called only after exact approval.

    Known capacity rejection is a completed diagnostic, not evidence that Jev
    accepted the maximum state. Ordinary smoke episodes still require valid
    selector responses. No truncation, fallback, or unmetered retry occurs.
    """
    import httpx
    from openwebrl.controlled_sft_state import SuiteState, digest
    root = Path(root).resolve()
    state = SuiteState(root)
    state.require_approved_allocation()
    job = os.environ["SLURM_JOB_ID"]
    directory = root / "scheduler-attempts" / job
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "jev-context-preflight.json"
    if target.exists():
        raise SelectorContractError("A preserved diagnostic exists for this allocation; do not replay blindly")
    key = os.getenv("TYPESAFE_API_KEY") or os.getenv("JEV_API_KEY")
    if not key:
        raise SelectorContractError("Missing Jev diagnostic credential")
    batch = maximal_jev_diagnostic_batch()
    request = selector_request(batch, "jev")
    journal = AttemptJournal(directory / "jev-context-events")
    request_path = journal("jev_context_preflight_request", dict(request=request,
        state_bytes=len(batch.state_json.encode("utf-8")), state_sha256=batch.state_sha256))
    reservation = state.reserve("jev_selector_requests", 1,
        purpose="maximum_shared_state_capacity_diagnostic", request=request)
    result = dict(completed=False, expected_model="jev-1.13.0", model_identity=None,
        request_sha256=digest(request), request_evidence_path=request_path,
        state_sha256=batch.state_sha256, state_bytes=STATE_BYTE_LIMIT,
        plan_sha256=state.plan_hash, job_id=job, reservation=reservation,
        browser_episodes=0, sft_proposals=0)
    try:
        async with httpx.AsyncClient(trust_env=False, timeout=180, follow_redirects=False) as client:
            transport = HTTPTransports(client, actor_base_url="http://127.0.0.1:1", kev_base_url="http://127.0.0.1:2",
                openai_key=None, jev_key=key, journal=journal, normalize_actor=None,
                on_failure=lambda reason, **details: state.halt(reason, purpose="jev_context_preflight", **details))
            try:
                raw = await transport.selector("jev", request)
                chosen = selector_result(batch, "jev", raw)
                result.update(completed=True, outcome="accepted", model_identity=raw["model"],
                    selected_displayed_index=chosen.displayed_index, selected_original_index=chosen.original_index)
            except JevContextCapacityExceeded:
                result.update(completed=True, outcome="known_capacity_rejection",
                    error_type="max_tokens_exceeded", http_status=400,
                    limitation="Maximum shared state exceeds observed provider capacity; no accepted-capacity claim.")
    except BaseException as exc:
        result.update(outcome="unconfirmed_error", error_type=type(exc).__name__)
        state.halt("jev_context_preflight_failed", error_type=type(exc).__name__, request_evidence_path=request_path)
        raise
    finally:
        with target.open("x") as handle:
            json.dump(result, handle, indent=2, sort_keys=True)
            handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--worker-id")
    parser.add_argument("--actor-base-url")
    parser.add_argument("--kev-base-url")
    parser.add_argument("--runtime-config")
    parser.add_argument("--jev-context-preflight", action="store_true")
    args = parser.parse_args()
    if args.jev_context_preflight:
        asyncio.run(run_jev_context_preflight(args.root))
        return
    if not all((args.worker_id, args.actor_base_url, args.kev_base_url, args.runtime_config)):
        parser.error("Workers require --worker-id, --actor-base-url, --kev-base-url and --runtime-config")
    async def supervised():
        task = asyncio.current_task()
        loop = asyncio.get_running_loop()
        for signum in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(signum, task.cancel)
        try:
            await run_worker(args.root, args.worker_id, args.actor_base_url, args.kev_base_url, args.runtime_config)
        finally:
            for signum in (signal.SIGTERM, signal.SIGINT):
                loop.remove_signal_handler(signum)
    asyncio.run(supervised())


if __name__ == "__main__":
    main()
