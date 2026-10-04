"""CPU-only Online-Mind2Web adapter for the pinned Jev Ultrafast agent.

Each task owns a process, browser and artifact directory. Upstream policy,
DOM extraction and guarded execution remain intact; only CDP transport changes.
No training imports, reference answers or screenshot inputs enter the actor.
"""
from __future__ import annotations

import ast
import base64
from contextlib import contextmanager
import gzip
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import statistics
import tempfile
import time
from urllib.parse import urlsplit

REPO = Path(__file__).resolve().parents[1]
UPSTREAM_COMMIT = "1231850a0bf1a0c0341fe408ef1668dbbfdfac46"
JUDGE_SOURCE = REPO / "openwebrl/eval/reward_online_mind2web.py"
UPSTREAM_FILES = ("agent.py", "browser.py", "model.py", "questions.py", "snapshot.js")
BROWSER_API = "https://api.browser-use.com/api/v2/browsers"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def append_json(path, value):
    with Path(path).open("a") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_tasks(path, limit=None):
    tasks = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        meta = row.get("metadata") or {}
        task = dict(task_id=str(meta.get("task_id") or row.get("task_id") or row.get("id") or ""),
                    intent=meta.get("intent") or row.get("task_name") or row.get("ques"),
                    start_url=meta.get("start_url") or row.get("website") or row.get("web"))
        if not all(task.values()) or urlsplit(task["start_url"]).scheme not in {"http", "https"}:
            raise ValueError("Task requires a stable ID, instruction and public HTTP(S) URL")
        # Explicit allowlist: never expose evaluator_reference or definite_answer.
        tasks.append(task)
    if not tasks or len({t["task_id"] for t in tasks}) != len(tasks):
        raise ValueError("Empty cohort or duplicate task IDs")
    if limit is not None and not 1 <= limit <= len(tasks):
        raise ValueError("Task limit is outside the dataset")
    return tasks[:limit]


def judge_protocol():
    """Read the canonical literals without importing the GPU training stack."""
    tree = ast.parse(JUDGE_SOURCE.read_text())
    values = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in {"SYSTEM_PROMPT", "USER_PROMPT"}:
                    values[target.id] = ast.literal_eval(node.value)
    if set(values) != {"SYSTEM_PROMPT", "USER_PROMPT"}:
        raise ValueError("Canonical AgentTrek prompt literals are missing")
    return values


def judge_messages(task, history, screenshot, terminal):
    protocol = judge_protocol()
    actions = []
    for entry in history:
        detail = entry["action"]
        if entry.get("text") is not None:
            detail += " " + json.dumps({"text": entry["text"]}, ensure_ascii=False)
        actions.append(detail.replace("\n\n", " "))
    # Jev has no chain of thought or generated final answer. Do not invent either.
    actions.append(f"Agent stopped: {terminal} (this is not a success verdict)")
    trajectory = "\n\n".join(f"Thought {i}: \nAction {i}: {a}" for i, a in enumerate(actions, 1))
    return [
        {"role": "system", "content": protocol["SYSTEM_PROMPT"]},
        {"role": "user", "content": [
            {"type": "text", "text": protocol["USER_PROMPT"].format(
                task=task["intent"], thoughts_and_actions=trajectory)},
            {"type": "image_url", "image_url": {
                "url": "data:image/jpeg;base64," + screenshot, "detail": "high"}},
        ]},
    ]


def parse_verdict(text):
    # Identical to reward_online_mind2web._parse_verdict, including its leniency.
    lower = text.lower()
    if "status:" not in lower:
        return None
    return 1.0 if "success" in lower.split("status:", 1)[1] else 0.0


class ProviderError(RuntimeError):
    def __init__(self, provider, status):
        self.provider, self.status = provider, status
        super().__init__(f"{provider} HTTP {status}")


class EpisodeTimeout(TimeoutError):
    pass


@contextmanager
def deadline(seconds):
    def expired(*_):
        raise EpisodeTimeout("Episode time budget exhausted")
    previous = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def load_credentials(env_file, config):
    from dotenv import dotenv_values
    values = {**dotenv_values(env_file), **os.environ}
    keys = {
        "TYPESAFE_API_KEY": ("local-kev" if config.get("decision_provider") == "kev"
                             else values.get("TYPESAFE_API_KEY") or values.get("JEV_API_KEY")),
        "TEXT_MODEL_API_KEY": values.get("TEXT_MODEL_API_KEY") or (
            values.get("OPENAI_API_KEY") if config["text_provider"] == "openai" else values.get("OPENROUTER_API_KEY")),
        "JUDGE_API_KEY": values.get("JUDGE_API_KEY") or values.get("OPENAI_API_KEY"),
        "BROWSER_USE_API_KEY": values.get("BROWSER_USE_API_KEY"),
    }
    required = {"TYPESAFE_API_KEY", "TEXT_MODEL_API_KEY", "JUDGE_API_KEY"}
    if config["browser"] == "browser-use":
        required.add("BROWSER_USE_API_KEY")
    missing = sorted(k for k in required if not keys[k])
    if missing:
        raise ValueError("Missing credentials: " + ", ".join(missing))
    os.environ.update({k: v for k, v in keys.items() if v})
    # Pin actual worker configuration, overriding inherited training settings.
    os.environ.update(TYPESAFE_MODEL=config["jev_model"], TEXT_MODEL=config["text_model"],
                      TEXT_MODEL_BASE_URL=config["text_base_url"], TEXT_MODEL_REASONING="none",
                      WANDB_PROJECT="openwebrl-evals")


def source_identity(upstream):
    import subprocess
    upstream = Path(upstream)
    commit = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    if commit != UPSTREAM_COMMIT:
        raise ValueError("Unexpected Jev Ultrafast commit; review before changing the pin")
    installed = Path(importlib.metadata.distribution("jev-ultrafast").locate_file("jev_ultrafast"))
    hashes = {}
    for name in UPSTREAM_FILES:
        original = subprocess.check_output(["git", "-C", str(upstream), "show", f"HEAD:jev_ultrafast/{name}"])
        if original != (installed / name).read_bytes() or original != (upstream / "jev_ultrafast" / name).read_bytes():
            raise ValueError(f"Installed/upstream source differs from pin: {name}")
        hashes[name] = hashlib.sha256(original).hexdigest()
    return dict(commit=commit, files_sha256=hashes,
                dependencies={name: importlib.metadata.version(name) for name in (
                    "jev-ultrafast", "browser-harness", "playwright", "httpx", "python-dotenv")})


class ModelTransport:
    """Record every HTTP attempt before sending, including retry usage/errors."""
    def __init__(self, root, config, client=None):
        import httpx
        self.root, self.config = Path(root), config
        self.client = client or httpx.Client(http2=True, follow_redirects=False)
        self.decision_provider = config.get("decision_provider", "jev")
        self.counts = {self.decision_provider: 0, "text": 0, "judge": 0}

    def post(self, url, key, body):
        provider = self.decision_provider if url == "https://api.typesafe.ai/v1/systemone" else "text"
        if provider == "kev":
            # Only the decision endpoint changes; never send TypeSafe credentials
            # to a locally served alternative model.
            url, key = self.config["decision_endpoint"], "local-kev"
        body = dict(body)
        if provider == "text" and self.config["text_provider"] == "openai":
            body.pop("reasoning", None)
            body.pop("thinking", None)
            body.update(temperature=0.6, top_p=0.95)
        timeout = self.config.get("decision_timeout_seconds", 25) if provider == "kev" else 25
        return self.request(provider, url, key, body, attempts=3, timeout=timeout)

    def request(self, provider, url, key, body, *, attempts, timeout):
        import httpx
        caps = {self.decision_provider: self.config["max_decisions"] * 3,
                "text": self.config["max_decisions"] * 3, "judge": 4}
        for attempt in range(attempts):
            if self.counts[provider] >= caps[provider]:
                raise RuntimeError(f"{provider} HTTP attempt budget exhausted")
            self.counts[provider] += 1
            call_id = f"{provider}-{self.counts[provider]:04d}"
            record = dict(call_id=call_id, provider=provider, model=body["model"],
                          attempt=attempt + 1, started_unix=time.time(), request=body)
            append_json(self.root / "api-attempts.jsonl", record)
            started = time.monotonic()
            try:
                response = self.client.post(url, headers={"Authorization": "Bearer " + key},
                                            json=body, timeout=timeout)
            except httpx.HTTPError as exc:
                append_json(self.root / "api-responses.jsonl", dict(
                    call_id=call_id, provider=provider, error_type=type(exc).__name__,
                    latency_seconds=time.monotonic() - started))
                # Ambiguous requests may have been billed; do not silently repeat them.
                raise ProviderError(provider, "transport") from None
            record = dict(call_id=call_id, provider=provider, http_status=response.status_code,
                          latency_seconds=time.monotonic() - started)
            if response.is_success:
                result = response.json()
                record["response"] = result
                append_json(self.root / "api-responses.jsonl", record)
                if provider == self.decision_provider and result.get("model") != self.config["jev_model"]:
                    raise ProviderError(provider, "model_identity_mismatch")
                if provider == "kev" and result.get("truncated"):
                    raise ProviderError(provider, "unexpected_state_truncation")
                return result
            append_json(self.root / "api-responses.jsonl", record)
            if response.status_code not in {429, 500, 502, 503, 504, 529} or attempt == attempts - 1:
                raise ProviderError(provider, response.status_code)
            time.sleep(min(10, 2 ** attempt))
        raise RuntimeError("Unreachable retry state")


class BrowserSession:
    """Own exactly one remote session or isolated local Chromium process."""
    def __init__(self, root, config):
        self.root, self.config = Path(root), config
        self.remote_id = None
        self.playwright = self.browser = self.client = None

    def open(self):
        import httpx
        from playwright.sync_api import sync_playwright
        self.client = httpx.Client(timeout=30, follow_redirects=False)
        self.playwright = sync_playwright().start()
        if self.config["browser"] == "browser-use":
            response = self.client.post(BROWSER_API, headers=self.headers, json=dict(
                timeout=12, proxyCountryCode=None, browserScreenWidth=1120, browserScreenHeight=780))
            if not response.is_success:
                raise ProviderError("browser-use", response.status_code)
            session = response.json()
            self.remote_id = session["id"]
            # Persist ownership before polling; never log capability-bearing CDP URLs.
            write_json(self.root / "browser-session.json", dict(id=self.remote_id, stopped=False))
            for _ in range(15):
                if session.get("cdpUrl"):
                    break
                time.sleep(2)
                response = self.client.get(BROWSER_API + "/" + self.remote_id, headers=self.headers)
                if not response.is_success:
                    raise ProviderError("browser-use", response.status_code)
                session = response.json()
            if not session.get("cdpUrl"):
                raise TimeoutError("Browser Use did not return a CDP connection")
            self.browser = self.playwright.chromium.connect_over_cdp(session["cdpUrl"], timeout=30000)
        else:
            self.browser = self.playwright.chromium.launch(headless=True, args=["--disable-dev-shm-usage", "--no-sandbox"])
        self.context = (self.browser.contexts[0] if self.config["browser"] == "browser-use" and self.browser.contexts
                        else self.browser.new_context(viewport={"width": 1120, "height": 780}))
        self.context.set_default_timeout(15000)
        return self

    @property
    def headers(self):
        return {"X-Browser-Use-API-Key": os.environ["BROWSER_USE_API_KEY"]}

    def close(self):
        errors = []
        for obj in (self.browser, self.playwright):
            if obj:
                try:
                    obj.close() if obj is self.browser else obj.stop()
                except Exception as exc:
                    errors.append(type(exc).__name__)
        if self.remote_id:
            try:
                response = self.client.patch(BROWSER_API + "/" + self.remote_id, headers=self.headers,
                                             json={"action": "stop"})
                stopped = response.is_success
                write_json(self.root / "browser-session.json", dict(id=self.remote_id, stopped=stopped,
                                                                    stop_http_status=response.status_code))
                if not stopped:
                    errors.append("RemoteStopFailed")
            except Exception as exc:
                errors.append(type(exc).__name__)
        if self.client:
            self.client.close()
        return errors


@contextmanager
def adapt_agent(session, transport, config):
    # Browser Harness initializes an IPC scratch directory during import even
    # though this adapter never starts its daemon. Keep it out of the user's HOME.
    os.environ.setdefault("BROWSER_HARNESS_HOME", str(Path(tempfile.gettempdir()) / f"openwebrl-jev-{os.getuid()}-{os.getpid()}"))
    import jev_ultrafast.agent as agent_module
    import jev_ultrafast.browser as browser_module
    import jev_ultrafast.model as model_module
    originals = (agent_module.Browser, agent_module.MAX_STEPS, browser_module.cdp, model_module.post_json)

    class CDPBrowser(browser_module.Browser):
        def __init__(self, url):
            self.page = session.context.new_page()
            self.session = session.context.new_cdp_session(self.page)
            self.target = True
            self.call("Emulation.setDeviceMetricsOverride", width=1120, height=780, deviceScaleFactor=1, mobile=False)
            self.call("Emulation.setFocusEmulationEnabled", enabled=True)
            response = self.call("Page.navigate", url=url)
            if response.get("errorText"):
                raise RuntimeError("Initial navigation failed")
            end = time.monotonic() + 15
            while time.monotonic() < end:
                if self.evaluate("document.readyState") == "complete":
                    break
                time.sleep(0.02)

        def close(self):
            if self.target:
                self.page.close()
                self.target = None

        def act(self, action, page, text=None):
            try:
                return super().act(action, page, text=text)
            except (browser_module.StalePage, RuntimeError) as exc:
                # Upstream tick swallows StalePage. Preserve its exact code-owned
                # cause, without recording arbitrary CDP exception bodies/secrets.
                message = str(exc)
                known = ("Page changed since this decision. Observe again.",
                         "Target changed or is covered. Observe again.",
                         "Dropdown execution was not confirmed; inspect before retrying.",
                         "Dropdown execution was interrupted; inspect before retrying.")
                append_json(transport.root / "execution-rejections.jsonl", dict(
                    action=action, fingerprint=page["fingerprint"], error_type=type(exc).__name__,
                    reason=message if message in known else "CDP execution failed", updated_unix=time.time()))
                raise

    def cdp(method, session_id=None, **params):
        if session_id is None:
            raise ValueError("CDP operation requires the owned task session")
        return session_id.send(method, params)

    agent_module.Browser = CDPBrowser
    agent_module.MAX_STEPS = config["max_steps"]
    browser_module.cdp = cdp
    model_module.post_json = transport.post
    try:
        from openwebrl.jev_harness import revision
        with revision(browser_module, config.get("harness_revision", "upstream-v1")):
            yield agent_module.Agent
    finally:
        agent_module.Browser, agent_module.MAX_STEPS, browser_module.cdp, model_module.post_json = originals


def save_snapshot(root, index, snapshot):
    target = Path(root) / f"state-{index:04d}.json.gz"
    with gzip.open(target.with_suffix(".tmp"), "wt") as handle:
        json.dump(snapshot, handle, ensure_ascii=False)
    target.with_suffix(".tmp").replace(target)
    screenshot = snapshot.get("page", {}).get("screenshot")
    if screenshot:
        (Path(root) / "final.jpg").write_bytes(base64.b64decode(screenshot, validate=True))


def stop_recorded_session(root):
    """Controller fallback after a worker dies; only its recorded session is touched."""
    marker = Path(root) / "browser-session.json"
    if not marker.exists():
        return []
    record = json.loads(marker.read_text())
    if record.get("stopped"):
        return []
    import httpx
    session = BrowserSession(root, {})
    session.remote_id = record["id"]
    session.client = httpx.Client(timeout=30, follow_redirects=False)
    return session.close()


def run_task(task, root, config):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "task.json", task)
    transport = ModelTransport(root, config)
    session = BrowserSession(root, config)
    agent, snapshot = None, None
    terminal, error, error_stage, provider_blocked = "error", None, "browser_start", False
    started = time.monotonic()
    write_json(root / "heartbeat.json", dict(stage="browser_start", updated_unix=time.time()))
    try:
        with deadline(config["task_timeout_seconds"]):
            session.open()
            with adapt_agent(session, transport, config) as Agent:
                agent = Agent(task["start_url"], task["intent"], screenshots=True)
                snapshot = agent.snapshot()
                save_snapshot(root, 0, snapshot)
                error_stage = "actor"
                for step in range(1, config["max_decisions"] + 1):
                    # A stale decision consumes a call but no browser action.
                    if len(agent.state["history"]) >= config["max_steps"]:
                        terminal = "max_steps"
                        break
                    snapshot = agent.command("tick")
                    save_snapshot(root, step, snapshot)
                    write_json(root / "heartbeat.json", dict(stage="actor", updated_unix=time.time(),
                                                             actions=len(snapshot["history"]), decisions=step))
                    if snapshot["status"] in {"done", "blocked"}:
                        terminal = snapshot["status"]
                        break
                else:
                    terminal = "max_decisions"
    except EpisodeTimeout as exc:
        terminal, error = "task_timeout", type(exc).__name__
    except Exception as exc:
        error = type(exc).__name__
        if isinstance(exc, ProviderError):
            error = str(exc)  # This class never includes bodies, URLs or credentials.
            provider_blocked = True
    finally:
        try:
            if agent:
                # Preserve executed actions even when the post-action observation failed.
                snapshot = agent.snapshot()
                try:
                    final_image = agent.browser.page.screenshot(type="jpeg", quality=72, timeout=10000)
                    snapshot["page"]["screenshot"] = base64.b64encode(final_image).decode()
                    snapshot["final_screenshot_fresh"] = True
                except Exception:
                    # Never judge a pre-action screenshot as the final outcome.
                    snapshot["page"].pop("screenshot", None)
                    snapshot["final_screenshot_fresh"] = False
                save_snapshot(root, config["max_decisions"] + 1, snapshot)
        finally:
            cleanup_errors = session.close()
        if cleanup_errors:
            provider_blocked = True
    actor_seconds = time.monotonic() - started
    history = snapshot.get("history", []) if snapshot else []
    screenshot = snapshot.get("page", {}).get("screenshot") if snapshot else None
    write_json(root / "trajectory.json", dict(task=task, terminal=terminal, history=history,
        decisions=snapshot.get("decisions", []) if snapshot else [],
        text_calls=snapshot.get("text_calls", []) if snapshot else [], error=error,
        actor_seconds=actor_seconds, cleanup_errors=cleanup_errors))
    score, judge_text, judge_error = None, "", None
    # Schema/helper/executor limitations are real actor failures, not site exclusions.
    actor_valid = screenshot is not None and not provider_blocked
    if actor_valid:
        write_json(root / "heartbeat.json", dict(stage="judge", updated_unix=time.time()))
        messages = judge_messages(task, history, screenshot, terminal)
        judge_request = dict(model="o4-mini", seed=42, messages=messages,
                             max_completion_tokens=config.get("judge_max_completion_tokens", 4096))
        write_json(root / "judge-request.json", judge_request)
        try:
            response = transport.request("judge", config["judge_base_url"] + "/chat/completions",
                os.environ["JUDGE_API_KEY"], judge_request,
                attempts=4, timeout=120)
            write_json(root / "judge-response.json", response)
            judge_text = response["choices"][0]["message"]["content"] or ""
            score = parse_verdict(judge_text)
            if score is None:
                judge_error = "unparseable_verdict"
        except Exception as exc:
            judge_error = str(exc) if isinstance(exc, ProviderError) else type(exc).__name__
            provider_blocked = isinstance(exc, ProviderError)
    else:
        judge_error = "missing_browser_evidence" if screenshot is None else "provider_error"
    record = dict(task_id=task["task_id"], completed=True, valid=actor_valid and score is not None,
        score=score, terminal=terminal, actor_error=error, error_stage=error_stage,
        judge_model="o4-mini", judge_prompt_variant="agenttrek", judge_text=judge_text,
        judge_error=judge_error, provider_blocked=provider_blocked, cleanup_errors=cleanup_errors,
        actor_seconds=actor_seconds, total_seconds=time.monotonic() - started,
        actions=len(history), decisions=len(snapshot.get("decisions", [])) if snapshot else 0,
        decision_provider=config.get("decision_provider", "jev"),
        decision_checkpoint=config.get("kev", {}).get("run"),
        api_attempts=transport.counts, requested_jev_model=config["jev_model"],
        resolved_jev_models=sorted({d["model"] for d in snapshot.get("decisions", [])}) if snapshot else [])
    write_json(root / "result.json", record)
    write_json(root / "heartbeat.json", dict(stage="complete", updated_unix=time.time()))
    transport.client.close()
    return record


def summarize(tasks, output, decision_provider="jev"):
    providers = (decision_provider, "text", "judge")
    rows, missing, usage = [], [], {p: {} for p in providers}
    for task in tasks:
        root = Path(output) / "tasks" / digest(task["task_id"])
        path = root / "result.json"
        if not path.exists():
            missing.append(task["task_id"])
            continue
        row = json.loads(path.read_text())
        if row["task_id"] != task["task_id"] or not (root / "trajectory.json").exists():
            raise ValueError("Result identity or trajectory archive mismatch")
        if row["valid"] and (not (root / "final.jpg").exists() or not row["judge_text"]):
            raise ValueError("Valid result lacks screenshot or verdict evidence")
        rows.append(row)
        responses = root / "api-responses.jsonl"
        if responses.exists():
            for line in responses.read_text().splitlines():
                response = json.loads(line)
                for key, value in response.get("response", {}).get("usage", {}).items():
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        totals = usage[response["provider"]]
                        totals[key] = totals.get(key, 0) + value
    valid = sum(bool(r["valid"]) for r in rows)
    successes = sum(r["valid"] and r["score"] == 1 for r in rows)
    blocked = any(r["provider_blocked"] for r in rows)
    return dict(planned=len(tasks), completed=len(rows), valid=valid, invalid=len(rows) - valid,
                successes=successes, overall_success_rate=successes / len(tasks),
                valid_only_success_rate=successes / valid if valid else None,
                missing_task_ids=missing, all_task_attempts_saved=not missing,
                verified_complete=not missing and not blocked, provider_blocked=blocked,
                usage=usage,
                mean_actor_seconds=statistics.mean(r["actor_seconds"] for r in rows) if rows else None,
                median_actor_seconds=statistics.median(r["actor_seconds"] for r in rows) if rows else None,
                api_attempts={p: sum(r["api_attempts"].get(p, 0) for r in rows) for p in providers},
                updated_unix=time.time())
