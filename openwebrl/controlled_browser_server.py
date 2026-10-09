"""Opt-in child server for the proposed controlled local-browser study.

Importing this file starts nothing. The existing local-process transport can
launch this module only with a pinned source hash. Runtime additionally requires
an explicitly supplied, hash-pinned browser manifest. Training defaults do not
import this extension.
"""
from __future__ import annotations

import asyncio
import base64
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import time

# The server retains its 15-second capture deadline. Receiving the serialized
# evidence and its durable receipts needs a separate, bounded transport margin.
OBSERVATION_RPC_TIMEOUT_SECONDS = 30
TERMINAL_CAPTURE_TIMEOUT_SECONDS = 35

PAGE_STATE_JS = """() => {
  const chars = Array.from(document.body?.innerText || '');
  return {url: location.href, title: document.title,
          text: chars.slice(0, 16000).join(''), text_characters: chars.length};
}"""


class ObservationMismatch(ValueError):
    pass


def operation_timeout_seconds(action):
    timeout = 60 if action["name"] in {"goto_url", "go_back", "new_tab"} else 30
    if action["name"] == "wait":
        try:
            # WebEnv waits int(seconds), default 3. Its deliberate delay must
            # finish before our watchdog; the outer RPC/episode caps still apply.
            seconds = int(action.get("args", {}).get("seconds", 3))
            timeout = max(timeout, seconds + 5)
        except (AttributeError, TypeError, ValueError, OverflowError):
            # Let the normal tool implementation report malformed arguments.
            pass
    return timeout


def transient_capture_error(exc):
    """Only the two observed navigation/capture races permit another capture."""
    if not type(exc).__module__.startswith("playwright."):
        return False
    return str(exc).startswith((
        "Page.screenshot: Protocol error (Page.captureScreenshot): Unable to capture screenshot",
        "Page.evaluate: Execution context was destroyed, most likely because of a navigation",
    ))


def validate_browser_manifest(manifest):
    expected = dict(viewport=[1280, 720], device_scale_factor=1, locale="en-US",
                    timezone="UTC", headless=True, proxy=False, stealth=False)
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError("Controlled browser manifest differs from the shared protocol")
    flags = manifest.get("launch_extra_args")
    if not isinstance(flags, list) or not all(isinstance(flag, str) for flag in flags):
        raise ValueError("Explicit pinned Chromium launch flags are required")
    artifacts = Path(manifest["artifact_directory"])
    if not artifacts.is_absolute() or artifacts.is_relative_to(Path(__file__).resolve().parents[1]):
        raise ValueError("Browser artifacts require an absolute private directory outside the repository")
    binary = Path(manifest["binary_path"])
    if hashlib.sha256(binary.read_bytes()).hexdigest() != manifest.get("binary_sha256"):
        raise ValueError("Controlled Chromium binary hash mismatch")
    return deepcopy(manifest)


def controlled_environment(base_class, manifest):
    """Create the opt-in subclass; ``base_class`` is WebEnv in the child only."""
    frozen = deepcopy(manifest)

    class ControlledWebEnv(base_class):
        def __init__(self, **kwargs):
            if (kwargs.get("width"), kwargs.get("height"), kwargs.get("dpr")) != (1280, 720, 1):
                raise ValueError("Controlled environment requires the fixed shared viewport")
            if not kwargs.get("policy", "").strip():
                raise ValueError("Controlled browser policy must not be empty")
            if kwargs.get("resize_output_coords") is not True or kwargs.get("resize_scale") != 1000:
                raise ValueError("SFT actions require the normalized-1000 coordinate contract")
            super().__init__(**kwargs)
            self.controlled_actions = []
            self.controlled_reason = None
            self.controlled_sequence = 0
            self.controlled_invalid = False
            self.controlled_capture_attempts = 0
            from openwebrl.controlled_sft_worker import AttemptJournal
            self.controlled_journal = AttemptJournal(frozen["artifact_directory"])
            self.browser_args = list(frozen["launch_extra_args"])
            self.timeout, self.init_navigation_timeout, self.screenshot_timeout = 30000, 60000, 15000

        async def setup(self):
            # Keep imports and actual browser work outside offline preparation.
            from playwright.async_api import async_playwright
            from openwebrl.env.browser_runtime import browser_process_environment
            validate_browser_manifest(frozen)
            self.playwright = await async_playwright().start()
            self.browser_type = self.playwright.chromium
            self.browser = await self.browser_type.launch(headless=True,
                executable_path=frozen["binary_path"], args=self.browser_args,
                env=browser_process_environment(self.browser_args))

        async def _initialize_context(self, enable_recording=False, start_url="about:blank", auth_info=None):
            if enable_recording or auth_info:
                raise ValueError("Controlled episodes require a fresh unauthenticated profile")
            if self.context is not None:
                raise ValueError("Do not reuse a controlled browser context across episodes")
            self.context = await self.browser.new_context(viewport={"width": 1280, "height": 720},
                device_scale_factor=1, is_mobile=False, locale="en-US", timezone_id="UTC")
            self.context.set_default_timeout(30000)
            for url in start_url.split(" |AND| "):
                page = await self.context.new_page()
                await page.goto(url, timeout=60000, wait_until="domcontentloaded")
            self.page = self.context.pages[0]
            await self.page.bring_to_front()
            self.setup_global_page_listener()
            self.setup_dialog_interceptor()

        async def controlled_observation(self):
            # Initial hydration and delayed overlays can invalidate one capture.
            # Retry only observation, retaining every rejected image; the same
            # exact consistency rule and one15-second deadline cover all arms.
            started = time.monotonic()
            for attempt in range(3):
                remaining = 15 - (time.monotonic() - started)
                if remaining <= 0:
                    raise ObservationMismatch("Shared capture exceeded its15-second deadline")
                try:
                    return await asyncio.wait_for(self._capture_once(), remaining)
                except Exception as exc:
                    if not isinstance(exc, ObservationMismatch):
                        if not transient_capture_error(exc):
                            raise
                        self.controlled_journal("capture_error", dict(
                            capture_attempt=self.controlled_capture_attempts,
                            error_type=type(exc).__name__, message=str(exc)[:4096],
                            elapsed_seconds=time.monotonic() - started))
                    if attempt == 2:
                        raise
                    remaining = 15 - (time.monotonic() - started)
                    if remaining <= 0:
                        raise
                    await asyncio.sleep(min(.2, remaining))

        async def _capture_once(self):
            started = time.monotonic()
            self.controlled_capture_attempts += 1
            page = self.page
            before = await page.evaluate(PAGE_STATE_JS)
            image = await asyncio.wait_for(self.get_screenshot(), 15)
            image_artifact = self.controlled_journal.blob(
                f"capture-{self.controlled_capture_attempts:05d}.image", image)
            elements = await self.get_a11ytree()
            tabs = await self.get_all_tab_urls()
            size = await self.get_screen_size()
            after = await page.evaluate(PAGE_STATE_JS)
            if (page is not self.page or before != after or not image
                    or list(size) != [1280, 720] or page.url != after.get("url")):
                self.controlled_journal("capture_rejected", dict(image=image_artifact,
                    before=before, after=after, reason="page_changed_during_capture"))
                raise ObservationMismatch("Page changed during the shared DOM/screenshot capture")
            self.controlled_sequence += 1
            shared = dict(after, interactive_elements=elements, tabs=tabs, screen_size=list(size))
            self.controlled_journal("capture_verified", dict(sequence=self.controlled_sequence,
                image=image_artifact, page=shared, action_attempts=len(self.controlled_actions)))
            return dict(screenshot=image, a11ytree=elements, screen_size=list(size),
                all_tab_url=tabs, active_tab_url=after["url"], selection_page=shared,
                controlled=dict(sequence=self.controlled_sequence,
                    screenshot_sha256=hashlib.sha256(image).hexdigest(),
                    page_sha256=hashlib.sha256(json.dumps(shared, sort_keys=True,
                        ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()).hexdigest(),
                    capture_started_monotonic=started, capture_finished_monotonic=time.monotonic(),
                    action_attempts=len(self.controlled_actions),
                    executed_actions=deepcopy(self.controlled_actions),
                    terminal_reason=self.controlled_reason, infrastructure_invalid=self.controlled_invalid))

        async def reset(self, url=None, auth_info=None):
            await self._initialize_context(start_url=url or self.start_url, auth_info=auth_info)
            observation = await self.controlled_observation()
            return observation, dict(tool_list=self.tool_list, policy=self.policy,
                env_message="Controlled browser reset", tool_responses=[])

        async def step(self, action_list):
            if self.controlled_reason is not None:
                raise ValueError("Controlled episode already reached a terminal state")
            if not isinstance(action_list, list) or not action_list:
                raise ValueError("Expected at least one dispatched browser operation")
            responses = []
            # Expand compound calls and count actual dispatched operations.
            for action in action_list:
                if len(self.controlled_actions) >= 30:
                    self.controlled_reason = "action_limit"
                    break
                if not isinstance(action, dict) or not isinstance(action.get("name"), str):
                    raise ValueError("Malformed browser operation")
                entry = dict(action=deepcopy(action), attempt=len(self.controlled_actions) + 1,
                             status="dispatched")
                self.controlled_actions.append(entry)
                self.controlled_journal("operation_dispatched", deepcopy(entry))
                timeout = operation_timeout_seconds(action)
                try:
                    if action["name"] == "call_user":
                        flag, message = False, "Autonomous episode cannot obtain additional user input."
                        self.controlled_reason = "requires_user_input"
                    else:
                        flag, message = await asyncio.wait_for(
                            super().execute_single_action(deepcopy(action)), timeout)
                    entry.update(status="returned", success=bool(flag), tool_response=message)
                except Exception as exc:
                    entry.update(status="error", error_type=type(exc).__name__)
                    flag, message = False, "Browser operation failed: " + type(exc).__name__
                    self.controlled_reason, self.controlled_invalid = "browser_operation_error", True
                self.controlled_journal("operation_finished", deepcopy(entry))
                responses.append(dict(tool_name=action["name"], tool_response=message))
                if action["name"] == "done" and flag:
                    self.controlled_reason = "done"
                if len(self.controlled_actions) >= 30 and self.controlled_reason is None:
                    self.controlled_reason = "action_limit"
                if self.controlled_reason is not None:
                    break
            observation = await self.controlled_observation()
            return observation, 0., self.controlled_reason == "done", self.controlled_reason not in (None, "done"), dict(
                tool_list=self.tool_list, policy=self.policy, tool_responses=responses,
                env_message=self.controlled_reason or "Controlled browser actions executed")

    return ControlledWebEnv


async def fresh_observation(client):
    """Use the existing owned local RPC client without dispatching a fake action."""
    data = await client._post("/observe", {"env_id": 0}, timeout_secs=OBSERVATION_RPC_TIMEOUT_SECONDS)
    observation = data["observation"]
    observation["screenshot"] = client._decode_screenshot(observation.get("screenshot"))
    return observation


def install_child_server(server, manifest):
    """Bind only this child server instance; never patch a parent worker's env."""
    server.WebEnv = controlled_environment(server.WebEnv, manifest)

    async def observe(request):
        async with server._lock:
            if server._env is None:
                raise server.HTTPException(400, "Controlled environment is not initialized")
            observation = await server._env.controlled_observation()
            return {"observation": server._serialize_observation(observation)}

    # A runtime type avoids an unresolved future-annotation reference to the
    # locally supplied server module when FastAPI inspects the endpoint.
    observe.__annotations__["request"] = server.EnvIdRequest
    server.app.post("/observe")(observe)


def main():
    path = Path(os.environ["OPENWEBRL_CONTROLLED_BROWSER_MANIFEST"])
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != os.environ["OPENWEBRL_CONTROLLED_BROWSER_MANIFEST_SHA256"]:
        raise ValueError("Controlled browser manifest hash mismatch")
    manifest = validate_browser_manifest(json.loads(raw))
    from openwebrl.docker import env_server
    install_child_server(env_server, manifest)
    artifacts = Path(manifest["artifact_directory"])
    artifacts.mkdir(parents=True, exist_ok=True)
    proc_stat = Path("/proc/self/stat").read_text().split(") ", 1)[1].split()
    ownership = dict(pid=os.getpid(), pgid=os.getpgid(0), parent_pid=os.getppid(), uid=os.getuid(),
        proc_start_ticks=int(proc_stat[19]), job_id=os.getenv("SLURM_JOB_ID"),
        manifest_sha256=hashlib.sha256(raw).hexdigest(), created_unix=time.time())
    with (artifacts / "server-process.json").open("x") as handle:
        json.dump(ownership, handle, sort_keys=True)
        handle.flush(); os.fsync(handle.fileno())
    try:
        env_server.main()
    finally:
        with (artifacts / "server-process-exited.json").open("x") as handle:
            json.dump(dict(ownership, exited_unix=time.time()), handle, sort_keys=True)



if __name__ == "__main__":
    main()
