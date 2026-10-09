import asyncio
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from openwebrl.controlled_browser_server import (
    ObservationMismatch, controlled_environment, install_child_server, validate_browser_manifest,
)
from openwebrl.env.local_process_env import _server_module
from openwebrl.env import local_process_env
from openwebrl.controlled_sft_worker import validate_observation


def manifest(tmp_path):
    binary = tmp_path / "fixture-chromium"  # Hash fixture only; never executed.
    binary.write_bytes(b"not-an-executable")
    return dict(viewport=[1280, 720], device_scale_factor=1, locale="en-US", timezone="UTC",
        headless=True, proxy=False, stealth=False, launch_extra_args=["--no-sandbox"],
        binary_path=str(binary), binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
        artifact_directory=str(tmp_path / "browser-artifacts"))


class FakePage:
    url = "https://example.com"
    def __init__(self): self.reads = 0; self.unstable = False
    async def evaluate(self, script):
        self.reads += 1
        return dict(url=self.url, title="Example", text="Changed" if self.unstable and self.reads % 2 == 0 else "Search",
                    text_characters=6)


class FakeWebEnv:
    def __init__(self, **kwargs):
        self.page = FakePage(); self.context = None; self.dispatched = []
        self.tool_list = kwargs.get("tool_list", []); self.policy = kwargs["policy"]
        self.start_url = "https://example.com"
    async def get_screenshot(self): return b"fixture-image"
    async def get_a11ytree(self): return [dict(textContent="Search", bbox=[.1, .1, .2, .2])]
    async def get_all_tab_urls(self):
        return [dict(url=self.page.url, title="Example", index=0, active=True)]
    async def get_screen_size(self): return (1280, 720)
    async def execute_single_action(self, action):
        self.dispatched.append(deepcopy(action))
        action.setdefault("args", {})["mutated_by_base"] = True
        return (action["name"] != "failed_action", "tool feedback")


def env(tmp_path):
    value = manifest(tmp_path)
    # Every environment/attempt owns a fresh private evidence directory.
    value["artifact_directory"] = str(tmp_path / f"browser-{len(list(tmp_path.glob('browser-*')))}")
    cls = controlled_environment(FakeWebEnv, value)
    return cls(width=1280, height=720, dpr=1, policy="Browser policy",
               resize_output_coords=True, resize_scale=1000)


def test_default_server_module_is_unchanged_and_extension_requires_its_source_hash():
    assert _server_module({}) == "openwebrl.docker.env_server"
    assert _server_module({"server_module": "openwebrl.docker.env_server"}) == "openwebrl.docker.env_server"
    path = Path(__file__).parents[1] / "openwebrl/controlled_browser_server.py"
    config = dict(server_module="openwebrl.controlled_browser_server",
                  server_module_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    assert _server_module(config) == config["server_module"]
    for bad in ({"server_module": "arbitrary.module"},
                {"server_module": config["server_module"]},
                dict(config, server_module_sha256="0" * 64)):
        with pytest.raises(ValueError): _server_module(bad)


def test_browser_manifest_pins_binary_and_locale_without_executing_it(tmp_path):
    value = manifest(tmp_path)
    assert validate_browser_manifest(value) == value
    for bad in (dict(value, locale="de-DE"), dict(value, viewport=[1120, 780]),
                dict(value, binary_sha256="wrong"), dict(value, proxy=True)):
        with pytest.raises(ValueError): validate_browser_manifest(bad)


def test_child_observation_captures_shared_page_image_and_execution_receipt(tmp_path):
    async def run():
        instance = env(tmp_path)
        first = await instance.controlled_observation()
        second = await instance.controlled_observation()
        assert validate_observation(first)["sequence"] == 1
        assert validate_observation(second)["sequence"] == 2
        assert first["selection_page"]["interactive_elements"] == first["a11ytree"]
        assert first["controlled"]["action_attempts"] == 0
        assert not instance.dispatched
    asyncio.run(run())


def test_observation_does_not_pair_image_with_changed_page_text(tmp_path):
    async def run():
        instance = env(tmp_path); instance.page.unstable = True
        with pytest.raises(ObservationMismatch): await instance.controlled_observation()
        assert instance.controlled_capture_attempts == 3
        assert instance.controlled_sequence == 0
        records = [json.loads(path.read_text()) for path in instance.controlled_journal.directory.glob('*.json')]
        assert len(records) == 3 and all(row['event'] == 'capture_rejected' for row in records)
    asyncio.run(run())


def test_initial_hydration_retries_only_capture_and_preserves_rejected_image(tmp_path):
    async def run():
        instance = env(tmp_path)
        reads = 0
        async def hydrate(script):
            nonlocal reads
            reads += 1
            text = 'Loading' if reads == 1 else 'Settled page'
            return dict(url=instance.page.url,title='Example',text=text,text_characters=len(text))
        instance.page.evaluate = hydrate
        result = await instance.controlled_observation()
        assert validate_observation(result)['sequence'] == 1
        assert instance.controlled_capture_attempts == 2 and not instance.dispatched
        rows = [json.loads(path.read_text()) for path in sorted(instance.controlled_journal.directory.glob('*.json'))]
        assert [row['event'] for row in rows] == ['capture_rejected','capture_verified']
        assert rows[0]['record']['before']['text'] == 'Loading'
        assert result['selection_page']['text'] == 'Settled page'
    asyncio.run(run())


@pytest.mark.parametrize("message", [
    "Page.screenshot: Protocol error (Page.captureScreenshot): Unable to capture screenshot",
    "Page.evaluate: Execution context was destroyed, most likely because of a navigation",
])
def test_observed_capture_transients_retry_without_repeating_an_operation(tmp_path, message):
    from playwright.async_api import Error
    async def run():
        instance = env(tmp_path)
        original = instance.get_screenshot
        calls = 0
        async def flaky_capture():
            nonlocal calls
            calls += 1
            if calls == 1:
                raise Error(message)
            return await original()
        instance.get_screenshot = flaky_capture
        result = await instance.step([{"name": "click", "args": {"point_2d": [10, 20]}}])
        assert validate_observation(result[0])["action_attempts"] == 1
        assert len(instance.dispatched) == 1 and calls == 2
        rows = [json.loads(p.read_text()) for p in sorted(instance.controlled_journal.directory.glob('*.json'))]
        errors = [r['record'] for r in rows if r['event'] == 'capture_error']
        assert len(errors) == 1 and errors[0]['message'] == message
    asyncio.run(run())


def test_transient_capture_retries_are_bounded_and_unknown_errors_fail_closed(tmp_path):
    from playwright.async_api import Error
    async def run():
        instance = env(tmp_path)
        instance.get_screenshot = AsyncMock(side_effect=Error(
            "Page.screenshot: Protocol error (Page.captureScreenshot): Unable to capture screenshot"))
        with pytest.raises(Error):
            await instance.controlled_observation()
        assert instance.get_screenshot.await_count == 3
        assert instance.controlled_sequence == 0 and not instance.dispatched
        records = [json.loads(p.read_text()) for p in instance.controlled_journal.directory.glob('*.json')]
        assert len(records) == 3 and all(r['event'] == 'capture_error' for r in records)
        other = env(tmp_path)
        other.get_screenshot = AsyncMock(side_effect=Error("Target page, context or browser has been closed"))
        with pytest.raises(Error):
            await other.controlled_observation()
        assert other.get_screenshot.await_count == 1
    asyncio.run(run())


def test_transient_capture_does_not_reset_total_deadline(tmp_path, monkeypatch):
    from playwright.async_api import Error
    from openwebrl import controlled_browser_server as module
    async def run():
        instance = env(tmp_path)
        now = 0.0
        monkeypatch.setattr(module.time, 'monotonic', lambda: now)
        # Replace only _capture_once; no running event-loop time is advanced.
        async def expensive_capture():
            nonlocal now
            now = 16.0
            raise Error("Page.screenshot: Protocol error (Page.captureScreenshot): Unable to capture screenshot")
        instance._capture_once = AsyncMock(side_effect=expensive_capture)
        with pytest.raises(Error):
            await instance.controlled_observation()
        assert instance._capture_once.await_count == 1
    asyncio.run(run())


def test_compound_operations_cannot_overrun_thirty_and_failed_actions_count(tmp_path):
    async def run():
        instance = env(tmp_path)
        actions = [{"name": "failed_action", "args": {}} for _ in range(35)]
        result = await instance.step(actions)
        assert len(instance.dispatched) == 30
        assert result[2] is False and result[3] is True
        assert result[0]["controlled"]["terminal_reason"] == "action_limit"
        assert all(entry["success"] is False for entry in instance.controlled_actions)
        assert all(action["args"] == {} for action in actions)
        assert all(entry["action"]["args"] == {} for entry in instance.controlled_actions)
        with pytest.raises(ValueError): await instance.step([{"name": "wait", "args": {}}])
    asyncio.run(run())


def test_explicit_wait_finishes_before_watchdog_and_is_counted_once(tmp_path, monkeypatch):
    from openwebrl import controlled_browser_server as module
    original_wait_for = asyncio.wait_for
    original_sleep = asyncio.sleep

    async def accelerated_wait_for(awaitable, timeout):
        return await original_wait_for(awaitable, timeout / 1000)

    async def delayed_action(self, action):
        self.dispatched.append(deepcopy(action))
        await original_sleep(int(action['args']['seconds']) / 1000)
        return True, 'wait finished'

    monkeypatch.setattr(module.asyncio, 'wait_for', accelerated_wait_for)
    monkeypatch.setattr(FakeWebEnv, 'execute_single_action', delayed_action)

    async def run():
        instance = env(tmp_path)
        result = await instance.step([dict(name='wait', args={'seconds': 30})])
        assert not result[0]['controlled']['infrastructure_invalid']
        assert result[0]['controlled']['terminal_reason'] is None
        assert len(instance.dispatched) == 1
        assert instance.controlled_actions[0]['status'] == 'returned'
        assert instance.controlled_actions[0]['success'] is True
    asyncio.run(run())


def test_wait_watchdog_keeps_nonwait_limits_and_handles_malformed_arguments():
    from openwebrl.controlled_browser_server import operation_timeout_seconds as timeout
    assert timeout(dict(name='wait', args={'seconds': '60'})) == 65
    assert timeout(dict(name='wait', args={'seconds': 30.9})) == 35
    for args in ({}, None, {'seconds': 'invalid'}, {'seconds': float('inf')}):
        assert timeout(dict(name='wait', args=args)) == 30
    assert timeout(dict(name='click', args={'seconds': 90})) == 30
    assert timeout(dict(name='goto_url', args={})) == 60


def test_done_stops_compound_sequence_and_call_user_does_not_wait_for_stdin(tmp_path):
    async def run():
        instance = env(tmp_path)
        result = await instance.step([{"name": "done", "args": {}}, {"name": "click", "args": {}}])
        assert len(instance.dispatched) == 1 and result[2] is True
        other = env(tmp_path)
        result = await other.step([{"name": "call_user", "args": {"question": "Which city?"}}])
        assert not other.dispatched and result[2] is False and result[3] is True
        assert result[0]["controlled"]["terminal_reason"] == "requires_user_input"
    asyncio.run(run())


def test_child_install_registers_typed_observe_endpoint_without_global_parent_patch(tmp_path):
    endpoints = {}
    class App:
        def post(self, route):
            def register(function): endpoints[route] = function; return function
            return register
    server = SimpleNamespace(WebEnv=FakeWebEnv, app=App(), EnvIdRequest=type("EnvIdRequest", (), {}))
    install_child_server(server, manifest(tmp_path))
    assert server.WebEnv is not FakeWebEnv
    assert endpoints["/observe"].__annotations__["request"] is server.EnvIdRequest


def test_operation_receipts_survive_failed_final_screenshot(tmp_path):
    async def run():
        instance = env(tmp_path)
        async def broken_image(): raise RuntimeError("Synthetic browser disconnection")
        instance.get_screenshot = broken_image
        with pytest.raises(RuntimeError):
            await instance.step([{"name": "click", "args": {"point_2d": [123, 456]}}])
        records = [json.loads(path.read_text()) for path in instance.controlled_journal.directory.glob("*.json")]
        assert {row["event"] for row in records} == {"operation_dispatched", "operation_finished"}
        assert all(row["record"]["action"]["args"]["point_2d"] == [123, 456] for row in records)
    asyncio.run(run())


@pytest.mark.parametrize("extension", [False, True])
def test_transport_launches_only_default_or_hash_pinned_child_module(monkeypatch, extension):
    async def run():
        m = local_process_env
        pool = m.LocalProcessSlotPool(1)
        lease = Mock(host="127.0.0.1", port=18100)
        process = SimpleNamespace(pid=12345, returncode=0)
        spawn = AsyncMock(return_value=process)
        monkeypatch.setattr(m.LocalProcessSlotPool, "get", AsyncMock(return_value=pool))
        monkeypatch.setattr(m, "_acquire_port", Mock(return_value=lease))
        monkeypatch.setattr(m, "_open_browser_log", AsyncMock(return_value=Mock()))
        monkeypatch.setattr(m, "_wait_until_healthy", AsyncMock())
        monkeypatch.setattr(m.asyncio, "create_subprocess_exec", spawn)
        config = {}
        module = "openwebrl.docker.env_server"
        if extension:
            module = "openwebrl.controlled_browser_server"
            source = Path(__file__).parents[1] / "openwebrl/controlled_browser_server.py"
            config.update(server_module=module, server_module_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
        instance = await m.create_local_process_env(config)
        assert spawn.call_args.args[1:3] == ("-m", module)
        await instance.exit()
        assert pool._active == 0
    asyncio.run(run())
