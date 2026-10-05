"""Real Playwright dispatcher regression, with no browser or network requests."""
import asyncio
import base64
from contextlib import contextmanager
from copy import deepcopy
import gzip
import json
import signal
from types import SimpleNamespace

from greenlet import greenlet
import httpx
from playwright._impl._sync_base import SyncBase

from openwebrl import jev_eval as evaluation


class StalledPage(SyncBase):
    def __init__(self):
        self._loop = asyncio.new_event_loop()
        self._dispatcher_fiber = greenlet(self._loop.run_forever)
        self.screenshot_calls = 0
        self.close_calls = 0

    def hang(self):
        async def pending():
            # Start the short timer only after the dispatcher has entered its
            # event loop; pytest's first inspect.stack() can otherwise use20ms.
            signal.setitimer(signal.ITIMER_REAL, .02)
            await asyncio.sleep(100)
        self._sync(pending())

    def screenshot(self, **kwargs):
        self.screenshot_calls += 1
        return self._sync(asyncio.sleep(0))

    def close(self):
        self.close_calls += 1
        return self._sync(asyncio.sleep(0))

    def dispose_for_test(self):
        for task in asyncio.all_tasks(self._loop):
            task.cancel()
        self._loop.run_until_complete(asyncio.sleep(0))
        self._loop.close()


def test_real_signal_timeout_kills_dispatcher_but_terminal_capture_skips_it():
    page = StalledPage()
    try:
        try:
            with evaluation.deadline(1): page.hang()
        except evaluation.EpisodeTimeout:
            pass
        assert evaluation.playwright_dispatcher_dead(page)
        # Prove the old cleanup pattern cannot run even a zero-duration task.
        try:
            with evaluation.deadline(.1): page.screenshot()
        except evaluation.EpisodeTimeout:
            pass
        else:
            raise AssertionError('Dead dispatcher unexpectedly ran cleanup')
        agent = SimpleNamespace(browser=SimpleNamespace(page=page),
            snapshot=lambda: {'page': {'screenshot': 'old'}, 'history': [{'action': 'preserved'}]})
        result = evaluation.terminal_snapshot(agent)
        assert page.screenshot_calls == 1  # Guarded path did not call it again.
        assert result['history'] == [{'action': 'preserved'}]
        assert not result['final_screenshot_fresh'] and 'screenshot' not in result['page']
        assert result['final_screenshot_error'] == 'playwright_dispatcher_unavailable'
    finally:
        page.dispose_for_test()


def test_timeout_writes_durable_invalid_and_stops_remote_before_local_cleanup(tmp_path, monkeypatch):
    page = StalledPage()
    requests = []
    def respond(request):
        requests.append(request)
        assert page.close_calls == 0
        return httpx.Response(200, json={'status': 'stopped'})
    monkeypatch.setenv('BROWSER_USE_API_KEY', 'offline-test')
    def open_session(self):
        self.browser = self.playwright = page
        self.remote_id = 'owned-offline-session'
        self.client = httpx.Client(transport=httpx.MockTransport(respond))
        return self
    monkeypatch.setattr(evaluation.BrowserSession, 'open', open_session)
    class Agent:
        def __init__(self, *args, **kwargs):
            self.browser = SimpleNamespace(page=page)
            self.state = {'page': {'screenshot': base64.b64encode(b'old image').decode()},
                          'history': [], 'decisions': [], 'text_calls': [], 'status': 'ready'}
        def snapshot(self): return deepcopy(self.state)
        def command(self, name): return page.hang()
    @contextmanager
    def adapter(*args): yield Agent
    monkeypatch.setattr(evaluation, 'adapt_agent', adapter)
    task = {'task_id': 'offline', 'start_url': 'https://example.invalid', 'intent': 'No request is made'}
    config = dict(task_timeout_seconds=1, max_steps=30, max_decisions=60, browser='browser-use',
                  decision_provider='kev', jev_model='kev-latest', kev={'run': 'offline-pinned'})
    try:
        result = evaluation.run_task(task, tmp_path, config)
        assert result['terminal'] == 'task_timeout' and result['actor_error'] == 'EpisodeTimeout'
        assert result['valid'] is False and result['score'] is None
        assert result['judge_error'] == 'missing_browser_evidence'
        assert result['cleanup_errors'] == ['PlaywrightDispatcherUnavailable']
        assert page.close_calls == page.screenshot_calls == 0
        assert len(requests) == 1 and requests[0].method == 'PATCH'
        assert json.loads(requests[0].content) == {'action': 'stop'}
        assert json.loads((tmp_path/'browser-session.json').read_text())['stopped'] is True
        assert (tmp_path/'result.json').exists() and (tmp_path/'trajectory.json').exists()
        assert not (tmp_path/'judge-request.json').exists()
        with gzip.open(tmp_path/'state-0061.json.gz', 'rt') as handle: final = json.load(handle)
        assert final['final_screenshot_fresh'] is False and 'screenshot' not in final['page']
    finally:
        page.dispose_for_test()


def test_live_terminal_capture_retains_fresh_bytes_and_history():
    seen = []
    page = SimpleNamespace(screenshot=lambda **kwargs: seen.append(kwargs) or b'fresh jpeg')
    agent = SimpleNamespace(browser=SimpleNamespace(page=page),
                            snapshot=lambda: {'page': {'screenshot': 'old'}, 'history': ['unchanged']})
    snapshot = evaluation.terminal_snapshot(agent)
    assert snapshot['final_screenshot_fresh'] is True
    assert base64.b64decode(snapshot['page']['screenshot']) == b'fresh jpeg'
    assert snapshot['history'] == ['unchanged']
    assert seen == [{'type': 'jpeg', 'quality': 72, 'timeout': 10000}]


def test_final_capture_has_independent_bound(monkeypatch):
    # Accelerate only the test finalization timer, not production constants.
    real_deadline = evaluation.deadline
    requested = []
    @contextmanager
    def short_deadline(seconds):
        requested.append(seconds)
        with real_deadline(1): yield
    monkeypatch.setattr(evaluation, 'deadline', short_deadline)
    page = StalledPage()
    page.screenshot = lambda **kwargs: page.hang()
    agent = SimpleNamespace(browser=SimpleNamespace(page=page), snapshot=lambda: {'page': {'screenshot': 'old'}})
    try:
        snapshot = evaluation.terminal_snapshot(agent)
        assert requested == [15] and not snapshot['final_screenshot_fresh']
        assert evaluation.playwright_dispatcher_dead(page)
    finally:
        page.dispose_for_test()
