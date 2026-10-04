import asyncio
import time
import unittest
from types import SimpleNamespace

from aiohttp import web

from openwebrl.env.local_process_env import _wait_until_healthy


class LocalBrowserHealthTest(unittest.IsolatedAsyncioTestCase):
    async def serve(self, handler):
        app = web.Application()
        app.router.add_get('/health', handler)
        runner = web.AppRunner(app, shutdown_timeout=0.05)
        await runner.setup()
        self.addAsyncCleanup(runner.cleanup)
        site = web.TCPSite(runner, '127.0.0.1', 0)
        await site.start()
        return f'http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}'

    async def test_healthy_response_can_take_more_than_one_second(self):
        async def health(request):
            await asyncio.sleep(1.2)
            return web.Response(text='ok')

        url = await self.serve(health)
        await _wait_until_healthy(url, SimpleNamespace(returncode=None),
                                  startup_timeout_secs=3, startup_poll_secs=0.01)

    async def test_unresponsive_server_respects_overall_deadline(self):
        async def health(request):
            await asyncio.Event().wait()

        url = await self.serve(health)
        start = time.monotonic()
        with self.assertRaisesRegex(TimeoutError, 'last_error=TimeoutError'):
            await _wait_until_healthy(url, SimpleNamespace(returncode=None),
                                      startup_timeout_secs=0.1, startup_poll_secs=0.01)
        self.assertLess(time.monotonic() - start, 0.5)

    async def test_server_can_become_ready_after_service_unavailable(self):
        calls = 0

        async def health(request):
            nonlocal calls
            calls += 1
            return web.Response(status=503 if calls == 1 else 200)

        url = await self.serve(health)
        await _wait_until_healthy(url, SimpleNamespace(returncode=None),
                                  startup_timeout_secs=1, startup_poll_secs=0.01)
        self.assertGreaterEqual(calls, 2)

    async def test_exited_child_is_reported_without_waiting(self):
        with self.assertRaisesRegex(RuntimeError, 'exited early with code 7'):
            await _wait_until_healthy('http://127.0.0.1:1', SimpleNamespace(returncode=7),
                                      startup_timeout_secs=1, startup_poll_secs=0.01)

    async def test_cancellation_propagates(self):
        started = asyncio.Event()

        async def health(request):
            started.set()
            await asyncio.Event().wait()

        url = await self.serve(health)
        task = asyncio.create_task(_wait_until_healthy(
            url, SimpleNamespace(returncode=None),
            startup_timeout_secs=3, startup_poll_secs=0.01))
        await asyncio.wait_for(started.wait(), timeout=1)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task


if __name__ == '__main__':
    unittest.main()
