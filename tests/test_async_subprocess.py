"""Exercise the background loop after SGLang's uvloop policy installation."""
import os
import subprocess
import sys
import textwrap
import unittest


class BackgroundSubprocessTest(unittest.TestCase):
    def test_subprocess_and_cancelled_http_under_uvloop_policy(self):
        code = textwrap.dedent("""\
            import asyncio
            import sys
            import uvloop
            from aiohttp import ClientSession, web
            uvloop.install()
            from slime.utils.async_utils import get_async_loop

            async def exercise():
                assert isinstance(asyncio.get_running_loop(), asyncio.SelectorEventLoop)
                async def child():
                    proc = await asyncio.create_subprocess_exec(
                        sys.executable, '-c', 'print(42)', stdout=asyncio.subprocess.PIPE)
                    stdout, _ = await proc.communicate()
                    assert proc.returncode == 0 and stdout.strip() == b'42'
                await asyncio.gather(*(child() for _ in range(16)))

                entered = 0
                all_entered = asyncio.Event()
                release = asyncio.Event()
                async def handler(request):
                    nonlocal entered
                    entered += 1
                    if entered == 32:
                        all_entered.set()
                    await release.wait()
                    return web.Response(text='done')
                app = web.Application()
                app.router.add_get('/', handler)
                runner = web.AppRunner(app)
                await runner.setup()
                site = web.TCPSite(runner, '127.0.0.1', 0)
                await site.start()
                port = site._server.sockets[0].getsockname()[1]
                try:
                    async with ClientSession() as session:
                        async def request():
                            async with session.get(f'http://127.0.0.1:{port}/') as response:
                                await response.read()
                        tasks = [asyncio.create_task(request()) for _ in range(32)]
                        await asyncio.wait_for(all_entered.wait(), 10)
                        for task in tasks:
                            task.cancel()
                        results = await asyncio.gather(*tasks, return_exceptions=True)
                        assert all(isinstance(r, asyncio.CancelledError) for r in results)
                        release.set()
                        await child()
                finally:
                    release.set()
                    await runner.cleanup()
                print('subprocess and HTTP cancellation passed')

            helper = get_async_loop()
            try:
                helper.run(exercise())
            finally:
                helper.loop.call_soon_threadsafe(helper.loop.stop)
                helper._thread.join(timeout=5)
                assert not helper._thread.is_alive()
                helper.loop.close()
        """)
        env = dict(os.environ, SLIME_ASYNC_USE_STDLIB_LOOP='1')
        result = subprocess.run([sys.executable, '-c', code], env=env,
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('subprocess and HTTP cancellation passed', result.stdout)


if __name__ == '__main__':
    unittest.main()
