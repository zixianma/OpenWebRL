import asyncio
import threading
import unittest
from unittest.mock import Mock, patch

from openwebrl.env import local_process_env as m


class BrowserLogIOTests(unittest.IsolatedAsyncioTestCase):
    async def test_slow_filesystem_open_does_not_block_event_loop(self):
        entered, release = threading.Event(), threading.Event()
        handle = Mock()

        def slow_open(*args, **kwargs):
            entered.set()
            release.wait(2)
            return handle

        with patch.object(m.os, 'makedirs'), patch('builtins.open', slow_open):
            task = asyncio.create_task(m._open_browser_log('/unused', '/unused/log'))
            try:
                self.assertTrue(await asyncio.to_thread(entered.wait, 1))
                # A separate event-loop callback must run while open is blocked.
                await asyncio.wait_for(asyncio.sleep(.01), timeout=.2)
                self.assertFalse(task.done())
            finally:
                release.set()
            self.assertIs(await task, handle)
        handle.close.assert_not_called()

    async def test_repeated_cancellation_closes_late_handle_once(self):
        entered, release = threading.Event(), threading.Event()
        handle = Mock()

        def slow_open(*args, **kwargs):
            entered.set()
            release.wait(2)
            return handle

        with patch.object(m.os, 'makedirs'), patch('builtins.open', slow_open):
            task = asyncio.create_task(m._open_browser_log('/unused', '/unused/log'))
            try:
                self.assertTrue(await asyncio.to_thread(entered.wait, 1))
                task.cancel()
                await asyncio.sleep(.01)
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
            finally:
                release.set()
            await asyncio.wait_for(asyncio.gather(*list(m._PENDING_CLEANUPS)), timeout=1)
        handle.close.assert_called_once()
        self.assertFalse(m._PENDING_CLEANUPS)

    async def test_open_failure_propagates(self):
        with patch.object(m.os, 'makedirs'), patch('builtins.open', side_effect=OSError('quota')):
            with self.assertRaisesRegex(OSError, 'quota'):
                await m._open_browser_log('/unused', '/unused/log')
        self.assertFalse(m._PENDING_CLEANUPS)


if __name__ == '__main__':
    unittest.main()
