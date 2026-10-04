import asyncio
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest

from openwebrl.durable_async_io import log_progress_async, run_durable_io


class DurableIO(unittest.IsolatedAsyncioTestCase):
    async def test_slow_filesystem_does_not_block_health_timers(self):
        started, release = threading.Event(), threading.Event()
        def slow_write():
            started.set()
            if not release.wait(2):
                raise TimeoutError('Heartbeat failed to release simulated I/O')
            return 'written'
        writing = asyncio.create_task(run_durable_io(slow_write))
        try:
            for _ in range(100):
                if started.is_set():
                    break
                await asyncio.sleep(.001)
            self.assertTrue(started.is_set())
            await asyncio.wait_for(asyncio.sleep(.01), .2)
            self.assertFalse(writing.done())
        finally:
            release.set()
        self.assertEqual(await writing, 'written')

    async def test_repeated_cancellation_waits_for_durable_label(self):
        started, release = threading.Event(), threading.Event()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'label.json'
            def write():
                started.set()
                if not release.wait(2):
                    raise TimeoutError('Test did not release writer')
                path.write_text(json.dumps({'selected_index': 3}))
            writing = asyncio.create_task(run_durable_io(write))
            try:
                while not started.is_set():
                    await asyncio.sleep(.001)
                for _ in range(2):
                    writing.cancel()
                    await asyncio.sleep(.01)
                    self.assertFalse(writing.done())
            finally:
                release.set()
            with self.assertRaises(asyncio.CancelledError):
                await writing
            self.assertEqual(json.loads(path.read_text()), {'selected_index': 3})

    async def test_telemetry_writers_are_serialized(self):
        current = {}; active = 0; maximum = 0; completed = 0
        def write(_):
            nonlocal active, maximum, completed
            active += 1; maximum = max(maximum, active)
            time.sleep(.01)
            completed += 1; active -= 1
        await asyncio.gather(*(log_progress_async(current, write) for _ in range(8)))
        self.assertEqual((maximum, completed), (1, 8))

    async def test_write_failure_is_not_hidden(self):
        def fail():
            raise OSError('quota')
        with self.assertRaisesRegex(OSError, 'quota'):
            await run_durable_io(fail)


if __name__ == '__main__':
    unittest.main()
