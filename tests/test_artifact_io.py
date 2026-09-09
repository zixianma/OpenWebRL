"""Slow writes must not stall browser tasks or outlive cancelled exporters."""
import asyncio
import base64
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from openwebrl.artifact_io import run_artifact_io
from openwebrl.arm_c2 import TurnExporter, sha


class ArtifactIOTests(unittest.IsolatedAsyncioTestCase):
    async def test_slow_write_yields_and_cancellation_drains_it(self):
        started, release, finished = threading.Event(), threading.Event(), threading.Event()
        def write():
            started.set()
            release.wait(5)
            finished.set()
        task = asyncio.create_task(run_artifact_io(write))
        try:
            for _ in range(100):
                if started.is_set(): break
                await asyncio.sleep(.01)
            self.assertTrue(started.is_set())
            task.cancel()
            await asyncio.sleep(.05)
            self.assertFalse(task.done())
            self.assertFalse(finished.is_set())
        finally:
            release.set()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertTrue(finished.is_set())

    async def test_exporters_share_identical_images_without_partial_reads(self):
        raw = b'content-addressed-image'
        image = 'data:image/png;base64,' + base64.b64encode(raw).decode()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            exporter = TurnExporter(root)
            for i in range(8): exporter.begin(str(i))
            writes = []
            original = Path.write_bytes
            def slow_write(path, data):
                if path.suffix == '.img':
                    import time
                    writes.append(path)
                    time.sleep(.03)
                return original(path, data)
            with patch.object(Path, 'write_bytes', slow_write):
                await asyncio.gather(*(run_artifact_io(exporter,
                    record=dict(task_id=str(i), turn=0), prompt='prefix', images=[image],
                    outputs=[('winner', [1, 2], [], 'stop')]) for i in range(8)))
            self.assertEqual(len(writes), 1)
            for path in root.glob('attempts/*/*/turn-0000.json'):
                captured = json.loads(path.read_text())
                self.assertEqual(captured['raw_candidates'][0]['response_token_ids'], [1, 2])
                self.assertEqual(sha(Path(captured['images'][0]['path']).read_bytes()), sha(raw))
            self.assertEqual(len(list(root.glob('attempts/*/*/turn-0000.json'))), 8)


if __name__ == '__main__': unittest.main()
