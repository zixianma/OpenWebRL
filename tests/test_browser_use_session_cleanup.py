"""No-network checks for owned Browser Use session lifecycle."""
import asyncio
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch
from openwebrl.env import browser_use_env as m


class SessionCleanupTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix='openwebrl-browseruse-cleanup-'))
        self.patches = [patch.object(m, '_BROWSER_USE_MANIFEST_DIR', str(self.root)),
                        patch.object(m, '_CLEANUP_STARTED', False)]
        for p in self.patches:
            p.start()
            self.addCleanup(p.stop)
        self.client = SimpleNamespace(browsers=SimpleNamespace(stop=AsyncMock(), get=AsyncMock()), close=AsyncMock())

    async def test_delayed_stop_confirms_before_archiving_receipt(self):
        m._save_session_id('owned')
        self.client.browsers.stop.return_value = SimpleNamespace(status=SimpleNamespace(value='active'))
        self.client.browsers.get.return_value = SimpleNamespace(status=SimpleNamespace(value='stopped'))
        with patch.object(m.asyncio, 'sleep', AsyncMock()):
            await m._stop_session(self.client, 'owned')
        self.assertTrue((self.root / 'stopped/owned').exists())
        self.assertEqual(m._list_session_ids(), [])

    async def test_failed_stop_retains_active_receipt(self):
        m._save_session_id('owned')
        self.client.browsers.stop.side_effect = RuntimeError('network')
        with self.assertRaises(RuntimeError):
            await m._stop_session(self.client, 'owned')
        self.assertEqual(m._list_session_ids(), ['owned'])

    async def test_concurrent_initialization_never_sweeps_new_session(self):
        m._save_session_id('old')
        async def stop(sid):
            m._save_session_id('new')
            await asyncio.sleep(0)
            return SimpleNamespace(status='stopped')
        self.client.browsers.stop.side_effect = stop
        with patch('browser_use_sdk.AsyncBrowserUse', return_value=self.client):
            await asyncio.gather(m.cleanup_existing_browser_use_sessions({'api_key': 'fixture'}),
                                 m.cleanup_existing_browser_use_sessions({'api_key': 'fixture'}))
        self.client.browsers.stop.assert_awaited_once_with('old')
        self.assertEqual(m._list_session_ids(), ['new'])
        self.client.close.assert_awaited_once()


if __name__ == '__main__':
    unittest.main()
