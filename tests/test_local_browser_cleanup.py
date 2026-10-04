import asyncio
import gc
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, AsyncMock, patch

spec=importlib.util.spec_from_file_location('local_browser_cleanup',Path(__file__).parents[1]/'openwebrl/env/local_process_env.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class CleanupTests(unittest.IsolatedAsyncioTestCase):
    async def build_env(self):
        env=object.__new__(m.LocalProcessWebEnv)
        env._closed=False;env._broken=True;env._current_task_id='test';env.proc=SimpleNamespace(returncode=None,pid=100)
        env.port_lease=Mock(port=12345);env.log_file=Mock();env.slot_pool=m.LocalProcessSlotPool(1)
        await env.slot_pool.acquire()
        return env

    async def test_cancelled_caller_keeps_cleanup_and_releases_once(self):
        env=await self.build_env();entered=asyncio.Event();finish=asyncio.Event()
        async def terminate():
            entered.set();await finish.wait();env.proc.returncode=0
        env._terminate_process_group=terminate
        caller=asyncio.create_task(env.exit());await entered.wait();caller.cancel()
        with self.assertRaises(asyncio.CancelledError):await caller
        del caller;gc.collect()
        self.assertEqual(env.slot_pool._active,1)
        finish.set()
        for _ in range(6):await asyncio.sleep(0)
        self.assertEqual(env.slot_pool._active,0)
        self.assertEqual(env.slot_pool._total_released,1)
        env.port_lease.release.assert_called_once();env.log_file.close.assert_called_once()
        await env.exit();self.assertEqual(env.slot_pool._total_released,1)
        self.assertFalse(m._PENDING_CLEANUPS)

    async def test_termination_exception_still_releases_ownership(self):
        env=await self.build_env()
        async def terminate():raise RuntimeError('synthetic wait error after process exit')
        env._terminate_process_group=terminate
        with self.assertRaisesRegex(RuntimeError,'synthetic'):await env.exit()
        self.assertEqual(env.slot_pool._active,0)
        env.port_lease.release.assert_called_once();env.log_file.close.assert_called_once()

    async def test_second_cancellation_during_failed_start_releases_slot(self):
        pool=m.LocalProcessSlotPool(1)
        health_started=asyncio.Event();reaping=asyncio.Event();reaped=asyncio.Event()
        proc=SimpleNamespace(returncode=None,pid=424242)
        async def wait():reaping.set();await reaped.wait();proc.returncode=-9
        proc.wait=wait
        async def health(*args,**kwargs):health_started.set();await asyncio.Event().wait()
        lease=Mock(port=18000);handle=Mock()
        with patch.object(m.LocalProcessSlotPool,'get',AsyncMock(return_value=pool)), \
             patch.object(m,'_acquire_port',return_value=lease), \
             patch.object(m,'_wait_until_healthy',health), \
             patch.object(m.asyncio,'create_subprocess_exec',AsyncMock(return_value=proc)), \
             patch.object(m.os,'makedirs'),patch('builtins.open',return_value=handle), \
             patch.object(m.os,'getpgid',return_value=424242),patch.object(m.os,'killpg'):
            task=asyncio.create_task(m.create_local_process_env({}))
            await health_started.wait();task.cancel();await reaping.wait();task.cancel()
            with self.assertRaises(asyncio.CancelledError):await task
            reaped.set()
            for _ in range(10):await asyncio.sleep(0)
        self.assertEqual(pool._active,0)
        self.assertEqual(pool._total_released,1)
        lease.release.assert_called_once();handle.close.assert_called_once()

if __name__=='__main__':unittest.main()
