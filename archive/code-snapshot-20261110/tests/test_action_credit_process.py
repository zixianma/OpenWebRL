"""Private artifact admission and actual child lifecycle; no API/GPU/browser."""
import asyncio
import json
import os
from pathlib import Path
import sys
import tempfile
from unittest import IsolatedAsyncioTestCase
from unittest.mock import patch

from openwebrl.action_credit_process import (_write, _private_file, _no_credentials,
    owned_process, _process_identity, _cleanup_detached)


class ProcessTests(IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir='/tmp')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_private_packet_checks_bytes_and_campaign_scope(self):
        ref = _write(self.root/'packet.json', {'source': 'CPU fixture'})
        self.assertEqual(_private_file(ref, self.root), self.root/'packet.json')
        with self.assertRaisesRegex(ValueError, 'exact campaign'):
            _private_file(ref, self.root/'another-campaign')
        (self.root/'packet.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'SHA256'):
            _private_file(ref, self.root)

    def test_aliased_and_nonprivate_files_are_rejected(self):
        ref = _write(self.root/'packet.json', {})
        os.chmod(ref['path'], 0o644)
        with self.assertRaisesRegex(ValueError, 'private unaliased'):
            _private_file(ref, self.root)
        os.chmod(ref['path'], 0o600)
        os.link(ref['path'], self.root/'alias')
        with self.assertRaisesRegex(ValueError, 'private unaliased'):
            _private_file(ref, self.root)

    def test_nested_secret_rejected_before_packet_persistence(self):
        _no_credentials({'approval_reference': {'path': '/tmp/approval.json'}, 'api_key': None})
        with self.assertRaisesRegex(ValueError, 'worker environment'):
            _no_credentials({'native_args': {'judge_api_key': 'CPU not a real credential'}})

    async def test_actual_child_failure_is_owned_and_does_not_retry(self):
        marker = self.root/'calls.txt'
        code = 'from pathlib import Path; import sys; Path(sys.argv[1]).write_text("one"); sys.exit(7)'
        with self.assertRaisesRegex(RuntimeError, 'exited 7'):
            await owned_process([sys.executable, '-B', '-c', code, str(marker)], cwd=self.root,
                env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'),
                log_path=self.root/'failure.log', deadline_seconds=10)
        self.assertEqual(marker.read_text(), 'one')

    async def test_timeout_reaps_child_before_return(self):
        pidfile = self.root/'pid'
        code = 'import os,time,sys; from pathlib import Path; Path(sys.argv[1]).write_text(str(os.getpid())); time.sleep(30)'
        with self.assertRaises(asyncio.TimeoutError):
            await owned_process([sys.executable, '-B', '-c', code, str(pidfile)], cwd=self.root,
                env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'),
                log_path=self.root/'timeout.log', deadline_seconds=.25)
        pid = int(pidfile.read_text())
        with self.assertRaises(ProcessLookupError):
            os.kill(pid, 0)

    def detached_worker(self, *, wait, ignore_term=False, grandchild_session=False):
        # A real detached server and grandchild, but no browser, model, or API.
        marker = self.root/'descendant-pids.json'
        grandchild = ('import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(60)'
                      if ignore_term else 'import time; time.sleep(60)')
        payload = ('import os,subprocess,sys,time,json,signal; from pathlib import Path; '
            + ('signal.signal(signal.SIGTERM,signal.SIG_IGN); ' if ignore_term else '')
            + f'child=subprocess.Popen([sys.executable,"-B","-c",{grandchild!r}],start_new_session={grandchild_session!r}); '
            'Path(sys.argv[1]).write_text(json.dumps([os.getpid(),child.pid])); time.sleep(60)')
        code = '\n'.join([
            'import asyncio,os,sys',
            'from pathlib import Path',
            'from openwebrl.action_credit_process import detached_process_scope',
            'async def run():',
            '    with detached_process_scope(os.environ["OPENWEBRL_ACTION_CREDIT_CHILD_REGISTRY"]):',
            f'        process=await asyncio.create_subprocess_exec(sys.executable,"-B","-c",{payload!r},{str(marker)!r},start_new_session=True)',
            f'        while not Path({str(marker)!r}).exists(): await asyncio.sleep(.01)',
            '        await asyncio.sleep(60)' if wait else '        return',
            'asyncio.run(run())'])
        return code, marker

    async def test_normal_exit_cleans_detached_server_and_grandchild(self):
        code, marker = self.detached_worker(wait=False)
        reference = await owned_process([sys.executable, '-B', '-c', code],
            cwd=Path(__file__).resolve().parents[1], env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'),
            log_path=self.root/'normal-detached.log', deadline_seconds=10,
            detached_registry=self.root/'registry', attempt_id='cpu-normal')
        receipt = json.loads(Path(reference['path']).read_text())
        self.assertTrue(receipt['owned_processes_closed'])
        self.assertEqual(len(receipt['detached_guardians']), 1)
        for pid in json.loads(marker.read_text()):
            identity = _process_identity(pid)
            self.assertTrue(identity is None or identity['state'] == 'Z')

    async def test_timeout_cleans_detached_startup_and_grandchild(self):
        code, marker = self.detached_worker(wait=True)
        with self.assertRaises(asyncio.TimeoutError):
            await owned_process([sys.executable, '-B', '-c', code],
                cwd=Path(__file__).resolve().parents[1], env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'),
                log_path=self.root/'timeout-detached.log', deadline_seconds=1,
                detached_registry=self.root/'registry', attempt_id='cpu-timeout')
        receipt = json.loads((self.root/'registry/cleanup.json').read_text())
        self.assertTrue(receipt['owned_processes_closed'])
        self.assertEqual(len(receipt['detached_guardians']), 1)
        for pid in json.loads(marker.read_text()):
            identity = _process_identity(pid)
            self.assertTrue(identity is None or identity['state'] == 'Z')

    async def test_birth_identity_mismatch_never_signals_foreign_process(self):
        registry = self.root/'registry'
        registry.mkdir()
        owner = dict(worker_pid=123, worker_start_ticks=456, nonce='cpu-owner')
        _write(registry/'owner.json', owner)
        actual = _process_identity(os.getpid())
        forged = dict(actual, start_ticks=actual['start_ticks']-1,
                      pgid=os.getpid(), session=os.getpid())
        _write(registry/f"child-{os.getpid()}-{forged['start_ticks']}.json",
               dict(owner, guardian=forged))
        with self.assertRaisesRegex(ValueError, 'PID was reused'):
            await _cleanup_detached(registry, owner)

    async def test_cancellation_during_spawn_still_reaps_real_worker(self):
        original = asyncio.create_subprocess_exec
        ready, release = asyncio.Event(), asyncio.Event()
        spawned = []
        async def delayed_spawn(*args, **kwargs):
            process = await original(*args, **kwargs)
            spawned.append(process)
            ready.set()
            await release.wait()
            return process
        with patch('asyncio.create_subprocess_exec', delayed_spawn):
            task = asyncio.create_task(owned_process(
                [sys.executable, '-B', '-c', 'import time; time.sleep(60)'], cwd=self.root,
                env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'),
                log_path=self.root/'cancel-start.log', deadline_seconds=10))
            await ready.wait()
            task.cancel()
            release.set()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertIsNotNone(spawned[0].returncode)
        self.assertIsNone(_process_identity(spawned[0].pid))

    async def test_guardian_force_reaps_descendants_ignoring_term(self):
        code, marker = self.detached_worker(wait=True, ignore_term=True, grandchild_session=True)
        with self.assertRaises(asyncio.TimeoutError):
            await owned_process([sys.executable, '-B', '-c', code],
                cwd=Path(__file__).resolve().parents[1], env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'),
                log_path=self.root/'forced-detached.log', deadline_seconds=1,
                detached_registry=self.root/'registry', attempt_id='cpu-forced')
        receipt = json.loads((self.root/'registry/cleanup.json').read_text())
        self.assertTrue(receipt['owned_processes_closed'])
        for pid in json.loads(marker.read_text()):
            identity = _process_identity(pid)
            self.assertTrue(identity is None or identity['state'] == 'Z')

    async def test_cancelled_worker_drains_late_guardian_registration(self):
        registering, payload = self.root/'registering', self.root/'payload-started'
        # Inject a delay only in this real CPU child process, after the guardian
        # has checked its parent but before its durable registration appears.
        guardian = '\n'.join([
            'import time', 'from pathlib import Path',
            'import openwebrl.action_credit_process as module',
            'original = module._write',
            'def delayed(path, *args, **kwargs):',
            '    if path.name.startswith("child-"):',
            f'        Path({str(registering)!r}).write_text("started")',
            '        time.sleep(.75)',
            '    return original(path, *args, **kwargs)',
            'module._write = delayed', 'module.main()'])
        proxy = '\n'.join([
            'import subprocess,sys', 'import openwebrl.action_credit_process as module',
            'original = subprocess.Popen',
            'def delayed_guardian(argv, **kwargs):',
            f'    return original([sys.executable,"-B","-c",{guardian!r},*argv[4:]],**kwargs)',
            'subprocess.Popen = delayed_guardian', 'module.main()'])
        worker = '\n'.join([
            'import asyncio,os,sys',
            'from openwebrl.action_credit_process import detached_process_scope',
            'original = asyncio.create_subprocess_exec',
            'async def proxy_spawn(*argv, **kwargs):',
            f'    return await original(sys.executable,"-B","-c",{proxy!r},*argv[4:],**kwargs)',
            'asyncio.create_subprocess_exec = proxy_spawn',
            'async def run():',
            '    with detached_process_scope(os.environ["OPENWEBRL_ACTION_CREDIT_CHILD_REGISTRY"]):',
            f'        process=await asyncio.create_subprocess_exec(sys.executable,"-B","-c",{("from pathlib import Path; Path("+repr(str(payload))+").write_text(\"bad\")")!r},start_new_session=True)',
            '        await asyncio.sleep(60)', 'asyncio.run(run())'])
        task = asyncio.create_task(owned_process([sys.executable, '-B', '-c', worker],
            cwd=Path(__file__).resolve().parents[1], env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'),
            log_path=self.root/'late-registration.log', deadline_seconds=10,
            detached_registry=self.root/'registry', attempt_id='cpu-late'))
        async with asyncio.timeout(5):
            while not registering.exists():
                if task.done():
                    await task
                await asyncio.sleep(.01)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        receipt = json.loads((self.root/'registry/cleanup.json').read_text())
        self.assertTrue(receipt['owned_processes_closed'])
        self.assertEqual(len(receipt['detached_proxies']), 1)
        self.assertEqual(len(receipt['detached_guardians']), 1)
        self.assertFalse(payload.exists())
