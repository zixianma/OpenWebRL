"""Real-process shared-state races and conservative accounting; no providers."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import multiprocessing
import tempfile
import time
import unittest
from unittest.mock import patch

from openwebrl.api_experiments.reproduce import shared_state as storage
from openwebrl.api_experiments.reproduce import stealth_study as study

LIMITS = dict(browser_target_usd='90', browser_admission_stop_usd='80', collection_calendar_seconds=604800)


def concurrent_writer(root, index, gate, active, maximum):
    # This remains correct when process-local flock provides no protection.
    with patch.object(study.pilot.fcntl, 'flock', return_value=None):
        control = study.GlobalControl(root)
        gate.wait(15)
        for turn in range(8):
            with control.locked() as state:
                with active.get_lock():
                    active.value += 1
                    maximum.value = max(maximum.value, active.value)
                try:
                    key = str(index) + '-' + str(turn)
                    time.sleep(.002)
                    state['creates'][key] = dict(shard=index % 2, payload_sha256=key, epoch=time.time())
                    state['sessions'][key] = dict(id=key, browserCost='0', proxyCost='.01', proxyUsedMb='1', status='stopped')
                finally:
                    with active.get_lock():
                        active.value -= 1


class SharedState(unittest.TestCase):
    def test_concurrent_spawned_writers_keep_every_admission_and_fee_without_flock(self):
        with tempfile.TemporaryDirectory(dir='/tmp') as temporary:
            root = Path(temporary) / 'shared'
            control = study.GlobalControl(root)
            control.initialize(time.time(), LIMITS)
            context = multiprocessing.get_context('spawn')
            gate, active, maximum = context.Barrier(4), context.Value('i', 0), context.Value('i', 0)
            workers = [context.Process(target=concurrent_writer, args=(root, index, gate, active, maximum)) for index in range(4)]
            try:
                for worker in workers:
                    worker.start()
                for worker in workers:
                    worker.join(30)
                    self.assertEqual(worker.exitcode, 0)
            finally:
                for worker in workers:
                    if worker.is_alive():
                        worker.terminate()
                    worker.join()
            with control.locked(readonly=True) as state:
                self.assertEqual(len(state['creates']), 32)
                self.assertEqual(len(state['sessions']), 32)
                self.assertEqual(study.GlobalControl.total(state), study.money('.32'))
            self.assertEqual(maximum.value, 1)

    def test_atomic_replace_uses_unique_temps_for_threads_and_complete_reader_records(self):
        with tempfile.TemporaryDirectory(dir='/tmp') as temporary:
            path = Path(temporary) / 'state.json'
            storage.atomic_write(path, dict(writer=0, body='0' * 10000))
            def writer(index):
                for _ in range(10):
                    storage.atomic_write(path, dict(writer=index, body=str(index) * 10000))
            with ThreadPoolExecutor(max_workers=4) as pool:
                futures = [pool.submit(writer, index) for index in range(4)]
                while not all(future.done() for future in futures):
                    row = json.loads(path.read_text())
                    self.assertEqual(row['body'], str(row['writer']) * 10000)
                for future in futures:
                    future.result()
            self.assertFalse(list(path.parent.glob('*.tmp')))

    def test_existing_lock_times_out_without_stale_takeover(self):
        with tempfile.TemporaryDirectory(dir='/tmp') as temporary:
            root = Path(temporary)
            lock = root / 'financial.namespace-lock'
            lock.mkdir()
            with self.assertRaisesRegex(TimeoutError, 'no stale takeover'):
                with storage.namespace_lock(root, timeout=.02):
                    self.fail('Foreign holder was bypassed')
            self.assertTrue(lock.is_dir())

    def test_replace_does_not_follow_a_destination_symlink_into_an_archive(self):
        with tempfile.TemporaryDirectory(dir='/tmp') as temporary:
            root = Path(temporary)
            archived = root / 'archive.json'
            archived.write_text('{"original":true}\n')
            destination = root / 'output.json'
            destination.symlink_to(archived)
            storage.atomic_write(destination, {'fresh': True})
            self.assertEqual(archived.read_text(), '{"original":true}\n')
            self.assertFalse(destination.is_symlink())
            self.assertEqual(json.loads(destination.read_text()), {'fresh': True})

    def fixture(self, root):
        control = study.GlobalControl(root)
        control.initialize(time.time(), LIMITS)
        with control.locked() as state:
            state['creates']['owned'] = dict(payload_sha256='same', shard=0)
            state['sessions']['owned'] = dict(id='native-owned', browserCost='1', proxyCost='2', proxyUsedMb='3', status='stopped')
        return control

    def test_ownership_caps_clock_and_fee_regressions_never_commit(self):
        changes = [lambda s:s['creates'].clear(), lambda s:s['sessions'].clear(),
                   lambda s:s['creates']['owned'].update(payload_sha256='changed'),
                   lambda s:s['sessions']['owned'].update(id='foreign'),
                   lambda s:s['sessions']['owned'].update(proxyCost='1'),
                   lambda s:s['sessions']['owned'].update(browserCost='NaN'),
                   lambda s:s['limits'].update(browser_target_usd='91'),
                   lambda s:s.update(deadline=s['deadline'] + 1),
                   lambda s:s.update(first_start=s['first_start'] + 1),
                   lambda s:s['sessions'].update(foreign=dict(id='x', browserCost='0', proxyCost='0', proxyUsedMb='0'))]
        for change in changes:
            with self.subTest(change=change), tempfile.TemporaryDirectory(dir='/tmp') as temporary:
                root = Path(temporary)
                control = self.fixture(root)
                before = (root / 'financial.json').read_bytes()
                with self.assertRaises(ValueError):
                    with control.locked() as state:
                        change(state)
                self.assertEqual((root / 'financial.json').read_bytes(), before)

    def test_halt_cannot_be_cleared_or_replaced(self):
        with tempfile.TemporaryDirectory(dir='/tmp') as temporary:
            root = Path(temporary)
            control = self.fixture(root)
            control.halt('unknown provider usage')
            before = (root / 'financial.json').read_bytes()
            for value in (None, dict(reason='different', epoch=0)):
                with self.assertRaises(ValueError):
                    with control.locked() as state:
                        state['abort'] = value
                self.assertEqual((root / 'financial.json').read_bytes(), before)

    def test_outside_writer_is_not_overwritten(self):
        with tempfile.TemporaryDirectory(dir='/tmp') as temporary:
            root = Path(temporary)
            control = self.fixture(root)
            with self.assertRaisesRegex(ValueError, 'outside its namespace lock'):
                with control.locked() as state:
                    outside = dict(json.loads(json.dumps(state)), abort=dict(reason='outside writer', epoch=0))
                    storage.atomic_write(root / 'financial.json', outside)
                    state['creates']['uncommitted'] = {}
            self.assertEqual(study.read(root / 'financial.json'), outside)

    def test_fresh_earlier_start_only_tightens_without_resetting_ownership_or_halt(self):
        with tempfile.TemporaryDirectory(dir='/tmp') as temporary:
            control = self.fixture(Path(temporary))
            control.halt('preserve me')
            with control.locked(readonly=True) as state:
                before = dict(state)
            deadline = control.initialize(before['first_start'] - 10, LIMITS)
            with control.locked(readonly=True) as state:
                self.assertEqual(state['creates'], before['creates'])
                self.assertEqual(state['sessions'], before['sessions'])
                self.assertEqual(state['abort'], before['abort'])
                self.assertEqual(deadline, before['deadline'] - 10)


if __name__ == '__main__':
    unittest.main()
