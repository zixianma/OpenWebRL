"""CPU integration checks for exact raw-prompt transport and upstream metrics."""
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import tempfile
import threading
import time
import unittest

from openwebrl.webprm_static_bench import run_official
from scripts.run_piotr_webprmbench import (
    LIMITS, RESOURCES, PARAMS, RawBackend, ProcessAdapter, audit_outputs, check_approval,
)

EVALUATOR = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-baselines-20261009/sources/arbiter/eval/WebPRMBench/script/evaluate_webprm_vllm.py')


class TestStaticBench(unittest.TestCase):
    def test_short_owned_ipc_path_binds_when_long_artifact_path_cannot(self):
        import shutil
        import socket
        from scripts.run_piotr_webprmbench import short_ipc_directory
        ipc = short_ipc_directory('123')
        self.addCleanup(shutil.rmtree, ipc)
        with tempfile.TemporaryDirectory(dir='/tmp') as tmp:
            long_path = Path(tmp) / ('x' * 100)
            long_path.mkdir()
            with socket.socket(socket.AF_UNIX) as sock:
                with self.assertRaises(OSError):
                    sock.bind(str(long_path/'socket'))
            with socket.socket(socket.AF_UNIX) as sock:
                sock.bind(str(ipc/'socket'))
                self.assertLessEqual(len(str(ipc/'socket').encode()), 107)

    def test_recovery_debits_failed_gpu_time_and_rejects_partial_inference(self):
        from unittest.mock import patch
        from scripts import run_piotr_webprmbench as worker
        from openwebrl.webprm_native_release import reference
        with tempfile.TemporaryDirectory(dir='/tmp') as tmp:
            root = Path(tmp)
            proof = root/'closed.json'
            value = dict(job_id='123', scheduler_state='FAILED', elapsed_seconds=24,
                         remaining_owned_processes={}, local_pair_generations=0)
            proof.write_text(json.dumps(value))
            approval = dict(prior_attempts=[dict(proof=reference(proof))], prior_gpu_seconds=24)
            with patch.object(worker, 'RUNTIME', root):
                self.assertEqual(worker.allocation_seconds(approval), 7140)
                self.assertEqual(worker.slurm_time(7140), '01:59:00')
                with self.assertRaisesRegex(ValueError, 'accounting differs'):
                    worker.allocation_seconds(dict(approval, prior_gpu_seconds=0))
                proof.write_text(json.dumps(dict(value, local_pair_generations=1)))
                with self.assertRaisesRegex(ValueError, 'proof changed'):
                    worker.allocation_seconds(approval)
                approval['prior_attempts'][0]['proof'] = reference(proof)
                with self.assertRaisesRegex(ValueError, 'zero generations'):
                    worker.allocation_seconds(approval)

    def test_real_upstream_loop_over_raw_http_and_independent_recount(self):
        calls = []
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                request = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                calls.append(request)
                # Labels are absent from the backend interface.
                self.send_response(200); self.end_headers()
                self.wfile.write(json.dumps([dict(text='<Answer>Response 1</Answer>')
                                             for _ in request['text']]).encode())
        server = HTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
        try:
            with tempfile.TemporaryDirectory(dir='/tmp') as tmp:
                folder = Path(tmp); (folder/'transport').mkdir()
                rows = [dict(source_name=s, state_idx=st, pair_idx=p, expected=1,
                             prompt=f'raw {s} {st} {p}')
                        for s, st in [('a', 0), ('b', 0), ('b', 1)] for p in range(4)]
                rows[-1]['expected'] = 2
                data = folder/'data.jsonl'
                data.write_text(''.join(json.dumps(r)+'\n' for r in rows))
                backend = RawBackend(f'http://127.0.0.1:{server.server_port}', folder/'transport', time.time()+30, lambda:None)
                run_official(EVALUATOR, model='fixture', data=data, output=folder, generate=backend, batch_size=5)
                audit = audit_outputs(data, EVALUATOR, folder)
                self.assertEqual(audit['metrics']['overall']['pairs_ok'], 11)
                self.assertEqual(audit['metrics']['overall']['state_acc'], .75)
                self.assertEqual([len(x['text']) for x in calls], [5, 5, 2])
                self.assertEqual(calls[0]['text'], [r['prompt'] for r in rows[:5]])
                self.assertEqual(set(calls[0]), {'text', 'sampling_params'})
                for mutation in ('raw', 'expected', 'state_idx'):
                    original = (folder/'pairs.jsonl').read_text()
                    saved = [json.loads(x) for x in original.splitlines()]
                    saved[0][mutation] = 'changed'
                    (folder/'pairs.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in saved))
                    with self.assertRaises(ValueError):
                        audit_outputs(data, EVALUATOR, folder)
                    (folder/'pairs.jsonl').write_text(original)
                (folder/'metrics.json').write_text('{}')
                with self.assertRaises(ValueError):
                    audit_outputs(data, EVALUATOR, folder)
        finally:
            server.shutdown(); thread.join(); server.server_close()

    def test_no_inference_retry_or_overwrite_after_transport_failure(self):
        class FailedTransport:
            def __init__(self):
                self.calls = 0
            def open(self, *args, **kwargs):
                self.calls += 1
                raise ConnectionError('synthetic failure')
        with tempfile.TemporaryDirectory(dir='/tmp') as tmp:
            backend = RawBackend('http://127.0.0.1:1', tmp, time.time()+10, lambda:None)
            backend.http = FailedTransport()
            with self.assertRaises(ConnectionError):
                backend(['raw prompt'], PARAMS)
            self.assertEqual(backend.http.calls, 1)
            self.assertEqual(backend.pairs, 1)
            self.assertTrue((Path(tmp)/'batch-0000.request.json').is_file())

    def test_caps_and_sampling_checked_before_transport(self):
        with tempfile.TemporaryDirectory(dir='/tmp') as tmp:
            backend = RawBackend('http://127.0.0.1:1', tmp, time.time()+10, lambda:None)
            with self.assertRaises(ValueError):
                backend(['a'], dict(PARAMS, temperature=1))
            backend.pairs = 4600
            with self.assertRaises(ValueError):
                backend(['a'], PARAMS)
            self.assertEqual(list(Path(tmp).iterdir()), [])
        with self.assertRaises(ValueError):
            RawBackend('https://example.org', '/tmp', time.time()+10, lambda:None)

    def test_request_is_not_an_approval(self):
        with tempfile.TemporaryDirectory(dir='/tmp') as tmp:
            path = Path(tmp)/'manifest.json'; path.write_text('{}')
            from openwebrl.webprm_native_release import reference
            approval = dict(status='approved', manifest_sha256=reference(path)['sha256'],
                            resources=RESOURCES, limits=LIMITS, job_id='123')
            check_approval(path, {}, approval, '123')
            for change in [dict(status='awaiting_exact_approval'), dict(job_id='124'),
                           dict(resources=dict(RESOURCES, seconds=14400)), dict(manifest_sha256='wrong')]:
                with self.assertRaises(ValueError):
                    check_approval(path, {}, dict(approval, **change), '123')

    def test_popen_adapter_closes_a_real_owned_child(self):
        import os
        import subprocess
        import sys
        from openwebrl.webprm_lifecycle import OwnedEpisodeTree, _processes, _subreaper
        previous = _subreaper(True)
        baseline = {pid for pid, row in _processes().items() if row['ppid'] == os.getpid()}
        child = subprocess.Popen([sys.executable, '-B', '-c', 'import time; time.sleep(30)'], start_new_session=True)
        try:
            tree = OwnedEpisodeTree(ProcessAdapter(child), baseline)
            tree.refresh()
            closure = tree.stop(.2)
            self.assertEqual(closure['remaining_owned_processes'], {})
            self.assertIsNotNone(child.poll())
        finally:
            if child.poll() is None:
                child.kill(); child.wait()
            _subreaper(previous)


if __name__ == '__main__':
    unittest.main()
